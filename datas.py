from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPTJForCausalLM, DataCollatorWithPadding
import datasets
from ml_utils.utils import try_key
import torch
import os

def owt_autoencode(examples, tokenizer, max_seq_len=20):
    """
    Simply predict the input sequence. Uses openwebtext.

    Inputs:
        examples: ?
            whatever huggingface uses when mapping an encoding function
            to a dataset
        tokenizer: huggingface tokenizer
        max_seq_len: int
            the length of the compression
    """
    tokenizer.padding_side = "left"
    cmps = tokenizer(
        examples["text"],
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt"
    )
    seqs = {
        "input_ids":      cmps["input_ids"].clone(),
        "attention_mask": cmps["attention_mask"].clone()
    }
    return {
        "input_ids":        cmps["input_ids"],
        "attention_mask":   cmps["attention_mask"],
        "output_ids":       seqs["input_ids"],
        "output_attn_mask": seqs["attention_mask"]
    }


def owt_causal_encode(examples, tokenizer, cmp_len=20, seq_len=100,
                                                       overlap=0,
                                                       min_seq=5,
                                                       model=None):
    """
    Output tokens are the continuation of a sequence of seq_len. Inputs
    are the starting cmp_len tokens of the sequence of len seq_len.

    Args:
        examples: ?
            whatever huggingface uses when mapping an encoding function
            to a dataset
        tokenizer: huggingface tokenizer
        cmp_len: int
            the length of the compression
        seq_len: int
            the length of the post compression, sequence chunk that
            will be processed by the transformer. So, the total
            predictive sequence will be seq_len+cmp_len tokens long.
        min_seq: int
            the minimum length predictive portion. total sequence
            lengths must be greater than or equal to cmp_len+min_seq
        overlap: int
            the number of overlapping tokens from the compression
            sequence to the rest of the sequence.
        model: SentenceAutoEncoder or None
            Will use model to generate output tokens. If None, ignored
    """
    tokenizer.padding_side = "right"
    cmps = tokenizer(
        examples["text"],
        padding="max_length",
        max_length=cmp_len+seq_len,
        truncation=True,
        return_tensors="pt"
    )
    if model is not None:
        device = model.get_device()
        with torch.no_grad():
            output_ids, output_logits = model.causal_lm(
                input_ids=cmps["input_ids"].to(device),
                attention_mask=cmps["attention_mask"].to(device),
                tforce=True,
                ret_logits=True
            )
        idx = seq_len + overlap
        seqs = {
            "output_ids": cmps["input_ids"][:,-idx:],
            "output_logits": output_logits[:,-idx:].data,
            "output_attn_mask": cmps["attention_mask"][:,-idx:]
        }
    else:
        seqs = {
          "output_ids": cmps["input_ids"][:, cmp_len-overlap:],
          "output_attn_mask":cmps["attention_mask"][:,cmp_len-overlap:],
        }
    cmps["input_ids"] = cmps["input_ids"][:,:cmp_len]
    cmps["attention_mask"] = cmps["attention_mask"][:,:cmp_len]

    return {**cmps, **seqs}

def get_loaders(hyps, tokenizer, model=None, val_only=False):
    """
    Use this function to get the training and validation loaders.

    Args:
        model: SentenceAutoEncoder
        val_only: bool
            if true, only returns test split
    """
    hyps["data_root"] = try_key(
        hyps,"data_root",hyps["save_root"]+"datasplits"
    )
    dset = hyps["dataset"]
    if hyps.get("gen_targs", False):
        dset += "modelgen"
        model.eval()
        hyps["n_data_procs"] = 1
    else: model = None

    # Name data path with number of samples
    path = os.path.join(hyps["data_root"],dset)
    abbrev = hyps.get("abbrev_len", None)
    save_threshold = 100000
    if abbrev is not None and abbrev>=save_threshold:
        if abbrev>=1000000:
            path = path + str(abbrev//1000000)+"m"
        elif abbrev>=1000:
            path = path + str(abbrev//1000)+"k"
    if hyps["cmp_len"]!=10 or hyps["seq_len"]!=20:
        path=path+"cmpr{}seq{}".format(hyps["cmp_len"],hyps["seq_len"])
    # Default is bigscience/bloomz-560m, so we only add if not that
    if hyps["model_string"]!="bigscience/bloomz-560m":
        path = path + hyps["model_string"].replace("/","")
    if not os.path.exists(path): os.makedirs(path)

    # Load previously saved data
    trpath = os.path.join(path,"train")
    if (abbrev is None or abbrev>=100000) and os.path.exists(trpath):
        if hyps.get("rank",0)==0:
            print("Loading data from", trpath)
        dataset = datasets.load_from_disk(trpath)
        val_path = os.path.join(path, "val")
        if hyps.get("rank",0)==0:
            print("Loading data from", val_path)
        valset = datasets.load_from_disk(val_path)
    # Make new data from glue
    elif hyps["dataset"]=="glue":
        encode_fxn = lambda x: pair_encode(
            x,
            tokenizer,
            max_seq_len=hyps["cmp_len"]
        )
        dataset = datasets.load_dataset(
            "glue","mrpc",split="train",
            cache_dir=try_key(hyps,"data_cache",None)
        )
        dataset = dataset.map(
            encode_fxn,
            batched=True,
            remove_columns=["sentence1", "sentence2", "idx", "label"]
        )
        dataset.set_format(type="torch")
        dataset.save_to_disk(os.path.join(path, "train"))
        if not val_only:
            valset = datasets.load_dataset(
                "glue", "mrpc", split="validation",
                cache_dir=try_key(hyps,"data_cache",None)
            )
            valset = valset.map(encode_fxn, batched=True)
            valset = valset.remove_columns(
                ["sentence1", "sentence2", "idx", "label"]
            )
            valset.save_to_disk(os.path.join(path, "val"))
    # Make new data from openwebtext
    elif hyps["dataset"]=="openwebtext":
        if hyps.get("rank",0)==0:
            print("Failed to find", trpath, "Manually loading dataset")
        if try_key(hyps,"rmb_only",False):
            if hyps.get("rank",0)==0: print("RMB Only")
            encode_fxn = lambda x: owt_autoencode(
                x,
                tokenizer=tokenizer,
                max_seq_len=hyps["cmp_len"]
            )
        else:
            encode_fxn = lambda x: owt_causal_encode(
                x,
                tokenizer=tokenizer,
                cmp_len=hyps["cmp_len"],
                seq_len=hyps["seq_len"],
                overlap=try_key(hyps,"seq_overlap",0),
                model=model
            )
        dataset = datasets.load_dataset(
            "openwebtext", split="train",
            cache_dir=try_key(hyps,"data_cache",None),
            num_proc=try_key(hyps,"n_data_procs",4)
        )
        dataset = dataset.shuffle()
        abrv = hyps.get("abbrev_len", 300)
        if hyps["exp_name"]=="test" or (abrv is not None and abrv>0):
            if abrv is None: abrv = 600
            dataset = dataset[:abrv]
            dataset = datasets.Dataset.from_dict(dataset)
        if hyps.get("rank",0)==0:
            print("Mapping Encoder Function")
        bsize = 1000 if model is None else\
                try_key(hyps,"data_batch_size",100)
        dataset = dataset.map(
            encode_fxn,
            batched=True,
            num_proc=try_key(hyps,"n_data_procs",4),
            remove_columns=["text"],
            batch_size=bsize
        )
        if not val_only:
            test_size = int(len(dataset)*.2)
            splt = dataset.train_test_split(test_size=test_size)
            dataset, valset = splt["train"], splt["test"]
            if abbrev is None or abbrev>=save_threshold:
                dataset.save_to_disk(os.path.join(path, "train"))
                valset.save_to_disk( os.path.join(path, "val")  )

    dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=hyps["batch_size"], shuffle=True
    )
    if not val_only:
        valset.set_format(type="torch")
        vsize = try_key(hyps, "val_batch_size", hyps["batch_size"])
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=vsize, shuffle=True
        )
        return dataset, valset, dataloader, valloader
    return dataset, dataloader

def glue_encode(examples, tokenizer, max_seq_len=100):
    tokenizer.padding_side = "left"
    inpts = tokenizer(
        examples["sentence1"],
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt"
    )
    outs = tokenizer(
        examples["sentence2"],
        padding="max_length",
        max_length=max_seq_len,
        truncation=True ,
        return_tensors="pt"
    )

    # If sentences are not semantically equivalent, use duplicate
    # sentence, and switch cls to eos
    examples["label"] = torch.LongTensor(examples["label"])
    idx = examples["label"]==0
    outs["input_ids"][idx] = inpts["input_ids"][idx].clone()
    outs["attention_mask"][idx] = inpts["attention_mask"][idx].clone()

    return {
        "input_ids":        inpts["input_ids"],
        "attention_mask":   inpts["attention_mask"],
        "output_ids":       outs["input_ids"],
        "output_attn_mask": outs["attention_mask"],
        "labels":           examples["label"],
    }


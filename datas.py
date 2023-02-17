from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPTJForCausalLM, DataCollatorWithPadding
import datasets
from ml_utils.utils import try_key
import torch

def owt_autoencode(examples, tokenizer, max_seq_len=20, cmp_token=None):
    """
    Simply predict the input sequence, bottlenecking through a single
    cls token. Uses openwebtext.

    Inputs:
        examples: ?
            whatever huggingface uses when mapping an encoding function
            to a dataset
        tokenizer: huggingface tokenizer
        max_seq_len: int
            the length of the compression
        cmp_token: str or None
            the compression token. if none, defaults to tokenizer
            cls_token
    """
    if cmp_token is None:
        cmp_token = tokenizer.cls_token
    cmp_id = tokenizer.encode(cmp_token)[0]
    cmps = tokenizer(
        examples["text"],
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt"
    )
    cmps["input_ids"][:,-1] = cmp_id
    cmps["attention_mask"][:,-1] = 1

    # Copy inputs and replace cmp token
    seqs = {
        "input_ids":      cmps["input_ids"].clone(),
        "attention_mask": cmps["attention_mask"].clone()
    }
    idx = seqs["input_ids"]==cmp_id
    seqs["input_ids"][idx] = tokenizer.eos_token_id

    return {
        "input_ids":        cmps["input_ids"],
        "attention_mask":   cmps["attention_mask"],
        "output_ids":       seqs["input_ids"],
        "output_attn_mask": seqs["attention_mask"]
    }


def owt_causal_encode(examples, tokenizer, cmp_len=20, seq_len=100,
                                                       min_seq=5,
                                                       cmp_token=None):
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
            the total length of the entire sequence chunk that will be
            processed by the transformer. So, the predictive sequence
            will be seq_len-cmp_len tokens long.
        min_seq: int
            the minimum length predictive portion. total sequence
            lengths must be greater than or equal to cmp_len+min_seq
        cmp_token: str or None
            the compression token. if none, defaults to tokenizer
            cls_token
    """
    if cmp_token is None:
        cmp_token = tokenizer.cls_token
    cmp_id = tokenizer.encode(cmp_token)[0]
    #cmp_list = []
    #seq_list = []
    #for s in examples["text"]:
    #    if len(s) >= cmp_len+min_seq:
    #        cmp_list.append(s[:cmp_len] + cmp_token)
    #        seq_list.append(s[cmp_len:seq_len])
    cmps = tokenizer(
        examples["text"],
        padding="max_length",
        max_length=cmp_len+seq_len,
        truncation=True,
        return_tensors="pt"
    )
    cmps["input_ids"][:,cmp_len-1] = cmp_id
    cmps["attention_mask"][:,cmp_len-1] = 1
    seqs = {
        "input_ids": cmps["input_ids"][:, cmp_len:],
        "attention_mask": cmps["attention_mask"][:, cmp_len:],
    }
    cmps["input_ids"] = cmps["input_ids"][:,:cmp_len]
    cmps["attention_mask"] = cmps["attention_mask"][:,:cmp_len]

    return {
        "input_ids":        cmps["input_ids"],
        "attention_mask":   cmps["attention_mask"],
        "output_ids":       seqs["input_ids"],
        "output_attn_mask": seqs["attention_mask"]
    }

def get_loaders(hyps, tokenizer):
    """
    Use this function to get the training and validation loaders.
    """
    if hyps["dataset"]=="glue":
        encode_fxn = lambda x: pair_encode(
            x,
            tokenizer,
            max_seq_len=hyps["cmp_len"],
            cmp_token=hyps["CMP_TOKEN"]
        )
        dataset = datasets.load_dataset("glue", "mrpc", split="train")
        dataset = dataset.map(encode_fxn, batched=True)
        dataset = dataset.remove_columns(
            ["sentence1", "sentence2", "idx", "label"]
        )
        dataset.set_format(type="torch")
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=hyps["batch_size"], shuffle=True
        )
        valset = datasets.load_dataset("glue", "mrpc", split="validation")
        valset = valset.map(encode_fxn, batched=True)
        valset = valset.remove_columns(
            ["sentence1", "sentence2", "idx", "label"]
        )
    elif hyps["dataset"]=="openwebtext":
        if try_key(hyps,"rmb_only",False):
            print("RMB Only")
            encode_fxn = lambda x: owt_autoencode(
                x,
                tokenizer=tokenizer,
                max_seq_len=hyps["cmp_len"],
                cmp_token=hyps["CMP_TOKEN"]
            )
        else:
            encode_fxn = lambda x: owt_causal_encode(
                x,
                tokenizer=tokenizer,
                cmp_len=hyps["cmp_len"],
                seq_len=hyps["seq_len"],
                cmp_token=hyps["CMP_TOKEN"]
            )
        dataset = datasets.load_dataset("openwebtext", split="train")
        if hyps["exp_name"]=="test" or try_key(hyps,"abbrev_data",False):
            dataset = dataset[:try_key(hyps,"abbrev_len",300)]
            dataset = datasets.Dataset.from_dict(dataset)
        dataset = dataset.map(encode_fxn, batched=True)
        dataset = dataset.remove_columns( ["text"] )
        dataset = dataset.shuffle()
        test_size = int(len(dataset)*.2)
        splt = dataset.train_test_split(test_size=test_size)
        dataset, valset = splt["train"], splt["test"]

    dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=hyps["batch_size"], shuffle=True
    )
    valset.set_format(type="torch")
    vsize = try_key(hyps, "val_batch_size", hyps["batch_size"])
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=vsize, shuffle=True
    )
    return dataset, valset, dataloader, valloader

def glue_encode(examples, tokenizer, max_seq_len=100, cmp_token=None):
    if cmp_token is None:
        cmp_token = tokenizer.cls_token
    cmp_id = tokenizer.encode(cmp_token)[0]
    sent1 = [s + cmp_token for s in examples["sentence1"]]
    inpts = tokenizer(
        sent1,
        padding="max_length",
        max_length=max_seq_len+1,
        truncation=True,
        return_tensors="pt"
    )
    sent2 = [s + tokenizer.eos_token for s in examples["sentence2"]]
    outs = tokenizer(
        sent2,
        padding="max_length",
        max_length=max_seq_len+1,
        truncation=True ,
        return_tensors="pt"
    )

    # Need to do some funny business in cases of truncation
    idx = inpts["input_ids"]==cmp_id
    idx = (idx).float().reshape(len(idx),-1).sum(-1)
    if idx.sum() != len(idx):
        inpts["input_ids"][idx==0,-1] = cmp_id

    # If sentences are not semantically equivalent, use duplicate
    # sentence, and switch cls to eos
    examples["label"] = torch.LongTensor(examples["label"])
    idx = examples["label"]==0
    outs["input_ids"][idx] = inpts["input_ids"][idx].clone()
    outs["attention_mask"][idx] = inpts["attention_mask"][idx].clone()
    idx = outs["input_ids"]==cmp_id
    outs["input_ids"][idx] = tokenizer.eos_token_id

    # Need to do some funny business in cases of truncation
    # This should largely be unnecessary because we already handled it
    # with the inputs, but we still need to worry about the cases when
    # the outputs are not equal to the inputs
    idx = outs["input_ids"]==tokenizer.eos_token_id
    idx = (idx).float().reshape(len(idx),-1).sum(-1)
    if idx.sum() != len(idx):
        outs["input_ids"][idx==0,-1] = tokenizer.eos_token_id

    return {
        "input_ids":        inpts["input_ids"],
        "attention_mask":   inpts["attention_mask"],
        "output_ids":       outs["input_ids"],
        "output_attn_mask": outs["attention_mask"],
        "labels":           examples["label"],
    }


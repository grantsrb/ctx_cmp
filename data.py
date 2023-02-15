from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPTJForCausalLM, DataCollatorWithPadding
from datasets import load_dataset

def autoencode(examples, tokenizer, cmp_len=20):
    """
    examples: ?
        whatever huggingface uses when mapping an encoding function to
        a dataset
    tokenizer: huggingface tokenizer
    cmp_len: int
        the length of the compression
    """
    cmps = []
    for s in examples["text"]:
        cmps.append(s[:cmp_len] + tokenizer.cls_token)
    cmps = tokenizer(
        cmps,
        padding="max_length",
        max_length=cmp_len,
        truncation=True,
        return_tensors="pt"
    )

    # Need to do some funny business in cases of truncation
    idx = cmps["input_ids"]==tokenizer.cls_token_id
    idx = (idx).float().reshape(len(idx),-1).sum(-1)
    if idx.sum() != len(idx):
        cmps["input_ids"][idx==0,-1] = tokenizer.cls_token_id

    # Copy inputs and replace cls token
    seqs = {
        "input_ids": cmps["input_ids"].clone(),
        "attention_mask": cmps["attention_mask"].clone()
    }
    idx = seqs["input_ids"]==tokenizer.cls_token_id
    seqs["input_ids"][idx] = tokenizer.eos_token_id

    return {
        "input_ids":        cmps["input_ids"],
        "attention_mask":   cmps["attention_mask"],
        "output_ids":       seqs["input_ids"],
        "output_attn_mask": seqs["attention_mask"]
    }

def encode(examples, tokenizer, cmp_len=20, seq_len=100, min_seq=5):
    """
    examples: ?
        whatever huggingface uses when mapping an encoding function to
        a dataset
    tokenizer: huggingface tokenizer
    cmp_len: int
        the length of the compression
    seq_len: int
        the total length of the entire sequence chunk that will be
        processed by the transformer. So, the predictive sequence will
        be seq_len-cmp_len tokens long.
    min_seq: int
        the minimum length predictive portion. total sequence lengths
        must be greater than or equal to cmp_len+min_seq
    """
    cmps = []
    seqs = []
    for s in examples["text"]:
        if len(s) >= cmp_len+min_seq:
            cmps.append(s[:cmp_len] + tokenizer.cls_token)
            seqs.append(s[cmp_len:seq_len] + tokenizer.eos_token)
    cmps = tokenizer(
        cmps,
        padding="max_length",
        max_length=cmp_len,
        truncation=True,
        return_tensors="pt"
    )
    seqs = tokenizer(
        seqs,
        padding="max_length",
        max_length=seq_len-cmp_len,
        truncation=True,
        return_tensors="pt"
    )

    # Need to do some funny business in cases of truncation
    idx = cmps["input_ids"]==tokenizer.cls_token_id
    idx = (idx).float().reshape(len(idx),-1).sum(-1)
    if idx.sum() != len(idx):
        cmps["input_ids"][idx==0,-1] = tokenizer.cls_token_id

    idx = seqs["input_ids"]==tokenizer.eos_token_id
    idx = (idx).float().reshape(len(idx),-1).sum(-1)
    if idx.sum() != len(idx):
        seqs["input_ids"][idx==0,-1] = tokenizer.eos_token_id

    return {
        "input_ids":        cmps["input_ids"],
        "attention_mask":   cmps["attention_mask"],
        "output_ids":       seqs["input_ids"],
        "output_attn_mask": seqs["attention_mask"]
    }

if __name__=="__main__":
    dataset = load_dataset("openwebtext", split="train")
    print(dataset)
    try:
        print(dataset[0])
    except:
        print("exception")
        print(dataset.keys())
    assert False

    dataset = dataset.map(encode_fxn, batched=True)
    dataset = dataset.remove_columns(
        ["sentence1", "sentence2", "idx", "label"]
    )
    dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bsize, shuffle=True
    )
    valset = load_dataset("glue", "mrpc", split="validation")
    valset = valset.map(encode_fxn, batched=True)
    valset = valset.remove_columns(
        ["sentence1", "sentence2", "idx", "label"]
    )
    valset.set_format(type="torch")
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=val_bsize, shuffle=True
    )

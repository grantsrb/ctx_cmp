import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPTJForCausalLM, DataCollatorWithPadding
from datasets import load_dataset
from ml_utils.utils import try_key
import ml_utils
import time

def encode(examples, tokenizer, max_seq_len=100):
    sent1 = [s + tokenizer.cls_token for s in examples["sentence1"]]
    inpts = tokenizer(sent1,padding="max_length", max_length=max_seq_len, truncation=True, return_tensors="pt")
    sent2 = [s + tokenizer.eos_token for s in examples["sentence2"]]
    outs = tokenizer(sent2,padding="max_length", max_length=max_seq_len, truncation=True , return_tensors="pt")

    examples["label"] = torch.LongTensor(examples["label"])
    idx = examples["label"]==0

    outs["input_ids"][idx] = inpts["input_ids"][idx].clone()
    outs["attention_mask"][idx] = inpts["attention_mask"][idx].clone()
    outs["input_ids"][outs["input_ids"]==tokenizer.cls_token_id] = tokenizer.eos_token_id

    # Need to do some funny business in cases of truncation
    idx = (inpts["input_ids"]==tokenizer.cls_token_id).float().reshape(len(idx),-1).sum(-1)
    if idx.sum() != len(idx):
        i = torch.argmax((idx==0).float())
        inpts["input_ids"][i,-1] = tokenizer.cls_token_id
    idx = (outs["input_ids"]==tokenizer.eos_token_id).float().reshape(len(examples["label"]),-1).sum(-1)
    if idx.sum() != len(examples["label"]):
        i = torch.argmax((idx==0).float())
        outs["input_ids"][i,-1] = tokenizer.eos_token_id
    
    return {
        "input_ids":        inpts["input_ids"],
        "attention_mask":   inpts["attention_mask"],
        "output_ids":       outs["input_ids"],
        "output_attn_mask": outs["attention_mask"],
        "labels":           examples["label"],
    }

class SentenceAutoEncoder(torch.nn.Module):
    """
    Trains a new token type to compress a sentence into a single vector representation
    """
    def __init__(self, model, tokenizer, *args, **kwargs):
        """
        model: hugging face transformer model
        """
        super().__init__()
        self.hface_model = model
        self.tokenizer = tokenizer
        self.CLS_ID = tokenizer.cls_token_id
        self.CLS = tokenizer.cls_token
        self.EOS_ID = tokenizer.eos_token_id
        self.EOS = tokenizer.eos_token

    def forward(self, data):
        """
        data: dict
            "input_ids": LongTensor (B,S1)
                the token indices of the input sequence. The CLS token should be appended to the end of each sentence.
            "attention_mask": LongTensor (B,S1)
                attention mask for padding purposes. 0s mean padding.
            "output_ids": LongTensor (B,S2)
                the token indices of the target sequence. An EOS token should be appended to the end of each sentence
            "output_attn_mask": LongTensor (B,S2)
                attention mask for padding purposes. 0s mean padding.
        """
        model = self.hface_model
        inpt_embs = model.transformer.wte(data["input_ids"]).data
        idx = data["input_ids"]==self.CLS_ID
        inpt_embs[idx] = 0
        inpt_embs[idx] += model.transformer.wte.weight[self.CLS_ID]
        out_embs =  model.transformer.wte(data["output_ids"]).data

        fx = model.transformer(
            inputs_embeds=inpt_embs,
            attention_mask=data["attention_mask"]
        )
        fx = fx["last_hidden_state"][idx][:,None]

        out_embs = torch.cat([fx,out_embs], dim=1)
        attn = torch.cat([
          torch.ones_like(data["output_attn_mask"][:,:1]),
          data["output_attn_mask"]
        ], dim=1)

        preds = model(inputs_embeds=out_embs,attention_mask=attn)
        return preds.logits

def train(rank, hyps, verbose=True, *args, **kwargs):
    if hyps["exp_name"]=="test": 
        model_string = "hf-internal-testing/tiny-random-gptj"
    else:
        model_string = hyps["model_string"]
    max_seq_len = hyps["max_seq_len"]
    bsize = hyps["batch_size"]
    lr = hyps["lr"]
    l2 = hyps["l2"]
    n_epochs = hyps["n_epochs"]
    hyps["seed"] = try_key(hyps, "seed", int(time.time()))
    if hyps["seed"] is None: hyps["seed"] = int(time.time())
    torch.manual_seed(hyps["seed"])

    tokenizer = GPT2Tokenizer.from_pretrained(model_string)
    hface_model = GPTJForCausalLM.from_pretrained(model_string)

    dataset = load_dataset("glue", "mrpc", split="train")
    new_tokens = {
        "pad_token": "[PAD]",
        "cls_token": "[CLS]", # Using CLS token as compression token
    }
    num_added = tokenizer.add_special_tokens(new_tokens)
    n,h = hface_model.transformer.wte.weight.shape
    hface_model.resize_token_embeddings(n+num_added)

    encode_fxn = lambda x: encode(x, tokenizer, max_seq_len)
    dataset = dataset.map(encode_fxn, batched=True)
    dataset = dataset.remove_columns(
        ["sentence1", "sentence2", "idx", "label"]
    )
    dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bsize, shuffle=True
    )

    model = SentenceAutoEncoder(hface_model, tokenizer)
    ml_utils.training.record_session(hyps, model)

    optimizer = torch.optim.Adam(
        model.hface_model.transformer.wte.parameters(),
        lr=lr,
        weight_decay=l2
    )
    loss_fxn = torch.nn.CrossEntropyLoss()

    model.cuda()
    for epoch in range(n_epochs):
        print()
        print("Beginning Epoch", epoch)
        model.train()
        avg_loss = 0
        avg_acc = 0
        for i,data in enumerate(dataloader):
            data = {k: v.cuda() for k,v in data.items()}
            preds = model(data)
            preds = preds[:,:-1].reshape(-1, preds.shape[-1])
            labels = data["output_ids"].reshape(-1)
            idx = labels!=tokenizer.pad_token_id
            loss = loss_fxn(preds[idx], labels[idx])
            avg_loss += loss.item()
            argmax = torch.argmax(preds[idx])
            acc = (argmax==labels[idx]).float().mean()
            avg_acc += acc.item()
            if i%10==0:
                l = round(loss.item(), 5)
                a = round(acc.item(), 5)
                print("Loss", l, "-- Acc:", a)
        l = round(avg_loss/i, 5)
        a = round(avg_acc/i, 5)
        print("Avg Loss:", l, "-- Avg Acc:", a)
        print()



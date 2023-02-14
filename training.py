import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPTJForCausalLM, DataCollatorWithPadding
from datasets import load_dataset

from ml_utils.utils import try_key
import ml_utils

def encode(examples, tokenizer, max_seq_len=100):
    sent1 = [s + tokenizer.cls_token for s in examples["sentence1"]]
    inpts = tokenizer(
        sent1,
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt"
    )
    sent2 = [s + tokenizer.eos_token for s in examples["sentence2"]]
    outs = tokenizer(
        sent2,
        padding="max_length",
        max_length=max_seq_len,
        truncation=True ,
        return_tensors="pt"
    )

    # Need to do some funny business in cases of truncation
    idx = inpts["input_ids"]==tokenizer.cls_token_id
    idx = (idx).float().reshape(len(idx),-1).sum(-1)
    if idx.sum() != len(idx):
        inpts["input_ids"][idx==0,-1] = tokenizer.cls_token_id

    # If sentences are not semantically equivalent, use duplicate
    # sentence, and switch cls to eos
    examples["label"] = torch.LongTensor(examples["label"])
    idx = examples["label"]==0
    outs["input_ids"][idx] = inpts["input_ids"][idx].clone()
    outs["attention_mask"][idx] = inpts["attention_mask"][idx].clone()
    idx = outs["input_ids"]==tokenizer.cls_token_id
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

class SentenceAutoEncoder(torch.nn.Module):
    """
    Trains a new token type to compress a sentence into a single vector
    representation
    """
    def __init__(self, model_string, rank=0, torch_dtype="float32",
                                             device_map="auto",
                                             *args, **kwargs):
        """
        model_string: str
            name of pretrained hugging face transformer model
        rank: int
            rank within distributed training
        """
        super().__init__()
        self.model_string = model_string
        if torch_dtype=="float32": torch_dtype = torch.float32
        elif torch_dtype=="float16": torch_dtype = torch.float16
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_string,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        self.rank = rank

    def forward(self, data):
        """
        data: dict
            "input_ids": LongTensor (B,S1)
                the token indices of the input sequence. The CLS token
                should be appended to the end of each sentence.
            "attention_mask": LongTensor (B,S1)
                attention mask for padding purposes. 0s mean padding.
            "output_ids": LongTensor (B,S2)
                the token indices of the target sequence. An EOS token
                should be appended to the end of each sentence
            "output_attn_mask": LongTensor (B,S2)
                attention mask for padding purposes. 0s mean padding.
        """
        model = self.hf_model
        model_embs = model.transformer.get_input_embeddings()
        inpt_embs = model_embs( data["input_ids"] ).data
        idx = data["input_ids"]==self.CLS_ID
        inpt_embs[idx] = 0
        inpt_embs[idx] += model_embs.weight[self.CLS_ID]
        out_embs =  model_embs(data["output_ids"]).data
        idx = data["output_ids"]==self.EOS_ID
        out_embs[idx] = 0
        out_embs[idx] += model_embs.weight[self.EOS_ID]

        fx = model.transformer(
            inputs_embeds=inpt_embs,
            attention_mask=data["attention_mask"]
        )
        early_fx_shape = fx["last_hidden_state"].shape
        fx = fx["last_hidden_state"][idx][:,None]

        # Concat compressed representation to beginning of sentence
        attn = torch.cat(
            [torch.ones_like(data["output_attn_mask"][:,:1]),
            data["output_attn_mask"]], dim=1
        )
        #attn = torch.pad(data["output_attn_mask"], (1,0))
        try:
            out_embs = torch.cat(
                [fx.to(self.rank),out_embs.to(self.rank)],
                dim=1
            )
            attn = torch.cat(
                [torch.ones_like(data["output_attn_mask"][:,:1]),
                data["output_attn_mask"]], dim=1
            ).to(self.rank)
        except:
            print("Data")
            for k in data: print(k, data[k].shape)
            print("idx sum:", idx.float().sum())
            print("In Embds", inpt_embs.shape)
            print("Out Embds", out_embs.shape)
            print("early FX", early_fx_shape)
            print("FX", fx.shape)
            assert False
        
        preds = model(inputs_embeds=out_embs, attention_mask=attn).logits
        return preds

class LossWrapper(torch.nn.Module):
    """
    This class wraps the model to keep the loss calculations distributed
    on all GPUs. Otherwise one gpu is overloaded with computational
    costs.
    """
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fxn = torch.nn.CrossEntropyLoss()

    def forward(self, data):
        """
        data: dict
            "input_ids": LongTensor (B,S1)
                the token indices of the input sequence. The CLS token
                should be appended to the end of each sentence.
            "attention_mask": LongTensor (B,S1)
                attention mask for padding purposes. 0s mean padding.
            "output_ids": LongTensor (B,S2)
                the token indices of the target sequence. An EOS token
                should be appended to the end of each sentence
            "output_attn_mask": LongTensor (B,S2)
                attention mask for padding purposes. 0s mean padding.
        """
        preds = self.model(data)
        preds = preds[:,:-1].reshape(-1, preds.shape[-1])
        labels = data["output_ids"].reshape(-1)

        idx = labels!=self.tokenizer.pad_token_id
        loss = self.loss_fxn(preds[idx], labels[idx])
        if self.training:
            loss.backward()
        argmax = torch.argmax(preds[idx])
        acc = (argmax==labels[idx]).float().mean()
        return loss, acc

def train(rank, hyps, verbose=True, *args, **kwargs):
    # Distributed Set Up
    torch.cuda.empty_cache()
    hyps["multi_gpu"] = try_key(hyps, "multi_gpu", False)
    if hyps["multi_gpu"]:
        world_size = try_key(hyps, "n_gpus", 1)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Hyperparameters
    if hyps["exp_name"]=="test": 
        hyps["model_string"] = hyps["testing"]
        print("turning 'save' to false")
        hyps["save"] = False
    model_string = hyps["model_string"]
    max_seq_len = hyps["max_seq_len"]
    bsize = hyps["batch_size"]
    val_bsize = try_key(hyps, "val_batch_size", bsize)
    lr = hyps["lr"]
    l2 = hyps["l2"]
    n_epochs = hyps["n_epochs"]
    hyps["seed"] = try_key(hyps, "seed", int(time.time()))
    if hyps["seed"] is None: hyps["seed"] = int(time.time())
    torch.manual_seed(hyps["seed"]*rank)

    kwargs = {
        "model_string": model_string,
        "rank": rank,
        "torch_dtype": hyps["torch_dtype"],
        "device_map": "auto" if hyps["model_parallel"] else None
    }
    model = SentenceAutoEncoder(**kwargs)
    if hyps["multi_gpu"]: ddp_model = DDP(model, device_ids=[rank])
    else: ddp_model = model

    # Make Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    new_tokens = {
        "pad_token": "<PAD>",
        "cls_token": "<CLS>", # Using CLS token as compression token
    }
    num_added = tokenizer.add_special_tokens(new_tokens)

    # Adjust Model Embeddings for new token types
    if hyps["multi_gpu"]: model = ddp_model.model
    else: model = ddp_model
    embs = model.hf_model.transformer.get_input_embeddings()
    n,h = embs.weight.shape
    model.hf_model.resize_token_embeddings(n+num_added)
    model.tokenizer = tokenizer
    model.CLS_ID = model.tokenizer.cls_token_id
    model.CLS =    model.tokenizer.cls_token
    model.EOS_ID = model.tokenizer.eos_token_id
    model.EOS =    model.tokenizer.eos_token

    # Make dataset
    encode_fxn = lambda x: encode(x, tokenizer, max_seq_len)
    dataset = load_dataset("glue", "mrpc", split="train")
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

    if rank==0: ml_utils.training.record_session(hyps, model)

    # Wrap model to distribute loss calculations
    wrapped_model = LossWrapper(ddp_model, tokenizer)
    if not hyps["model_parallel"]:
        wrapped_model.to(rank)
    #wrapped_model.to(rank)
    # Mayber better to parallelize after wrap, unsure at this point
    #ddp_model = DDP(wrapped_model, device_ids=[rank])

    # This first line is crucial, otherwise referencing stale embs
    embs = model.hf_model.transformer.get_input_embeddings()
    optimizer = torch.optim.Adam(
        embs.parameters(),
        lr=lr,
        weight_decay=l2
    )
    # Turn off gradient calculations for everything except for the
    # embedding matrix.
    params = set(embs.parameters())
    for name, p in ddp_model.named_parameters():
        if p not in params:
            p.requires_grad = False

    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        if rank==0 and verbose:
            print()
            print("Beginning Epoch", epoch, "--", hyps["save_folder"])
        ddp_model.train()
        avg_loss = 0
        avg_acc = 0
        optimizer.zero_grad()
        for i,data in enumerate(dataloader):
            data = {k: v.to(rank) for k,v in data.items()}

            loss, acc = wrapped_model(data)

            avg_acc += acc.item()
            avg_loss += loss.item()

            if i%hyps["n_grad_loops"]==0 or i==len(dataloader)-1:
                optimizer.step()
                optimizer.zero_grad()

            if i%10==0 and rank==0 and verbose:
                l = round(loss.item(), 5)
                a = round(acc.item(), 5)
                c = round(100*i/len(dataloader), 2)
                s = "Loss: {} -- Acc: {} -- {}%".format( l,a,c )
                print(s, end="         " + len(s)*" " + "\r")
            if hyps["exp_name"]=="test" and i>=30: break
        train_loss = round(avg_loss/i, 5)
        train_acc = round(avg_acc/i, 5)
        if rank==0 and verbose:
            print("Avg Loss:", train_loss, "-- Avg Acc:", train_acc)
            print("Validating...")

        # Validation
        avg_loss = 0
        avg_acc = 0
        if rank==0:
            wrapped_model.eval()
            with torch.no_grad():
                for i,data in enumerate(valloader):
                    data = {k: v.to(rank) for k,v in data.items()}
                    loss, acc = wrapped_model(data)

                    avg_loss += loss.item()
                    avg_acc += acc.item()
                    if hyps["exp_name"]=="test" and i>=3: break
            keys = list(data.keys())
            for k in keys: del data[k]
            val_loss = round(avg_loss/i, 5)
            val_acc = round(avg_acc/i, 5)
            if rank==0 and verbose:
                print("Val Loss:", val_loss, "-- Val Acc:", val_acc)
                print()

            optimizer.zero_grad()
            save_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss": val_loss,
                "val_acc":  val_acc,
                "state_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
                "hyps": hyps,
            }
            if try_key(hyps, "save", True):
                ml_utils.save_io.save_checkpt(
                    save_dict=save_dict,
                    save_folder=hyps["save_folder"],
                    save_name="checkpt",
                    epoch=epoch,
                    ext=".pt"
                )
            else: print("NOT SAVING MODEL!!!!")
        if hyps["exp_name"]=="test" and epoch==2: break

    if hyps["multi_gpu"]: dist.destroy_process_group()



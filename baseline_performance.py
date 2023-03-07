import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import pandas as pd
import time
from transformers import AutoTokenizer
from tqdm import tqdm
import sys

from ml_utils.utils import try_key
import ml_utils
import datas
from models import *
from training import print_examples

hyps = {
    "exp_name": "_",
    "results_file": "./baseline_results.csv",
    "save_root": "/mnt/fs0/grantsrb/sa_saves/",
    "data_root": "/mnt/fs0/grantsrb/datasplits/",
    "data_cache": None,
    "model_string": "gpt2",
    "_model_string": "bigscience/bloomz-560m",
    "model_parallel": False,
    "torch_dtype": "float32",
    "seed": 123456,

    "dataset": "openwebtext",
    "abbrev_len": 1000,
    "data_batch_size": 1000,

    "batch_size": 50,

    "cmp_len": 15,
    "seq_len": 20,
    "seq_overlap": 0,
}

def get_metrics(hyps, model, inpts, loss_fxn, seed_len=3, tforce=True):
    """
    Calculates the loss and accuracy using causal language modeling

    Args:
        hyps: dict
        model: SentenceAutoEncoder
        inpts: dict {str: tensor}
            "input_ids": tensor (B,S)
            "attention_mask": tensor (B,S)
                this is a padding mask. 1's mean non-padded input. 0's
                mean token is padding
        loss_fxn: torch loss function
        tforce: bool
            if true, predictions are teacher forced
    """
    # Make predictions
    preds, logits = model.causal_lm(
      **inpts, tforce=tforce, ret_logits=True, seed_len=seed_len
    )
    logits = logits[:, seed_len:]
    # Calculate loss
    loss, acc = loss_and_acc(
        logits, inpts["input_ids"][:,seed_len+1:],
        attn=inpts["attention_mask"][:,seed_len+1:],
        loss_fxn=loss_fxn,
        loss_scale=1
    )
    return loss, acc

if __name__=="__main__":
    rank = 0
    # Hyperparameters
    if len(sys.argv)>1:
        hyps = ml_utils.save_io.load_json(sys.argv[1])
        hyps["results_file"] = "./baseline_results.csv"
        hyps["abbrev_len"] = 1000
        print(hyps)
    rank = 0
    verbose = True
    hyps["seed"] = hyps.get("seed", int(time.time()))
    if hyps["seed"] is None: hyps["seed"] = int(time.time())
    torch.manual_seed(hyps["seed"])

    hyps["device_map"] = "auto" if hyps["model_parallel"] else None
    model = SentenceAutoEncoder(**hyps)
    model.to(rank)
    model.eval()

    # Make Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hyps["model_string"])
    tokenizer.truncation_side = "right"

    # Add important tokens
    num_added = 0
    if tokenizer.pad_token is None:
        num_added += tokenizer.add_special_tokens(
            {"pad_token": hyps.get("pad_token", tokenizer.eos_token)}
        )
        # Adjust Model Embeddings for new token types
        model.add_embeddings(num_added)

    # Make dataset
    if verbose and rank==0:
        print("Collecting Data")
    dataset, dataloader = datas.get_loaders(
        hyps,
        tokenizer,
        val_only=True
    )


    loss_fxn = torch.nn.CrossEntropyLoss()
    avgs = {
        "tlow_loss": 0,
        "low_loss": 0,
        "thigh_loss": 0,
        "high_loss": 0,
        "tlow_acc": 0,
        "low_acc": 0,
        "thigh_acc": 0,
        "high_acc": 0,
    }
    for i,data in tqdm(enumerate(dataloader)):
        data = {k: v.to(rank) for k,v in data.items()}
        with torch.no_grad():
            low_inpts = {
                "input_ids": data["output_ids"],
                "attention_mask": data["output_attn_mask"],
            }
            model.train()
            tlow_loss, tlow_acc = get_metrics(
              hyps, model, low_inpts, loss_fxn,
              tforce=True, seed_len=0
            )
            avgs["tlow_loss"] += tlow_loss.item()
            avgs["tlow_acc"] += tlow_acc.item()

            model.eval()
            low_loss, low_acc = get_metrics(
                hyps, model, low_inpts, loss_fxn,
                tforce=False, seed_len=max(hyps.get("seq_overlap",3),3)
            )
            avgs["low_loss"] += low_loss.item()
            avgs["low_acc"] +=  low_acc.item()

            high_inpts = {
              "input_ids": torch.cat([
                data["input_ids"],data["output_ids"]
              ], dim=1),
              "attention_mask": torch.cat([ 
                data["attention_mask"], data["output_attn_mask"]
              ], dim=1)
            }
            model.train()
            thigh_loss, thigh_acc = get_metrics(
              hyps, model, high_inpts, loss_fxn, tforce=True, seed_len=0
            )
            avgs["thigh_loss"] += thigh_loss.item()
            avgs["thigh_acc"] += thigh_acc.item()

            model.eval()
            high_loss, high_acc = get_metrics(
                hyps, model, high_inpts, loss_fxn, tforce=False,
                seed_len=data["input_ids"].shape[1]
            )
            avgs["high_loss"] += high_loss.item()
            avgs["high_acc"] +=  high_acc.item()
    for k,v in avgs.items():
        avgs[k] = v/len(dataloader)
    print("TFrce Low Loss: {} -- Acc: {}".format(
        round(avgs['tlow_loss'],dec), round(avgs['tlow_acc'],dec)
    ))
    print("Low Loss: {} -- Acc: {}".format(
        round(avgs['low_loss'],dec), round(avgs['low_acc'],dec)
    ))
    print("TFrce High Loss: {} -- Acc: {}".format(
        round(avgs['thigh_loss'],dec), round(avgs['thigh_acc'],dec)
    ))
    print("High Loss: {} -- Acc: {}".format(
        round(avgs['high_loss'],dec), round(avgs['high_acc'],dec)
    ))

    for k,v in avgs.items():
        avgs[k] = [v]
    df = pd.DataFrame(avgs)
    for k,v in hyps.items():
        try:
            df[k] = v
        except: print("error for", k)
    df.to_csv(hyps["results_file"], mode="a")


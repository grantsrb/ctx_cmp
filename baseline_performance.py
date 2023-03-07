import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
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
    "results_path": "./baseline_results.csv",
    "model_string": "gpt2",
    "_model_string": "bigscience/bloomz-560m",
    "model_parallel": False,
    "torch_dtype": "float32",
    "seed": 123456,

    "dataset": "openwebtext",
    "abbrev_len": 1000,

    "batch_size": 50,

    "cmp_len": 15,
    "seq_len": 20,
    "seq_overlap": 0,
}

if __name__=="__main__":
    # Hyperparameters
    if len(sys.argv)>1:
        hyps = ml_utils.save_io.load_json(sys.argv[1])
        hyps["results_path"] = "./baseline_results.csv"
        hyps["abbrev_len"] = 1000
        print(hyps)
    rank = 0
    verbose = True
    hyps["seed"] = hyps.get("seed", int(time.time()))
    if hyps["seed"] is None: hyps["seed"] = int(time.time())
    torch.manual_seed(hyps["seed"])

    hyps["device_map"] = "auto" if hyps["model_parallel"] else None
    model = SentenceAutoEncoder(**hyps)

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
    rank = 0
    avgs = {
        "tlow_loss": 0,
        "low_loss": 0,
        "thigh_loss": 0,
        "high_loss": 0,
    }
    for i,data in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            low_inpts = {
                "input_ids": data["output_ids"].to(rank),
                "attention_mask": data["output_attn_mask"].to(rank),
            }
            tlow_loss, tlow_acc = get_metrics(
                hyps, model, low_inpts, loss_fxn, tforce=True
            )
            avgs["tlow_loss"] += tlow_loss.item()
            avgs["tlow_acc"] += tlow_acc.item()

            low_loss, low_acc = get_metrics(
                hyps, model, low_inpts, loss_fxn, tforce=False
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
            thigh_loss, thigh_acc = get_metrics(
                hyps, model, high_inpts, loss_fxn, tforce=True
            )
            avgs["thigh_loss"] += thigh_loss.item()
            avgs["thigh_acc"] += thigh_acc.item()

            high_loss, high_acc = get_metrics(
                hyps, model, high_inpts, loss_fxn, tforce=False
            )
            avgs["high_loss"] += high_loss.item()
            avgs["high_acc"] +=  high_acc.item()
    for k,v in avgs.items():
        avgs[k] = v/len(dataloader)
    print("TFrce Low Loss: {} -- Acc: {}".format(
        avgs['tlow_loss'], avgs['tlow_acc']
    ))
    print("Low Loss: {} -- Acc: {}".format(
        avgs['low_loss'], avgs['low_acc']
    ))
    print("TFrce High Loss: {} -- Acc: {}".format(
        avgs['thigh_loss'], avgs['thigh_acc']
    ))
    print("High Loss: {} -- Acc: {}".format(
        avgs['high_loss'], avgs['high_acc']
    ))

    for k,v in avgs.items():
        avgs[k] = [v]
    df = pd.DataFrame(avgs)
    for k,v in hyps.items():
        try:
            df[k] = v
        except: print("error for", k)
    df.to_csv(hyps["save_file"])


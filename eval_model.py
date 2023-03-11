"""
This script is used to evaluate trained models. Argue a model folder
or a path to a specific checkpoint. The results will be saved to a
csv called model_results.csv unless otherwise specified.

$ python3 eval_model.py path/to/model_folder

Or:

$ python3 eval_model.py path/to/model_checkpt.pt

If you would like to run the untrained model to see the baseline
performance, either use the `baseline_performance.py` script or
include `untrained` in the bash command.

WARNING!!! THE FOLLOWING LINE IS FOR BASELINE RESULTS:
$ python3 eval_model.py path/to/model_folder untrained
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import pandas as pd
import time
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
import os

from ml_utils.utils import try_key
import ml_utils.save_io as io
import datas
from models import *
from training import print_examples

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
    verbose = True
    results_file = "model_results.csv"
    abbrev_len = 1000
    untrained = False # Detemines if model should load saved checkpt
    bsize = None # Determines batch size of evaluation

    path = sys.argv[1]
    for arg in sys.argv[1:]:
        if arg=="untrained": untrained = True
        elif io.is_model_folder(arg): path = arg
        else:
            try:
                bsize = int(arg)
            except:
                print("Unrecognized arg", arg)
    checkpt = io.load_checkpoint(path)
    hyps = checkpt["hyps"]
    if abbrev_len is not None: hyps["abbrev_len"] = abbrev_len
    hyps["results_file"] = results_file
    hyps["seed"] = hyps.get("seed", int(time.time()))
    if hyps["seed"] is None: hyps["seed"] = int(time.time())
    torch.manual_seed(hyps["seed"])
    hyps["loss_scale"] = 1./hyps["n_grad_loops"]
    hyps["csl_task"] = False # CSL is handled by high_loss and high_acc
    if bsize is not None:
        hyps["batch_size"] = bsize
        hyps["val_batch_size"] = bsize

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

    hyps["device_map"] = "auto" if hyps["model_parallel"] else None
    model = SentenceAutoEncoder(**hyps)
    if not untrained: model.load_state_dict(checkpt["state_dict"])

    # Wrap model and place on gpu
    wrapped_model = LossWrapper( model, tokenizer, hyps=hyps )
    if not hyps["model_parallel"]: wrapped_model.to(rank)

    # Make dataset
    if verbose and rank==0:
        print("Collecting Data", hyps["dataset"], hyps["abbrev_len"])

    dataset, valset, dataloader, valloader = datas.get_loaders(
        hyps,
        tokenizer,
    )
    abrv = hyps.get("abbrev_len", None)
    if abrv is not None and abrv<100000:
        valset = dataset
        valloader = dataloader

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

        "tpred_loss": 0,
        "pred_loss": 0,
        "tpred_acc": 0,
        "pred_acc": 0,

        "trmb_loss": 0,
        "rmb_loss": 0,
        "trmb_acc": 0,
        "rmb_acc": 0,
    }
    for i,data in tqdm(enumerate(dataloader)):
        if not hyps["model_parallel"]:
            data = {k: v.to(rank) for k,v in data.items()}
        with torch.no_grad():
            wrapped_model.train()
            model.train()

            # Low Training Type
            low_inpts = {
                "input_ids": data["output_ids"],
                "attention_mask": data["output_attn_mask"],
            }
            tlow_loss, tlow_acc = get_metrics(
              hyps, model, low_inpts, loss_fxn,
              tforce=True, seed_len=0
            )
            avgs["tlow_loss"] += tlow_loss.item()*hyps["loss_scale"]
            avgs["tlow_acc"] += tlow_acc.item()

            # Regular Model Training Type
            package = wrapped_model(
                data,
                ret_preds=True,
                seq_len=hyps["seq_len"],
                tforce=True,
                gen_ids=try_key(hyps, "gen_ids", False),
                no_grad=True
            )
            avgs["tpred_loss"] = package["loss"].item()
            avgs["tpred_acc"] = package["acc"].item()
            if "rmb_loss" in package:
                avgs["trmb_loss"] += package["rmb_loss"].item()
                avgs["trmb_acc"]  += package["rmb_acc"].item()

            # High Train Type
            high_inpts = {
              "input_ids": torch.cat([
                data["input_ids"],data["output_ids"]
              ], dim=1),
              "attention_mask": torch.cat([ 
                data["attention_mask"], data["output_attn_mask"]
              ], dim=1)
            }
            thigh_loss, thigh_acc = get_metrics(
              hyps, model, high_inpts, loss_fxn, tforce=True, seed_len=0
            )
            avgs["thigh_loss"] += thigh_loss.item()*hyps["loss_scale"]
            avgs["thigh_acc"] += thigh_acc.item()


            if valloader==dataloader:
                wrapped_model.eval()
                model.eval()
                # Low Eval Type
                low_loss, low_acc = get_metrics(
                  hyps, model, low_inpts, loss_fxn,
                  tforce=False, seed_len=max(hyps.get("seq_overlap",1),1)
                )
                avgs["low_loss"] += low_loss.item()*hyps["loss_scale"]
                avgs["low_acc"] +=  low_acc.item()
                # Regular Model Eval Type
                package = wrapped_model(
                    data,
                    ret_preds=True,
                    seq_len=hyps["seq_len"],
                    tforce=False,
                    gen_ids=try_key(hyps, "gen_ids", False),
                    no_grad=True
                )
                avgs["pred_loss"] = package["loss"].item()
                avgs["pred_acc"] = package["acc"].item()
                if "rmb_loss" in package:
                    avgs["rmb_loss"] += package["rmb_loss"].item()
                    avgs["rmb_acc"]  += package["rmb_acc"].item()

                # High Eval Type
                high_loss, high_acc = get_metrics(
                    hyps, model, high_inpts, loss_fxn, tforce=False,
                    seed_len=data["input_ids"].shape[1]
                )
                avgs["high_loss"] += high_loss.item()*hyps["loss_scale"]
                avgs["high_acc"] +=  high_acc.item()
        if (i+1) > hyps.get("n_train_loops", np.inf): break
    for k,v in avgs.items():
        avgs[k] = round(v/(i+1), 4)
    print("TFrce Low Loss: {} -- Acc: {}".format(
        avgs['tlow_loss'], avgs['tlow_acc']
    ))
    print("TFrce Pred Loss: {} -- Acc: {}".format(
        avgs['tpred_loss'], avgs['tpred_acc']
    ))
    if hyps["rmb_task"]:
        print("TFrce RMB Loss: {} -- Acc: {}".format(
            avgs['trmb_loss'], avgs['trmb_acc']
        ))
    print("TFrce High Loss: {} -- Acc: {}".format(
        avgs['thigh_loss'], avgs['thigh_acc']
    ))

    if valloader==dataloader:
        print("Low Loss: {} -- Acc: {}".format(
            avgs['low_loss'], avgs['low_acc']
        ))
        print("Pred Loss: {} -- Acc: {}".format(
            avgs['pred_loss'], avgs['pred_acc']
        ))
        if hyps["rmb_task"]:
            print("RMB Loss: {} -- Acc: {}".format(
                avgs['rmb_loss'], avgs['rmb_acc']
            ))
        print("High Loss: {} -- Acc: {}".format(
            avgs['high_loss'], avgs['high_acc']
        ))

    if valloader!=dataloader:
        print("Using Separate Validation Loader")
        for i,data in tqdm(enumerate(valloader)):
            data = {k: v.to(rank) for k,v in data.items()}
            with torch.no_grad():
                model.eval()
                wrapped_model.eval()

                # Low Eval Type
                low_inpts = {
                    "input_ids": data["output_ids"],
                    "attention_mask": data["output_attn_mask"],
                }
                low_loss, low_acc = get_metrics(
                  hyps, model, low_inpts, loss_fxn,
                  tforce=False, seed_len=max(hyps.get("seq_overlap",1),1)
                )
                avgs["low_loss"] += low_loss.item()*hyps["loss_scale"]
                avgs["low_acc"] +=  low_acc.item()

                # Regular Model Eval Type
                package = wrapped_model(
                    data,
                    ret_preds=True,
                    seq_len=hyps["seq_len"],
                    tforce=False,
                    gen_ids=try_key(hyps, "gen_ids", False),
                    no_grad=True
                )
                avgs["pred_loss"] = package["loss"].item()
                avgs["pred_acc"] = package["acc"].item()
                if "rmb_loss" in package:
                    avgs["rmb_loss"] += package["rmb_loss"].item()
                    avgs["rmb_acc"]  += package["rmb_acc"].item()

                # High Eval Type
                high_inpts = {
                  "input_ids": torch.cat([
                    data["input_ids"],data["output_ids"]
                  ], dim=1),
                  "attention_mask": torch.cat([ 
                    data["attention_mask"], data["output_attn_mask"]
                  ], dim=1)
                }
                high_loss, high_acc = get_metrics(
                    hyps, model, high_inpts, loss_fxn, tforce=False,
                    seed_len=data["input_ids"].shape[1]
                )
                avgs["high_loss"] += high_loss.item()*hyps["loss_scale"]
                avgs["high_acc"] +=  high_acc.item()
            if (i+1) > hyps.get("max_val_loops", np.inf): break
        for k,v in avgs.items():
            if "t"!=k[0]:
                avgs[k] = round(v/(i+1), 4)
        print("Low Loss: {} -- Acc: {}".format(
            avgs['low_loss'], avgs['low_acc']
        ))
        print("Pred Loss: {} -- Acc: {}".format(
            avgs['pred_loss'], avgs['pred_acc']
        ))
        if hyps["rmb_task"]:
            print("RMB Loss: {} -- Acc: {}".format(
                avgs['rmb_loss'], avgs['rmb_acc']
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
    if os.path.exists(hyps["results_file"]):
        og_df = pd.read_csv(hyps["results_file"])
        df = og_df.append(df, sort=True)
    df.to_csv(hyps["results_file"], mode="w", index=False, header=True)


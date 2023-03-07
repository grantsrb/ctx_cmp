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
    "save_path": "./baseline_results.csv",
    "model_string": "bigscience/bloomz-560m",
    "model_parallel": False,
    "torch_dtype": "float32",
    "seed": 123456,

    "dataset": "openwebtext",
    "abbrev_len": 300,

    "batch_size": 6,

    "cmp_len": 15,
    "seq_len": 30,
    "seq_overlap": 3,
}

if __name__=="__main__":
    # Hyperparameters
    rank = 0
    verbose = True
    model_string = hyps["model_string"]
    hyps["seed"] = try_key(hyps, "seed", int(time.time()))
    if hyps["seed"] is None: hyps["seed"] = int(time.time())
    torch.manual_seed(hyps["seed"])

    kwargs = {
        "model_string": model_string,
        "rank": rank,
        "torch_dtype": hyps["torch_dtype"],
        "device_map": "auto" if hyps["model_parallel"] else None,
        "rmb_task": False
    }
    model = SentenceAutoEncoder(**kwargs)

    # Make Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    tokenizer.truncation_side = "right"

    # Add important tokens
    num_added = 0
    if tokenizer.pad_token is None:
        num_added += tokenizer.add_special_tokens(
            {"pad_token": "|<PAD>|"}
        )
    hyps["pad_token"] = tokenizer.pad_token

    # Adjust Model Embeddings for new token types
    model.add_embeddings(num_added)

    # Make dataset
    if verbose and rank==0:
        print("Collecting Data")
    dataset, valset, dataloader, valloader = datas.get_loaders(
        hyps,
        tokenizer
    )



    for i,data in enumerate(dataloader):
        print()
        print("Input Ids")
        print_examples(
            data["input_ids"],
            data["output_ids"],
            tokenizer,
            n_samps=5
        )
        print()
        print("Output Ids")
        print_examples(
            data["output_ids"],
            data["output_ids"],
            tokenizer,
            n_samps=5
        )
        assert False



    # Wrap model to distribute loss calculations
    if verbose and rank==0:
        print("Wrapping Model")
    wrapped_model = LossWrapper(
        model,
        tokenizer,
        hyps=hyps,
        loss_scale=1/hyps["n_grad_loops"]
    )
    if not hyps["model_parallel"]:
        if verbose and rank==0:
            print("Putting Model On GPU")
        wrapped_model.to(rank)

    if verbose:
        print("Collecting Baseline Scores...")
    with torch.no_grad():
        wrapped_model.eval()
        val_baselines = collect_baselines(
            hyps,
            model,
            valloader,
            wrapped_model.loss_fxn,
            tforce=False,
            seed_len=try_key(hyps,"infr_seed_len",3),
            rank=rank
        )
        torch.cuda.empty_cache()
        wrapped_model.train()
        train_baselines = collect_baselines(
            hyps,
            model,
            dataloader,
            wrapped_model.loss_fxn,
            tforce=True,
            seed_len=try_key(hyps,"infr_seed_len",3),
            rank=rank
        )
        torch.cuda.empty_cache()
    if verbose and rank==0:
        print("Training Baseline Predictions")
        print("Low Preds")
        examples = print_examples(
            train_baselines["labels"],
            train_baselines["low_preds"],
            tokenizer
        )
        print()
        print("High Preds")
        examples = print_examples(
            train_baselines["labels"],
            train_baselines["high_preds"],
            tokenizer
        )
        print()
        print("Validation Baseline Predictions")
        print("Low Preds")
        examples = print_examples(
            val_baselines["labels"],
            val_baselines["low_preds"],
            tokenizer
        )
        print()
        print("High Preds")
        examples = print_examples(
            val_baselines["labels"],
            val_baselines["high_preds"],
            tokenizer
        )
        print()

def collect_baselines(hyps, model, loader, loss_fxn, tforce=False,
                                                     rank=0,
                                                     seed_len=3,
                                                     verbose=True):
    """
    This function handles producing baseline predictions from the model
    The lower predictions come from simply predicting the sequence
    without any previous representations. The upper bound comes from
    predicting the sequence with the complete previous context.

    Args:
        model: SentenceAutoEncoder
            must have member variable `hf_model`
        loader: torch DataLoader
        loss_fxn: torch loss function
        tforce: bool
            if true, model will use teacher forcing
        rank: int
            the rank of the process if using data_parallel. otherwise
            the device number
        seed_len: int
            the number of inputs to seed the lower bound
            predictions.
    """
    hf_model = model.hf_model
    avg_high_loss = 0
    avg_low_loss = 0
    avg_high_acc = 0
    avg_low_acc = 0
    with torch.no_grad():
        nloops = try_key(hyps,"max_val_loops",1000)
        for i,data in enumerate(loader):
            low_inpts = {
                "input_ids": data["output_ids"].to(rank),
                "attention_mask": data["output_attn_mask"].to(rank),
            }
            og_low_preds =  model.causal_lm(
                **low_inpts , tforce=tforce, seed_len=seed_len
            )
            low_preds = og_low_preds[:,seed_len-1:-1]
            idx = data["output_attn_mask"][:,seed_len:].bool().to(rank)
            low_preds = low_preds[idx]
            for k,v in low_inpts.items(): low_inpts[k] = v.cpu()
            torch.cuda.empty_cache()

            high_inpts = {
                "input_ids": torch.cat(
                  [
                    data["input_ids"].to(rank),
                    data["output_ids"].to(rank)
                  ],
                  dim=1
                ),
                "attention_mask": torch.cat(
                    [
                        data["attention_mask"].to(rank),
                        data["output_attn_mask"].to(rank)
                    ],
                    dim=1
                )
            }

            og_high_preds = model.causal_lm(
                **high_inpts,
                tforce=tforce,
                seed_len=data["input_ids"].shape[1]+seed_len
            )
            startx = data["input_ids"].shape[1]
            og_high_preds = og_high_preds[:,startx:]
            for k,v in high_inpts.items(): high_inpts[k] = v.cpu()
            high_preds = og_high_preds[:,seed_len-1:-1]
            high_preds = high_preds[idx]

            labels = data["output_ids"][:, seed_len:].to(rank)[idx]

            low_loss = loss_fxn(low_preds, labels)
            high_loss = loss_fxn(high_preds, labels)

            low_acc =  (low_preds.argmax(-1)==labels).float().mean()
            high_acc = (high_preds.argmax(-1)==labels).float().mean()

            avg_low_loss +=  low_loss.item()
            avg_high_loss += high_loss.item()
            avg_low_acc +=  low_acc.item()
            avg_high_acc += high_acc.item()

            if hyps["exp_name"]=="test" and i>=3: break
            if nloops is not None and i>nloops: break
            if i%25==0 and verbose:
                print(str(round(i/len(loader)*100))+"%", end="     \r")
    return {
        "low_loss": round(avg_low_loss/i, 5),
        "low_acc":  round(avg_low_acc/i, 5),
        "high_loss": round(avg_high_loss/i, 5),
        "high_acc":  round(avg_high_acc/i, 5),
        "low_preds":  og_low_preds.cpu(),
        "high_preds": og_high_preds.cpu(),
        "labels":     data["output_ids"].cpu()
    }



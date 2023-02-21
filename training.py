import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import time

from transformers import AutoTokenizer

from ml_utils.utils import try_key
import ml_utils
import datas
from models import *

RMB = "|<RMB>|" # Extra characters are to ensure uniqueness
CMP = "|<CMP{}>|"
SOS = "|<SOS>|"


def train(rank, hyps, verbose=True, *args, **kwargs):
    # Distributed Set Up
    torch.cuda.empty_cache()
    hyps["multi_gpu"] = try_key(hyps, "multi_gpu", False)
    if hyps["multi_gpu"]:
        world_size = try_key(hyps, "n_gpus", 1)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Hyperparameters
    if hyps["exp_name"]=="test": hyps["model_string"] = hyps["testing"]
    model_string = hyps["model_string"]
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
        "device_map": "auto" if hyps["model_parallel"] else None,
        "rmb_task": try_key(hyps, "rmb_task", False),
    }
    model = SentenceAutoEncoder(**kwargs)
    if hyps["multi_gpu"]: ddp_model = DDP(model, device_ids=[rank])
    else: ddp_model = model

    # Make Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    tokenizer.truncation_side = "right"
    token_names = {
        "RMB": RMB,
        "SOS": SOS,
        "CMPS": [CMP.format(i) for i in range(hyps["n_cmps"])]
    }
    new_tokens = {
      "additional_special_tokens": [RMB, SOS] + token_names["CMPS"]
    }
    num_added = tokenizer.add_special_tokens(new_tokens)
    if tokenizer.pad_token is None:
        num_added += tokenizer.add_special_tokens(
            {"pad_token": "|<PAD>|"}
        )
    if tokenizer.eos_token is None:
        num_added += tokenizer.add_special_tokens(
            { "eos_token": "|<EOS>|" }
        )
    token_names["PAD"] = tokenizer.pad_token
    token_names["EOS"] = tokenizer.eos_token
    print("Tokenizer Keys:")
    print("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("BOS:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("PAD:", tokenizer.pad_token, tokenizer.pad_token_id)

    # Adjust Model Embeddings for new token types
    if hyps["multi_gpu"]: model = ddp_model.model
    else: model = ddp_model
    model.add_embeddings(num_added)

    # Add token names and ids to model and hyps
    model.add_attrs(token_names)
    token_ids = {}
    for k,v in token_names.items():
        if type(v)==type([]):
            token_ids[k[:-1]+"_IDS"] = [
                int(tokenizer.encode(x)[0]) for x in v
            ]
        else:
            token_ids[k+"_ID"] = int(tokenizer.encode(v)[0])
    model.add_attrs(token_ids)
    hyps = {**hyps, "token_names": token_names, "token_ids": token_ids}
    print("Token Names:", token_names)
    print("Token IDs:", token_ids)

    # Make dataset
    print("Collecting Data")
    dataset, valset, dataloader, valloader = datas.get_loaders(
        hyps,
        tokenizer
    )

    print("Recording Session")
    if rank==0: ml_utils.training.record_session(hyps, model)

    # Wrap model to distribute loss calculations
    print("Wrapping Model")
    wrapped_model = LossWrapper(
        ddp_model,
        tokenizer,
        loss_scale=1/hyps["n_grad_loops"]
    )
    if not hyps["model_parallel"]:
        print("Putting Model On GPU")
        wrapped_model.to(rank)
    # Mayber better to parallelize after wrap, unsure at this point
    #ddp_model = DDP(wrapped_model, device_ids=[rank])

    # This line is crucial, otherwise you will reference stale embs
    print("Creating Optimizer")
    embs = model.hf_model.transformer.get_input_embeddings()
    optimizer = torch.optim.Adam(
        embs.parameters(),
        lr=lr,
        weight_decay=l2
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, threshold=0.001
    )
    # Turn off gradient calculations for everything except for the
    # embedding matrix.
    print("Turning Off Gradients")
    params = set(embs.parameters())
    for name, p in ddp_model.named_parameters():
        if p not in params:
            p.requires_grad = False

    print("Beginning Training")
    for epoch in range(n_epochs):
        print("Emptying Trash")
        torch.cuda.empty_cache()
        if rank==0 and verbose:
            print()
            print("Beginning Epoch", epoch, "--", hyps["save_folder"])
        wrapped_model.train()
        avg_loss = 0
        avg_acc = 0
        rmb_avg_loss = 0
        rmb_avg_acc = 0
        nloops = try_key(hyps,"n_train_loops", None)
        nloops = np.inf if nloops is None else nloops
        checkpt_mod = try_key(hyps, "checkpt_mod", None)
        checkpt_mod = np.inf if checkpt_mod is None else checkpt_mod
        optimizer.zero_grad()
        for i,data in enumerate(dataloader):
            starttime = time.time()
            data = {k: v.to(rank) for k,v in data.items()}

            package = wrapped_model(data,ret_preds=True)
            loss = package["loss"]
            acc = package["acc"]

            avg_acc += acc.item()
            avg_loss += loss.item()
            if "rmb_loss" in package:
                rmb_avg_loss += package["rmb_loss"].item()
                rmb_avg_acc  += package["rmb_acc"].item()

            if i%hyps["n_grad_loops"]==0 or i==len(dataloader)-1:
                optimizer.step()
                optimizer.zero_grad()

            if i%10==0 and rank==0 and verbose:
                l = round(loss.item(), 5)
                a = round(acc.item(), 5)
                c = round(100*i/len(dataloader), 2)
                t = round(time.time()-starttime, 3)
                s = "Loss:{} -- Acc:{}".format(l,a)
                if "rmb_loss" in package:
                    l = round(package["rmb_loss"].item(), 5)
                    a = round(package["rmb_acc"].item(), 5)
                    s += " -- RmbLoss:{} -- RmbAc:{}".format(l,a)
                s += " -- {}% -- {}s".format(c,t)
                #s = "Loss: {} -- Acc: {} -- {}% -- {}s".format(l,a,c,t)
                print(s, end="          " + len(s)*" " + "\r")
            if hyps["exp_name"]=="test" and i>=30: break
            if i>nloops: break
            if i>0 and i%checkpt_mod==0 and rank==0:
                if try_key(hyps, "save", True):
                    if verbose:
                        print()
                        print("Checkpt Training Predictions")
                        examples = print_examples(
                            data["output_ids"],
                            package["preds"],
                            tokenizer
                        )
                    train_loss = round(avg_loss/i, 5)
                    train_acc = round(avg_acc/i, 5)
                    save_dict = {
                        "mid_epoch": True,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc":  train_acc,
                        "val_loss": None,
                        "val_acc":  None,
                        "state_dict": model.state_dict(),
                        "optim_dict": optimizer.state_dict(),
                        "hyps": hyps,
                        "examples": examples,
                    }
                    ml_utils.save_io.save_checkpt(
                        save_dict=save_dict,
                        save_folder=hyps["save_folder"],
                        save_name="checkpt",
                        epoch=epoch,
                        ext=".pt"
                    )
        train_loss = round(avg_loss/i, 5)
        train_acc = round(avg_acc/i, 5)
        if "rmb_loss" in package:
            rmb_train_loss = round(rmb_avg_loss/i, 5)
            rmb_train_acc = round(rmb_avg_acc/i, 5)
        if rank==0 and verbose:
            print()
            print("Example Predictions On Training")
            examples = print_examples(
                data["output_ids"], package["preds"], tokenizer
            )
            print("Validating...")
        del package["preds"]

        # Validation
        avg_loss = 0
        avg_acc = 0
        rmb_avg_loss = 0
        rmb_avg_acc = 0
        if rank==0:
            wrapped_model.eval()
            with torch.no_grad():
                nloops = try_key(hyps,"max_val_loops",None)
                for i,data in enumerate(valloader):
                    data = {k: v.to(rank) for k,v in data.items()}
                    package = wrapped_model(
                        data, ret_preds=True, tforce=False
                    )
                    loss = package["loss"]
                    acc = package["acc"]
                    preds = package["preds"]
                    if "rmb_loss" in package:
                        rmb_avg_loss += package["rmb_loss"].item()
                        rmb_avg_acc  += package["rmb_acc"].item()

                    avg_loss += loss.item()
                    avg_acc += acc.item()
                    if hyps["exp_name"]=="test" and i>=3: break
                    if nloops is not None and i>nloops: break
            val_loss = round(avg_loss/i, 5)
            val_acc = round(avg_acc/i, 5)
            if "rmb_loss" in package:
                rmb_val_loss = round(rmb_avg_loss/i, 5)
                rmb_val_acc = round(rmb_avg_acc/i, 5)
            if rank==0 and verbose:
                print()
                print("Example Predictions On Validation")
                examples = print_examples(
                    data["output_ids"], preds, tokenizer
                )
                print()
                print("Final Stats Epoch", epoch)
                print("Train Loss:",train_loss,"-- Train Acc:",train_acc)
                print("Val Loss:", val_loss, "-- Val Acc:", val_acc)
                if "rmb_loss" in package:
                    print("RMB Train Loss:",
                        rmb_train_loss,
                        "-- RMB Train Acc:",
                        rmb_train_acc)
                    print("RMB Val Loss:",
                        rmb_val_loss,
                        "-- RMB Val Acc:",
                        rmb_val_acc)
                print()
                print()
                print()
                print()

            keys = list(data.keys())
            for k in keys: del data[k]
            optimizer.zero_grad()
            if rank==0 and try_key(hyps, "save", True):
                save_dict = {
                    "mid_epoch": False,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc":  train_acc,
                    "val_loss":   val_loss,
                    "val_acc":    val_acc,
                    "rmb_train_loss": rmb_train_loss,
                    "rmb_train_acc":  rmb_train_acc,
                    "rmb_val_loss":   rmb_val_loss,
                    "rmb_val_acc":    rmb_val_acc,
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                    "hyps": hyps,
                    "examples": examples,
                }
                ml_utils.save_io.save_checkpt(
                    save_dict=save_dict,
                    save_folder=hyps["save_folder"],
                    save_name="checkpt",
                    epoch=epoch,
                    ext=".pt"
                )
            else: print("NOT SAVING MODEL!!!!")
            scheduler.step(val_loss)
        if hyps["exp_name"]=="test" and epoch==2: break

    if hyps["multi_gpu"]: dist.destroy_process_group()


def print_examples(targs, preds, tokenizer, n_samps=5):
    """
    Helper function to print the model's predictions

    Args:
        targs: torch tensor (B,S)
            the target tokens
        preds: torch tensor (B,S,P)
            the prediction logits
        tokenizer: huggingface tokenizer
        n_samps: int
            the number of samples to print and collect
    Returns:
        examples: list of dicts of str
            a list of the printed examples. the dicts have keys of
            "targs" and "preds"
    """
    examples = []
    for i in range(min(n_samps, len(preds))):
        pred = tokenizer.decode(preds[i].argmax(-1))
        targ = tokenizer.decode(targs[i])
        print("Samp", i)
        print(
            "Targ:",
            targ.replace(tokenizer.pad_token, "").replace("\n", "\\n")
        )
        print(
            "Pred:",
            pred.replace(tokenizer.pad_token, "").replace("\n", "\\n")
        )
        print()
        examples.append({"targ": targ, "pred": pred})
    return examples


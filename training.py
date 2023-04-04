import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import time
from transformers import AutoTokenizer
from tqdm import tqdm
import os

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
    assert not hyps.get("csl_task",False) or\
        (hyps.get("train_lmhead",False) or hyps.get("train_embs", False))

    # Hyperparameters
    if hyps["exp_name"]=="test" and hyps["test_model_str"] is not None:
        hyps["model_string"] = hyps["test_model_str"]
    model_string = hyps["model_string"]
    lr = hyps["lr"]
    l2 = hyps["l2"]
    n_epochs = hyps["n_epochs"]
    hyps["seed"] = try_key(hyps, "seed", int(time.time()))
    if hyps["seed"] is None: hyps["seed"] = int(time.time())
    torch.manual_seed(hyps["seed"]*rank)
    hyps["rank"] = rank

    hyps["device_map"] = "auto" if hyps["model_parallel"] else None
    model = SentenceAutoEncoder(**hyps)
    if not hyps["model_parallel"]: model.to(rank)

    # Make Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    tokenizer.truncation_side = "right"

    # Add important tokens
    num_added = 0
    if tokenizer.pad_token is None:
        print("No Pad Token")
        print("EOS:", tokenizer.eos_token)
        print("BOS:", tokenizer.bos_token)
        print("CLS:", tokenizer.cls_token)
        if tokenizer.eos_token is not None:
            tokenizer.add_special_tokens(
                {"pad_token": tokenizer.eos_token}
            )
        else:
            num_added += tokenizer.add_special_tokens(
                {"pad_token": "|<PAD>|"}
            )
        #num_added += tokenizer.add_special_tokens({
        #    "pad_token": "|<PAD>|",
        #    "eos_token": "|<EOS>|",
        #})
        print("PAD:", tokenizer.pad_token)
    hyps["pad_token"] = tokenizer.pad_token

    # Adjust Model Embeddings for new token types
    model.add_embeddings(num_added)


    # Make dataset
    if verbose and rank==0:
        print("Collecting Data")
    dataset, valset, dataloader, valloader = datas.get_loaders(
        hyps,
        tokenizer,
        model=model
    )

    if verbose and rank==0:
        print("Recording Session")
    if rank==0: ml_utils.training.record_session(hyps, model)

    # Wrap model to distribute loss calculations
    if verbose and rank==0:
        print("Wrapping Model")
    wrapped_model = LossWrapper( model, tokenizer, hyps=hyps )
    if not hyps["model_parallel"]:
        if verbose and rank==0:
            print("Putting Model On GPU")
        wrapped_model.to(rank)

    # Turn off gradient calculations for everything except for the
    # embedding matrix.
    embs = model.embs
    if verbose and rank==0:
        print("Turning Off Gradients")
    params = set(embs.parameters())
    if try_key(hyps,"train_embs",False):
        mod = model.hf_model.transformer.get_input_embeddings()
        params = params.union(mod.parameters())
    if try_key(hyps,"train_lmhead",False):
        mod = model.hf_model.lm_head
        params = params.union(mod.parameters())
    if hyps.get("proj_cmpr", False):
        params = params.union(model.proj_cmpr.parameters())
    for name, p in model.named_parameters():
        if p not in params:
            p.requires_grad = False
        elif rank==0 and verbose:
            print(name, "gradients on")

    if verbose and rank==0:
        print("Creating Optimizer")
    optimizer = torch.optim.Adam(
        params,
        lr=lr,
        weight_decay=l2
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, threshold=0.001, patience=try_key(hyps,"patience",10)
    )

    if hyps["multi_gpu"]:
      if rank==0 and verbose: print("Putting model on multiple GPUs")
      ddp_model = DDP(
            wrapped_model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True
      )
    else: ddp_model = wrapped_model

    if rank==0 and verbose: print("Beginning Training")
    for epoch in range(n_epochs):
        epochtime = time.time()
        torch.cuda.empty_cache()
        if rank==0 and verbose:
            print("Emptying Trash")
            print()
            s = "Beginning Epoch {} - {}".format(
                epoch, hyps["save_folder"]
            )
            print(s)
            logstr = s + "\n"
        ddp_model.train()
        avg_loss = 0
        avg_acc = 0
        rmb_avg_loss = 0
        rmb_avg_acc = 0
        csl_avg_loss = 0
        csl_avg_acc = 0
        nloops = try_key(hyps,"n_train_loops", None)
        nloops = len(dataloader) if nloops is None else nloops
        nloops = min(nloops,len(dataloader))
        checkpt_mod = try_key(hyps, "checkpt_mod", None)
        checkpt_mod = np.inf if checkpt_mod is None else checkpt_mod
        val_mod = try_key(hyps, "val_mod", 1)
        optimizer.zero_grad()
        for i,data in enumerate(dataloader):
            starttime = time.time()
            if not hyps["model_parallel"]:
                data = {k: v.to(rank) for k,v in data.items()}
            #if hyps["dtype"]!="float32":
            #  data = {
            #    k: v.to(getattr(torch,hyps["dtype"])) for\
            #                           k,v in data.items()
            #  }

            package = ddp_model(
                data,
                ret_preds=True,
                seq_len=hyps["seq_len"],
                tforce=True,
                gen_ids=try_key(hyps, "gen_ids", False)
            )
            loss = package["loss"]
            acc = package["acc"]

            avg_acc += acc.item()
            avg_loss += loss.item()
            if "rmb_loss" in package:
                rmb_avg_loss += package["rmb_loss"].item()
                rmb_avg_acc  += package["rmb_acc"].item()
            if "csl_loss" in package:
                csl_avg_loss += package["csl_loss"].item()
                csl_avg_acc  += package["csl_acc"].item()

            if i%hyps["n_grad_loops"]==0 or i==len(dataloader)-1:
                if try_key(hyps,"grad_scaling",False):
                    model.embs.weight.grad.data = temp/temp.norm(2)
                #temp = model.embs.weight.grad.data
                #print("abs grad mean:",  temp.abs().mean(-1))
                #print("grad norm:",      temp.norm(2))
                #print()
                if hyps.get("grad_clip",0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        ddp_model.parameters(), hyps["grad_clip"]
                    )
                optimizer.step()
                optimizer.zero_grad()

            if verbose and i%10==0 and rank==0:
                dec = 4
                l = round(loss.item(), dec)
                a = round(acc.item(), dec)
                c = round(100*i/nloops, 2)
                t = round(time.time()-starttime, 3)
                s = "Loss: {} -Acc: {}".format(l,a)
                if "rmb_loss" in package:
                    l = round(package["rmb_loss"].item(), dec)
                    a = round(package["rmb_acc"].item(), dec)
                    s += " -RMB: {} -RMBAc: {}".format(l,a)
                if "csl_loss" in package:
                    l = round(package["csl_loss"].item(), dec)
                    a = round(package["csl_acc"].item(), dec)
                    s += " -CSL: {} -CSLAc: {}".format(l,a)
                s += " - {}% {}s   ".format(c,t)
                #s = "Loss: {} -- Acc: {} -- {}% -- {}s".format(l,a,c,t)
                print(s, end=int(len(s)/2)*" " + "\r")
            if hyps["exp_name"]=="test" and i>=30: break
            if i>=(nloops-1): break
            if i>0 and i%checkpt_mod==0 and rank==0:
                if try_key(hyps, "save", True):
                    if verbose:
                        print()
                        s = "Checkpt Training Predictions"
                        print(s)
                        low_preds, high_preds = get_baselines(
                            model,data,hyps,
                            rank=rank,tforce=True,to_cpu=True,
                            calc_high="csl_preds" not in package
                        )
                        high_preds=package.get("csl_preds",high_preds)
                        inpt_dict = {
                            "input_ids":  data["input_ids"],
                            "output_ids": data["output_ids"],
                            **package,
                            "low": low_preds, "high": high_preds
                        }
                        examples, s = print_examples(
                            inpt_dict,
                            tokenizer
                        )
                        keys = list(inpt_dict.keys())
                        for k in keys:
                            inpt_dict[k] = inpt_dict[k].cpu()
                        del inpt_dict
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
                    ep = round(epoch+i/len(dataloader), 3)
                    ml_utils.save_io.save_checkpt(
                        save_dict=save_dict,
                        save_folder=hyps["save_folder"],
                        save_name="checkpt",
                        epoch=ep,
                        ext=".pt"
                    )
        div = (i+1)
        train_loss = round(avg_loss/div, 5)
        train_acc  = round(avg_acc/div, 5)
        if "rmb_loss" in package:
            rmb_train_loss = round(rmb_avg_loss/div, 5)
            rmb_train_acc = round(rmb_avg_acc/div, 5)
        if "csl_loss" in package:
            csl_train_loss = round(csl_avg_loss/div, 5)
            csl_train_acc = round(csl_avg_acc/div, 5)
        if rank==0 and verbose:
            print()
            s = "Example Predictions On Training"
            print(s)
            logstr += s + "\n"
            low_preds, high_preds = get_baselines(
                model,data,hyps, rank=rank,tforce=True,to_cpu=True,
                calc_high="csl_preds" not in package
            )
            high_preds=package.get("csl_preds",high_preds)
            inpt_dict = {
                "input_ids":  data["input_ids"],
                "output_ids": data["output_ids"],
                **package,
                "low": low_preds, "high": high_preds
            }
            examples,s = print_examples( inpt_dict, tokenizer )
            logstr += s + "\n"
            keys = list(inpt_dict.keys())
            for k in keys:
                inpt_dict[k] = inpt_dict[k].cpu()
            del inpt_dict
        del package["preds"]

        # Validation
        avg_loss = 0
        avg_acc = 0
        rmb_avg_loss = 0
        rmb_avg_acc = 0
        csl_avg_loss = 0
        csl_avg_acc = 0
        if rank==0 and epoch%val_mod==0:
            ddp_model.eval()
            if verbose:
                print("Validating...")
            with torch.no_grad():
                nloops = try_key(hyps,"max_val_loops",None)
                if nloops is None: nloops = len(valloader)
                for i,data in enumerate(valloader):
                    starttime = time.time()
                    if not hyps["model_parallel"]:
                        data = {k: v.to(rank) for k,v in data.items()}
                    #if hyps["dtype"]!="float32":
                    #  data = {
                    #    k: v.to(getattr(torch,hyps["dtype"])) for\
                    #                          k,v in data.items()
                    #  }
                    package = ddp_model(
                        data,
                        ret_preds=True,
                        tforce=False,
                        gen_targs=try_key(hyps, "gen_targs", False),
                        seq_len=hyps["seq_len"],
                        gen_ids=try_key(hyps, "gen_ids", False)
                    )
                    loss = package["loss"]
                    acc = package["acc"]
                    preds = package["preds"]
                    if "rmb_loss" in package:
                        rmb_avg_loss += package["rmb_loss"].item()
                        rmb_avg_acc  += package["rmb_acc"].item()
                    if "csl_loss" in package:
                        csl_avg_loss += package["csl_loss"].item()
                        csl_avg_acc  += package["csl_acc"].item()

                    avg_loss += loss.item()
                    avg_acc += acc.item()
                    if hyps["exp_name"]=="test" and i>=3: break
                    if i>=nloops-l: break
                    if verbose and i%20==0:
                        p = round(100*i/nloops)
                        t = time.time()-starttime
                        print("{}% -- {}s".format(p,t), end="     \r")
            div = (i+1)
            val_loss = round(avg_loss/div, 5)
            val_acc = round(avg_acc/div, 5)
            if "rmb_loss" in package:
                rmb_val_loss = round(rmb_avg_loss/div, 5)
                rmb_val_acc = round(rmb_avg_acc/div, 5)
            if "csl_loss" in package:
                csl_val_loss = round(csl_avg_loss/div, 5)
                csl_val_acc = round(csl_avg_acc/div, 5)
            if rank==0 and verbose:
                print()
                s = "Example Predictions On Validation"
                print(s)
                logstr += s + "\n"
                low_preds, high_preds = get_baselines(
                    model,data,hyps,rank=rank,tforce=False,to_cpu=True,
                    calc_high="csl_preds" not in package
                )
                high_preds=package.get("csl_preds",high_preds)
                inpt_dict = {
                    "input_ids": data["input_ids"],
                    "output_ids": data["output_ids"],
                    **package,
                    "low": low_preds, "high": high_preds
                }
                examples,s = print_examples( inpt_dict, tokenizer )
                keys = list(inpt_dict.keys())
                for k in keys:
                    inpt_dict[k] = inpt_dict[k].cpu()
                del inpt_dict
                logstr += s + "\n"
                print()
                s = "Final Stats, Epoch: {}".format(epoch)
                print(s)
                logstr += "\n" + s + "\n"

                s = "Train Loss: {} -Train Acc: {}".format(
                    train_loss,train_acc
                )
                print(s)
                logstr += s + "\n"
                s = "Val Loss: {} -Val Acc: {}".format(
                    val_loss,val_acc
                )
                print(s)
                logstr += s + "\n"
                if "rmb_loss" in package:
                    s = "RMB Train Loss: {} -RMB Train Acc: {}".format(
                        rmb_train_loss, rmb_train_acc
                    )
                    print(s)
                    logstr += s + "\n"
                    s = "RMB Val Loss: {} -RMB Val Acc: {}".format(
                        rmb_val_loss, rmb_val_acc
                    )
                    print(s)
                    logstr += s + "\n"
                if "csl_loss" in package:
                    s = "CSL Train Loss: {} -CSL Train Acc: {}".format(
                        csl_train_loss, csl_train_acc
                    )
                    print(s)
                    logstr += s + "\n"
                    s = "CSL Val Loss: {} -CSL Val Acc: {}".format(
                        csl_val_loss, csl_val_acc
                    )
                    print(s)
                    logstr += s + "\n"
                s = "Epoch Dur: {}s".format(round(time.time()-epochtime))
                logstr += s + "\n\n\n\n"
                print(s)
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
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                    "hyps": hyps,
                    "examples": examples,
                }
                if "rmb_loss" in package:
                    save_dict["rmb_train_loss"] = rmb_train_loss
                    save_dict["rmb_train_acc"] =  rmb_train_acc
                    save_dict["rmb_val_loss"] =   rmb_val_loss
                    save_dict["rmb_val_acc"] =    rmb_val_acc
                if "csl_loss" in package:
                    save_dict["csl_train_loss"] = csl_train_loss
                    save_dict["csl_train_acc"] =  csl_train_acc
                    save_dict["csl_val_loss"] =   csl_val_loss
                    save_dict["csl_val_acc"] =    csl_val_acc
                ml_utils.save_io.save_checkpt(
                    save_dict=save_dict,
                    save_folder=hyps["save_folder"],
                    save_name="checkpt",
                    epoch=epoch,
                    ext=".pt"
                )
                save_training_log(hyps, logstr)
            scheduler.step(val_loss)
        keys = list(package.keys())
        for k in keys: del package[k]
        if hyps["exp_name"]=="test" and epoch==2: break
    if hyps["multi_gpu"]: dist.destroy_process_group()


def print_examples(inpt_dict, tokenizer, n_samps=5):
    """
    Helper function to print the model's predictions

    Args:
        inpt_dict: dict {str: tensor (B,Sn)}
            input_ids: torch tensor (B,S1)
                the ground truth of the compressed context ids
            output_ids: torch tensor (B,S2)
                the target ids
            rmb_pred: torch tensor (B,S1,L)
                the predicted compressed context logits
            low: torch tensor (B,S1)
                the predicted target ids with no prepended information
            pred: torch tensor (B,S1,L)
                the predicted compressed context logits
            high: torch tensor (B,S1)
                the predicted target ids with all prepended information
        tokenizer: huggingface tokenizer
        n_samps: int
            the number of samples to print and collect
    Returns:
        examples: list of dicts of str
            a list of the printed examples. the dicts have keys of
            "targs" and "preds"
        logstr: str
            a single string of one printout loop
    """
    tensors = []
    if "input_ids" in inpt_dict:
        ctxs = inpt_dict["input_ids"]
        tensors.append(ctxs)
    if "rmb_preds" in inpt_dict:
        rmbs = inpt_dict["rmb_preds"].argmax(-1)
        tensors.append(rmbs)
    targs = inpt_dict["output_ids"]
    tensors.append(targs)
    preds = dict()
    for k in ["low", "preds", "high"]:
        if len(inpt_dict[k].shape)==3:
            preds[k] = inpt_dict[k].argmax(-1)
        else: preds[k] = inpt_dict[k]
    tensors = tensors + [v for v in preds.values()]

    lens = []
    l = min([len(t) for t in tensors])
    logstr = ""
    examples = []
    for i in range(min(n_samps, l)):
        examp = {}
        print("Samp", i)
        if "input_ids" in inpt_dict:
            ctx = tokenizer.decode(ctxs[i], skip_special_tokens=False)
            ctx = ctx.replace(tokenizer.pad_token,"").replace("\n","\\n")
            examp["ctx"] = ctx
            s = "Ctx: " +  ctx
            if i == 0:
                logstr += s + "\n"
            print(s)

        if "rmb_preds" in inpt_dict:
            rmb = tokenizer.decode(rmbs[i], skip_special_tokens=False)
            rmb = rmb.replace(tokenizer.pad_token,"").replace("\n","\\n")
            examp["rmb"] = rmb
            s = "RMB: " +  rmb
            if i == 0:
                logstr += s + "\n"
            print(s)

        targ = tokenizer.decode(targs[i], skip_special_tokens=False)
        targ = targ.replace(tokenizer.pad_token, "").replace("\n","\\n")
        s = "Targ: " +  targ
        if i == 0:
            logstr += s + "\n"
        print(s)
        examp["targ"] = targ
        for k,v in preds.items():
            pred = tokenizer.decode(
                v[i], skip_special_tokens=False
            ).replace("\n", "\\n")
            s = k + ": " + pred
            if i == 0:
                logstr += s + "\n"
            print(s)
            examp[k] = pred
        print()
        examples.append(examp)
    return examples, logstr

def get_baselines(model, data, hyps, rank=0, tforce=True,
                                             to_cpu=True,
                                             calc_high=True):
    """
    Args:
        model: SentenceAutoEncoder
        data: dict {str: tensor}
            input_ids: tensor
            attention_mask: tensor
        hyps: dict
        rank: int
        tforce: bool
        to_cpu: bool
            if true, returns tensors on cpu
        calc_high: bool
            if true, calculates the high preds. otherwise returns None
            for high_preds

    Returns:
        low_preds: torch tensor (B,S,L)
            logits predicted with minimal token seed (i.e. almost no
            token context from cmp sequence)
        high_preds: torch tensor (B,S,L)
            logits predicted with maximal token seed (i.e. complete
            token context from cmp sequence)
            returns tensor of zeros if calc_high is false
    """
    with torch.no_grad():
        low_inpts = {
            "input_ids": data["output_ids"],
            "attention_mask": data["output_attn_mask"],
        }
        if not hyps["model_parallel"]:
            low_inpts = {k: v.to(rank) for k,v in low_inpts.items()}
        low_preds =  model.causal_lm(
            **low_inpts,
            tforce=tforce,
            ret_logits=False,
            seed_len=max(3,hyps["seq_overlap"])
        )

        if calc_high:
            high_inpts = {
                "input_ids": torch.cat([
                    data["input_ids"], data["output_ids"]
                ], dim=1),
                "attention_mask": torch.cat([
                    data["attention_mask"], data["output_attn_mask"]
                ], dim=1)
            }
            if not hyps["model_parallel"]:
                high_inpts = {k: v.to(rank) for k,v in high_inpts.items()}
            seed_len=data["input_ids"].shape[1]+max(0,hyps["seq_overlap"])
            high_preds = model.causal_lm(
                **high_inpts,
                tforce=tforce,
                ret_logits=False,
                seed_len=seed_len
            )
            high_preds = high_preds[:,data["input_ids"].shape[1]:]
        else: high_preds = torch.zeros(1,1,1)
    if to_cpu:
        low_preds = low_preds.cpu()
        high_preds = high_preds.cpu()
    return low_preds, high_preds

def save_training_log(hyps, logstr, fname="training_log.txt", reset=False):
    """
    Saves the logstr to the save folder under the name training_log.txt

    hyps: dict
    logstr: str
        the string to save
    fname: str
        the name of the file to save to
    reset: bool
        if true, resets the training log and then writes. otherwise
        appends to training log
    """
    mode = "w" if reset else "a"
    with open(os.path.join(hyps["save_folder"], fname),mode) as f:
        f.write(logstr)

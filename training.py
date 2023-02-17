import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPTJForCausalLM, DataCollatorWithPadding
import datasets

from ml_utils.utils import try_key
import ml_utils
import datas

RMB = "|<RMB>|" # Extra characters are to ensure uniqueness
CMP = "|<CMP>|"
SOS = "|<SOS>|"


class SentenceAutoEncoder(torch.nn.Module):
    """
    Trains a new token type to compress a sentence into a single vector
    representation
    """
    def __init__(self, model_string, rank=0, torch_dtype="float32",
                                             device_map="auto",
                                             cmp_layer="half",
                                             rmb_task=False,
                                             *args, **kwargs):
        """
        model_string: str
            name of pretrained hugging face transformer model
        rank: int
            rank within distributed training
        torch_dtype: torch type or str
            the floating point precision to use
        device_map: str
            determines whether you want to use model parallel or not
        cmp_layer: int, str, or None
            the layer from the transformer to use for the compression
            token. str argument can be 'half' denoting the middle layer
            of the transformer. None defaults to last layer.
        rmb_task: bool
            if true, will assume that there is an auxiliary
            memory reconstruction objective.
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
        self.cmp_layer = cmp_layer
        self.rmb_task = rmb_task

    def compress(self, input_ids, attention_mask, *args, **kwargs):
        """
        Compresses the input ids to a single vector.

        Args: 
            input_ids: LongTensor (B,S)
                the token indices of the input sequence. The CMP token
                should be appended to the end of each sentence.
            attention_mask: LongTensor (B,S)
                attention mask for padding purposes. 0s mean padding.
        Returns:
            cmpr: torch tensor (B,1,H)
                the compressed representations
        """
        model = self.hf_model
        model_embs = model.transformer.get_input_embeddings()
        inpt_embs = model_embs( input_ids ).data
        # Need to do this to prevent backprop into all other parameters
        idx = input_ids==self.CMP_ID
        inpt_embs[idx] = model_embs.weight[self.CMP_ID]

        fx = model.transformer(
            inputs_embeds=inpt_embs,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Get the representational layer of choice
        if self.cmp_layer is None:
            cmpr = fx["last_hidden_state"][idx][:,None]
        elif self.cmp_layer=="half" or self.cmp_layer=="middle":
            n_layers = len(fx["hidden_states"])
            cmpr = fx["hidden_states"][n_layers//2][idx][:,None]
        elif isinstance(self.cmp_layer, int):
            cmpr = fx["hidden_states"][self.cmp_layer][idx][:,None]
        else:
            raise NotImplemented
        return cmpr

    def forward(self, data):
        """
        Args:
          data: dict
            "input_ids": LongTensor (B,S1)
                the token indices of the input sequence. The CMP token
                should be appended to the end of each sentence.
            "attention_mask": LongTensor (B,S1)
                attention mask for padding purposes. 0s mean padding.
            "output_ids": LongTensor (B,S2)
                the token indices of the target sequence. An EOS token
                should be appended to the end of each sentence
            "output_attn_mask": LongTensor (B,S2)
                attention mask for padding purposes. 0s mean padding.
        Returns:
            preds: tensor (B,S2,H)
        """
        cmpr = self.compress(data["input_ids"], data["attention_mask"])

        model = self.hf_model
        model_embs = model.transformer.get_input_embeddings()
        out_embs =  model_embs(data["output_ids"]).data

        # Concat compressed representation and start of sentence token
        # to beginning of sentence and pad attention accordingly
        sos = model_embs.weight[self.SOS_ID][None,None]
        out_embs = torch.cat(
            [
                cmpr.to(self.rank),
                sos.repeat(len(cmpr),1,1),
                out_embs.to(self.rank)
            ],
            dim=1
        )
        attn = torch.nn.functional.pad(
            data["output_attn_mask"], (2,0), value=1
        )
        preds = model(inputs_embeds=out_embs,attention_mask=attn).logits

        # Handle Auxiliary, Rememberance Predictions
        if self.rmb_task:
            # TODO: instead of eos, you should pad. don't forget the attn
            data["input_ids"][data["input_ids"]==self.CMP_ID]=self.EOS_ID
            out_embs = model_embs( data["input_ids"] ).data
            try:
                rmb = model_embs.weight[self.RMB_ID][None,None]
                out_embs = torch.cat(
                    [
                        cmpr.to(self.rank),
                        rmb.repeat(len(cmpr),1,1),
                        out_embs
                    ],
                    dim=1
                )
                attn = torch.nn.functional.pad(
                    data["attention_mask"], (2,0), value=1
                )
                rmb_preds = model(
                    inputs_embeds=out_embs,
                    attention_mask=attn
                ).logits
                return preds, rmb_preds
            except:
                print("og_rmb:", model_embs.weight[self.RMB_ID].shape)
                print("rmb:", rmb.shape)
                print("Out Embds", out_embs.shape)
                print("attn", attn.shape)
                print("FX", cmpr.shape)
                print("rmb_preds", rmb_preds.shape)
                assert False
        return preds

    def infer(self, data, pred_len=None, rmb_task=False):
        """
        Performs inference without teacher forcing.
        Args:
            data: dict (keys: str, vals: tensors)
              "input_ids": LongTensor (B,S1)
                  the token indices of the input sequence. The CMP token
                  should be appended to the end of each sentence.
              "attention_mask": LongTensor (B,S1)
                  attention mask for padding purposes. 0s mean padding.
              "output_ids": LongTensor (B, 1 or S2)
                  the token indices of the target sequence.
            pred_len: int or None
                the number of prediction steps to perform
            rmb_task: bool
                

        """
        cmpr = self.compress(**data)
        out_embs = [ cmpr.to(self.rank) ]

        embs = self.hf_model.transformer.get_input_embeddings()
        if rmb_task:
            tsk = model_embs.weight[self.RMB_ID][None,None]
        else:
            tsk = model_embs.weight[self.SOS_ID][None,None]
        out_embs.append( tsk.repeat(len(cmpr),1,1) )
        out_embs.append( embs(data["output_ids"][:,:1]).data )
        out_embs = torch.cat( out_embs, dim=1 )

        if pred_len is None: pred_len = data["output_ids"].shape[1]
        for i in range(pred_len):
            pred = self.hf_model(input_embeds=out_embs).logits
            # TODO: ALLOW SAMPLING WITH TEMPERATURE
            pred = embs( pred[:,-1:].argmax(-1).to(self.rank) )
            out_embs = torch.cat(
                [out_embs, pred], dim=1
            )
        return out_embs

class LossWrapper(torch.nn.Module):
    """
    This class wraps the model to keep the loss calculations distributed
    on all GPUs. Otherwise one gpu is overloaded with computational
    costs.
    """
    def __init__(self, model, tokenizer, loss_scale=1.):
        """
        loss_scale: float
            the loss is multiplied by this value on every iteration.
            useful as a way to normalize the learning rate when
            performing multiple gradient computations before each
            gradient step.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.loss_scale = loss_scale
        self.loss_fxn = torch.nn.CrossEntropyLoss()

    def forward(self, data, ret_preds=False):
        """
        Args:
            data: dict
                "input_ids": LongTensor (B,S1)
                    the token indices of the input sequence. The CMP
                    token should be appended to the end of each sentence.
                "attention_mask": LongTensor (B,S1)
                    attention mask for padding purposes. 0s mean padding.
                "output_ids": LongTensor (B,S2)
                    the token indices of the target sequence. An EOS
                    token should be appended to the end of each sentence
                "output_attn_mask": LongTensor (B,S2)
                    attention mask for padding purposes. 0s mean padding.
            ret_preds: bool
                if true, will return the predictions
        Returns:
            ret_dict: dict (keys: str, vals: torch tensor)
                "loss": torch tensor (1,)
                "rmb_loss": torch tensor (1,)
                    only returned if `rmb_task` is true
                "acc": torch tensor (1,)
                    the raw accuracy for the non-rmb task
                "preds": torch tensor (B,S,P)
                    the prediction logits. only returned if ret_preds is
                    true
                "rmb_preds": torch tensor (B,S,P)
                    the rmb prediction logits. only returned if ret_preds
                    is true
        """
        if self.model.rmb_task:
            preds, rmb_preds = self.model(data)
            # Need to remove CMP representation, the first element
            ps = rmb_preds[:,1:-1].reshape(-1, preds.shape[-1])
            labels = data["input_ids"].reshape(-1)
            idx = (labels!=self.tokenizer.pad_token_id)
            idx = idx&(labels!=self.model.CMP_ID)
            rmb_loss = self.loss_fxn(ps[idx],labels[idx])*self.loss_scale
        else:
            preds = self.model(data)
        # Need to remove CMP representation, the first element
        ps = preds[:,1:-1].reshape(-1, preds.shape[-1])
        labels = data["output_ids"].reshape(-1)

        idx = labels!=self.tokenizer.pad_token_id
        loss = self.loss_fxn(ps[idx], labels[idx])*self.loss_scale
        if self.training: loss.backward()
        argmax = torch.argmax(ps[idx])
        acc = (argmax==labels[idx]).float().mean()
        ret_dict = {
            "loss": loss,
            "acc": acc,
        }
        if self.model.rmb_task:
            ret_dict["rmb_loss"] = rmb_loss
            if ret_preds:
                ret_dict["rmb_preds"] = rmb_preds[:,1:-1]
        if ret_preds:
            ret_dict["preds"] = preds[:,1:-1]
        return ret_dict

    def infer(self, data, ret_preds=False):
        """
        Use this function to make predictions during validation or
        testing. Does not use teacher forcing.

        Args:
            data: dict
                "input_ids": LongTensor (B,S1)
                    the token indices of the input sequence. The CMP
                    token should be appended to the end of each sentence.
                "attention_mask": LongTensor (B,S1)
                    attention mask for padding purposes. 0s mean padding.
                "output_ids": LongTensor (B,S2)
                    the token indices of the target sequence. An EOS
                    token should be appended to the end of each sentence
                "output_attn_mask": LongTensor (B,S2)
                    attention mask for padding purposes. 0s mean padding.
            ret_preds: bool
                if true, will return the predictions
        Returns:
            loss
        """
        if self.model.rmb_task:
            preds, rmb_preds = self.model(data)
            # Need to take off first element for RMB token
            ps = rmb_preds[:,1:-1].reshape(-1, preds.shape[-1])
            labels = data["input_ids"].reshape(-1)
            idx = (labels!=self.tokenizer.pad_token_id)
            idx = idx&(labels!=self.model.CMP_ID)
            rmb_loss = self.loss_fxn(ps[idx],labels[idx])*self.loss_scale
        else:
            preds = self.model(data)
        ps = preds[:,1:-1].reshape(-1, preds.shape[-1])
        labels = data["output_ids"].reshape(-1)

        idx = labels!=self.tokenizer.pad_token_id
        loss = self.loss_fxn(ps[idx], labels[idx])*self.loss_scale
        if self.training: loss.backward()
        argmax = torch.argmax(ps[idx])
        acc = (argmax==labels[idx]).float().mean()
        ret_dict = {
            "loss": loss,
            "acc": acc,
        }
        if self.model.rmb_task:
            ret_dict["rmb_loss"] = rmb_loss
            if ret_preds:
                ret_dict["rmb_preds"] = rmb_preds[:,1:-1]
        if ret_preds:
            ret_dict["preds"] = preds[:,1:-1]
        return ret_dict

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
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    new_tokens = {"additional_special_tokens": [RMB, CMP, SOS]}
    num_added = tokenizer.add_special_tokens(new_tokens)
    print("Tokenizer Initial Keys:")
    print("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("BOS:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("PAD:", tokenizer.pad_token, tokenizer.pad_token_id)
    if tokenizer.pad_token is None:
        num_added += tokenizer.add_special_tokens(
            {"pad_token": "|<PAD>|"}
        )
    if tokenizer.eos_token is None:
        num_added += tokenizer.add_special_tokens(
            { "eos_token": "|<EOS>|" }
        )

    # Adjust Model Embeddings for new token types
    if hyps["multi_gpu"]: model = ddp_model.model
    else: model = ddp_model
    embs = model.hf_model.transformer.get_input_embeddings()
    n,h = embs.weight.shape
    model.hf_model.resize_token_embeddings(n+num_added)
    model.tokenizer = tokenizer
    model.CMP_ID = int(tokenizer.encode(CMP)[0])
    model.CMP =    CMP
    hyps["CMP_TOKEN"] = CMP
    hyps["CMP_ID"] = model.CMP_ID
    model.RMB_ID = int(tokenizer.encode(RMB)[0])
    model.RMB =    RMB
    hyps["RMB_TOKEN"] = RMB
    hyps["RMB_ID"] = model.RMB_ID
    model.SOS_ID = int(tokenizer.encode(SOS)[0])
    model.SOS =    SOS
    hyps["SOS_TOKEN"] = SOS
    hyps["SOS_ID"] = model.SOS_ID
    model.EOS_ID = tokenizer.eos_token_id
    model.EOS =    tokenizer.eos_token

    # Make dataset
    dataset, valset, dataloader, valloader = datas.get_loaders(
        hyps,
        tokenizer
    )

    if rank==0: ml_utils.training.record_session(hyps, model)

    # Wrap model to distribute loss calculations
    wrapped_model = LossWrapper(
        ddp_model,
        tokenizer,
        loss_scale=1/hyps["n_grad_loops"]
    )
    if not hyps["model_parallel"]:
        wrapped_model.to(rank)
    # Mayber better to parallelize after wrap, unsure at this point
    #ddp_model = DDP(wrapped_model, device_ids=[rank])

    # This line is crucial, otherwise you will reference stale embs
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
        wrapped_model.train()
        avg_loss = 0
        avg_acc = 0
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

            if i%hyps["n_grad_loops"]==0 or i==len(dataloader)-1:
                optimizer.step()
                optimizer.zero_grad()

            if i%10==0 and rank==0 and verbose:
                l = round(loss.item(), 5)
                a = round(acc.item(), 5)
                c = round(100*i/len(dataloader), 2)
                t = round(time.time()-starttime, 3)
                s = "Loss: {} -- Acc: {} -- {}% -- {}s".format(l,a,c,t)
                print(s, end="             " + len(s)*" " + "\r")
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
        if rank==0:
            wrapped_model.eval()
            with torch.no_grad():
                nloops = try_key(hyps,"max_val_loops",None)
                for i,data in enumerate(valloader):
                    data = {k: v.to(rank) for k,v in data.items()}
                    package = wrapped_model(data, ret_preds=True)
                    loss = package["loss"]
                    acc = package["acc"]
                    preds = package["preds"]

                    avg_loss += loss.item()
                    avg_acc += acc.item()
                    if hyps["exp_name"]=="test" and i>=3: break
                    if nloops is not None and i>nloops: break
            val_loss = round(avg_loss/i, 5)
            val_acc = round(avg_acc/i, 5)
            if rank==0 and verbose:
                print()
                print("Example Predictions On Validation")
                examples = print_examples(
                    data["output_ids"], preds, tokenizer
                )
                print("Train Loss:",train_loss,"-- Train Acc:",train_acc)
                print("Val Loss:", val_loss, "-- Val Acc:", val_acc)
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
                    "val_loss": val_loss,
                    "val_acc":  val_acc,
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
        print("Preds:", pred.replace(tokenizer.pad_token, "").replace("\n", "\\n"))
        print("Targs:", targ.replace(tokenizer.pad_token, ""))
        print()
        examples.append({"targ": targ, "pred": pred})
    return examples

import torch
from transformers import AutoModelForCausalLM
import numpy as np


class SentenceAutoEncoder(torch.nn.Module):
    """
    Trains a new token type to compress a sentence into a single vector
    representation
    """
    def __init__(self, model_string, rank=0, dtype="float32",
                                             device_map="auto",
                                             cmp_layer="half",
                                             rmb_task=False,
                                             n_cmps=3,
                                             n_tsks=2,
                                             train_embs=False,
                                             proj_cmpr=False,
                                             *args, **kwargs):
        """
        model_string: str
            name of pretrained hugging face transformer model
        rank: int
            rank within distributed training
        dtype: torch type or str
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
        n_embs: int
            the number of compression tokens
        n_tsks: int
            the number of task tokens. for the task ids, rmb is 1, sos
            is 0
        train_embs: bool
            if false, uses data of transformer embedding parameters
            instead of embedding parameters directly.
        proj_cmpr: bool
            if true, projects the cmpr' representations using a linear
            weight matrix before using them as input to the forward/
            auxiliary tasks.
        """
        super().__init__()
        self.model_string = model_string
        if dtype=="float32": dtype = torch.float32
        elif dtype=="float16": dtype = torch.float16
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_string,
            torch_dtype=dtype,
            device_map=device_map
        )
        self.rank = rank
        self.cmp_layer = cmp_layer
        self.rmb_task = rmb_task
        self.n_cmps = n_cmps
        self.n_tsks = n_tsks

        t = self.hf_model.transformer
        tembs = t.get_input_embeddings().weight
        hsize = tembs.shape[-1]
        self.embs = torch.nn.Embedding(self.n_cmps+self.n_tsks, hsize)
        self.embs.to(dtype)
        if tembs.get_device()>-1:
            self.embs.to(tembs.get_device())
        self.proj_cmpr = None
        if proj_cmpr:
            self.proj_cmpr = torch.nn.Linear(hsize, hsize)

        self.cmp_ids = [i for i in range(self.n_cmps)]
        # sos is 0, rmb is 1
        self.tsk_ids = [i+self.n_cmps for i in range(self.n_tsks)]
        self.train_embs = train_embs

    def get_device(self):
        t = self.hf_model.transformer
        return t.get_input_embeddings().weight.get_device()

    def add_attrs(self, new_attrs):
        """
        Adds new attributes to the model.

        Args:
            new_attrs: dict {str: val}
                "new_attr": new_value
        """
        for k,v in new_attrs.items():
            setattr(self, k, v)

    def get_embeddings(self):
        """
        Returns a reference to the transformer embeddings.
        """
        return self.hf_model.transformer.get_input_embeddings()

    def add_embeddings(self, n_embs):
        """
        Args:
            n_embs: int
                the number of embeddings to add
        """
        if n_embs <= 0: return
        embs = self.get_embeddings()
        n,h = embs.weight.shape
        self.hf_model.resize_token_embeddings(n+n_embs)

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
        # inpt_embs are padded on left side, so we can append the cmp
        # tokens to the right side and pad the attention
        inpt_embs = model_embs( input_ids )
        if not self.train_embs: inpt_embs = inpt_embs.data
        cmp_embs = self.embs.weight[self.cmp_ids[0]:self.cmp_ids[-1]+1]
        inpt_embs = torch.cat(
            [inpt_embs, cmp_embs[None].repeat(len(inpt_embs),1,1)],
            dim=1
        )
        attention_mask = torch.nn.functional.pad(
            attention_mask, (0, self.n_cmps)
        )

        fx = model.transformer(
            inputs_embeds=inpt_embs,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Get the representational layer of choice and take only the
        # compression representations
        if self.cmp_layer is None:
            cmpr = fx["last_hidden_state"][:,-self.n_cmps:]
        elif self.cmp_layer=="half" or self.cmp_layer=="middle":
            n_layers = len(fx["hidden_states"])
            cmpr = fx["hidden_states"][n_layers//2][:,-self.n_cmps:]
        elif isinstance(self.cmp_layer, int):
            cmpr = fx["hidden_states"][self.cmp_layer][:,-self.n_cmps:]
        # Can use linear combo of embeddings as compressed representations
        elif self.cmp_layer=="output":
            logits = fx["last_hidden_state"]
            shape = logits.shape
            logits = model.lm_head(logits.reshape(-1,shape[-1]))
            emb = model_embs.weight
            if not self.train_embs: emb = emb.data
            cmpr = torch.nn.functional.softmax(
                logits.reshape(-1, logits.shape[-1]), dim=-1
            )
            cmpr = torch.mm(cmpr, emb)
            cmpr = cmpr.reshape(*shape[:-1], cmpr.shape[-1])
            cmpr = cmpr/shape[1]
        if self.proj_cmpr is not None:
            shape = cmpr.shape
            cmpr = cmpr.reshape(-1,shape[-1])
            try:
                proj = self.proj_cmpr(cmpr)
            except:
                self.proj_cmpr.to(cmpr.get_device())
                proj = self.proj_cmpr(cmpr)
            cmpr = proj.reshape(shape)
        return cmpr

    def forward(self, data, tforce=True):
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
          tforce: bool
            if true, uses teacher forcing
        Returns:
            preds: tensor (B,S2,H)
        """
        cmpr = self.compress(**data)
        if not tforce:
            preds = self.infer(
                data,
                pred_len=data["output_ids"].shape[1],
                rmb_task=False,
                cmpr=cmpr,
                ret_logits=True
            )["logits"]
            if self.rmb_task:
                rmb_preds = self.infer(
                    data,
                    pred_len=data["input_ids"].shape[1],
                    rmb_task=True,
                    ret_logits=True,
                    cmpr=cmpr
                )["logits"]
                return preds, rmb_preds
            return preds

        model = self.hf_model
        model_embs = model.transformer.get_input_embeddings()
        out_embs =  model_embs(data["output_ids"])
        if not self.train_embs: out_embs = out_embs.data
        # Concat compressed representation and start of sentence token
        # to beginning of sentence and pad attention accordingly
        sos = self.embs.weight[self.tsk_ids[0]][None,None]
        out_embs = torch.cat(
            [
                cmpr.to(self.rank),
                sos.repeat(len(cmpr),1,1),
                out_embs.to(self.rank)
            ],
            dim=1
        )
        npad = out_embs.shape[1] - data["output_attn_mask"].shape[1]
        attn = torch.nn.functional.pad(
            data["output_attn_mask"], (npad,0), value=1
        )
        preds = model(inputs_embeds=out_embs,attention_mask=attn).logits
        preds = preds[:,cmpr.shape[1]:-1]

        # Optionally Handle Auxiliary, Rememberance Predictions
        if self.rmb_task:
            out_embs = model_embs( data["input_ids"] )
            rmb = self.embs.weight[self.tsk_ids[1]][None,None]
            out_embs = torch.cat(
                [
                    cmpr.to(self.rank),
                    rmb.repeat(len(cmpr),1,1),
                    out_embs
                ],
                dim=1
            )
            npad = out_embs.shape[1] - data["attention_mask"].shape[1]
            attn = torch.nn.functional.pad(
                data["attention_mask"], (npad,0), value=1
            )
            rmb_preds = model(
                inputs_embeds=out_embs,
                attention_mask=attn
            ).logits
            rmb_preds = rmb_preds[:,cmpr.shape[1]:-1]
            return preds, rmb_preds
        return preds

    def infer(self, data, pred_len=None, rmb_task=False,
                                         temperature=1,
                                         ret_logits=False,
                                         ret_embs=False,
                                         cmpr=None):
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
                a bool indicating whether this is a rememberance task
                or a prediction task. true means rememberance.
            temperature: float
                a temperature parameter for softmax sampling. Set to
                low number for high confidence sampling, high value
                for low confidence sampling
            ret_logits: bool
                if true, will return the logits as well as the
                predicted ids
            ret_embs: bool
                if true, will return the embeddings as well as the
                predictions
            cmpr: None or torch tensor
                optional argument to reuse past compressions
        Returns:
            ret: dict {str: tensor}
                preds: tensor (B,S)
                    the prediction ids
                logits: tensor (B,S,N)
                    the pre-softmax id predictions
                embs: tensor (B,S,H)
                    the embeddings of the predictions
        """
        if cmpr is None:
            cmpr = self.compress(**data)

        out_embs = [ cmpr.to(self.rank) ]
        if rmb_task:
            tsk = self.embs.weight[self.tsk_ids[1]][None,None]
        else:
            tsk = self.embs.weight[self.tsk_ids[0]][None,None]
        out_embs.append( tsk.repeat(len(cmpr),1,1) )
        out_embs = torch.cat( out_embs, dim=1 )
        shape = (len(cmpr), pred_len+cmpr.shape[1])
        attn = torch.ones(shape,device=cmpr.get_device())

        preds = []
        logits = []
        emb_list = []

        past_key_values = None
        n_cmps = self.n_cmps
        t_embs = self.hf_model.transformer.get_input_embeddings()
        if pred_len is None: pred_len = data["output_ids"].shape[1]
        for i in range(pred_len):
            ret = self.hf_model(
              inputs_embeds=out_embs,
              attention_mask=attn[:,:out_embs.shape[1]],
              past_key_values=past_key_values,
              use_cache=True
            )
            past_key_values = ret.past_key_values
            logits.append(ret.logits[:,-1:])

            pred = logits[-1][:,-1]
            pred = torch.nn.functional.softmax(pred/temperature,dim=-1)
            pred = torch.multinomial(pred,1,replacement=True)
            preds.append(pred)
            out_embs = t_embs( pred.to(self.rank) )
            if not self.train_embs: out_embs = out_embs.data
            emb_list.append(out_embs)

        ret = { "preds": torch.cat(preds,dim=1) }
        if ret_logits:
            ret["logits"] = torch.cat(logits, dim=1)
        if ret_embs:
            ret["embs"] = torch.cat(emb_list, dim=1)
        return ret

    def causal_lm(self, input_ids=None, attention_mask=None,
                                        inputs_embeds=None,
                                        tforce=True,
                                        seed_len=None,
                                        pred_len=None,
                                        ret_logits=False,
                                        temperature=1):
        """
        Performs the traditional causal language modeling with or
        without teacher forcing.

        Args:
            input_ids: LongTensor (B,S)
                the token indices of the input sequence. The CMP token
                should be appended to the end of each sentence.
            attention_mask: LongTensor (B,S)
                attention mask for padding purposes. 0s mean padding.
            inputs_embeds: FloatTensor (B,S,H) optional
                the embeddings of the inputs. If both input_ids and
                this is argued, input_ids takes priority
            tforce: bool
                determines whether model should use teacher forcing for
                predictions or not.
            seed_len: int or None, must be greater than 0 if not None
                the number of inputs to seed the non-teacher forced
                predictions. Only applies if tforce is false. If None
                is argued, uses the whole input_ids or inputs_embeds
                as the seed.
            pred_len: int or None
                the number of prediction loops to perform. If None is
                argued, assumes the sequence length of the input_ids
                minus the seed_len.
            ret_logits: bool
                if true, will return logits as well as prediction idxs
            temperature: float
                a temperature parameter for softmax sampling. Set to
                low number for high confidence sampling, high value
                for low confidence sampling

        Returns:
            preds: torch tensor (B,seed_len + pred_len - 1)
                the prediction ids
            logits: torch tensor (B,seed_len + pred_len - 1,H)
                the prediction logits
        """
        if tforce:
            logits = self.hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            ).logits
            logits = logits[:,:-1]
            preds = logits.argmax(-1)
            if ret_logits:
                return preds, logits
            return preds

        t_embs = self.hf_model.transformer.get_input_embeddings()
        if input_ids is not None:
            if seed_len is None: seed_len = input_ids.shape[1]
            elif seed_len == 0:
                seed_len = 1
                if pred_len is not None: pred_len -= 1
            input_ids = input_ids[:,:seed_len]
            inputs_embeds = t_embs(input_ids)
        else:
            if seed_len is None: seed_len = inputs_embeds.shape[1]
            elif seed_len == 0:
                seed_len = 1
                if pred_len is not None: pred_len -= 1
            inputs_embeds = inputs_embeds[:,:seed_len]
            input_ids = torch.zeros_like(inputs_embeds[:,:,0])
        if not self.train_embs: inputs_embeds = inputs_embeds.data

        if pred_len is None:
            n_loops = attention_mask.shape[1]-seed_len
        else: n_loops = pred_len
        past_key_values = None
        logits = []
        preds = []
        for i in range(n_loops):
            ret = self.hf_model(
              inputs_embeds=inputs_embeds,
              attention_mask=attention_mask[:,:inputs_embeds.shape[1]],
              past_key_values=past_key_values,
              use_cache=True
            )
            past_key_values = ret.past_key_values
            logit = ret.logits
            if i==0:
                logits.append(logit[:,:-1])
                preds.append( input_ids[:,1:].to(logit.get_device()) )
            logits.append(logit[:,-1:])
            pred = logit[:,-1]
            pred = torch.nn.functional.softmax(pred/temperature,dim=-1)
            pred = torch.multinomial(pred,1,replacement=True)
            preds.append(pred)
            inputs_embeds = t_embs( pred.to(self.rank) )
            if not self.train_embs: inputs_embeds = inputs_embeds.data
        preds = torch.cat(preds,dim=1)
        if ret_logits:
            return preds, torch.cat(logits, dim=1)
        return preds


class LossWrapper(torch.nn.Module):
    """
    This class wraps the model to keep the loss calculations distributed
    on all GPUs. Otherwise one gpu is overloaded with computational
    costs.
    """
    def __init__(self, model, tokenizer, hyps, *args, **kwargs):
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
        self.loss_scale = 1./hyps.get("n_grad_loops",1)
        self.csl_scale = hyps.get("csl_scale",1.)
        self.loss_fxn = torch.nn.CrossEntropyLoss()
        self.hyps = hyps
        self.grad_clip = self.hyps.get("grad_clip", 10)

    def forward(self, data, ret_preds=False, seq_len=30,
                                             tforce=True,
                                             gen_targs=False,
                                             gen_ids=False,
                                             no_grad=False,
                                             temperature=1.):
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
            seq_len: int
                the length of the output sequence.
            ret_preds: bool
                if true, will return the predictions
            tforce: bool
                determines whether model should use teacher forcing for
                predictions or not.
            gen_targs: bool
                if true, the model will generate the ground truth labels
                on the fly
            gen_ids: bool
                if true, the model will use the generated ids rather than
                the logits as the ground truth
            temperature: float
                a temperature parameter for softmax sampling. Set to
                low number for high confidence sampling, high value
                for low confidence sampling
            no_grad: bool
                if true, this function will not call .backward() on
                the loss. If false, this function will only call
                .backward if in training mode.
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
        if gen_targs: 
            print("Generating Targets")
            with torch.no_grad():
                output_ids, output_logits = self.model.causal_lm(
                    input_ids=data["input_ids"],
                    attention_mask=data["attention_mask"],
                    tforce=tforce,
                    pred_len=seq_len,
                    ret_logits=True
                )
            idx = seq_len + self.hyps["seq_overlap"]
            data["output_ids"] = output_ids[:,-idx:].data
            del output_ids
            data["output_attn_mask"] = torch.ones_like(data["output_ids"])
            data["output_logits"] = output_logits[:,-idx:].data
            del output_logits

        n_cmps = self.model.n_cmps
        if self.hyps["rmb_task"]:
            preds, rmb_preds = self.model(data, tforce=tforce)
            # Need to remove CMP representation, the first n elements
            rmb_loss, rmb_acc = loss_and_acc(
                preds=rmb_preds, labels=data["input_ids"],
                attn=data["attention_mask"], loss_fxn=self.loss_fxn,
                loss_scale=self.loss_scale
            )
        else:
            preds = self.model(data, tforce=tforce)

        if gen_ids or "output_logits" not in data:
            loss, acc = loss_and_acc(
                preds, data["output_ids"], attn=data["output_attn_mask"],
                loss_fxn=self.loss_fxn, loss_scale=self.loss_scale
            )
        else:
            ps = preds.reshape(-1, preds.shape[-1])
            labels = data["output_ids"].reshape(-1)
            target = data["output_logits"]
            target = target.reshape(-1, output_logits.shape[-1])
            loss = self.loss_fxn(ps, target)*self.loss_scale
            argmax = torch.argmax(ps, dim=-1)
            acc = (argmax==labels).float().mean()

        sum_loss = loss
        if self.training and not no_grad:
            if self.hyps["rmb_task"]: sum_loss += rmb_loss
            sum_loss.backward()

        ret_dict = { "loss": loss, "acc": acc, }
        if self.hyps["rmb_task"]:
            ret_dict["rmb_loss"] = rmb_loss
            ret_dict["rmb_acc"] = rmb_acc
            if ret_preds:
                ret_dict["rmb_preds"] = rmb_preds
        if ret_preds:
            ret_dict["preds"] = preds

        # Do causal lm task after everything else to save space on
        # gpu.
        if self.hyps.get("csl_task",False):
            # Create complete sequence
            causal_inpts = {
              "input_ids": torch.cat([
                data["input_ids"],data["output_ids"]
              ], dim=1),
              "attention_mask": torch.cat([ 
                data["attention_mask"], data["output_attn_mask"]
              ], dim=1)
            }
            # Make predictions
            _, preds = self.model.causal_lm(
              **causal_inpts, tforce=True, ret_logits=True, seed_len=0
            )
            # Calculate loss
            loss, acc = loss_and_acc(
                preds, causal_inpts["input_ids"][:,1:],
                attn=causal_inpts["attention_mask"][:,1:],
                loss_fxn=self.loss_fxn,
                loss_scale=self.loss_scale
            )
            loss = loss*self.csl_scale
            # Backpropagate error
            if self.training:
                loss.backward()
            ret_dict["csl_loss"] = loss
            ret_dict["csl_acc"] = acc
            if ret_preds:
                ret_dict["csl_preds"] = preds
        return ret_dict

def loss_and_acc(preds, labels, attn, loss_fxn, loss_scale=1):
    """
    preds: torch float tensor (B,S,L)
        prediction logits
    labels: torch long tensor (B,S)
        prediction ids
    attn: torch tensor (B,S)
        padding mask. 1s mean include these tokens, 0 means ignore them
    loss_fxn: function
        the loss function for the predictions
    loss_scale: float
        a scalar that scales the loss
    """
    ps = preds.reshape(-1,preds.shape[-1])
    device = ps.get_device()
    try:
        labels = labels.reshape(-1).to(device)
        idx = attn.bool().reshape(-1).to(device)
    except:
        device = "cpu"
        labels = labels.reshape(-1).to(device)
        idx = attn.bool().reshape(-1).to(device)
    loss = loss_fxn(ps[idx],labels[idx])*loss_scale
    argmax = torch.argmax(ps[idx], dim=-1)
    acc = (argmax==labels[idx]).float().mean()
    return loss, acc

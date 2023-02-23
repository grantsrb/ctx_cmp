import torch
from transformers import AutoModelForCausalLM
import numpy as np

class SentenceAutoEncoder(torch.nn.Module):
    """
    Trains a new token type to compress a sentence into a single vector
    representation
    """
    def __init__(self, model_string, rank=0, torch_dtype="float32",
                                             device_map="auto",
                                             cmp_layer="half",
                                             rmb_task=False,
                                             n_cmps=3,
                                             n_tsks=2,
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
        n_embs: int
            the number of compression tokens
        n_tsks: int
            the number of task tokens. for the task ids, rmb is 1, sos
            is 0
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
        self.n_cmps = n_cmps
        self.n_tsks = n_tsks
        t = self.hf_model.transformer
        hsize = t.get_input_embeddings().weight.shape[-1]
        self.embs = torch.nn.Embedding(self.n_cmps+self.n_tsks, hsize)
        self.cmp_ids = [i for i in range(self.n_cmps)]
        # rmb is 1, sos is 0
        self.tsk_ids = [i+self.n_cmps for i in range(self.n_tsks)]

    def add_attrs(self, new_attrs):
        """
        Adds new attributes to the model.

        Args:
            new_attrs: dict {str: val}
                "new_attr": new_value
        """
        for k,v in new_attrs.items():
            setattr(self, k, v)

    def add_embeddings(self, n_embs):
        """
        Args:
            n_embs: int
                the number of embeddings to add
        """
        if n_embs <= 0: return
        embs = self.hf_model.transformer.get_input_embeddings()
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
        else:
            raise NotImplemented
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
                cmpr=cmpr
            )
            if self.rmb_task:
                rmb_preds = self.infer(
                    data,
                    pred_len=data["input_ids"].shape[1],
                    rmb_task=True,
                    cmpr=cmpr
                )
                return preds, rmb_preds
            return preds
        model = self.hf_model
        model_embs = model.transformer.get_input_embeddings()
        out_embs =  model_embs(data["output_ids"])
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

        # Optionally Handle Auxiliary, Rememberance Predictions
        if self.rmb_task:
            out_embs = model_embs( data["input_ids"] ).data
            rmb = model_embs.weight[self.tsk_ids[1]][None,None]
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
            return preds, rmb_preds
        return preds

    def infer(self, data, pred_len=None, rmb_task=False,
                                         temperature=1,
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
            ret_embs: bool
                if true, will return the embeddings as well as the
                predictions
            cmpr: None or torch tensor
                optional argument to reuse past compressions
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

        n_cmps = self.n_cmps
        t_embs = self.hf_model.transformer.get_input_embeddings()
        if pred_len is None: pred_len = data["output_ids"].shape[1]
        for i in range(pred_len):
            pred = self.hf_model(
              inputs_embeds=out_embs,attention_mask=attn[:,:i+n_cmps+1]
            ).logits
            if i == 0:
                preds = [ pred ]
            else:
                preds.append(pred[:,-1:])
            pred = pred[:,-1]
            pred = torch.nn.functional.softmax(pred/temperature,dim=-1)
            pred = torch.multinomial(pred,1,replacement=True)
            pred = t_embs( pred.to(self.rank) )
            out_embs = torch.cat( [out_embs, pred], dim=1 )
        preds.append(torch.zeros_like(preds[-1]))
        if ret_embs:
            return torch.cat(preds, dim=1), out_embs
        return torch.cat(preds, dim=1)

    def causal_lm(self, input_ids=None, attention_mask=None,
                                        inputs_embeds=None,
                                        tforce=True,
                                        seed_len=3,
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
            seed_len: int
                the number of inputs to seed the non-teacher forced
                predictions. Only applies if tforce is false
            temperature: float
                a temperature parameter for softmax sampling. Set to
                low number for high confidence sampling, high value
                for low confidence sampling

        Returns:
            preds: torch tensor (B,S,H)
                the prediction logits
        """
        if tforce:
            logits = self.hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            ).logits
            return logits

        t_embs = self.hf_model.transformer.get_input_embeddings()
        if input_ids is not None:
            inputs_embeds = t_embs(input_ids[:,:seed_len])

        n_loops = attention_mask.shape[1]-seed_len
        inputs_embeds = inputs_embeds[:,:seed_len]
        for i in range(n_loops):
            pred = self.hf_model(
              inputs_embeds=inputs_embeds,
              attention_mask=attention_mask[:,:i+seed_len]
            ).logits
            if i == 0:
                preds = [ pred ]
            else:
                preds.append(pred[:,-1:])
            pred = pred[:,-1]
            pred = torch.nn.functional.softmax(pred/temperature,dim=-1)
            pred = torch.multinomial(pred,1,replacement=True)
            pred = t_embs( pred.to(self.rank) )
            inputs_embeds = torch.cat( [inputs_embeds, pred], dim=1 )
        # We generally remove the final output during training, so this
        # helps reuse the same code
        preds.append(torch.zeros_like(preds[-1]))
        return torch.cat(preds, dim=1)


class LossWrapper(torch.nn.Module):
    """
    This class wraps the model to keep the loss calculations distributed
    on all GPUs. Otherwise one gpu is overloaded with computational
    costs.
    """
    def __init__(self, model, tokenizer, hyps, loss_scale=1.):
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
        self.hyps = hyps

    def forward(self, data, ret_preds=False, tforce=True):
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
            tforce: bool
                determines whether model should use teacher forcing for
                predictions or not.
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
        n_cmps = self.model.n_cmps
        if self.hyps["rmb_task"]:
            preds, rmb_preds = self.model(data, tforce=tforce)
            # Need to remove CMP representation, the first n elements
            ps = rmb_preds[:,n_cmps:-1].reshape(-1,rmb_preds.shape[-1])
            labels = data["input_ids"].reshape(-1)
            idx = (labels!=self.tokenizer.pad_token_id)
            rmb_loss = self.loss_fxn(ps[idx],labels[idx])*self.loss_scale
            argmax = torch.argmax(ps[idx], dim=-1)
            rmb_acc = (argmax==labels[idx]).float().mean()
        else:
            preds = self.model(data, tforce=tforce)
        # Need to remove CMP representation, the first n elements
        ps = preds[:,n_cmps:-1].reshape(-1, preds.shape[-1])
        labels = data["output_ids"].reshape(-1)
        idx = labels!=self.tokenizer.pad_token_id
        loss = self.loss_fxn(ps[idx], labels[idx])*self.loss_scale
        sum_loss = loss
        if self.training:
            if self.hyps["rmb_task"]: sum_loss += rmb_loss
            sum_loss.backward()
        argmax = torch.argmax(ps[idx], dim=-1)
        acc = (argmax==labels[idx]).float().mean()
        ret_dict = { "loss": loss, "acc": acc, }
        if self.hyps["rmb_task"]:
            ret_dict["rmb_loss"] = rmb_loss
            ret_dict["rmb_acc"] = rmb_acc
            if ret_preds:
                ret_dict["rmb_preds"] = rmb_preds[:,n_cmps:-1]
        if ret_preds:
            ret_dict["preds"] = preds[:,n_cmps:-1]
        return ret_dict


import torch
import torch.nn.functional as F
from typing import (
    Dict, 
    Any, 
    Optional,
)
from transformers import AutoModelForCausalLM
from omegaconf.dictconfig import DictConfig
from templates.model import AbstractSentenceAutoEncoder


class SentenceAutoEncoder(AbstractSentenceAutoEncoder):

    def __init__(self, 
                 model_cls: AutoModelForCausalLM,
                 args: DictConfig,
                 ):
        super().__init__()  
        self.args = args # see arguments.py
        self.model = model_cls # generic hf lm_head (or custom if custom model is used)
        self.t = self.model.transformer # lm_head transformer architecture 

        self.local_rank = self.args.training.local_rank if self.args.training.local_rank != -1 else "cpu" # uncomment / remove for gpu

        self.wte = self.t.get_input_embeddings().weight  # (wte) shape: [vocab_size, n_embs] # weights of the token embeddings 
        self.n_embs = self.wte.shape[1] # n_embs for each token
        
        # custom embedding for cmpr and tsk tokens (c CMPR toekn)
        embedding_size = args.training.n_cmps + args.training.n_tsks + int(args.training.sep_cmpr)
        embedding_dtype = torch.float32 if args.model.dtype == 'float32' else torch.float16
        self.model.embs = torch.nn.Embedding(embedding_size, self.n_embs, dtype=embedding_dtype) # shape: [embedding_size, n_embs] # TODO: check whether this should be self.model.embs vs self.embs
        
        # set device    
        if self.wte.get_device() > -1: 
            self.model.embs.to(self.wte.get_device())  

        # compression ids, task ids and the separator id
        self.cmp_ids = list(range(args.training.n_cmps)) 
        self.tsk_ids = list(range(args.training.n_cmps, args.training.n_cmps + args.training.n_tsks))
        self.sep_id = args.training.n_tsks + args.training.n_cmps  

        # input_ids = torch.LongTensor([2953, 262, 3726, 286, 262, 10037, 444, 8200, 5701, 1097, 16009, 2275, 11999, 373, 19233, 656, 262, 23327, 13735, 543])
        # attention_mask = torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) 
        # self.causal_lm(input_ids, attention_mask)
        
    @property
    def get_device(self) -> torch.device:
        return self.model.transformer.get_input_embeddings().weight.device
    
    def get_embeddings(self) -> torch.tensor:
        return self.model.transformer.get_input_embeddings()

    def add_embeddings(self,
                       new_embs: int,
                       ) -> None:
        if new_embs <= 0: return
        vocab_size, _ = self.get_embeddings().weight.shape 
        self.model.resize_token_embeddings(vocab_size + new_embs) # increase vocab size

    def compress(self,
                 input_ids: torch.LongTensor,
                 attention_mask: torch.LongTensor,
                 *args,
                 **kwargs) -> torch.tensor:
        """
        Compresses the input ids to a single vector. See BaseClass for details (Ignoring train_embs and sep_cmpr for now).
        """
        model_embs = self.get_embeddings()  # shape: [vocab_size, n_embs]
        inpt_embs = model_embs(input_ids)   # shape [batch_size, cmp_len, n_embs]

        cmp_embs = self.model.embs.weight[self.cmp_ids[0]:self.cmp_ids[-1]+1]  # shape: [n_cmps, n_embs]
        cmp_embs = cmp_embs[None].repeat(len(inpt_embs), 1, 1)  # shape: [batch_size, n_cmps, n_embs]

        inpt_embs = torch.cat([inpt_embs, cmp_embs], dim=1) # shape [batch_size, cmp_len + n_cmps, n_embs] # = k + c, i.e. compression sequence and concatenated compression tokens to right.
        
        # Pad attention_mask to match new input_emb dimension
        pad_dim = (0, self.args.training.n_cmps + int(self.args.training.sep_cmpr)) # (0, n_cmps + int(sep_cmpr))
        attention_mask = F.pad(attention_mask, pad_dim) # shape: [batch_size, cmp_len + n_cmps] # attention mask adapted such that final compression tokens c can attend to the entire sequence k + c

        fx = self.model.transformer( # see BaseModelOutputWithPastAndCrossAttentions (https://huggingface.co/docs/transformers/main_classes/output)
            inputs_embeds=inpt_embs,
            attention_mask=attention_mask,
            output_hidden_states=True, # if false, does not return hidden states
        ) 
        # Select the representational layer of choice, defaulting to the last layer
        n_layers = len(fx["hidden_states"]) # shape: n_blocks (eg 6 GPT2Blocks) + 1 (output_embedding) * [batch_size, cmp_len + n_cmps, n_embs]
        layer_dict = {'half': n_layers//2, None: -1}
        layer = self.args.model.cmp_layer
        chosen_layer = layer_dict.get(layer, layer)
        # `We take the representations at the CMPR locations (:,-self.args.training.n_cmps:) from that intermediate layer (chosen_layer) as the representation of the compressed sequence (k + c). 
        # We call this representation CMPR’. We discard the other k vectors (:,-self.args.training.n_cmps).`
        cmpr = fx['hidden_states'][chosen_layer][:,-self.args.training.n_cmps:] # shape: [batch_size, n_cmps, n_embs] the final n_cmps hidden states of the chosen layer, disregarding k
        return cmpr

    def forward(self,
                data: Dict[str, torch.LongTensor],
                ) -> torch.tensor:
        """
        Runs the forward pass. See BaseClass for details.
        """
        # skipping self.args.training.tforce for now (i.e. default = True)
        cmpr = self.compress(**data) # shape: [batch_size, n_cmps, n_embs]
        if not self.args.training.tforce:
            raise NotImplementedError(f"Not tfoce has to be implemented")
        model_embs = self.get_embeddings() # shape: [vocab_size, n_embs]
        out_embs =  model_embs(data["output_ids"]) # shape: [batch_size, seq_len, n_embs] -> this is of length n, the forward sequence
        sos = self.model.embs.weight[self.tsk_ids[0]][None,None] # shape: [1, 1, n_embs]
        out_embs = torch.cat( 
            [
                cmpr.to(self.local_rank), # the CMPR' 
                sos.repeat(len(cmpr),1,1), # The FWD
                out_embs.to(self.local_rank), # the n length forward context of len sequence len
            ],
            dim=1,
        ) # shape: [batch_size, n_cmps + sos + seq_len, n_embs]
        npad = out_embs.shape[1] - data["output_attn_mask"].shape[1] # int = (n_cmps + sos + seq_len) - seq_len 
        attn = F.pad(
            data["output_attn_mask"], 
            (npad, 0), # padding attention mask with 1s to be able to attend to the new tokens
            value=1, # inserts 1s
        ) # shape: [batch_size, n_cmps + sos + seq_len]
        preds = self.model(inputs_embeds=out_embs, attention_mask=attn).logits # shape [batch_size, n_cmps + sos + seq_len, vocab_size]
        preds = preds[:,cmpr.shape[1]:-1] # shape [batch_size, seq_len, vocab_size] # get rid of cmpr and sos on left of seq

        return preds
    
    def infer(self):
        """
        Performs inference without teacher forcing.
        """
        pass
    
    def causal_lm(self,
                  input_ids: torch.LongTensor,
                  attention_mask: torch.LongTensor,
                  inputs_embeds: Optional[torch.FloatTensor] = None, # TODO: what is the point of this?
                  ) -> torch.tensor:
        """
        Performs traditional causal language modeling with or
        without teacher forcing. See BaseClass for details.
        """
        if self.args.training.tforce:
            logits = self.model(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                inputs_embeds=inputs_embeds, # dont understand what it means if this is not None
            ).logits # shape: [batch_size, seq_len, vocab_size]
        else:
            raise NotImplementedError(f"tforce has to be implemented")
        
        logits = logits[:,:-1] # shape: [batch_size, seq_len, vocab_size] # get rid of last token from input_ids
        preds = logits.argmax(dim=-1) # shape: [batch_size, seq_len]
        if self.args.training.ret_logits:
            return preds, logits

        return preds


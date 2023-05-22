from abc import ABC, abstractmethod
from typing import (
    Dict,
    Optional,
)

import torch


# Abstract model classes
class AbstractSentenceAutoEncoder(ABC, torch.nn.Module):
    """
    Abstract base class for a sentence autoencoder.
    """
    @abstractmethod
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def get_device(self) -> torch.device:
        """
        Returns the device of the hf_model.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_embeddings(self) -> torch.tensor:
        """
        Returns a reference to the transformer embeddings.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_embeddings(self, 
                       new_embs: int) -> None:
        """
        Adds or resizes hf_model token embeddings shape to n_embs + new_embs.
        """
        raise NotImplementedError()

    @abstractmethod
    def compress(self,
                 input_ids: torch.LongTensor,
                 attention_mask: torch.LongTensor,
                 *args,
                 **kwargs) -> torch.tensor:
        """
        Compresses the input ids to a single vector.

        Args: 
            input_ids: LongTensor of shape [args.data.batch_size, args.compression.cmp_len]
                the token indices of the input sequence. The CMP token
                should be appended to the end of each sentence.
            attention_mask: LongTensor of shape [args.data.batch_size, args.compression.cmp_len]
                attention mask for padding purposes. 0s mean padding.
        Returns:
            cmpr: torch.tensor of shape [args.data.batch_sizee, 1, n_embs)
                the compressed representations
        """
        raise NotImplementedError()

    @abstractmethod
    def forward(self,
                data: Dict[str, torch.LongTensor],
                ) -> torch.tensor: # shape: [batch_size, seq_len, n_embs]
        """
        Args:
          data: dict
            "input_ids": LongTensor of shape [args.data.batch_size, args.compression.cmp_len]
                the token indices of the input sequence. The CMP token
                should be appended to the end of each sentence.
            "attention_mask": LongTensor of shape [args.data.batch_size, args.compression.cmp_len]
                attention mask for padding purposes. 0s mean padding.
            "output_ids": LongTensor of shape [args.data.batch_size, args.compression.seq_len]
                the token indices of the target sequence. An EOS token
                should be appended to the end of each sentence
            "output_attn_mask": LongTensor of shape [args.data.batch_size, args.compression.seq_len]
                attention mask for padding purposes. 0s mean padding.
        Returns:
            preds: tensor [batch_size, seq_len, vocab_size]
        """
        raise NotImplementedError()
    
    @abstractmethod
    def infer(self):
        """
        Performs inference without teacher forcing.
        """
        raise NotImplementedError()

    @abstractmethod
    def causal_lm(self,
                  input_ids: torch.LongTensor,
                  attention_mask: torch.LongTensor,
                  inputs_embeds: torch.FloatTensor,
                 ) -> torch.tensor:
        """
        Performs traditional causal language modeling with or without teacher forcing.
        Args:
            input_ids: LongTensor shape: [batch_size, seq_len]
                the token indices of the input sequence. The CMP token
                should be appended to the end of each sentence. 
            attention_mask: LongTensor shape: [batch_size, seq_len]
                attention mask for padding purposes. 0s mean padding.
            inputs_embeds: FloatTensor shape: [batch_size, seq_len, n_embs]
                the embeddings of the inputs. If both input_ids and
                this is argued, input_ids takes priority
        """
        raise NotImplementedError()
    

class AbstractLossWrapper(ABC, torch.nn.Module):
    """
    This class wraps the model to keep the loss calculations distributed
    on all GPUs. Otherwise one gpu is overloaded with computational
    costs.
    """
    @abstractmethod
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, 
                data: Dict[str, torch.LongTensor],
                seq_len: int,
                ret_preds: bool,
                tforce: bool,
                gen_targs: bool,
                gen_ids: bool,
                no_grad: bool,
                kl_scale: float, 
                temperature: float, 
                top_k: Optional[int]) -> Dict[str, torch.tensor]:
        """
        Runs forward pass.
        """
        raise NotImplementedError()
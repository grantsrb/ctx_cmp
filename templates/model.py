from abc import ABC, abstractmethod
from typing import (
    Dict,
    Optional,
    Any,
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
    def device(self) -> torch.device:
        """
        Returns the device of the hf_model.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_attrs(self, new_attrs: Dict[str, Any]) -> None:
        """
        Adds new attributes to the model.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_embeddings(self) -> torch.tensor:
        """
        Returns a reference to the transformer embeddings.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_embeddings(self, n_embs: int) -> None:
        """
        Adds or resizes hf_model token embeddings shape to n + n_embs.
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
        """
        raise NotImplementedError()

    @abstractmethod
    def forward(self,
                data: Dict[str, torch.LongTensor],
                tforce: bool) -> torch.tensor:
        """
        Runs the forward pass.
        """
        raise NotImplementedError()

    @abstractmethod
    def infer(self,
              data: Dict[str, torch.LongTensor],
              pred_len: Optional[int],
              rmb_task: bool,
              temperature: float,
              ret_logits: bool,
              ret_embs: bool,
              cmpr: Optional[torch.tensor]) -> Dict[str, torch.tensor]:
        """
        Performs inference without teacher forcing.
        """
        raise NotImplementedError()

    @abstractmethod
    def causal_lm(self,
                  input_ids: torch.LongTensor,
                  inputs_embeds: torch.FloatTensor,
                  tforce: bool,
                  seed_len: Optional[int],
                  pred_len: Optional[int],
                  ret_logits: bool,
                  temperature: float) -> torch.tensor:
        """
        Performs traditional causal language modeling with or without teacher forcing.
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
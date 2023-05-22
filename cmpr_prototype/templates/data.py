from typing import Dict, Tuple


from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset
from omegaconf.dictconfig import DictConfig

from transformers import AutoTokenizer
from datasets import DatasetDict, load_dataset

import torch    


class CompressionTokenizer():
    """
    Output tokens are the continuation of a sequence of seq_len. Inputs
    are the starting cmp_len tokens of the sequence of len seq_len.
    """
    def __init__(self,
                 tokenizer: AutoTokenizer, 
                 args: DictConfig):
        self.args = args
        self.cmp_len = args.compression.cmp_len
        self.seq_len = args.compression.seq_len
        self.overlap = args.compression.overlap
        self.min_seq = args.compression.min_seq
        self.tokenizer = tokenizer

        # Load dataset
        self.dataset = load_dataset(args.data.dataset_name,
            split=args.data.load_split,
            cache_dir=args.data.cache_dir,
            num_proc=args.data.num_proc)
        
        # Split into train/val
        split = self.dataset.train_test_split(test_size=args.data.val_size)
        self.train_set, self.validation_set = split["train"], split["test"]

        # Package into DatasetDict
        self.dataset_dict = DatasetDict({
            'train': self.train_set,
            'val': self.validation_set
        })

    def tokenize(self, 
                 examples: Dict[str, str], # ? 
                ) -> Dict:
        """
        Tokenize the examples.
        """
        self.tokenizer.padding_side = "right"
        cmps = self.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=self.cmp_len + self.seq_len,
            truncation=True,
            return_tensors="pt"
        )
        seqs = {
              "output_ids": cmps["input_ids"][:, self.cmp_len-self.overlap:],
              "output_attn_mask":cmps["attention_mask"][:,self.cmp_len-self.overlap:],
            }
        cmps["input_ids"] = cmps["input_ids"][:,:self.cmp_len]
        cmps["attention_mask"] = cmps["attention_mask"][:,:self.cmp_len]

        return {**cmps, **seqs}

    def get_data_loaders(self, 
                         ) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
        """
        Get data loaders.
        """
        # Tokenize
        if self.args.data.dataset_name == "stas/openwebtext-10k":
            tokenized_dataset = self.dataset_dict.map(
                self.tokenize,
                batched=self.args.data.batched,
                num_proc=self.args.data.num_proc,
                remove_columns=self.args.data.remove_columns,
                batch_size=self.args.data.batch_size,
            )
        else:
             raise NotImplementedError(f"Unknown dataset name {self.args.data.dataset_name}") # TODO: @add other datasets + funcs
        # Create data loaders
        data_loader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size=100, shuffle=True)
        val_loader = torch.utils.data.DataLoader(tokenized_dataset['val'], batch_size=100, shuffle=True)

        return tokenized_dataset["train"], tokenized_dataset["val"], data_loader, val_loader


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
        self.tokenizer = tokenizer
       
        # Load original dataset
        self.dataset = load_dataset(args.data.dataset_name,
                                    split=args.data.load_split,
                                    cache_dir=args.data.cache_dir,
                                    num_proc=args.data.num_proc)

        # Resize dataset (i.e. sample a subset of size args.data.size)
        if args.data.resize:
            self.dataset = self.dataset.shuffle(args.training.data_seed).select(range(args.data.size))

        # Resize block size for tokenizer (critical if tokenizer.model_max_length too large for GPU memory)
        if args.model.resize:
            self.block_size = args.model.block_size
        else:
            self.block_size = tokenizer.model_max_length
        
        # Split into train and validation sets
        split = self.dataset.train_test_split(test_size=args.data.val_size)
        self.train_set, self.validation_set = split["train"], split["test"]

        # Package into DatasetDict
        self.dataset_dict = DatasetDict({
            'train': self.train_set,
            'validation': self.validation_set
        })

    def tokenize(self, 
                 examples: Dict[str, str], # not sure if str, str is correct
                ) -> Dict:
        """
        Tokenize the examples.
        """
        self.tokenizer.padding_side = self.args.model.padding_side
        cmps = self.tokenizer(
            examples[self.args.data.text_column_name],
            padding=self.args.model.padding,
            max_length=self.args.compression.cmp_len + self.args.compression.seq_len,
            truncation=self.args.model.truncation,
            return_tensors=self.args.model.return_tensors,
        )
        seqs = {
              "output_ids": cmps["input_ids"][:, self.args.compression.cmp_len-self.args.compression.overlap:],
              "output_attn_mask": cmps["attention_mask"][:,self.args.compression.cmp_len-self.args.compression.overlap:],
            }
        cmps["input_ids"] = cmps["input_ids"][:,:self.args.compression.cmp_len]
        cmps["attention_mask"] = cmps["attention_mask"][:,:self.args.compression.cmp_len]

        seqs["labels"] = cmps["input_ids"].clone() # this was required by the trainer otherwise it crashed

        return {**cmps, **seqs}

    def get_data_loaders(self, 
                         ) -> Tuple[DatasetDict, DataLoader, DataLoader]:
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
        data_loader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size=self.args.data.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(tokenized_dataset['validation'], batch_size=self.args.data.batch_size, shuffle=True)
    
        return tokenized_dataset, data_loader, val_loader
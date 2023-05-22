# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training script, adapted from huggingface's run_clm.py example and from https://github.com/jayelm/gisting
"""
import logging
import os


import hydra
import torch
import math
from omegaconf.dictconfig import DictConfig
from transformers import (
    AutoTokenizer,
    AutoConfig, # need to add later
    AutoModelForCausalLM,
    Trainer, 
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from arguments import Arguments, global_setup
from data import CompressionTokenizer
from model import SentenceAutoEncoder

# Will error if the minimal version of Transformers is not installed. 
check_min_version("4.28.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


RMB = "|<RMB>|" # Extra characters are to ensure uniqueness
CMP = "|<CMP{}>|"
SOS = "|<SOS>|"


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(args: DictConfig) -> None:
    args: Arguments = global_setup(args)
    
    # TODO: add checkpoints/snapshots

    # Set seed before initializing model.
    set_seed(args.training.seed)

    # 1 Tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.model.cache_dir,
        "use_fast": args.model.use_fast_tokenizer,
        "truncation_side": args.model.truncation_side,
    }
    if args.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.tokenizer_name, **tokenizer_kwargs
        )
    elif args.model.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.model_name_or_path, **tokenizer_kwargs
        )
    # check for special tokens (need to chat how this relates to the different models' tokenizers and make it more clear)
    num_added = 0
    if tokenizer.pad_token is None:
        print(f"No Pad Token in {args.model.model_name_or_path}")
        print(f"EOS: {tokenizer.eos_token}")
        print(f"CLS: {tokenizer.cls_token}")
        print(f"SEP: {tokenizer.sep_token}")
        if tokenizer.eos_token is not None: # this is unclear 
            tokenizer.add_special_tokens(
                {"pad_token": tokenizer.eos_token}
            )
        else:
            num_added += tokenizer.add_special_tokens(
                {"pad_token": "|<PAD>|"}
            )
         #num_added += tokenizer.add_special_tokens({ ????
        #    "pad_token": "|<PAD>|",
        #    "eos_token": "|<EOS>|",
        #})
    args.model.pad_token = tokenizer.pad_token

    # 2 Datasets
    cmpr_tokenizer = CompressionTokenizer(tokenizer, args)
    tokenized_dataset, data_loader, val_loader = cmpr_tokenizer.get_data_loaders()

    # 3 Model 
    is_bloom = "bloomz" in args.model.model_name_or_path.lower().replace('/', '-').split('-')
    is_gpt2 = "distilgpt2" in args.model.model_name_or_path.lower().replace('/', '-').split('-') 
    is_tiny = "tinystories" in args.model.model_name_or_path.lower().replace('/', '-').split('-') # for debugging on cpu


    if is_bloom:
        model_cls = AutoModelForCausalLM.from_pretrained(args.model.model_name_or_path,
                                                         torch_dtype=torch.float32 if args.model.dtype == 'float32' else torch.float16) # half precision does not work on cpu                                   
    elif is_gpt2:
        model_cls = AutoModelForCausalLM.from_pretrained(args.model.model_name_or_path,
                                                         torch_dtype=torch.float32 if args.model.dtype == 'float32' else torch.float16) # half precision does not work on cpu                         
    elif is_tiny:
        model_cls = AutoModelForCausalLM.from_pretrained(args.model.model_name_or_path,
                                                         torch_dtype=torch.float32 if args.model.dtype == 'float32' else torch.float16) # half precision does not work on cpu                         
    else:
        raise ValueError(f"Model type {args.model.model_name_or_path} not supported")
    
    custom_lm = SentenceAutoEncoder(model_cls, args)
    if num_added > 0: custom_lm.add_embeddings(num_added) # resize vocab size if new special tokens have been added

    print("--------MODEL--------")
    print(model_cls)
    # get size of model
    print("--------MODEL SIZE--------")
    print(f"Model size: {sum(p.numel() for p in custom_lm.model.parameters() if p.requires_grad)}")
    print(custom_lm.model) # custom_lm now has two transformers: .t which is the original transformer
                           # and .model which is the custom one with an additional (embs): nn.Embedding(args.training.n_cmps + args.training.n_tsks, n_embs)
    print("---------------------")


    # # 4 Training
    # TODO: use pytorch instead to do ddp and build your own custom trainer and run with accelerator
    trainer = Trainer( # TODO: 
        model=custom_lm.model, # this is prob wrong? NOTE: it has the new (embs) layers added, so the architecture has changed but unclear to me if correct.
        args=args.training,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )   

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

if __name__ == "__main__":
    main()

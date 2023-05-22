"""
Based on 
    - https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/trainer#transformers.TrainingArguments 
    - https://github.com/jayelm/gisting/blob/main/src/arguments.py
    - https://github.com/grantsrb/ctx_cmp/tree/master 

Script currently contains three types of arguments:
    - ModelArguments
    - CustomTrainingArguments (Based on HuggingFace TrainingArguments)
    - DataArguments
    - CompressionArguments: Custom arguments related to the token compression

To overwrite default values, use ./config/config.yaml
"""
import logging
import socket
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Any

import datasets
import transformers
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from transformers import TrainingArguments


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Model Arguments.
    """
    cache_dir: Optional[str] = field(
        default="/data/jphilipp/research-projects/ctx-cmp-2023/.cache/huggingface",
        metadata={
            "help": (
                "Where to store pre-trained models and datasets."
            )
        },
    )
    model_name_or_path: str = field(
        default="bigscience/bloomz-560m", 
        metadata={
            "help": "The model checkpoint for weights initialization."
        }
    )
    pretrained: bool = field(
        default=True,
        metadata={
            "help": (
                "Use pretrained model. This replaces old run_clm.py script "
                "`model_type` argument."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name_or_path."
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use one of the fast tokenizer (backed by the tokenizers "
                "library) or not."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": (
                "The specific model version to use (can be a branch name, tag name "
                "or commit id)."
            )
        },
    )
    truncation: bool = field(
        default=True,
        metadata={
            "help": (
                "Truncation strategy to use. Choose from [True, False]."
            )
        },
    )
    truncation_side: str = field(
        default="right",
        metadata={
            "help": (
                "Truncation side to use. Choose from ['left', 'right', 'longest_first', 'do_not_truncate']."
            )
        },
    )
    padding_side: str = field(
        default="right",
        metadata={
            "help": (
                "Padding side to use."
            )
        },
    )
    padding: str = field(
        default="max_length",
        metadata={
            "help": (
                "Padding strategy to use for tokenizer()."
            )
        },
    )
    return_tensors: str = field(
        default="pt",
        metadata={
            "help": (
                "Return tensors argument for tokenizer()."
            )
        },
    )
    block_size: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "Adjusted block size if model_max_length is too long for GPU."
            )
        },
    )
    resize: bool = field(
        default=False,
        metadata={
            "help": (
                "If we want to resize the block size."
            )
        },
    )
    dtype: str = field(
        default="float32",
        metadata={
            "help": (
                "Floating point precision."
            )
        },
    )
    device_map : str = field(
        default="auto",
        metadata={
            "help": (
                "Determines whether you want to use model parallel or not."
            )
        },
    )
    cmp_layer: Any = field(
        default="half",
        metadata={
            "help": (
                "The layer from the transformer to use for the compression"
                "token. Str argument can be 'half' denoting the middle layer"
                "of the transformer. None defaults to last layer. Int indexes layer."
            )
        },
    )
    rmb_task: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, will assume that there is an auxiliary"
                "memory reconstruction objective."
            )
        },
    )
    pad_token: Optional[str] = field(
        default="|<PAD>|",
        metadata={
            "help": (
                "Padding token."
            )
        },
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Training Arguments (non-exhaustive, currently only including relevant ones + custome, missing args will default to HF TrainingArguments)
    """
    output_dir: str = field(
        default="../output_dir",
        metadata={
            "help": (
                    "The output directory where the model predictions and checkpoints will be written."
            )   
        },
    )
    seed: Optional[int] = field(
        default=42,
        metadata={
            "help": (
                "Random seed to be used with data samplers. If not set, random generators for data sampling will use the same seed as seed." 
                "This can be used to ensure reproducibility of data sampling, independent of the model seed."
            )
        },
    )
    data_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Random seed to be used with data samplers. If not set, random generators for data sampling will use the same seed as seed." 
                "This can be used to ensure reproducibility of data sampling, independent of the model seed."
            )
        },
    )
    generation_max_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The max_length to use on each evaluation loop when predict_with_generate=True."
                "Will default to the max_length value of the model configuration."
            )
        },
    )
    max_steps: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "If set to a positive number, the total number of training steps to perform."
                "Overrides num_train_epochs. In case of using a finite iterable dataset the training" 
                "may stop before reaching the set number of steps when all data is exhausted."
            )
        },
    )
    n_cmps: int = field(
        default=3,
        metadata={
            "help": (
                "Number of compression tokens."
            )
        },
    )
    n_tsks: int = field(
        default=2,
        metadata={
            "help": (
                "The number of task tokens. for the task ids, rmb is 1, sos is 0."
            )
        },
    )
    train_embs: bool = field(
        default=False,
        metadata={
            "help": (
                "If false, uses data of transformer embedding parameters"
                "instead of embedding parameters directly."
            )
        },
    )
    proj_cmpr: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, projects the cmpr' representations using a linear"
                "weight matrix before using them as input to the forward/"
                "auxiliary tasks."
            )
        },
    )
    proj_hid_mult: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If an integer value is argued, the projection will use a 2"
                "layer neural net instead of a single linear projection."
                "This argument specifies the size of the hidden layer as"
                "proj_hid_mult*h_size"
            )
        },
    )
    sep_cmpr: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, an additional embedding is inserted between"
                "the cmpr token and the compression sequence."
            )
        },
    )
    tforce: bool = field(
        default=True,
        metadata={
            "help": (
                "If true, will use teacher forcing during training."
            )
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": (
                "A temperature parameter for softmax sampling. Set to"
                "low number for high confidence sampling, high value"
                "for low confidence sampling."
            )
        },
    )
    ret_logits: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If true, will return logits as well as prediction idxs when performing model.causal_lm()"
            )
        },
    )
    backend: str = field(
        default="nccl",
        metadata={
            "help": (
                "The backend to use for distributed training."
            )
        },
    )

    _run_post_init: bool = False

    def __post_init__(self):
        # Don't run post-init until ready to convert to TrainingArgs (https://github.com/jayelm/gisting/blob/main/src/arguments.py)
        if self._run_post_init:
            super().__post_init__()
    

@dataclass
class DataArguments:
    """
    Data Arguments.
    """
    cache_dir: Optional[str] = field(
        default="/data/jphilipp/research-projects/ctx-cmp-2023/.cache/huggingface",
        metadata={
            "help": (
                "Where to store/cache downloaded dataset."
            )
        },
    )
    dataset_name: Optional[str] = field(
        default="openwebtext",
        metadata={
            "help": (
                "The name of the dataset for datasets.load_dataset(dataset_name)."
            ),
        },
    )
    load_split: Optional[str] = field(
        default="train",
        metadata={
            "help": (
                "The split of the dataset loaded using datasets.load_dataset()."
            ),
        },
    )
    num_proc: Optional[int] = field(
        default=4,
        metadata={
            "help": (
                "Multiprocessing significantly speeds up processing by parallelizing processes on the CPU."
                "Sets the num_proc parameter in map() to set the number of processes to use:"
            )
        },
    )
    batched: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                 "Operate on batches by setting batched=True. The default batch size is 1000, but you can adjust it with the batch_size parameter." 
            )
        },
    )
    batch_size: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                 "Size of each batch to be processed by map()." 
            )
        },
    )
    remove_columns: Optional[List[str]] = field(
        default_factory=lambda: ["text"],
        metadata={
            "help": (
                "Columns to remove from the dataset after tokenization."
            )
        },
    )
    resize: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Resize the dataset to a smaller size."
            )
        },
    )
    size: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "Size of random sample of dataset to use."
            )
        },
    )
    val_size: Optional[float] = field(
        default=0.2,
        metadata={
            "help": (
                "Size of the validation set."
            )
        },
    )
    text_column_name: str = field(
        default="text",
        metadata={
            "help": (
                "where in examples to find the data we are interested in."
            )
        },
    )
    

@dataclass
class CompressionArguments:
    """
    Compression Arguments.
    """
    cmp_len: Optional[int] = field(
        default=20,
        metadata={
            "help": (
                "The length of the compreesion."
            )
        },
    )
    seq_len: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "The length of the post compression, sequence chunk that will be processed by the transformer."
                 "So, the total predictive sequence will be seq_len+cmp_len tokens long."
            )
        },
    )
    overlap: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "The number of overlapping tokens from the compression sequence to the rest of the sequence."
            )
        },
    )
    min_seq: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "The minimum length predictive portion. Total sequence lengths must be greater than or equal to cmp_len + min_seq."
            )
        },
    )



@dataclass
class Arguments:
    model: ModelArguments = ModelArguments()
    training: CustomTrainingArguments = CustomTrainingArguments()
    data: DataArguments = DataArguments() 
    compression: CompressionArguments = CompressionArguments()


cs = ConfigStore.instance()
cs.store(name="base_config", node=Arguments) # needs to be same name as defaults in /config/config.yaml


def global_setup(args: DictConfig) -> Arguments:
    """Global setup of arguments."""
    # Print the hostname
    hostname = socket.gethostname()
    logger.info(f"Running on {hostname}")

     # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Convert args to the actual dataclass object, to enable methods.  Need to
    # delete _n_gpu, a property that TrainingArgs init doesn't expect.
    del args.training._n_gpu
    # Dirty hack: only run post init when we're ready to convert to TrainingArgs
    args.training._run_post_init = True
    args = OmegaConf.to_object(args)

    log_level = args.training.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    #  uncomment once running
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.training.local_rank}, device: {args.training.device}, "
        f"n_gpu: {args.training.n_gpu}"
        f" distributed training: {bool(args.training.local_rank != -1)}, 16-bits "
        f"training: {args.training.fp16}, bf16 training: {args.training.bf16}"
    )
    logger.info(f"Training/evaluation parameters {args.training}")


    return args
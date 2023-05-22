"""
Based on https://github.com/jayelm/gisting/blob/main/src/arguments.py.

Script currently contains three types of arguments:
    - ModelArguments: Arguments related to the model.
    - CustomTrainingArguments: Arguments related to training.
    - DataArguments: Arguments related to the data.
"""
import logging
import socket
from dataclasses import dataclass, field
from typing import Optional, List

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
    truncation_side: str = field(
        default="right",
        metadata={
            "help": (
                "Truncation side to use. Choose from ['left', 'right', 'longest_first', 'do_not_truncate']."
            )
        },
    )
    

@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Training Arguments with type fixes (TODO: need to adjust the types to strs so that they typecheck works).
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

    _run_post_init: bool = False

    def __post_init__(self):
        # Don't run post-init until ready to convert to TrainingArgs (check jesse's repo for details on converting args to str/optional)
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
        default=1000,
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
    val_size: Optional[float] = field(
        default=0.2,
        metadata={
            "help": (
                "Size of the validation set."
            )
        },
    )

# # adding custom arguments for  self.cmp_len = args.compression.cmp_len
#         self.seq_len = args.compression.seq_len
#         self.overlap = args.compression.overlap

#         self.min_seq = args.compression.min_seq

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
cs.store(name="base_config", node=Arguments)


def global_setup(args: DictConfig) -> Arguments:
    """Global setup of arguments."""
    # Print the hostname
    hostname = socket.gethostname()
    logger.info(f"Running on {hostname}")

    args = OmegaConf.to_object(args)

    return args
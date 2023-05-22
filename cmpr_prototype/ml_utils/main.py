"""
This is an example of how to use this repository. Create a training
function like the example below. Then import it and use `run_training`
to perform trainings.
"""

import torch
from .training import run_training
import torch.multiprocessing as mp

torch.autograd.set_detect_anomaly(True)

def train(rank, hyps, verbose=True, *args, **kwargs):
    """
    This is an example of a main training function that will be used
    in the `run_training` function.

    Args:
        rank: int
            the index of the distributed training system.
        hyps: dict
            a dict of hyperparams
            keys: str
            vals: object
        verbose: bool
            determines if the function should print status updates
    """
    raise NotImplemented

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    run_training(train)


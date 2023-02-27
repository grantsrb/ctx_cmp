"""
This is an example of how to use this repository. Create a training
function like the example below. Then import it and use `run_training`
to perform trainings.
"""

from training import train
import torch
from ml_utils.training import run_training
import torch.multiprocessing as mp

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    run_training(train)


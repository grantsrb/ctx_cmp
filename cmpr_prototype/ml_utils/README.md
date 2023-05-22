# Machine Learning Utils
This repo contains functions and classes that can be used in deep learning projects.

## Vocab
`save_root`: path to the folder in which the experiment folder resides
    or will reside. i.e. `/home/user/project_saves/`

`exp_folder`: this is the path to the experiment folder (the folder
    that contains all model training folders for a given `exp_name`.)
    i.e. `/home/user/project_saves/my_exp_name/`

`model_folder`: refers to the model folder without the full path. This
    is the folder name that contains the checkpt files.
    i.e. `my_exp_name_0_lr0.001`

`save_folder`: this is the full path to where the checkpoint files are
    actually saved. the `hyperparams.json` and the `hyperparams.txt` are
    both also saved to this path.
    i.e. `/home/user/project_saves/my_exp_name/my_exp_name_0_lr0.001`

## Setup
Clone this repo as a submodule in your project. Then install all
necessary packages locally:
```sh
python3 -m pip install --user -r requirements.txt
```








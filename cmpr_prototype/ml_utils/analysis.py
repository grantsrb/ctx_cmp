import torch
import os
import pandas as pd
import ml_utils.save_io as io
from tqdm import tqdm

def get_stats_dataframe(model_folders,
                        names=None,
                        incl_hyps=False,
                        verbose=False):
    """
    Sorts through all checkpoints of all models contained within the
    model_folders array.  Returns a dataframe with all of their stats.
    Includes the following columns for ease of use:

        epoch: the epoch of the row in the dataframe
        model_path: the full path to the model folder
        model_num: the number associated with this model (the number
            surrounded by underscores following the experiment name)
        name: a name for the model. if names is not none, will draw
            from the corresponding entry in that array. Otherwise
            defaults to "model0", "model1", ...

    ASSUMES ALL CHECKPOINTS HAVE A "stats" KEY THAT CONTAINS THE SAME
    KEYS OVER ALL CHECKPOINTS OF ALL MODELS

    Args:
        model_folders: list of str
            the full paths to each of the model experiments to be
            included in the dataframe
        names: list of str or None
            optional argument to name each model. If not None, the
            array must be as long as the model_folders array. if None,
            each name defaults to "model<int>" where <int> is repaced
            by the order in which this model was processed in the loop
        incl_hyps: bool
            if true, the hyperparameters will also be loaded into
            the dataframe
    """
    extras = ["epoch", "model_path", "model_num", "name"]
    if len(model_folders) == 0:
        return pd.DataFrame({ k: [] for k in extras })
    if names is None: 
        names = ["model"+str(i) for i in range(len(model_folders))]
    assert len(names) == len(model_folders)
    
    main_df = None
    
    # Add Each Model's Data to Main Data Frame
    z = zip(model_folders, names)
    if verbose: z = tqdm(z)
    for model_folder, name in z:
        checkpts = io.get_checkpoints(model_folder)
        if len(checkpts) == 0: continue
        df = None
        for ci,path in enumerate(checkpts):
            checkpt = io.load_checkpoint(path)
            if ci==0:
                df = {k:[] for k in checkpt["stats"].keys()}
                if incl_hyps:
                    hyps = {k:[] for k in checkpt["hyps"].keys()}
                df["phase"] = []
                df["epoch"] = []
            for k in {*checkpt["stats"].keys(), *df.keys()}:
                if k not in df:
                    df[k] = [None for i in range(ci)]
                if k not in checkpt["stats"]:
                    try:
                        if k == "phase": df[k].append(checkpt["phase"])
                        elif k == "epoch": df[k].append(checkpt["epoch"])
                        else: df[k].append(None)
                    except:
                        if k == "epoch":
                            epoch=int(path.split(".")[0].split("_")[-1])
                            df[k].append(epoch)
                            checkpt[k] = epoch
                            torch.save(checkpt, os.path.expanduser(path))
                else:
                    df[k].append(checkpt["stats"][k])
            if incl_hyps:
                for k in hyps.keys():
                    if k in checkpt["hyps"]:
                        hyps[k].append(checkpt["hyps"][k])
        try:
            if incl_hyps: df = {**df, **hyps}
            df = pd.DataFrame(df)
        except:
            for k in df.keys():
                print(k, "-", len(df[k]))
        df["model_path"] = model_folder
        df["name"] = name
        df["model_num"] = checkpt["hyps"]["exp_num"]
        if main_df is None: main_df = df
        else: main_df = main_df.append(df, sort=True)
    return main_df

def get_hyps_dataframe(model_folders, names=None):
    """
    Sorts through each model contained within the model_folders array.
    Returns a dataframe with all of their hyperparameters.
    Includes the following columns for ease of use:

        model_path: the full path to the model folder
        model_num: the number associated with this model (the number
            surrounded by underscores following the experiment name)
        name: a name for the model. if names is not none, will draw
            from the corresponding entry in that array. Otherwise
            defaults to "model0", "model1", ...

    ASSUMES ALL CHECKPOINTS HAVE A "hyps" KEY THAT CONTAINS THE SAME
    KEYS OVER ALL CHECKPOINTS OF ALL MODELS

    Args:
        model_folders: list of str
            the full paths to each of the model experiments to be
            included in the dataframe
        names: list of str or None
            optional argument to name each model. If not None, the
            array must be as long as the model_folders array. if None,
            each name defaults to "model<int>" where <int> is repaced
            by the order in which this model was processed in the loop
    """
    extras = ["model_path", "model_num", "name"]
    if len(model_folders) == 0:
        return pd.DataFrame({ k: [] for k in extras })
    if names is None: 
        names = [None for i in range(len(model_folders))]
    assert len(names) == len(model_folders)
    
    checkpt = None
    i = 0
    while checkpt is None and i < len(model_folders):
        checkpt = io.load_checkpoint(model_folders[i])
        i+=1
    if checkpt is None: return pd.DataFrame({ k: [] for k in extras })
    hyps = checkpt["hyps"]
    keys = [k for k in hyps.keys()]
    df = {k:[] for k in keys+extras}
    main_df = pd.DataFrame(df)
    
    for model_folder, name in zip(model_folders, names):
        checkpt = io.load_checkpoint(model_folder)
        df = {k:[checkpt["hyps"][k]] for k in checkpt["hyps"].keys()}
        df = pd.DataFrame(df)
        df["model_path"] = model_folder
        num = checkpt["hyps"]["exp_num"]
        if name is None:
            name = checkpt["hyps"]["exp_name"] + str(num)
        df["name"] = name
        df["model_num"] = num
        main_df = main_df.append(df, sort=True)
    return main_df

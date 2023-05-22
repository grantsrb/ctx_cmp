import torch
import pickle
import os
import json

BEST_CHECKPT_NAME = "best_checkpt_0.pt.best"

def save_checkpt(save_dict, save_folder, save_name, epoch, ext=".pt",
                                                    del_prev_sd=True,
                                                    best=False):
    """
    Saves a dictionary that contains a statedict

    save_dict: dict
        a dictionary containing all the things you want to save
    save_folder: str
        the full path to save the checkpt file to
    save_name: str
        the name of the file that the save dict will be saved to. This
        function will automatically append the epoch to the end of the
        save name followed by the extention, `ext`.
    epoch: int
        an integer to be associated with this checkpoint
    ext: str
        the extension of the file
    del_prev_sd: bool
        if true, the state_dict of the previous checkpoint will be
        deleted
    best: bool
        if true, additionally saves this checkpoint as the best
        checkpoint under the filename set by BEST_CHECKPT_NAME
    """
    if del_prev_sd and epoch is not None:
        prev_paths = get_checkpoints(save_folder)
        if len(prev_paths) > 0:
            prev_path = prev_paths[-1]
            delete_sds(prev_path)
        elif epoch != 0:
            print("Failed to find previous checkpoint")
    if epoch is None: epoch = 0
    path = "{}_{}{}".format(save_name,epoch,ext)
    path = os.path.join(save_folder, path)
    path = os.path.abspath(os.path.expanduser(path))
    torch.save(save_dict, path)
    if best: save_best_checkpt(save_dict, save_folder)

def delete_sds(checkpt_path):
    """
    Deletes the state_dicts from the argued checkpt path.

    Args:
        checkpt_path: str
            the full path to the checkpoint
    """
    if not os.path.exists(checkpt_path): return
    checkpt = load_checkpoint(checkpt_path)
    keys = list(checkpt.keys())
    for key in keys:
        if "state_dict" in key or "optim_dict" in key:
            del checkpt[key]
    torch.save(checkpt, checkpt_path)

def save_best_checkpt(save_dict, folder):
    """
    Saves the checkpoint under the name set in BEST_CHECKPT_PATH to the
    argued folder

    save_dict: dict
        a dictionary containing all the things you want to save
    folder: str
        the path to the folder to save the dict to.
    """
    path = os.path.join(folder,BEST_CHECKPT_NAME)
    path = os.path.abspath(path)
    torch.save(save_dict,path)

def get_checkpoints(folder, checkpt_exts={'p', 'pt', 'pth'}):
    """
    Returns all .p, .pt, and .pth file names contained within the
    folder. They're sorted by their epoch.

    BEST_CHECKPT_PATH is not included in this list. It is excluded using
    the assumption that it has the extension ".best"

    folder: str
        path to the folder of interest
    checkpt_exts: set of str
        a set of checkpoint extensions to include in the checkpt search.

    Returns:
        checkpts: list of str
            the full paths to the checkpoints contained in the folder
    """
    folder = os.path.expanduser(folder)
    assert os.path.isdir(folder)
    checkpts = []
    for f in os.listdir(folder):
        splt = f.split(".")
        if len(splt) > 1 and splt[-1] in checkpt_exts:
            path = os.path.join(folder,f)
            checkpts.append(path)
    def sort_key(x): return int(x.split(".")[-2].split("_")[-1])
    checkpts = sorted(checkpts, key=sort_key)
    return checkpts

def foldersort(x):
    """
    A sorting key function to order folder names with the format:
    <path_to_folder>/<exp_name>_<exp_num>_<ending_folder_name>/

    Assumes that the experiment number will always be the rightmost
    occurance ofan integer surrounded by underscores (i.e. _1_)

    x: str
    """
    if x[-1] == "/": x = x[:-1]
    splt = x.split("/")
    if len(splt) > 1: splt = splt[-1].split("_")
    else: splt = splt[0].split("_")
    for s in reversed(splt[1:]):
        try:
            return int(s)
        except:
            pass
    print("Folder sort splt:", x)
    assert False

def prep_search_keys(s):
    """
    Removes unwanted characters from the search keys string.
    """
    return s.replace(
            " ", ""
        ).replace(
            "[", ""
        ).replace(
            "]", ""
        ).replace(
            "\'", ""
        ).replace(
            ",", ""
        ).replace(
            "/", ""
        )

def is_model_folder(path, exp_name=None):
    """
    checks to see if the argued path is a model folder or otherwise.
    i.e. does the folder contain checkpt files and a hyperparams.json?

    path: str
        path to check
    exp_name: str or None
    """
    check_folder = os.path.expanduser(path)
    if exp_name is not None:
        # Remove ending slash if there is one
        if check_folder[-1]=="/": check_folder = check_folder[:-1]
        folder_splt = check_folder.split("/")[-1]
        # Need to split on underscores and check for entirety of
        # exp_name because exp_name is only the first part of any
        # model folder
        name_splt = exp_name.split("_")
        folder_splt = folder_splt.split("_")
        match = True
        for i in range(len(name_splt)):
            if i >= len(folder_splt) or name_splt[i] != folder_splt[i]:
                match = False
                break
        if match: return True
    contents = os.listdir(check_folder)
    for content in contents:
        if ".pt" in content or "hyperparams" in content:
            return True
    return False

def get_model_folders(exp_folder, incl_full_path=False):
    """
    Returns a list of paths to the model folders contained within the
    argued exp_folder

    exp_folder - str
        full path to experiment folder
    incl_full_path: bool
        include extension flag. If true, the expanded paths are
        returned. otherwise only the end folder (i.e.  <folder_name>
        instead of exp_folder/<folder_name>)

    Returns:
        list of folder names (see incl_full_path for full path vs end
        point)
    """
    folders = []
    exp_folder = os.path.expanduser(exp_folder)
    if exp_folder[-1]=="/":
        exp_name = exp_folder[:-1].split("/")[-1]
    else:
        exp_name = exp_folder.split("/")[-1]
    if ".pt" in exp_folder[-4:]:
        # if model file, return the corresponding folder
        folders = [ "/".join(exp_folder.split("/")[:-1]) ]
    else:
        for d, sub_ds, files in os.walk(exp_folder):
            for sub_d in sub_ds:
                check_folder = os.path.join(d,sub_d)
                if is_model_folder(check_folder, exp_name=exp_name):
                    if incl_full_path:
                        folders.append(check_folder)
                    else:
                        folders.append(sub_d)
        if is_model_folder(exp_folder): folders.append(exp_folder)
    folders = list(set(folders))
    if incl_full_path: folders = [os.path.expanduser(f) for f in folders]
    return sorted(folders, key=foldersort)

def load_checkpoint(path, use_best=False):
    """
    Loads the save_dict into python. If the path is to a model_folder,
    the loaded checkpoint is the BEST checkpt if available, otherwise
    the checkpt of the last epoch

    Args:
        path: str
            path to checkpoint file or model_folder
        use_best: bool
            if true, will load the best checkpt based on validation metrics
    Returns:
        checkpt: dict
            a dict that contains all the valuable information for the
            training.
    """
    path = os.path.expanduser(path)
    hyps = None
    if os.path.isdir(path):
        best_path = os.path.join(path,BEST_CHECKPT_NAME)
        if use_best and os.path.exists(best_path):
            path = best_path 
        else:
            checkpts = get_checkpoints(path)
            if len(checkpts)==0: return None
            path = checkpts[-1]
    data = torch.load(path, map_location=torch.device("cpu"))
    data["loaded_path"] = path
    if "hyps" not in data: 
        data["hyps"] = get_hyps(path)
    if "epoch" not in data:
        # Untested!!
        ext = path.split(".")[-1]
        data["epoch"] = int(path.split("."+ext)[0].split("_")[-1])
        torch.save(data, path) 
    return data

def load_model(path, models, load_sd=True, use_best=False,
                                           hyps=None,
                                           verbose=True):
    """
    Loads the model architecture and state dict from a .pt or .pth
    file. Or from a training save folder. Defaults to the last check
    point file saved in the save folder.

    path: str or dict
        either .pt,.p, or .pth checkpoint file; or path to save folder
        that contains multiple checkpoints. if dict, must be a checkpt
        dict.
    models: dict
        A dict of the potential model classes. This function is
        easiest if you import each of the model classes in the calling
        script and simply pass `globals()` as the argument for this
        parameter. If None is argued, `globals()` is used instead.
        (can usually pass `globals()` as the arg assuming you have
        imported all of the possible model classes into the script
        that calls this function)

        keys: str
            the class names of the potential models
        vals: Class
            the potential model classes
    load_sd: bool
        if true, the saved state dict is loaded. Otherwise only the
        model architecture is loaded with a random initialization.
    use_best: bool
        if true, will load the best model based on validation metrics
    hyps: dict (optional)
        if you would like to argue your own hyps, you can do that here
    """
    if type(path) == type(str()):
        path = os.path.expanduser(path)
        hyps = None
        data = load_checkpoint(path,use_best=use_best)
    else: data = path
    if 'hyps' in data:
        kwargs = data['hyps']
    elif 'model_hyps' in data:
        kwargs = data['model_hyps']
    else:
        kwargs = get_hyps(path)
    if models is None: models = globals()
    model = models[kwargs['model_type']](**kwargs)
    if "state_dict" in data and load_sd:
        print("loading state dict")
        try:
            model.load_state_dict(data["state_dict"])
        except:
            print("failed to load state dict, attempting fix")
            sd = data["state_dict"]
            keys = {*sd.keys(), *model.state_dict().keys()}
            for k in keys:
                if k not in sd: sd[k] = getattr(model, k)
                if k not in model: setattr(model, k, sd[k])
            print("succeeded!")
    else:
        print("state dict not loaded!")
    return model

def get_hyps(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        folder = get_model_folders(folder)[0]
    hyps_json = os.path.join(folder, "hyperparams.json")
    hyps = load_json(hyps_json)
    return hyps

def load_hyps(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    return get_hyps(folder)

def exp_num_exists(exp_num, exp_folder):
    """
    Determines if the argued experiment number already exists for the
    argued experiment name.

    exp_num: int
        the number to be determined if preexisting
    exp_folder: str
        path to the folder that contains the model folders
    """
    folders = get_model_folders(exp_folder)
    for folder in folders:
        num = foldersort(folder)
        if exp_num == num:
            return True
    return False

def make_save_folder(hyps, incl_full_path=False):
    """
    Creates the save name for the model. Will add exp_num to hyps if
    it does not exist when argued.

    hyps: dict
        keys:
            exp_save_path: str
                path to the experiment folder where all experiments
                sharing the same `exp_name` are saved.
                i.e. /home/user/all_saves/exp_name/
            exp_name: str
                the experiment name
            exp_num: int
                the experiment id number
            search_keys: str
                the identifying keys for this hyperparameter search
    incl_full_path: bool
        if true, prepends the exp_save_path to the save_folder.
    """
    return get_save_folder(hyps, incl_full_path=incl_full_path)

def get_save_folder(hyps, incl_full_path=False):
    """
    Creates the save name for the model. Will add exp_num to hyps if
    it does not exist when argued.

    hyps: dict
        keys:
            exp_folder: str
                path to the experiment folder where all experiments
                sharing the same `exp_name` are saved.
                i.e. /home/user/all_saves/exp_name/
            exp_name: str
                the experiment name
            exp_num: int
                the experiment id number
            search_keys: str
                the identifying keys for this hyperparameter search
    incl_full_path: bool
        if true, prepends the exp_folder to the save_folder.
    """
    if "exp_num" not in hyps:
        hyps["exp_num"] = get_new_exp_num(
            hyps["exp_folder"], hyps["exp_name"]
        )
    model_folder = "{}_{}".format( hyps["exp_name"], hyps["exp_num"] )
    model_folder += prep_search_keys(hyps["search_keys"])
    if incl_full_path: 
        return os.path.join(hyps["exp_folder"], model_folder)
    return model_folder

def get_new_exp_num(exp_folder, exp_name, offset=0):
    """
    Finds the next open experiment id number by searching through the
    existing experiment numbers in the folder.

    If an offset is argued, it is impossible to have an exp_num that is
    less than the value of the offset. The returned exp_num will be
    the next available experiment number starting with the value of the
    offset.

    Args:
        exp_folder: str
            path to the main experiment folder that contains the model
            folders (should not include the experiment name as the final
            directory)
        exp_name: str
            the name of the experiment
        offset: int
            a number to offset the experiment numbers by.

    Returns:
        exp_num: int
    """
    name_splt = exp_name.split("_")
    namedex = 1
    if len(name_splt) > 1:
        namedex = len(name_splt)
    exp_folder = os.path.expanduser(exp_folder)
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
    _, dirs, _ = next(os.walk(exp_folder))
    exp_nums = set()
    for d in dirs:
        splt = d.split("_")
        if len(splt) >= 2:
            num = None
            for i in reversed(range(len(splt))):
                try:
                    num = int(splt[i])
                    break
                except:
                    pass
            if namedex > 1 and i > 1:
                name = "_".join(splt[:namedex])
            else: name = splt[0]
            if name == exp_name and num is not None:
                exp_nums.add(num)
    for i in range(len(exp_nums)):
        if i+offset not in exp_nums:
            return i+offset
    return len(exp_nums) + offset

def load_json(file_name):
    """
    Loads a json file as a python dict

    file_name: str
        the path of the json file
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name,'r') as f:
        s = f.read()
        j = json.loads(s)
    return j

def save_json(data, file_name):
    """
    saves a dict to a json file

    data: dict
    file_name: str
        the path that you would like to save to
    """
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


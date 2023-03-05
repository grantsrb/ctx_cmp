# Leveraging LLMs for Context Compression

## Using this repo
After cloning, you will first need to initialize the `ml_utils`
submodule. You can do this with the following commands at the terminal:

    $ cd ml_utils
    $ git submodule init
    $ git submodule update

Next, you will need to make sure you have all necessary pacakges
installed.

Lastly, you can run a training by creating a hyperparameters.json and
then running the following command:

    # python main.py hyperparameters.json

## Hyperparameters

    "exp_name": str
        the name of the folder in which the hyperparameter search will
        be saved to. This is different than the path. If you would like
        to save the experiment to a different folder than the one in
        which you run `main.py`, use the hyperparemter called `save_root`
    "save_root": str
        this value is prepended to the exp_name when creating the save
        folder for the hyperparameter search.
    "data_root": str
        the path to where the processed datasets are saved
    "data_cache": str
        path to where to cache the downloaded datasets
    "model_string": str
        the pretrained huggingface model you would like to use. For
        example: "bigscience/bloomz-560m"
    "testing": str
        a model string to use for testing. you will probably just want
        to use "hf-internal-testing/tiny-random-gptj"
    "multi_gpu": bool
        if true, the script will try a data parallel approach, splitting
        the batches accross multiple gpus
    "model_parallel": bool
        if true, the script will use Huggingface's auto device map
        feature.
    "torch_dtype": str
        the floating point precision to use. for example: "float32"
    "seed": int
        the random seed for all stochastic processes
    "dataset": str
        a string of the huggingface datasets dataset you would like to
        use. Currently only support "openwebtext" and "glue"
    "gen_targs": bool
        if true, the model will generate the target tokens
    "gen_ids": bool
        if true, the model will use the generated ids rather than
        the logits as the ground truth. only applies if `gen_targs` is
        true
    "n_data_procs": int
        the number of parallel processes to use for the initial
        encoding of the data.
    "max_val_loops": int
        enforces a limit on the number of validation iterations. This
        is useful for speeding up trainings
    "n_train_loops": int
        the number of loops per epoch. this is useful if you want to
        validate more often.
    "checkpt_mod": int or None
        during training, the model will be saved every `checkpt_mod`
        iterations

    "train_lmhead": bool
        if true, the final output layer is trainable
    "train_embs": bool
        if true, the embedding layer is trainable

    "n_epochs": int
        the total number of training iterations
    "batch_size": int
        the size of batches for stochastic gradient descent
    "lr": float
        the learning rate
    "l2": float
        the l2 norm regularization. aka weight decay
    "cmp_len": int
        the length of the sequence that will be compressed. if doing
        `rmb_only`, you should use `seq_len` instead as `cmp_len` is
        ignored.
    "seq_len": int
        the data sequence length to use. for causal modeling, `seq_len`
        refers to the sequence length post compression, so the model will
        compress `cmp_len` tokens and then predict `seq_len` tokens.
        if doing rmb_only, `cmp_len` is ignored
    "seq_overlap": int
        the number of non-compressed tokens overlapping with the
        sequence of tokens that is being compressed. This is kind of
        like a stride parameter. 0 means the stride is the full
        compression sequence length

    "n_cmps": int
        the number of compression tokens to use for compression
    "cmp_layer": str or int
        the layer of the transformer from which to take the compression
        representation. -1 will take the last layer, "half" will take
        from the middle layer. All integer arguments must be within
        the number of layers of the model.
    "rmb_rask": bool
        if true, and using openwebtext dataset, model will have
        auxiliary compression autoencoding task to reconstruct the
        compressed sequence.
    "rmb_only": bool
        if true, model will only perform autoencoder task.

    "n_grad_loops": int
        the number of backprop loops to perform before performing an
        optimizer step. the loss is divided by this quantity so as to
        be equivalent to stepping the optimizer once per iteration with
        a batch size of batch_size*n_grad_loops


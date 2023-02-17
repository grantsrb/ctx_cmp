# Leveraging LLMs for Context Compression

## Hyperparameters

    "exp_name": str
        the name of the folder in which the hyperparameter search will
        be saved to. This is different than the path. If you would like
        to save the experiment to a different folder than the one in
        which you run `main.py`, use the hyperparemter called `save_root`
    "save_root": str
        this value is prepended to the exp_name when creating the save
        folder for the hyperparameter search.
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
    "max_val_loops": int
        enforces a limit on the number of validation iterations. This
        is useful for speeding up trainings
    "n_train_loops": int
        the number of loops per epoch. this is useful if you want to
        validate more often.
    "checkpt_mod": int or None
        during training, the model will be saved every `checkpt_mod`
        iterations

    "n_epochs": int
        the total number of training iterations
    "batch_size": int
        the size of batches for stochastic gradient descent
    "lr": float
        the learning rate
    "l2": float
        the l2 norm regularization. aka weight decay
    "seq_len": int
        the data sequence length to use. for causal modeling, `cmp_len`
        tokens are subtracted from this length, so the model will
        compress `cmp_len` tokens and then predict `seq_len`-`cmp_len`
        tokens. if doing rmb_only, `cmp_len` is ignored
    "cmp_len": int
        the length of the sequence that will be compressed. `cmp_len`
        tokens are subtracted from the original sequence, so the model
        will compress `cmp_len` tokens and then predict
        `seq_len`-`cmp_len` tokens. if doing rmb_only, `cmp_len` is
        ignored
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

    "cmp_layer": str or int
        the layer of the transformer from which to take the compression
        representation. -1 will take the last layer, "half" will take
        from the middle layer. All integer arguments must be within
        the number of layers of the model.

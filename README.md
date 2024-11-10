# JaxMARLCustom
A fork of [JaxMARL](https://github.com/FLAIROx/JaxMARL) containing custom environments and algorithms.
## Setup
1. Clone this repository and follow the installation instruction from [JaxMARL](https://github.com/FLAIROx/JaxMARL).
## Configurations
Scripts are supplied Hydra config management. Consult `.yaml` files in the `custom` module for lists of default parameters.

The file `registry.py` lists available environments and algorithms within `custom`.
## Some Scripts to Run
- Run the training script with `python -m custom.experiments.train`. Must specify an algorithm via argument `+alg=<ALGORITHM_NAME>`. Parameters can be overridden via arguments (e.g., `alg.GAMMA=0.5`). Alternatively, arguments from a json file can be used instead (e.g., `'+ENV_PATH=custom/environments/sample_args.json'`). Arguments' dictionary structure to initialise environment must be compatible with the signature of constructor of `CustomeMPE`. Additionally, a pre-trained model can be loaded from a .safetensor file with `alg.INIT_PARAM_PATH=models/filename_without_extension` to initialise training with (i.e., fine-tuning mode).
- Run the testing script with `python -m custom.experiments.test`. Must specify an algorithm, whose pre-trained model(s) must be present at the given directory. This can be done with argument `+algname=alg_name`. Models will be loaded from `MODEL_PATH/customMPE_algname_x.safetensors` where `x` goes from `0` to `NUM_TRAIN_SEEDS-1`.
- Run training script with `'OFFLINE_DATA_PATH=fullpathtorunjson'` initialise training buffer with offline data. Episodes are duplicated up to a multiple of `alg.NUM_ENVS`, action sequences' lengths are made to match `alg.NUM_STEPS`. Overwrites some training environment parameters.
- Examples commands:
    - `python -m custom.experiments.train +alg=independent_ql alg.RNN_LAYER=True NUM_SEEDS=10 WANDB_MODE=online`
    - `python3 -m custom.experiments.train +alg=independent_ql alg.RNN_LAYER=True alg.INIT_PARAM_PATH=models/customMPE_independent_ql_0 '+ENV_KWARGS.tar_resolve_rad=[0,0.4,0.5]' 'ENV_KWARGS.damping=[0.3]' 'ENV_KWARGS.accel_agents=[4.0]' 'ENV_KWARGS.rad_tar=[0.05,0.1]' alg.EPSILON_ANNEAL_TIME=20000 +SAVE_FILE_NAME=iql_tuned`
    - `python3 -m custom.experiments.test +algname=independent_ql ‘+ENV_PATH=custom/environments/sample_args.json' NUM_TRAIN_SEEDS=1`
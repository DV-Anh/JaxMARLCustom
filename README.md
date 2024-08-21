# JaxMARLCustom
A fork of [JaxMARL](https://github.com/FLAIROx/JaxMARL) containing custom environments and algorithms.
## Setup
1. Clone this repository and follow the installation instruction from [JaxMARL](https://github.com/FLAIROx/JaxMARL).
## Configurations
Scripts are supplied Hydra config management. Consult `.yaml` files in the `custom` module for lists of default parameters.

The file `registry.py` lists available environments and algorithms within `custom`.
## Some Scripts to Run
- Run the training script with `python -m custom.experiments.train`. Must specify an algorithm via argument `+alg=<ALGORITHM_NAME>`. Parameters can be overridden via arguments (e.g., `alg.GAMMA=0.5`).
- Run the testing script with `python -m custom.experiments.test`. Must specify an algorithm, whose pre-trained model(s) must be present at the given directory.
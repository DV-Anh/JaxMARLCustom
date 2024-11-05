import os
import jax
import hydra
from omegaconf import OmegaConf

import wandb
from jaxmarl.wrappers.baselines import (
    LogWrapper,
    save_params,
    load_params,
)

from custom.registry import make_env, make_alg

import json
import time


def train_procedure(config):
    # set hyperparameters:
    env_name = config["ENV_NAME"]
    if config.get(
        "ENV_PATH", None
    ):  # read env arg list from path, TODO: exception check
        f = open(config["ENV_PATH"], "r")
        benchmark_dict = json.load(f)
        f.close
        if isinstance(benchmark_dict, list):
            benchmark_dict = benchmark_dict[0]  # only one arg for training
        env_name = benchmark_dict["env_name"]
        config["ENV_KWARGS"] = benchmark_dict["args"]
    if config["alg"].get("DISTRIBUTION_Q", False):
        alg_name = f'{config["alg"]["NAME"]}_dist'
    else:
        alg_name = config["alg"]["NAME"]
    wandb_mode = config.get("WANDB_MODE", "disabled")
    config["alg"]["WANDB_ONLINE_REPORT"] = wandb.login() & config["alg"].get("WANDB_ONLINE_REPORT", False) & (wandb_mode != 'disabled')
    if config["SAVE_PATH"] is not None:
        os.makedirs(config["SAVE_PATH"], exist_ok=True)
        f = open(f'{config["SAVE_PATH"]}/{env_name}_{alg_name}_config.json', "w")
        f.write(json.dumps(config, separators=(",", ":")))
        f.close()
    env = make_env(env_name, **config["ENV_KWARGS"])
    env = LogWrapper(env)
    rng = jax.random.PRNGKey(config["SEED"])
    if config["alg"].get("INIT_PARAM_PATH", None):
        init_param = {"INIT_PARAM":load_params(f'{config["alg"]["INIT_PARAM_PATH"]}.safetensors')}
    else:
        init_param = {}
    train_fn = make_alg(alg_name, config["alg"]|init_param, env) # single-seed for sequential runs, wandb doesn't allow parallel logging
    # train_vjit = jax.jit(jax.vmap(train_fn)) # use this variable for parallel runs across seeds
    # rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_fn = jax.jit(train_fn)
    # p = [] # store params to debug
    tags = [
        alg_name.upper(),
        env_name.upper(),
        "RNN" if config["alg"].get("RNN_LAYER", True) else "NO_RNN",
        "TD_LOSS" if config["alg"].get("TD_LAMBDA_LOSS", True) else "DQN_LOSS",
        "PRIORITIZED_EXP_REPLAY" if config["alg"].get("PRIORITIZED_EXPERIENCE_REPLAY", False) else "UNIFORM_EXP_REPLAY",
        f"jax_{jax.__version__}",
    ]
    group_str = "|".join(tags) + "|" + str(hex(int(time.time() / 1000)))[2:]
    for i in range(config["NUM_SEEDS"]):
        wandb_run = wandb.init(
            #    entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=tags,
            name=f"{alg_name}|{env_name}|run{i}",
            config=config,
            mode=wandb_mode,
            group=group_str,
            job_type="train",
            reinit=True,
        )
        rng, _rng = jax.random.split(rng, 2)
        start = time.time()
        outs = jax.block_until_ready(train_fn(_rng))
        print(f"Train time: {time.time() - start}")
        wandb_run.finish()
        params = outs["runner_state"][0].params # sequential runs
        if 'agent' in params.keys(): # if there are other modules not belonging to agent (e.g., actor-critic), only take agent module
            params = params['agent']
        # params = jax.tree_util.tree_map(lambda x: x[0], outs["runner_state"][0].params)  # save only params of run 0, used after parallel runs via vmap
        if config["SAVE_PATH"] is not None:
            file_name = f'{env_name}_{alg_name}_{i}' if config.get('SAVE_FILE_NAME',None) is None else config['SAVE_FILE_NAME']
            save_path = f'{config["SAVE_PATH"]}/{file_name}.safetensors'
            save_params(params, save_path)
            print(f"Parameters of batch {i} saved in {save_path}")
        # p.append(params.copy())
        del outs, params  # save memory


@hydra.main(version_base=None, config_path="./config", config_name="config_train")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    assert config.get("alg", None), "Must supply an algorithm"
    train_procedure(config)


if __name__ == "__main__":
    main()

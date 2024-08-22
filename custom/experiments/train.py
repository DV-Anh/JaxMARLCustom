import os
import jax
import hydra
from omegaconf import OmegaConf

import wandb
from jaxmarl.wrappers.baselines import (
    LogWrapper,
    save_params,
)

from custom.registry import make_env, make_alg

import json
import time


def train_procedure(config):
    # set hyperparameters:
    env_name = config["ENV_NAME"]
    if config["alg"].get("DISTRIBUTION_Q", False):
        alg_name = f'{config["alg"]["NAME"]}_dist'
    else:
        alg_name = config["alg"]["NAME"]
    if config["SAVE_PATH"] is not None:
        os.makedirs(config["SAVE_PATH"], exist_ok=True)
        f = open(f'{config["SAVE_PATH"]}/{env_name}_{alg_name}_config.json', "w")
        f.write(json.dumps(config, separators=(",", ":")))
        f.close()
    env = make_env(env_name, **config["ENV_KWARGS"])
    env = LogWrapper(env)
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = make_alg(alg_name, config["alg"], env)
    p = []
    tags = [
        alg_name.upper(),
        env_name.upper(),
        "RNN" if config["alg"].get("RNN_LAYER", True) else "NO_RNN",
        "TD_LOSS" if config["alg"].get("TD_LAMBDA_LOSS", True) else "DQN_LOSS",
        "PRIORITIZED_EXP_REPLAY" if config["alg"].get("PRIORITIZED_EXPERIENCE_REPLAY", False) else "UNIFORM_EXP_REPLAY",
        f"jax_{jax.__version__}",
    ]
    group_str = "|".join(tags) + "|" + str(hex(int(time.time() / 1000)))[2:]
    wandb.login()
    for i in range(config["NUM_SEEDS"]):
        wandb_run = wandb.init(
            #    entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=tags,
            name=f"{alg_name}|{env_name}|run{i}",
            config=config,
            mode=config["WANDB_MODE"],
            group=group_str,
            job_type="train",
            reinit=True,
        )
        start = time.time()
        outs = jax.block_until_ready(train_vjit(jax.random.split(rngs[i], 1)))
        print(f"Train time: {time.time() - start}")
        wandb_run.finish()
        params = jax.tree_util.tree_map(lambda x: x[0], outs["runner_state"][0].params)  # save only params of run 0
        if config["SAVE_PATH"] is not None:
            save_path = f'{config["SAVE_PATH"]}/{env_name}_{alg_name}_{i}.safetensors'
            save_params(params, save_path)
            print(f"Parameters of batch {i} saved in {save_path}")
        p.append(params.copy())
        del outs, params  # save memory


@hydra.main(version_base=None, config_path="./config", config_name="config_train")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    assert config.get("alg", None), "Must supply an algorithm"
    train_procedure(config)


if __name__ == "__main__":
    main()

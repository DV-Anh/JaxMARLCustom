import os
import jax
import hydra
from omegaconf import OmegaConf
import numpy as np
import jax.numpy as jnp
import math

import wandb
from jaxmarl.wrappers.baselines import (
    LogWrapper,
    CTRolloutManager,
    save_params,
    load_params,
)

from custom.environments.customMPE import init_obj_to_array
from custom.registry import make_env, make_alg

import json
import time

def rollout_multi_ep_with_actions(env_list, action_list_list, uniform_ep_length, batch_size, key):
    # each episode has separate env object, envs should share all params other than init config
    assert len(env_list)==len(action_list_list), f'Number of initial states ({len(env_list)}) does not match number of action sequences ({len(action_list_list)})'
    # parallel size and number of insersion
    wrap_size = math.ceil(batch_size/len(env_list))
    list_size = math.ceil(len(env_list)*wrap_size/batch_size)
    def rollout_single_ep_with_actions(env, action_list, key):
        # put actions in dict format, leaf matrices should have uniform_ep_length at dim 0
        acts = {agent:jnp.array([[actions[i]]*wrap_size for actions in action_list]) for i,agent in enumerate(env.agents)}
        # generate initial observation and dones
        key, key_ = jax.random.split(key)
        wrapped_env = CTRolloutManager(LogWrapper(env), batch_size=wrap_size, preprocess_obs=False)
        obs, state = wrapped_env.batch_reset(key_)
        # rollout
        # env should include initial state configuration
        def _env_step_with_action(step_state, actions):
            env_state, rng = step_state
            rng, key_s = jax.random.split(rng)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
            return (env_state, rng), {"obs":obs, "actions":actions, "rewards":rewards, "dones":dones, "infos":infos}
        key, key_ = jax.random.split(key)
        step_state = (state, key_)
        _, trajectory = jax.lax.scan(_env_step_with_action, step_state, acts)
        return trajectory
    def fix_ep_length(ep_dat_list,fix_length):
        # uniform horizon
        # if longer than fix_length, truncated to fix_length; if shorter, concatenate
        a, b = fix_length//len(ep_dat_list), fix_length%len(ep_dat_list)
        return [*(ep_dat_list*a),*(ep_dat_list[:b])]
    action_list_uniform = [fix_ep_length(ep,uniform_ep_length) for ep in action_list_list]
    # store episode data in list
    trajectory_list = []
    for a,b in zip(env_list,action_list_uniform):
        key, key_ = jax.random.split(key, 2)
        trajectory_list.append(rollout_single_ep_with_actions(a,b,key_))
    # convert list to dict for jax-compatibility, concat episodes along dim 1 of leaves
    trajectory_dict = trajectory_list[0]
    trajectory_dict_list = []
    for i in range(1, len(env_list)):
        trajectory_dict = jax.tree_util.tree_map(lambda x,y:jnp.concatenate([x,y],axis=1), trajectory_dict, trajectory_list[i])
    for i in range(list_size-1):
        trajectory_dict_list.append(jax.tree_util.tree_map(lambda x:x[:,(i*batch_size):((i+1)*batch_size)], trajectory_dict))
    trajectory_dict_list.append(jax.tree_util.tree_map(lambda x:x[:,-batch_size:], trajectory_dict)) # split into equal size batch for inserting into buffer
    return trajectory_dict_list # rootkeys: "obs", "actions", "rewards", "dones", "infos"; leaf shape: (timestep, episode, ...)

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
    rng = jax.random.PRNGKey(config["SEED"])
    if config["OFFLINE_DATA_PATH"] is not None:
        f = open(config["OFFLINE_DATA_PATH"], "r")
        run_data = json.load(f)
        run_data_main = run_data['env'].copy()
        run_data_main.pop('num_obs', None)
        run_data_main.pop('num_tar', None)
        run_data_main.pop('objects', None)
        env_args = config["ENV_KWARGS"]|run_data_main
        env_list, act_list = [], []
        for run_dict in run_data['runs']:
            init_p, init_v, num_obj_dict = init_obj_to_array(run_dict['states'][0]['objects'])
            env_args_w_init = env_args|{'init_p':init_p, 'init_v':init_v, 'num_agents':num_obj_dict['agent'], 'num_obs':num_obj_dict['obstacle'], 'num_tar':num_obj_dict['target']}
            env_list.append(make_env(env_name, **env_args_w_init))
            act_list.append(run_dict['action_ids'])
        env_args |= {'num_agents':num_obj_dict['agent']}
        env = make_env(env_name, **env_args)
        print(f'Offline data provided, override env params (except num_obs and num_tar) with {config["OFFLINE_DATA_PATH"]}')
        rng, _rng = jax.random.split(rng, 2)
        offline_sample = rollout_multi_ep_with_actions(env_list, act_list, config["alg"]["NUM_STEPS"], config["alg"]["NUM_ENVS"], _rng)
        print(f"Rolled out {offline_sample[0]['obs'][env.agents[0]].shape[1]*len(offline_sample)} offline episodes over {offline_sample[0]['obs'][env.agents[0]].shape[0]} steps")
        f.close()
    else:
        env = make_env(env_name, **config["ENV_KWARGS"])
        offline_sample = None
    env = LogWrapper(env)
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
        outs = jax.block_until_ready(train_fn(_rng, offline_sample))
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

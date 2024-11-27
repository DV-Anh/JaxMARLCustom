from .environments import (
    customMPE,
)
from .qlearning import (
    learner
)

def make_env(env_id: str, **env_kwargs):
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered custom environments.")
    if env_id == "customMPE":
        env = customMPE.CustomMPE(**env_kwargs)
    return env

def make_alg(alg_id: str, config: dict, env, init_param=None):
    """Train fn signature: [rng_key, offline_data] -> run_data_dict"""
    if alg_id not in registered_algs:
        raise ValueError(f"{alg_id} is not in registered custom algorithms.")
    if alg_id in ["independent_ql","independent_ql_dist"]:
        alg = learner.IndependentQL(config, env).train_fn
    elif alg_id == 'vdn':
        alg = learner.VDN(config, env).train_fn
    elif alg_id == 'qmix':
        alg = learner.QMIX(config, env).train_fn
    elif alg_id == 'sunrise':
        alg = learner.SUNRISE(config, env).train_fn
    return alg

def make_alg_runner(alg_id: str, config: dict, env, init_param=None):
    """Runner fn signature: [policy, state, obs, dones, hstate, rng_key] -> [[policy, state, obs, dones, hstate, rng_key], [rewards, dones, infos, actions]]"""
    if alg_id not in registered_algs:
        raise ValueError(f"{alg_id} is not in registered custom algorithms.")
    if alg_id in ["independent_ql","independent_ql_dist"]:
        runner = learner.IndependentQL(config, env)._greedy_env_step
    elif alg_id == 'vdn':
        runner = learner.VDN(config, env)._greedy_env_step
    elif alg_id == 'qmix':
        runner = learner.QMIX(config, env)._greedy_env_step
    elif alg_id == 'sunrise':
        runner = learner.SUNRISE(config, env)._greedy_env_step
    return runner

def make_alg_action_chooser(alg_id: str, config: dict, env, init_param=None):
    """Action fn signature: [params, hstate, obs, dones] -> actions, keys in actions match keys in obs"""
    if alg_id not in registered_algs:
        raise ValueError(f"{alg_id} is not in registered custom algorithms.")
    if alg_id in ["independent_ql","independent_ql_dist"]:
        runner = learner.IndependentQL(config, env)._greedy_choice
    elif alg_id == 'vdn':
        runner = learner.VDN(config, env)._greedy_choice
    elif alg_id == 'qmix':
        runner = learner.QMIX(config, env)._greedy_choice
    elif alg_id == 'sunrise':
        runner = learner.SUNRISE(config, env)._greedy_choice
    return runner

registered_envs = [
    "customMPE",
]

registered_algs = [
    "independent_ql",
    "independent_ql_dist",
    'vdn',
    'qmix',
    'sunrise',
]

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

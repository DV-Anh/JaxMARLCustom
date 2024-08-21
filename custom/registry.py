import jax
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

def make_alg(alg_id: str, config, env, init_param=None):
    if alg_id not in registered_algs:
        raise ValueError(f"{alg_id} is not in registered custom algorithms.")

    if alg_id in ["independent_ql","independent_ql_dist"]:
        alg = jax.jit(jax.vmap(learner.IndependentQL.make_train(config, env)))
    elif alg_id == 'vdn':
        alg = jax.jit(jax.vmap(learner.VDN.make_train(config, env)))

    return alg

registered_envs = [
    "customMPE",
]

registered_algs = [
    "independent_ql",
    "independent_ql_dist",
    'vdn',
]

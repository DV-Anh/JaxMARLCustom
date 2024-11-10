import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple

import chex
import wandb

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

def homogeneous_pass_ps(agent, params, hidden_state, obs, dones):
    """foward pass with parameter sharing"""
    # concatenate agents and parallel envs to process them in one batch
    agents, flatten_agents_obs = zip(*obs.items())
    original_shape = flatten_agents_obs[0].shape  # assumes obs shape is the same for all agents
    batched_input = (
        jnp.concatenate(flatten_agents_obs, axis=1),  # (time_step, n_agents*n_envs, obs_size)
        jnp.concatenate([dones[agent] for agent in agents], axis=1),  # ensure to not pass other keys (like __all__)
    )
    hidden_state, q_vals = agent.apply(params, hidden_state, batched_input)
    q_vals = jnp.reshape(
        q_vals,
        (original_shape[0], len(agents), *original_shape[1:-1]) + q_vals.shape[(len(original_shape) - len(q_vals.shape) - 1) :],
    )  # (time_steps, n_agents, n_envs, action_dim)
    q_vals = {a: q_vals[:, i] for i, a in enumerate(agents)}
    return hidden_state, q_vals

def homogeneous_pass_nops(agent, params, hidden_state, obs, dones):
    """foward pass without parameter sharing"""
    # homogeneous pass vmapped in respect to the agents parameters (i.e., no parameter sharing)
    agents, flatten_agents_obs = zip(*obs.items())
    batched_input = (
        jnp.stack(flatten_agents_obs, axis=0),  # (n_agents, time_step, n_envs, obs_size)
        jnp.stack([dones[agent] for agent in agents], axis=0),  # ensure to not pass other keys (like __all__)
    )
    # computes the q_vals with the params of each agent separately by vmapping
    hidden_state, q_vals = jax.vmap(agent.apply, in_axes=0)(params, hidden_state, batched_input)
    q_vals = {a: q_vals[i] for i, a in enumerate(agents)}
    return hidden_state, q_vals

def q_of_action(q, u, ax=-1):
    """index the q_values with action indices"""
    q_u = jnp.take_along_axis(q, u, axis=ax)
    return q_u  # jnp.squeeze(q_u, axis=-1)

def q_softmax_from_dis(q):
    """convert q distribution to softmax probs"""
    q = jax.tree_util.tree_map(lambda x: jnp.exp(x), q)
    return jax.tree_util.tree_map(lambda x: x / x.sum(-1, keepdims=True), q)

def q_expectation_from_dis(q, atoms):
    """get q expectation from q distribution"""
    return jax.tree_util.tree_map(lambda x: (x * atoms).sum(-1), q_softmax_from_dis(q))

def td_targets(target_max_qvals, rewards, dones, td_max_steps, _lambda, _gamma, is_multistep=True):
    """calculate truncated td-lambda targets"""
    if is_multistep:
        trace = _lambda*_gamma
        trace_prev, trace_future = _gamma-trace, trace**td_max_steps
        # time difference loss
        def _td_lambda_target(carry, values):
            ret, counter = carry
            reward, done, target_qs = values
            counter = (counter + 1) * (~done)
            ret = reward + (trace * ret + trace_prev * target_qs) * (~done)
            return (ret, counter), (ret, counter >= td_max_steps)

        ret = target_max_qvals[-1]  # * (1-dones[-1])
        _, (td_targets, flags) = jax.lax.scan(
            _td_lambda_target,
            (ret, jnp.zeros(dones.shape[1:], dtype=int)),
            (rewards[-2::-1], dones[-2::-1], target_max_qvals[-1::-1]),
        )
        td_targets = td_targets[::-1]
        flags = flags[::-1][:-td_max_steps] * trace_future
        # padding = jnp.zeros((td_max_steps,)+td_targets.shape[1:])
        targets = jnp.concatenate(
            [
                td_targets[:-td_max_steps] + flags * (target_max_qvals[(td_max_steps - 1) : -1] - td_targets[td_max_steps:]),
                td_targets[-td_max_steps:],
            ],
            axis=0,
        )
    else:
        # standard DQN loss
        targets = rewards[:-1] + _gamma * (~dones[:-1]) * target_max_qvals
    return targets

def callback_wandb_report(metrics, infos):
    info_metrics = {k: v[..., 0][infos["returned_episode"][..., 0]].mean() for k, v in infos.items() if k != "returned_episode"}
    wandb.log(
        {
            "timestep": metrics["timesteps"],
            "updates": metrics["updates"],
            "gradients": metrics["gradients"],
            "loss": metrics["loss"],
            **{"return_" + k: v.mean() for k, v in metrics["rewards"].items()},
            **info_metrics,
            **{k: v.mean() for k, v in metrics["test_metrics"].items()},
        }
    )

class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[..., np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(jax.random.PRNGKey(0), (*batch_size, hidden_size))


class EpsilonGreedy:
    """Epsilon Greedy action selection"""

    def __init__(self, start_e: float, end_e: float, duration: int, act_type_idx):
        self.start_e = start_e
        self.end_e = end_e
        self.duration = duration
        self.slope = (end_e - start_e) / duration
        self.act_type_idx = act_type_idx

    @partial(jax.jit, static_argnums=0)
    def get_epsilon(self, t: int):
        e = self.slope * t + self.start_e
        return jnp.clip(e, self.end_e)

    @partial(jax.jit, static_argnums=0)
    def choose_actions(self, q_vals: dict, t: int, rng: chex.PRNGKey):
        def explore(q, eps, key):
            key_a, key_e = jax.random.split(key, 2)  # a key for sampling random actions and one for picking
            greedy_actions = jnp.argmax(q, axis=-1)  # get the greedy actions
            random_actions = jax.random.randint(key_a, shape=greedy_actions.shape, minval=0, maxval=q.shape[-1])  # sample random actions
            pick_random = jax.random.uniform(key_e, greedy_actions.shape) < eps  # pick which actions should be random
            chosed_actions = jnp.where(pick_random, random_actions, greedy_actions)
            return chosed_actions

        eps = self.get_epsilon(t)
        keys = dict(zip(q_vals.keys(), jax.random.split(rng, len(q_vals))))  # get a key for each agent
        chosen_actions = jax.tree_util.tree_map(
            lambda q, k: jnp.stack(
                jax.tree_util.tree_map(lambda x: explore(q[..., x], eps, k), self.act_type_idx),
                axis=-1,
            ),
            q_vals,
            keys,
        )
        return chosen_actions
class UCB:
    """Upper Confidence Bound action selection"""

    def __init__(self, std_coeff: float, act_type_idx):
        self.l = std_coeff
        self.act_type_idx = act_type_idx

    @partial(jax.jit, static_argnums=0)
    def choose_actions(self, q_vals: dict):
        def choose_action_inner(q):
            q_mean = q.mean(-1, keepdims=True)
            greedy_actions = jnp.argmax(q_mean.squeeze(-1) + self.l*jnp.sqrt(((q - q.mean)**2).mean(-1)), axis=-1)  # get argmax over UCB
            return greedy_actions

        chosen_actions = jax.tree_util.tree_map(
            lambda q: jnp.stack(
                jax.tree_util.tree_map(lambda x: choose_action_inner(q[..., x]), self.act_type_idx),
                axis=-1,
            ),
            q_vals,
        )
        return chosen_actions

class Transition(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    infos: dict
    def augment_reward_invariant_inner(self, obs_keys, dones_keys, infos_keys, trans_obs_func, trans_acts_func, trans_no, axis=1):
        obs = jax.tree_util.tree_map(lambda x: jnp.concatenate([x, trans_obs_func(x, axis)], axis=axis), self.obs if (obs_keys is None) else {a: self.obs[a] for a in obs_keys})
        actions = jax.tree_util.tree_map(
            lambda x: jnp.concatenate([x, trans_acts_func(x, axis)], axis=axis),
            self.actions,
        )
        # rewards=jax.tree_util.tree_map(lambda x:jnp.tile(x,(1,)*axis+(trans_no,)+(1,)*(x.ndim-axis-1)),self.rewards)
        infos = jax.tree_util.tree_map(
            lambda x: jnp.tile(x, (1,) * axis + (trans_no,) + (1,) * (x.ndim - axis - 1)),
            self.infos if (infos_keys is None) else {a: self.infos[a] for a in infos_keys},
        )
        dones = jax.tree_util.tree_map(
            lambda x: jnp.tile(x, (1,) * axis + (trans_no,) + (1,) * (x.ndim - axis - 1)),
            self.dones if (dones_keys is None) else {a: self.dones[a] for a in dones_keys},
        )
        return obs, actions, dones, infos
    def augment_reward_invariant(self, obs_keys, dones_keys, infos_keys, trans_obs_func, trans_acts_func, trans_no, axis=1):
        obs, actions, dones, infos = self.augment_reward_invariant_inner(obs_keys, dones_keys, infos_keys, trans_obs_func, trans_acts_func, trans_no, axis)
        return Transition(obs=obs, actions=actions, rewards=self.rewards, dones=dones, infos=infos)

class TransitionBootstrap(Transition):
    bootstrap_mask: chex.Array

    def augment_reward_invariant_inner(self, obs_keys, dones_keys, infos_keys, trans_obs_func, trans_acts_func, trans_no, axis=1):
        obs, actions, dones, infos = super().augment_reward_invariant_inner(obs_keys, dones_keys, infos_keys, trans_obs_func, trans_acts_func, trans_no, axis)
        bootstrap_mask = jax.tree_util.tree_map(
            lambda x: jnp.tile(x, (1,) * axis + (trans_no,) + (1,) * (x.ndim - axis - 1)),
            self.bootstrap_mask,
        )
        return obs, actions, dones, infos, bootstrap_mask
    def augment_reward_invariant(self, obs_keys, dones_keys, infos_keys, trans_obs_func, trans_acts_func, trans_no, axis=1):
        obs, actions, dones, infos, bootstrap_mask = self.augment_reward_invariant_inner(obs_keys, dones_keys, infos_keys, trans_obs_func, trans_acts_func, trans_no, axis)
        return TransitionBootstrap(obs=obs, actions=actions, rewards=self.rewards, dones=dones, infos=infos, bootstrap_mask=bootstrap_mask)

class AgentRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    atom_dim: int  # equals 1 for standard q-learning
    init_scale: float
    act_type_idx: list
    use_rnn: bool
    layer_name: tuple = (
        "input_0",
        "input_1",
        "hidden_GRU",
        "hidden",
        "output",
        "s_hidden",
        "s_output",
    )
    common_layers: tuple = (0, 1, 2)
    reset_layers: tuple = (3, 4, 5, 6)

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        # layer 1
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
            name=self.layer_name[0],
        )(obs)
        embedding = nn.relu(embedding)
        if not self.use_rnn:
            # layer 2
            embedding = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
                name=self.layer_name[1],
            )(embedding)
            embedding = nn.relu(embedding)
        else:
            # GRU layer
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN(name=self.layer_name[2])(hidden, rnn_in)
        # main hidden
        q_vals = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
            name=self.layer_name[3],
        )(embedding)
        q_vals = nn.relu(q_vals)
        # advantage layer
        q_vals = nn.Dense(
            self.action_dim * self.atom_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
            name=self.layer_name[4],
        )(q_vals)
        q_vals = jnp.reshape(q_vals, q_vals.shape[:-1] + (self.atom_dim, self.action_dim))  # (...,atoms,actions)
        # state-value hidden layer
        s_val = nn.relu(
            nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0),
                name=self.layer_name[5],
            )(embedding)
        )
        # state-value output
        s_val = nn.Dense(
            len(self.act_type_idx) * self.atom_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
            name=self.layer_name[6],
        )(s_val)
        s_val = jnp.reshape(s_val, s_val.shape[:-1] + (self.atom_dim, len(self.act_type_idx)))  # (...,atoms,action_types)
        # combine state-value with advantages, splicing by atoms
        q_vals = jnp.concatenate(
            [s_val[..., i, None] + q_vals[..., x] - q_vals[..., x].mean(-1, keepdims=True) for i, x in enumerate(self.act_type_idx)],
            axis=-1,
        )
        # rearrange q_values index, only necessary if indices are not consecutive
        # q_vals = q_vals[...,np.concatenate(self.act_type_idx)]
        # return atom dimension to last dimension
        q_vals = q_vals.swapaxes(-1, -2)
        # if raw q-value instead of distribution, remove redundant dimension
        if self.atom_dim == 1:
            q_vals = q_vals.squeeze(-1)
        return hidden, q_vals

class HyperNetwork(nn.Module):
    """HyperNetwork for generating weights of QMix' mixing network."""
    hidden_dim: int
    output_dim: int
    init_scale: float
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        return x

class MixingNetwork(nn.Module):
    """
    Mixing network for projecting individual agent Q-values into Q_tot. Follows the original QMix implementation.
    """
    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float
    @nn.compact
    def __call__(self, q_vals, states):

        n_agents, time_steps, batch_size, act_dim = q_vals.shape
        q_vals = jnp.transpose(q_vals, (1, 2, 3, 0))  # (time_steps, batch_size, act_dim, n_agents)
        # q_vals = jnp.reshape(q_vals, (time_steps, batch_size, n_agents * act_dim)) # merge act dim

        # hypernetwork
        w_1 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim * n_agents * act_dim,
            init_scale=self.init_scale,
        )(states)
        b_1 = nn.Dense(
            self.embedding_dim * act_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(states)
        w_2 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim * act_dim,
            init_scale=self.init_scale,
        )(states)
        b_2 = HyperNetwork(
            hidden_dim=self.embedding_dim, output_dim=act_dim, init_scale=self.init_scale
        )(states)

        # monotonicity and reshaping
        w_1 = jnp.abs(jnp.reshape(w_1, (time_steps, batch_size, act_dim, n_agents, self.embedding_dim)))
        b_1 = jnp.reshape(b_1, (time_steps, batch_size, act_dim, 1, self.embedding_dim))
        w_2 = jnp.abs(jnp.reshape(w_2, (time_steps, batch_size, act_dim, self.embedding_dim, 1)))
        b_2 = jnp.reshape(b_2, (time_steps, batch_size, act_dim, 1, 1))

        # mix
        hidden = nn.elu(jnp.matmul(q_vals[:, :, :, None, :], w_1) + b_1)
        q_tot = jnp.matmul(hidden, w_2) + b_2

        return q_tot.squeeze()  # (time_steps, batch_size, act_dim)

class MLP(nn.Module):
    # Base MLP
    layer_dim: list
    init_scale: float

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.layer_dim)-1):
            x = nn.relu(nn.Dense(self.layer_dim[i],kernel_init=orthogonal(self.init_scale),bias_init=constant(0.0),name=f'hidden_{i}')(x))
        # last layer has no activation function
        x = nn.Dense(self.layer_dim[-1],kernel_init=orthogonal(self.init_scale),bias_init=constant(0.0),name='output')(x)
        return x
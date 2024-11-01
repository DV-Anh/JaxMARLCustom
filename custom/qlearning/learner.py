import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


import optax
from flax.training.train_state import TrainState
import flashbax as fbx
from jaxmarl.wrappers.baselines import CTRolloutManager

from custom.qlearning.common import ScannedRNN, AgentRNN, MixingNetwork, EpsilonGreedy, Transition, homogeneous_pass_ps, homogeneous_pass_nops, q_of_action, q_softmax_from_dis, q_expectation_from_dis, td_targets, callback_wandb_report

class BaseQL:
    def __init__(self, config: dict, env) -> None:
        self.config = config
        self.num_updates = self.config["TOTAL_TIMESTEPS"] // self.config["NUM_STEPS"] // self.config["NUM_ENVS"]
        self.td_max_steps = min(self.config["NUM_STEPS"] - 1, self.config["TD_MAX_STEPS"])
        self.wrapped_env = CTRolloutManager(env, batch_size=self.config["NUM_ENVS"], preprocess_obs=False)
        self.test_env = CTRolloutManager(env, batch_size=self.config["NUM_TEST_EPISODES"], preprocess_obs=False)  # batched env for testing (has different batch size)
        if self.config.get('PRIORITIZED_EXPERIENCE_REPLAY', False):
            self.buffer = fbx.make_prioritised_trajectory_buffer(
                max_length_time_axis=self.config['BUFFER_SIZE']//self.config['NUM_ENVS'],
                min_length_time_axis=self.config['BUFFER_BATCH_SIZE'],
                sample_batch_size=self.config['BUFFER_BATCH_SIZE'],
                add_batch_size=self.config['NUM_ENVS'],
                sample_sequence_length=1,
                period=1,
                priority_exponent=self.config['PRIORITY_EXPONENT']
            )
        else:
            self.buffer = fbx.make_trajectory_buffer(
                max_length_time_axis=self.config['BUFFER_SIZE']//self.config['NUM_ENVS'],
                min_length_time_axis=self.config['BUFFER_BATCH_SIZE'],
                sample_batch_size=self.config['BUFFER_BATCH_SIZE'],
                add_batch_size=self.config['NUM_ENVS'],
                sample_sequence_length=1,
                period=1,
            )
        # accept initial params if any
        self.init_param = config.get("INIT_PARAM", None)
        # set agent params structure (if init_param is used, make sure the structures are identical)
        self.act_type_idx_offset = jnp.array(self.wrapped_env._env.act_type_idx_offset)
        if self.config.get("DISTRIBUTION_Q", False):
            self.agent = AgentRNN(
                action_dim=self.wrapped_env.max_action_space,
                atom_dim=self.config["DISTRIBUTION_ATOMS"],
                hidden_dim=self.config["AGENT_HIDDEN_DIM"],
                init_scale=self.config["AGENT_INIT_SCALE"],
                act_type_idx=self.wrapped_env._env.act_type_idx,
                use_rnn=self.config["RNN_LAYER"],
            )
            self.dis_support_step = (self.config["DISTRIBUTION_RANGE"][1] - self.config["DISTRIBUTION_RANGE"][0]) / (self.config["DISTRIBUTION_ATOMS"] - 1)
            self.dis_support = jnp.arange(
                self.config["DISTRIBUTION_RANGE"][0],
                self.config["DISTRIBUTION_RANGE"][1] + self.dis_support_step,
                self.dis_support_step,
            )
        else:
            self.agent = AgentRNN(
                action_dim=self.wrapped_env.max_action_space,
                atom_dim=1,
                hidden_dim=self.config["AGENT_HIDDEN_DIM"],
                init_scale=self.config["AGENT_INIT_SCALE"],
                act_type_idx=self.wrapped_env._env.act_type_idx,
                use_rnn=self.config["RNN_LAYER"],
            )
        # depending if using parameters sharing or not, q-values are computed using one or multiple parameters
        if self.config.get("PARAMETERS_SHARING", True):
            self.foward_pass = partial(homogeneous_pass_ps, self.agent)
        else:
            self.foward_pass = partial(homogeneous_pass_nops, self.agent)
        def linear_schedule(count):
            frac = 1.0 - (count / self.num_updates)
            return self.config["LR"] * frac
        # INIT LOSS OPTIMIZER
        lr = linear_schedule if self.config.get("LR_LINEAR_DECAY", False) else self.config["LR"]
        self.tx = optax.chain(
            optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=self.config["EPS_ADAM"]),
        )
        # INIT REWEIGHTING SCHEDULE
        self.importance_weights_sch = optax.linear_schedule(
            self.config["PRIORITY_IMPORTANCE_EXPONENT_START"],
            self.config["PRIORITY_IMPORTANCE_EXPONENT_END"],
            self.num_updates,
            self.config["BUFFER_BATCH_SIZE"],
        )
        # INIT EXPLORATION STRATEGY
        self.explorer = EpsilonGreedy(
            start_e=self.config["EPSILON_START"],
            end_e=self.config["EPSILON_FINISH"],
            duration=self.config["EPSILON_ANNEAL_TIME"],
            act_type_idx=self.wrapped_env._env.act_type_idx,
        )
        # preparing target function for loss calc
        self.target_fn = partial(td_targets, _lambda=self.config["TD_LAMBDA"], td_max_steps=self.td_max_steps, _gamma=self.config["GAMMA"], is_multistep=self.config.get("TD_LAMBDA_LOSS", True))

    def _env_sample_step(self, env_state_and_key, unused):
        env_state, rng = env_state_and_key
        rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)  # use a dummy rng here
        key_a = jax.random.split(key_a, self.wrapped_env._env.num_agents)
        actions = {agent: self.wrapped_env.batch_sample(key_a[i], agent) for i, agent in enumerate(self.wrapped_env._env.agents)}
        obs, env_state, rewards, dones, infos = self.wrapped_env.batch_step(key_s, env_state, actions)
        transition = Transition(obs, actions, rewards, dones, infos)
        return (env_state, rng), transition

    def _env_step(self, step_state, unused):
        params, env_state, last_obs, last_dones, hstate, rng, t = step_state

        # prepare rngs for actions and step
        rng, key_a, key_s = jax.random.split(rng, 3)

        # SELECT ACTION
        # add a dummy time_step dimension to the agent input
        obs_ = {a: last_obs[a] for a in self.wrapped_env._env.agents}  # ensure to not pass the global state (obs["__all__"]) to the network
        obs_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], obs_)
        dones_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], last_dones)
        # get the q_values from the agent network
        hstate, q_vals = self.foward_pass(params, hstate, obs_, dones_)
        if self.config.get("DISTRIBUTION_Q", False):
            q_vals = q_expectation_from_dis(q_vals, self.dis_support)
        # remove the dummy time_step dimension and index qs by the valid actions of each agent
        valid_q_vals = jax.tree_util.tree_map(
            lambda q, valid_idx: q.squeeze(0)[..., valid_idx],
            q_vals,
            self.wrapped_env.valid_actions,
        )
        # explore with epsilon greedy_exploration
        actions = self.explorer.choose_actions(valid_q_vals, t, key_a)

        # STEP ENV
        obs, env_state, rewards, dones, infos = self.wrapped_env.batch_step(key_s, env_state, actions)
        transition = Transition(last_obs, actions, rewards, dones, infos)

        step_state = (params, env_state, obs, dones, hstate, rng, t + 1)
        return step_state, transition

    def _greedy_env_step(self, step_state, unused):
        params, env_state, last_obs, last_dones, hstate, rng = step_state
        rng, key_s = jax.random.split(rng)
        obs_ = {a: last_obs[a] for a in self.test_env._env.agents}
        obs_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], obs_)
        dones_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], last_dones)
        hstate, q_vals = self.foward_pass(params, hstate, obs_, dones_)
        if self.config.get("DISTRIBUTION_Q", False):
            q_vals = q_expectation_from_dis(q_vals, self.dis_support)
        valid_q_vals = jax.tree_util.tree_map(
            lambda q, valid_idx: q.squeeze(0)[..., valid_idx],
            q_vals,
            self.test_env.valid_actions,
        )
        actions = jax.tree_util.tree_map(
            lambda q: jnp.stack(
                [jnp.argmax(q[..., x], axis=-1) for x in self.test_env._env.act_type_idx],
                axis=-1,
            ),
            valid_q_vals,
        )  # one argmax per action type
        obs, env_state, rewards, dones, infos = self.test_env.batch_step(key_s, env_state, actions)
        step_state = (params, env_state, obs, dones, hstate, rng)
        return step_state, (rewards, dones, infos)

    def _get_greedy_metrics(self, rng, params, time_state):
        """Help function to test greedy policy during training"""

        rng, _rng = jax.random.split(rng)
        init_obs, env_state = self.test_env.batch_reset(_rng)
        init_dones = {agent: jnp.zeros((self.config["NUM_TEST_EPISODES"]), dtype=bool) for agent in self.wrapped_env._env.agents + ["__all__"]}
        rng, _rng = jax.random.split(rng)
        if self.config["PARAMETERS_SHARING"]:
            hstate = ScannedRNN.initialize_carry(
                self.config["AGENT_HIDDEN_DIM"],
                len(self.wrapped_env._env.agents) * self.config["NUM_TEST_EPISODES"],
            )  # (n_agents*n_envs, hs_size)
        else:
            hstate = ScannedRNN.initialize_carry(
                self.config["AGENT_HIDDEN_DIM"],
                len(self.wrapped_env._env.agents),
                self.config["NUM_TEST_EPISODES"],
            )  # (n_agents, n_envs, hs_size)
        step_state = (
            params,
            env_state,
            init_obs,
            init_dones,
            hstate,
            _rng,
        )
        step_state, (rewards, dones, infos) = jax.lax.scan(self._greedy_env_step, step_state, None, self.config["NUM_STEPS"])

        # compute the metrics of the first episode that is done for each parallel env
        def first_episode_returns(rewards, dones):
            first_done = jax.lax.select(jnp.argmax(dones) == 0.0, dones.size, jnp.argmax(dones))
            first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
            return jnp.where(first_episode_mask, rewards, 0.0).sum() / (first_done + 1)

        all_dones = dones["__all__"]
        first_returns = jax.tree_util.tree_map(
            lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones),
            rewards,
        )
        first_infos = jax.tree_util.tree_map(
            lambda i: jax.vmap(first_episode_returns, in_axes=1)(i[..., 0], all_dones),
            infos,
        )
        metrics = {
            **{"test_returns_" + k: v.mean() for k, v in first_returns.items()},
            **{"test_" + k: v for k, v in first_infos.items()},
        }
        if self.config.get("VERBOSE", False):

            def callback(timestep, updates, gradients, val):
                print(f"Timestep: {timestep}, updates: {updates}, gradients: {gradients}, return: {val}")

            jax.debug.callback(
                callback,
                time_state["timesteps"] * self.config["NUM_ENVS"],
                time_state["updates"],
                time_state["gradients"],
                first_returns["__all__"].mean() / len(self.wrapped_env._env.agents),
            )
        return metrics

    def batchify(self, x: dict):
        return jnp.stack([x[agent] for agent in self.wrapped_env._env.agents], axis=0)

    def unbatchify(self, x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(self.wrapped_env._env.agents)}
    
class IndependentQL(BaseQL):
    def __init__(self, config: dict, env):
        super().__init__(config, env)
        def train(rng):
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = self.wrapped_env.batch_reset(_rng)
            init_dones = {agent: jnp.zeros((self.config["NUM_ENVS"]), dtype=bool) for agent in self.wrapped_env._env.agents + ["__all__"]}

            # INIT BUFFER
            # to initalize the buffer is necessary to sample a trajectory to know its strucutre
            _, sample_traj = jax.lax.scan(self._env_sample_step, (env_state,_rng), None, self.config["NUM_STEPS"])
            sample_traj_unbatched = jax.tree_util.tree_map(lambda x: x[:, 0][:, np.newaxis], sample_traj)  # remove the NUM_ENV dim, add dummy dim 1
            sample_traj_unbatched = sample_traj_unbatched.augment_reward_invariant(
                self.wrapped_env._env.agents,
                self.wrapped_env._env.agents,
                self.wrapped_env._env.reward_invariant_transform_obs,
                self.wrapped_env._env.reward_invariant_transform_acts,
                self.wrapped_env._env.transform_no + 1,
                1,  # append augment data at dim 1
            )
            buffer_state = self.buffer.init(sample_traj_unbatched)

            # INIT NETWORK
            rng, _rng = jax.random.split(rng)
            if self.config.get("PARAMETERS_SHARING", True):
                init_x = (
                    jnp.zeros((1, 1, self.wrapped_env.obs_size)),  # (time_step, batch_size, obs_size)
                    jnp.zeros((1, 1)),  # (time_step, batch size)
                )
                init_hs = ScannedRNN.initialize_carry(self.config["AGENT_HIDDEN_DIM"], 1)  # (batch_size, hidden_dim)
                if self.init_param is None:
                    network_params = self.agent.init(_rng, init_hs, init_x)
                else:
                    network_params = self.init_param
            else:
                init_x = (
                    jnp.zeros((len(self.wrapped_env._env.agents), 1, 1, self.wrapped_env.obs_size)),  # (time_step, batch_size, obs_size)
                    jnp.zeros((len(self.wrapped_env._env.agents), 1, 1)),  # (time_step, batch size)
                )
                init_hs = ScannedRNN.initialize_carry(self.config["AGENT_HIDDEN_DIM"], len(self.wrapped_env._env.agents), 1)  # (n_agents, batch_size, hidden_dim)
                rngs = jax.random.split(_rng, len(self.wrapped_env._env.agents))  # a random init for each agent
                if self.init_param is None:
                    network_params = jax.vmap(self.agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)
                else:
                    network_params = jax.vmap(lambda x,y: y.copy(), in_axes=(0))(rngs, self.init_param)

            # INIT TRAIN STATE AND OPTIMIZER
            train_state = TrainState.create(
                apply_fn=self.agent.apply,
                params=network_params,
                tx=self.tx,
            )
            # target network params
            target_agent_params = jax.tree_util.tree_map(lambda x: jnp.copy(x), train_state.params)

            # TRAINING LOOP
            def _update_step(runner_state, unused):
                (
                    train_state,
                    target_agent_params,
                    env_state,
                    buffer_state,
                    time_state,
                    init_obs,
                    init_dones,
                    test_metrics,
                    rng,
                ) = runner_state

                # EPISODE STEP
                # prepare the step state and collect the episode trajectory
                rng, _rng = jax.random.split(rng)
                if self.config.get("PARAMETERS_SHARING", True):
                    hstate = ScannedRNN.initialize_carry(self.config["AGENT_HIDDEN_DIM"], len(self.wrapped_env._env.agents) * self.config["NUM_ENVS"])  # (n_agents*n_envs, hs_size)
                else:
                    hstate = ScannedRNN.initialize_carry(self.config["AGENT_HIDDEN_DIM"], len(self.wrapped_env._env.agents), self.config["NUM_ENVS"])  # (n_agents, n_envs, hs_size)

                step_state = (
                    train_state.params,
                    env_state,
                    init_obs,
                    init_dones,
                    hstate,
                    _rng,
                    time_state["timesteps"],  # t is needed to compute epsilon
                )

                step_state, traj_batch = jax.lax.scan(self._env_step, step_state, None, self.config["NUM_STEPS"])
                # BUFFER UPDATE: save the collected trajectory in the buffer
                buffer_traj_batch = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis, :, np.newaxis],  # put the batch dim first, add a dummy sequence dim, add dummy dim for augment
                    traj_batch,
                )  # (num_envs, 1, time_steps, ...)
                buffer_traj_batch = buffer_traj_batch.augment_reward_invariant(
                    self.wrapped_env._env.agents,
                    self.wrapped_env._env.agents,
                    self.wrapped_env._env.reward_invariant_transform_obs,
                    self.wrapped_env._env.reward_invariant_transform_acts,
                    self.wrapped_env._env.transform_no + 1,
                    3,  # append augment data at dim 3
                )
                buffer_state = self.buffer.add(buffer_state, buffer_traj_batch)

                if self.config.get("PARAMETERS_SHARING", True):

                    def _loss_fn(params, target_q_vals, init_hs, learn_traj, importance_weights):
                        # obs_={a:learn_traj.obs[a] for a in self.wrapped_env._env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                        _, q_vals = self.foward_pass(params, init_hs, learn_traj.obs, learn_traj.dones)
                        if not self.config.get("DISTRIBUTION_Q", False):
                            # get the q_vals of the taken actions (with exploration) for each agent
                            chosen_action_qvals = jax.tree_util.tree_map(
                                lambda q, u: q_of_action(q, u + jnp.broadcast_to(self.act_type_idx_offset, u.shape))[:-1],  # avoid last timestep
                                q_vals,
                                learn_traj.actions,
                            )
                            # get the target for each agent (assumes every agent has a reward)
                            valid_q_vals = jax.tree_util.tree_map(
                                lambda q, valid_idx: q[..., valid_idx],
                                q_vals,
                                self.wrapped_env.valid_actions,
                            )
                            target_max_qvals = jax.tree_util.tree_map(
                                lambda t_q, q: q_of_action(
                                    t_q,
                                    jnp.stack(
                                        [jnp.argmax(q[..., x], axis=-1) + self.act_type_idx_offset[i] for i, x in enumerate(self.wrapped_env._env.act_type_idx)],
                                        axis=-1,
                                    ),
                                )[1:],  # avoid first timestep
                                target_q_vals,
                                jax.lax.stop_gradient(valid_q_vals),
                            )
                            # get reward vector along action types
                            rewards_vec = {
                                a: jnp.stack(
                                    [learn_traj.infos[f"reward_{x}"][..., i] for x in range(len(self.wrapped_env._env.act_type_idx))],
                                    axis=-1,
                                )
                                for i, a in enumerate(self.wrapped_env._env.agents)
                            }
                            dones = {
                                a: jnp.tile(
                                    learn_traj.dones[a][..., None],
                                    (target_max_qvals[a].shape[-1],),
                                )
                                for a in self.wrapped_env._env.agents
                            }
                        else:
                            q_vals = q_softmax_from_dis(q_vals)
                            target_q_vals = q_softmax_from_dis(target_q_vals)
                            chosen_action_qvals = jax.tree_util.tree_map(
                                lambda q, u: q_of_action(
                                    q,
                                    (u + jnp.broadcast_to(self.act_type_idx_offset, u.shape))[..., None],
                                    -2,
                                )[:-1],  # avoid last timestep
                                q_vals,
                                learn_traj.actions,
                            )
                            # get the target for each agent (assumes every agent has a reward)
                            valid_q_vals = jax.tree_util.tree_map(
                                lambda q, valid_idx: (q * self.dis_support).sum(-1)[..., valid_idx],
                                q_vals,
                                self.wrapped_env.valid_actions,
                            )  # get expectation of q-value
                            target_max_qvals_dis = jax.tree_util.tree_map(
                                lambda t_q, q: q_of_action(
                                    t_q,
                                    jnp.stack(
                                        [jnp.argmax(q[..., x], axis=-1) + self.act_type_idx_offset[i] for i, x in enumerate(self.wrapped_env._env.act_type_idx)],
                                        axis=-1,
                                    )[..., None],
                                    -2,
                                )[1:],  # avoid first timestep
                                target_q_vals,
                                jax.lax.stop_gradient(valid_q_vals),
                            )
                            target_max_qvals = jax.tree_util.tree_map(
                                lambda x: jnp.tile(self.dis_support, x.shape[:-1] + (1,)),
                                target_max_qvals_dis,
                            )
                            # get reward vector along action types
                            rewards_vec = {
                                a: jnp.stack(
                                    [learn_traj.infos[f"reward_{x}"][..., i] for x in range(len(self.wrapped_env._env.act_type_idx))],
                                    axis=-1,
                                )[..., None]
                                for i, a in enumerate(self.wrapped_env._env.agents)
                            }
                            dones = {
                                a: jnp.tile(
                                    learn_traj.dones[a][..., None],
                                    (target_max_qvals[a].shape[-2],),
                                )[..., None]
                                for a in self.wrapped_env._env.agents
                            }
                        # compute a single loss for all the agents in one pass (parameter sharing)
                        targets = jax.tree_util.tree_map(
                            self.target_fn,
                            target_max_qvals,
                            rewards_vec,  # {agent:learn_traj.rewards[agent] for agent in self.wrapped_env._env.agents}, # rewards and agents could contain additional keys
                            dones,
                        )
                        chosen_action_qvals = jnp.concatenate(list(chosen_action_qvals.values()))
                        targets = jnp.concatenate(list(targets.values()))
                        if not self.config.get("DISTRIBUTION_Q", False):
                            importance_weights = importance_weights[(None, ...) + (None,) * (targets.ndim - importance_weights.ndim - 1)]
                            mean_axes = tuple(range(targets.ndim - 1))
                            err = chosen_action_qvals - jax.lax.stop_gradient(targets)
                            if self.config.get("TD_LAMBDA_LOSS", True):
                                loss = jnp.mean(0.5 * (err**2) * importance_weights, axis=mean_axes)
                            else:
                                loss = jnp.mean((err**2) * importance_weights, axis=mean_axes)
                            err = jnp.clip(jnp.abs(err), 1e-7, None)
                            err_axes = tuple(range(err.ndim))
                            return loss.mean(), err.mean(axis=err_axes[0:1] + err_axes[2:])  # maintain 1 abs error for each batch
                        else:

                            def dis_shift(dis, base_support, new_support, cell_radius):  # assuming dis and support span the last dimension
                                coef = jnp.clip(
                                    cell_radius - jnp.abs(base_support[..., None] - new_support),
                                    0.0,
                                    None,
                                )  # projection coefficients
                                return (dis[..., None] * coef).sum(-2) / cell_radius  # project back onto initial support

                            target_max_qvals_dis = jnp.concatenate(list(target_max_qvals_dis.values()))
                            targets = dis_shift(
                                target_max_qvals_dis,
                                jnp.clip(targets, self.config["DISTRIBUTION_RANGE"][0], self.config["DISTRIBUTION_RANGE"][1]),
                                self.dis_support,
                                self.dis_support_step,
                            )
                            mean_axes = tuple(range(targets.ndim - 2))
                            loss = -((jax.lax.stop_gradient(targets)) * jnp.log(chosen_action_qvals)).sum(-1)  # cross-entropy
                            importance_weights = importance_weights[(None, ...) + (None,) * (loss.ndim - importance_weights.ndim - 1)]
                            err = jnp.clip(loss, 1e-7, None)
                            err_axes = tuple(range(err.ndim))
                            return (loss * importance_weights).mean(axis=mean_axes).mean(), err.mean(axis=err_axes[0:1] + err_axes[2:])/len(self.wrapped_env._env.agents)  # maintain 1 abs error for each batch
                else:
                    # without parameters sharing, a different loss must be computed for each agent via vmap
                    def _loss_fn(
                        params,
                        target_q_vals,
                        init_hs,
                        obs,
                        dones,
                        actions,
                        valid_actions,
                        rewards,
                        type_idx,
                    ):
                        _, q_vals = self.agent.apply(params, init_hs, (obs, dones))
                        chosen_action_qvals = q_of_action(q_vals, actions)[:-1]
                        valid_actions = valid_actions.reshape(*[1] * len(q_vals.shape[:-1]), -1)  # reshape to match q_vals shape
                        valid_argmax = jnp.argmax(
                            jnp.where(
                                valid_actions.astype(bool),
                                jax.lax.stop_gradient(q_vals),
                                -1000000.0,
                            ),
                            axis=-1,
                        )
                        target_max_qvals = q_of_action(target_q_vals, valid_argmax)[1:]  # target q_vals of greedy actions, avoiding first time step
                        targets = self.target_fn(target_max_qvals, rewards, dones)
                        return jnp.mean((chosen_action_qvals - jax.lax.stop_gradient(targets)) ** 2).mean(), jnp.abs(chosen_action_qvals - targets)[..., type_idx].sum(-1)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                def _learn_phase(carry, _):
                    train_state, buffer_state, rng = carry
                    # sample a batched trajectory from the buffer and set the time step dim in first axis
                    rng, _rng = jax.random.split(rng)
                    learn_traj_batch = self.buffer.sample(buffer_state, _rng)  # (batch_size, 1, max_time_steps, ...)
                    learn_traj = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(x[:, 0], 0, 1),  # remove the dummy sequence dim (1) and swap batch and temporal dims
                        learn_traj_batch.experience,
                    )  # (max_time_steps, batch_size, ...)
                    importance_weights = (
                        learn_traj_batch.priorities ** (-self.importance_weights_sch(time_state["updates"]))
                        if self.config.get("PRIORITIZED_EXPERIENCE_REPLAY", False)
                        else jnp.ones((self.config['BUFFER_BATCH_SIZE'],))
                    )
                    importance_weights /= importance_weights.max()

                    # for iql the loss must be computed differently with or without parameters sharing
                    if self.config.get("PARAMETERS_SHARING", True):
                        init_hs = ScannedRNN.initialize_carry(
                            self.config["AGENT_HIDDEN_DIM"],
                            len(self.wrapped_env._env.agents) * self.config["BUFFER_BATCH_SIZE"],
                            (self.wrapped_env._env.transform_no + 1),
                        )  # (n_agents*batch_size, hs_size, augment_size)
                        _, target_q_vals = self.foward_pass(target_agent_params, init_hs, learn_traj.obs, learn_traj.dones)
                        # compute loss and optimize grad
                        grad_results = grad_fn(
                            train_state.params,
                            target_q_vals,
                            init_hs,
                            learn_traj,
                            importance_weights,
                        )
                        loss, grads = grad_results
                        loss, priorities = loss
                        # grads = jax.tree_util.tree_map(lambda *x: jnp.array(x).sum(0), *grads)
                    else:
                        obs_, dones_ = self.batchify(learn_traj.obs), self.batchify(learn_traj.dones)
                        init_hs = ScannedRNN.initialize_carry(
                            self.config["AGENT_HIDDEN_DIM"],
                            len(self.wrapped_env._env.agents),
                            self.config["BUFFER_BATCH_SIZE"],
                        )  # (n_agents, batch_size, hs_size)
                        _, target_q_vals = self.agent.apply(target_agent_params, init_hs, (obs_, dones_))

                        loss, grads = jax.vmap(grad_fn, in_axes=0)(
                            train_state.params,
                            target_q_vals,
                            init_hs,
                            obs_,
                            dones_,
                            self.batchify(learn_traj.actions),
                            self.batchify(self.wrapped_env.valid_actions_oh),
                            self.batchify(learn_traj.rewards),
                        )
                        loss, priorities = loss
                        loss, priorities = loss.mean(), priorities.mean(0)

                    # apply gradients
                    # rescale_factor = 1/np.sqrt(len(self.wrapped_env._env.act_type_idx)+1)
                    # for x in self.agent.common_layers:
                    #     if self.agent.layer_name[x] in grads['params'].keys():
                    #         grads['params'][self.agent.layer_name[x]]=jax.tree_util.tree_map(lambda z:z*rescale_factor,grads['params'][self.agent.layer_name[x]])
                    train_state = train_state.apply_gradients(grads=grads)
                    # rescale_factor = 1/np.sqrt(len(self.wrapped_env._env.act_type_idx)+1)
                    # for x in grads:
                    #     train_state = train_state.apply_gradients(grads=jax.tree_util.tree_map(lambda z:z*rescale_factor,x))
                    # update priorities of sampled batch
                    if self.config.get("PRIORITIZED_EXPERIENCE_REPLAY", False):
                        buffer_state = self.buffer.set_priorities(buffer_state, learn_traj_batch.indices, priorities)
                    return (train_state, buffer_state, rng), loss

                is_learn_time = self.buffer.can_sample(buffer_state)
                rng, _rng = jax.random.split(rng)
                (train_state, buffer_state, rng), loss = jax.lax.cond(
                    is_learn_time,
                    lambda train_state, buffer_state, rng: jax.lax.scan(
                        _learn_phase,
                        (train_state, buffer_state, rng),
                        None,
                        self.config["NUM_EPOCHS"],
                    ),
                    lambda train_state, buffer_state, rng: (
                        (train_state, buffer_state, rng),
                        jnp.zeros(self.config["NUM_EPOCHS"]),
                    ),
                    train_state,
                    buffer_state,
                    _rng,
                )
                # UPDATE THE VARIABLES AND RETURN
                # reset the environment
                rng, _rng = jax.random.split(rng)
                init_obs, env_state = self.wrapped_env.batch_reset(_rng)
                init_dones = {agent: jnp.zeros((self.config["NUM_ENVS"]), dtype=bool) for agent in self.wrapped_env._env.agents + ["__all__"]}

                # update the states
                time_state["timesteps"] = step_state[-1]
                time_state["updates"] = time_state["updates"] + 1
                time_state["gradients"] = time_state["gradients"] + is_learn_time * self.config["NUM_EPOCHS"]

                # update the target network if necessary
                target_agent_params = jax.lax.cond(
                    time_state["updates"] % self.config["TARGET_UPDATE_INTERVAL"] == 0,
                    lambda _: optax.incremental_update(train_state.params, target_agent_params, self.config["TAU"]),
                    lambda _: target_agent_params,
                    operand=None,
                )

                # update the greedy rewards
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    time_state["updates"] % (self.config["TEST_INTERVAL"] // self.config["NUM_STEPS"] // self.config["NUM_ENVS"]) == 0,
                    lambda _: self._get_greedy_metrics(_rng, train_state.params, time_state),
                    lambda _: test_metrics,
                    operand=None,
                )

                # update the returning metrics
                metrics = {
                    "timesteps": time_state["timesteps"] * self.config["NUM_ENVS"],
                    "updates": time_state["updates"],
                    "gradients": time_state["gradients"],
                    "loss": jax.lax.select(is_learn_time, loss.mean(), np.nan),
                    "rewards": jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0).mean(), traj_batch.rewards),
                }
                metrics["test_metrics"] = test_metrics  # add the test metrics dictionary

                if self.config.get("WANDB_ONLINE_REPORT", False):
                    jax.debug.callback(callback_wandb_report, metrics, traj_batch.infos)

                # reset param if necessary
                if self.config.get("PARAM_RESET", False):
                    rng, _rng = jax.random.split(rng)
                    if self.config.get("PARAMETERS_SHARING", True):
                        network_params = self.agent.init(_rng, init_hs, init_x)
                    else:
                        rngs = jax.random.split(_rng, len(self.wrapped_env._env.agents))  # a random init for each agent
                        network_params = jax.vmap(self.agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)
                    for sub_param in self.agent.reset_layers:
                        layer_name = self.agent.layer_name[sub_param]
                        param_to_copy = network_params["params"][layer_name]
                        train_state.params["params"][layer_name] = jax.tree_util.tree_map(
                            lambda x, y: jax.lax.select(
                                time_state["updates"] % self.config["PARAM_RESET_INTERVAL"] == 0,
                                jnp.copy(x),
                                y,
                            ),
                            param_to_copy,
                            train_state.params["params"][layer_name],
                        )
                        target_agent_params["params"][layer_name] = jax.tree_util.tree_map(
                            lambda x, y: jax.lax.select(
                                time_state["updates"] % self.config["PARAM_RESET_INTERVAL"] == 0,
                                jnp.copy(x),
                                y,
                            ),
                            param_to_copy,
                            target_agent_params["params"][layer_name],
                        )
                runner_state = (
                    train_state,
                    target_agent_params,
                    env_state,
                    buffer_state,
                    time_state,
                    init_obs,
                    init_dones,
                    test_metrics,
                    rng,
                )

                return runner_state, metrics

            time_state = {
                "timesteps": jnp.array(0),
                "updates": jnp.array(0),
                "gradients": jnp.array(0),
            }
            rng, _rng = jax.random.split(rng)
            test_metrics = self._get_greedy_metrics(_rng, train_state.params, time_state)  # initial greedy metrics

            # train
            rng, _rng = jax.random.split(rng)
            runner_state = (
                train_state,
                target_agent_params,
                env_state,
                buffer_state,
                time_state,
                init_obs,
                init_dones,
                test_metrics,
                _rng,
            )
            runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, self.num_updates)
            return {"runner_state": runner_state, "metrics": metrics}

        self.train_fn = train

class VDN(BaseQL):
    def __init__(self, config: dict, env):
        super().__init__(config, env)
        def train(rng):
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            self.wrapped_env = CTRolloutManager(self.wrapped_env._env, batch_size=self.config["NUM_ENVS"],preprocess_obs=False)
            self.test_env = CTRolloutManager(self.wrapped_env._env, batch_size=self.config["NUM_TEST_EPISODES"],preprocess_obs=False) # batched env for testing (has different batch size)
            init_obs, env_state = self.wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((self.config["NUM_ENVS"]), dtype=bool) for agent in self.wrapped_env._env.agents+['__all__']}

            # INIT BUFFER
            # to initalize the buffer is necessary to sample a trajectory to know its structure
            _, sample_traj = jax.lax.scan(self._env_sample_step, (env_state, _rng), None, self.config["NUM_STEPS"])
            sample_traj_unbatched = jax.tree_util.tree_map(lambda x: x[:, 0][:, np.newaxis], sample_traj) # remove the NUM_ENV dim, add dummy dim 1
            sample_traj_unbatched = sample_traj_unbatched.augment_reward_invariant(
                self.wrapped_env._env.agents, None,
                self.wrapped_env._env.reward_invariant_transform_obs,
                self.wrapped_env._env.reward_invariant_transform_acts,
                self.wrapped_env._env.transform_no+1,
                1 # append augment data at dim 1
            )
            buffer_state = self.buffer.init(sample_traj_unbatched) 

            # INIT NETWORK
            rng, _rng = jax.random.split(rng)
            if self.config.get("PARAMETERS_SHARING", True):
                init_x = (
                    jnp.zeros((1, 1, self.wrapped_env.obs_size)),  # (time_step, batch_size, obs_size)
                    jnp.zeros((1, 1)),  # (time_step, batch size)
                )
                init_hs = ScannedRNN.initialize_carry(self.config["AGENT_HIDDEN_DIM"], 1)  # (batch_size, hidden_dim)
                if self.init_param is None:
                    network_params = self.agent.init(_rng, init_hs, init_x)
                else:
                    network_params = self.init_param
            else:
                init_x = (
                    jnp.zeros((len(self.wrapped_env._env.agents), 1, 1, self.wrapped_env.obs_size)),  # (time_step, batch_size, obs_size)
                    jnp.zeros((len(self.wrapped_env._env.agents), 1, 1)),  # (time_step, batch size)
                )
                init_hs = ScannedRNN.initialize_carry(self.config["AGENT_HIDDEN_DIM"], len(self.wrapped_env._env.agents), 1)  # (n_agents, batch_size, hidden_dim)
                rngs = jax.random.split(_rng, len(self.wrapped_env._env.agents))  # a random init for each agent
                if self.init_param is None:
                    network_params = jax.vmap(self.agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)
                else:
                    network_params = jax.vmap(lambda x,y: y.copy(), in_axes=(0))(rngs, self.init_param)

            # INIT TRAIN STATE AND OPTIMIZER
            train_state = TrainState.create(
                apply_fn=self.agent.apply,
                params=network_params,
                tx=self.tx,
            )
            # target network params
            target_agent_params = jax.tree_util.tree_map(lambda x: jnp.copy(x), train_state.params)

            # TRAINING LOOP
            def _update_step(runner_state, unused):

                train_state, target_agent_params, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state

                # EPISODE STEP
                # prepare the step state and collect the episode trajectory
                rng, _rng = jax.random.split(rng)
                if self.config.get('PARAMETERS_SHARING', True):
                    hstate = ScannedRNN.initialize_carry(self.config['AGENT_HIDDEN_DIM'], len(self.wrapped_env._env.agents)*self.config["NUM_ENVS"]) # (n_agents*n_envs, hs_size)
                else:
                    hstate = ScannedRNN.initialize_carry(self.config['AGENT_HIDDEN_DIM'], len(self.wrapped_env._env.agents), self.config["NUM_ENVS"]) # (n_agents, n_envs, hs_size)

                step_state = (
                    train_state.params,
                    env_state,
                    init_obs,
                    init_dones,
                    hstate, 
                    _rng,
                    time_state['timesteps'] # t is needed to compute epsilon
                )

                step_state, traj_batch = jax.lax.scan(self._env_step, step_state, None, self.config["NUM_STEPS"])
                # LEARN PHASE
                def _loss_fn(params, target_q_vals, init_hs, learn_traj, importance_weights):
                    _, q_vals = self.foward_pass(params, init_hs, learn_traj.obs, learn_traj.dones)
                    if not self.config.get('DISTRIBUTION_Q',False):
                        # get the q_vals of the taken actions (with exploration) for each agent
                        chosen_action_qvals = jax.tree_util.tree_map(
                            lambda q, u: q_of_action(q, u+jnp.broadcast_to(jnp.array(self.act_type_idx_offset),u.shape))[:-1], # avoid last timestep
                            q_vals,
                            learn_traj.actions
                        )
                        # get the target q values of the greedy actions
                        valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q[..., valid_idx], q_vals, self.wrapped_env.valid_actions)
                        target_max_qvals = jax.tree_util.tree_map(
                            lambda t_q, q: q_of_action(t_q, jnp.stack([jnp.argmax(q[...,x], axis=-1)+self.act_type_idx_offset[i] for i,x in enumerate(self.wrapped_env._env.act_type_idx)],axis=-1))[1:], # get the greedy actions and avoid first timestep
                            target_q_vals,
                            valid_q_vals
                        )
                        # VDN: computes q_tot as the sum of the agents' individual q values
                        chosen_action_qvals_sum = jnp.stack(list(chosen_action_qvals.values())).sum(axis=0)
                        target_max_qvals_sum = jnp.stack(list(target_max_qvals.values())).sum(axis=0)
                        # get centralized reward vector along action types
                        rewards_vec=jnp.stack([jnp.stack([learn_traj.infos[f'reward_{x}'][...,i] for x in range(len(self.wrapped_env._env.act_type_idx))],axis=-1) for i,_ in enumerate(self.wrapped_env._env.agents)],axis=0).sum(0)
                        # compute the centralized targets using the "__all__" rewards and dones
                        dones=jnp.tile(learn_traj.dones['__all__'][...,None],(target_max_qvals_sum.shape[-1],))
                        # rewards_=jnp.tile(learn_traj.rewards['__all__'][...,None],(target_max_qvals_sum.shape[-1],))
                        mean_axes=tuple(range(chosen_action_qvals_sum.ndim-1))
                    else:
                        q_vals=q_softmax_from_dis(q_vals)
                        target_q_vals=q_softmax_from_dis(target_q_vals)
                        chosen_action_qvals = jax.tree_util.tree_map(
                            lambda q, u: q_of_action(q, (u+jnp.broadcast_to(self.act_type_idx_offset,u.shape))[...,None],-2)[:-1], # avoid last timestep
                            q_vals,
                            learn_traj.actions
                        )
                        # get the target for each agent (assumes every agent has a reward)
                        valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: (q*self.dis_support).sum(-1)[..., valid_idx], q_vals, self.wrapped_env.valid_actions)# get expectation of q-value
                        target_max_qvals_dis = jax.tree_util.tree_map(
                            lambda t_q, q: q_of_action(t_q, jnp.stack([jnp.argmax(q[...,x], axis=-1)+self.act_type_idx_offset[i] for i,x in enumerate(self.wrapped_env._env.act_type_idx)],axis=-1)[...,None],-2)[1:], # avoid first timestep
                            target_q_vals,
                            jax.lax.stop_gradient(valid_q_vals)
                        )
                        # TODO: calculate sum of distributions along axis 0
                        chosen_action_qvals_sum = jnp.stack(list(chosen_action_qvals.values()))
                        target_max_qvals=jax.tree_util.tree_map(lambda x:jnp.tile(self.dis_support,x.shape[:-1]+(1,)),target_max_qvals_dis)
                        # get centralized reward vector along action types
                        rewards_vec=jnp.stack([jnp.stack([learn_traj.infos[f'reward_{x}'][...,i] for x in range(len(self.wrapped_env._env.act_type_idx))],axis=-1)[...,None] for i,_ in enumerate(self.wrapped_env._env.agents)],axis=0).sum(0)
                        # compute the centralized targets using the "__all__" rewards and dones
                        dones=jnp.tile(learn_traj.dones['__all__'][...,None],(target_max_qvals_sum.shape[-1],))[...,None]
                        # rewards_=jnp.tile(learn_traj.rewards['__all__'][...,None],(target_max_qvals_sum.shape[-1],))
                        mean_axes=tuple(range(chosen_action_qvals_sum.ndim-2))
                    targets = jax.tree_util.tree_map(
                        self.target_fn,
                        target_max_qvals_sum,
                        rewards_vec,  # {agent:learn_traj.rewards[agent] for agent in self.wrapped_env._env.agents}, # rewards and agents could contain additional keys
                        dones,
                    )
                    if not self.config.get('DISTRIBUTION_Q',False):
                        err = chosen_action_qvals_sum - jax.lax.stop_gradient(targets)
                        loss = (0.5 if self.config.get('TD_LAMBDA_LOSS', True) else 1)*(err**2)
                        err=jnp.abs(err)
                    else:
                        def dis_shift(dis,base_support,new_support,cell_radius):# assuming dis and support span the last dimension
                            coef=jnp.clip(cell_radius-jnp.abs(base_support[...,None]-new_support),0.0,None)# projection coefficients
                            return (dis[...,None]*coef).sum(-2)/cell_radius# project back onto initial support
                        targets=dis_shift(target_max_qvals_dis,jnp.clip(targets,self.config['DISTRIBUTION_RANGE'][0],self.config['DISTRIBUTION_RANGE'][1]),self.dis_support,self.dis_support_step)
                        loss=-((jax.lax.stop_gradient(targets))*jnp.log(chosen_action_qvals_sum)).sum(-1)# cross-entropy
                        err=loss
                    err_axes=tuple(range(err.ndim))
                    importance_weights = importance_weights[(None, ...) + (None,) * (loss.ndim - importance_weights.ndim - 1)]
                    return (loss*importance_weights).mean(axis=mean_axes).mean(), err.mean(axis=err_axes[0:1]+err_axes[2:])/len(self.wrapped_env._env.agents)

                # BUFFER UPDATE: save the collected trajectory in the buffer
                buffer_traj_batch = jax.tree_util.tree_map(
                    lambda x:jnp.swapaxes(x, 0, 1)[:, np.newaxis,:, np.newaxis], # put the batch dim first, add a dummy sequence dim, add dummy dim for augment
                    traj_batch
                ) # (num_envs, 1, time_steps, ...)
                buffer_traj_batch = buffer_traj_batch.augment_reward_invariant(
                    self.wrapped_env._env.agents, None,
                    self.wrapped_env._env.reward_invariant_transform_obs,
                    self.wrapped_env._env.reward_invariant_transform_acts,
                    self.wrapped_env._env.transform_no+1,
                    3 # append augment data at dim 3
                )
                buffer_state = self.buffer.add(buffer_state, buffer_traj_batch)
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                def _learn_phase(carry,_):
                    train_state, buffer_state, rng = carry
                    # sample a batched trajectory from the buffer and set the time step dim in first axis
                    rng, _rng = jax.random.split(rng)
                    learn_traj_batch = self.buffer.sample(buffer_state, _rng) # (batch_size, 1, max_time_steps, ...)
                    learn_traj = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(x[:, 0], 0, 1), # remove the dummy sequence dim (1) and swap batch and temporal dims
                        learn_traj_batch.experience
                    ) # (max_time_steps, batch_size, ...)  # (max_time_steps, batch_size, ...)
                    importance_weights = (
                        learn_traj_batch.priorities
                        ** (-self.importance_weights_sch(time_state["updates"]))
                        if self.config.get("PRIORITIZED_EXPERIENCE_REPLAY", False)
                        else jnp.ones((self.config['BUFFER_BATCH_SIZE'],))
                    )
                    importance_weights /= importance_weights.max()
                    if self.config.get('PARAMETERS_SHARING', True):
                        init_hs = ScannedRNN.initialize_carry(self.config['AGENT_HIDDEN_DIM'], len(self.wrapped_env._env.agents)*self.config["BUFFER_BATCH_SIZE"], (self.wrapped_env._env.transform_no+1)) # (n_agents*batch_size, hs_size)
                    else:
                        init_hs = ScannedRNN.initialize_carry(self.config['AGENT_HIDDEN_DIM'], len(self.wrapped_env._env.agents), self.config["BUFFER_BATCH_SIZE"]) # (n_agents, batch_size, hs_size)
                    _, target_q_vals = self.foward_pass(target_agent_params, init_hs, learn_traj.obs, learn_traj.dones)
                    # compute loss and optimize grad
                    grad_results = grad_fn(train_state.params, target_q_vals, init_hs, learn_traj, importance_weights)
                    loss, grads = grad_results
                    loss, priorities = loss
                    # rescale_factor = 1/np.sqrt(len(self.wrapped_env._env.act_type_idx)+1)
                    train_state = train_state.apply_gradients(grads=grads)
                    # update priorities of sampled batch
                    if self.config.get('PRIORITIZED_EXPERIENCE_REPLAY', False):
                        buffer_state = self.buffer.set_priorities(buffer_state, learn_traj_batch.indices, priorities)
                    return (train_state,buffer_state,rng),loss
                is_learn_time = (self.buffer.can_sample(buffer_state))
                rng,_rng=jax.random.split(rng)
                (train_state,buffer_state,rng),loss=jax.lax.cond(
                    is_learn_time,
                    lambda train_state,buffer_state,rng:jax.lax.scan(_learn_phase,(train_state,buffer_state,rng),None,self.config['NUM_EPOCHS']),
                    lambda train_state,buffer_state,rng:((train_state,buffer_state,rng),jnp.zeros(self.config["NUM_EPOCHS"])),
                    train_state,buffer_state,_rng
                )
                # UPDATE THE VARIABLES AND RETURN
                # reset the environment
                rng, _rng = jax.random.split(rng)
                init_obs, env_state = self.wrapped_env.batch_reset(_rng)
                init_dones = {agent:jnp.zeros((self.config["NUM_ENVS"]), dtype=bool) for agent in self.wrapped_env._env.agents+['__all__']}

                # update the states
                time_state['timesteps'] = step_state[-1]
                time_state['updates']   = time_state['updates'] + 1
                time_state['gradients'] = time_state['gradients'] + is_learn_time*self.config['NUM_EPOCHS']

                # update the target network if necessary
                target_agent_params = jax.lax.cond(
                    time_state['updates'] % self.config['TARGET_UPDATE_INTERVAL'] == 0,
                    lambda _: optax.incremental_update(train_state.params,target_agent_params,self.config["TAU"]),
                    lambda _: target_agent_params,
                    operand=None
                )

                # update the greedy rewards
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    time_state['updates'] % (self.config["TEST_INTERVAL"] // self.config["NUM_STEPS"] // self.config["NUM_ENVS"]) == 0,
                    lambda _: self._get_greedy_metrics(_rng, train_state.params, time_state),
                    lambda _: test_metrics,
                    operand=None
                )

                # update the returning metrics
                metrics = {
                    'timesteps': time_state['timesteps']*self.config['NUM_ENVS'],
                    'updates' : time_state['updates'],
                    'gradients' : time_state['gradients'],
                    'loss': jax.lax.select(is_learn_time,loss.mean(),np.nan),
                    'rewards': jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0).mean(), traj_batch.rewards),
                }
                metrics['test_metrics'] = test_metrics # add the test metrics dictionary

                if self.config.get('WANDB_ONLINE_REPORT', False):
                    jax.debug.callback(callback_wandb_report, metrics, traj_batch.infos)
                # reset param if necessary
                if self.config.get('PARAM_RESET', False):
                    rng, _rng = jax.random.split(rng)
                    if self.config.get('PARAMETERS_SHARING', True):
                        network_params = self.agent.init(_rng, init_hs, init_x)
                    else:
                        rngs = jax.random.split(_rng, len(self.wrapped_env._env.agents)) # a random init for each agent
                        network_params = jax.vmap(self.agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)
                    for sub_param in self.agent.reset_layers:
                        layer_name=self.agent.layer_name[sub_param]
                        param_to_copy=network_params['params'][layer_name]
                        train_state.params['params'][layer_name]=jax.tree_util.tree_map(lambda x,y:jax.lax.select(time_state['updates']%self.config['PARAM_RESET_INTERVAL']==0,jnp.copy(x),y),param_to_copy,train_state.params['params'][layer_name])
                        target_agent_params['params'][layer_name]=jax.tree_util.tree_map(lambda x,y:jax.lax.select(time_state['updates']%self.config['PARAM_RESET_INTERVAL']==0,jnp.copy(x),y),param_to_copy,target_agent_params['params'][layer_name])
                runner_state = (
                    train_state,
                    target_agent_params,
                    env_state,
                    buffer_state,
                    time_state,
                    init_obs,
                    init_dones,
                    test_metrics,
                    rng
                )

                return runner_state, metrics
            
            time_state = {
                'timesteps':jnp.array(0),
                'updates':  jnp.array(0),
                'gradients':jnp.array(0)
            }
            rng, _rng = jax.random.split(rng)
            test_metrics = self._get_greedy_metrics(_rng, train_state.params, time_state) # initial greedy metrics
            
            # train
            rng, _rng = jax.random.split(rng)
            runner_state = (
                train_state,
                target_agent_params,
                env_state,
                buffer_state,
                time_state,
                init_obs,
                init_dones,
                test_metrics,
                _rng
            )
            runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, self.num_updates)
            return {'runner_state':runner_state, 'metrics':metrics}
        
        self.train_fn = train

class QMIX(BaseQL):
    def __init__(self, config: dict, env):
        super().__init__(config, env)
        self.foward_pass = partial(homogeneous_pass_ps, self.agent) # enforce sharing agent params
        self.mixer = MixingNetwork(
            config["MIXER_EMBEDDING_DIM"],
            config["MIXER_HYPERNET_HIDDEN_DIM"],
            config["MIXER_INIT_SCALE"],
        )
        def train(rng):
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            self.wrapped_env = CTRolloutManager(self.wrapped_env._env, batch_size=self.config["NUM_ENVS"],preprocess_obs=False)
            self.test_env = CTRolloutManager(self.wrapped_env._env, batch_size=self.config["NUM_TEST_EPISODES"],preprocess_obs=False) # batched env for testing (has different batch size)
            init_obs, env_state = self.wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((self.config["NUM_ENVS"]), dtype=bool) for agent in self.wrapped_env._env.agents+['__all__']}

            # INIT BUFFER
            # to initalize the buffer is necessary to sample a trajectory to know its structure
            _, sample_traj = jax.lax.scan(self._env_sample_step, (env_state, _rng), None, self.config["NUM_STEPS"])
            sample_traj_unbatched = jax.tree_util.tree_map(lambda x: x[:, 0][:, np.newaxis], sample_traj) # remove the NUM_ENV dim, add dummy dim 1
            sample_traj_unbatched = sample_traj_unbatched.augment_reward_invariant(
                self.wrapped_env._env.agents, None,
                self.wrapped_env._env.reward_invariant_transform_obs,
                self.wrapped_env._env.reward_invariant_transform_acts,
                self.wrapped_env._env.transform_no+1,
                1 # append augment data at dim 1
            )
            buffer_state = self.buffer.init(sample_traj_unbatched) 

            # INIT AGENT NETWORK
            rng, _rng = jax.random.split(rng)
            init_x = (
                jnp.zeros((1, 1, self.wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)) # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(self.config['AGENT_HIDDEN_DIM'], 1) # (batch_size, hidden_dim)
            if self.init_param is None:
                agent_params = self.agent.init(_rng, init_hs, init_x)
            else:
                agent_params = self.init_param
            # INIT MIXER
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros((len(self.wrapped_env._env.agents), 1, 1, len(self.wrapped_env._env.act_type_idx))) # q vals: agents, time, batch, act_dim
            state_size = sample_traj.obs["__all__"].shape[-1] # get the state shape from the buffer
            init_state = jnp.zeros((1, 1, state_size)) # (time_step, batch_size, obs_size)
            mixer_params = self.mixer.init(_rng, init_x, init_state)
            
            # INIT TRAIN STATE AND OPTIMIZER
            network_params = {'agent':agent_params, 'mixer':mixer_params}
            train_state = TrainState.create(
                apply_fn=self.agent.apply,
                params=network_params,
                tx=self.tx,
            )
            # target network params
            target_agent_params = jax.tree_util.tree_map(lambda x: jnp.copy(x), train_state.params)

            # TRAINING LOOP
            def _update_step(runner_state, unused):

                train_state, target_agent_params, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state

                # EPISODE STEP
                # prepare the step state and collect the episode trajectory
                rng, _rng = jax.random.split(rng)
                hstate = ScannedRNN.initialize_carry(self.config['AGENT_HIDDEN_DIM'], len(self.wrapped_env._env.agents)*self.config["NUM_ENVS"]) # (n_agents*n_envs, hs_size)

                step_state = (
                    train_state.params['agent'],
                    env_state,
                    init_obs,
                    init_dones,
                    hstate, 
                    _rng,
                    time_state['timesteps'] # t is needed to compute epsilon
                )

                step_state, traj_batch = jax.lax.scan(self._env_step, step_state, None, self.config["NUM_STEPS"])
                # BUFFER UPDATE: save the collected trajectory in the buffer
                buffer_traj_batch = jax.tree_util.tree_map(
                    lambda x:jnp.swapaxes(x, 0, 1)[:, np.newaxis,:, np.newaxis], # put the batch dim first, add a dummy sequence dim, add dummy dim for augment
                    traj_batch
                ) # (num_envs, 1, time_steps, ...)
                buffer_traj_batch = buffer_traj_batch.augment_reward_invariant(
                    self.wrapped_env._env.agents, None,
                    self.wrapped_env._env.reward_invariant_transform_obs,
                    self.wrapped_env._env.reward_invariant_transform_acts,
                    self.wrapped_env._env.transform_no+1,
                    3 # append augment data at dim 3
                )
                buffer_state = self.buffer.add(buffer_state, buffer_traj_batch)
                # LEARN PHASE
                def _loss_fn(params, target_q_vals, target_mixer_param, init_hs, learn_traj, obs_all, importance_weights):
                    _, q_vals = self.foward_pass(params['agent'], init_hs, learn_traj.obs, learn_traj.dones)
                    # get the q_vals of the taken actions (with exploration) for each agent
                    chosen_action_qvals = jax.tree_util.tree_map(
                        lambda q, u: q_of_action(q, u+jnp.broadcast_to(jnp.array(self.act_type_idx_offset),u.shape))[:-1], # avoid last timestep
                        q_vals,
                        learn_traj.actions
                    )
                    # get the target q values of the greedy actions
                    valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q[..., valid_idx], q_vals, self.wrapped_env.valid_actions)
                    target_max_qvals = jax.tree_util.tree_map(
                        lambda t_q, q: q_of_action(t_q, jnp.stack([jnp.argmax(q[...,x], axis=-1)+self.act_type_idx_offset[i] for i,x in enumerate(self.wrapped_env._env.act_type_idx)],axis=-1))[1:], # get the greedy actions and avoid first timestep
                        target_q_vals,
                        valid_q_vals
                    )
                    # pass concatenated observation+q_val via mixer net
                    chosen_action_qvals = jnp.stack(list(chosen_action_qvals.values()))
                    batch_size, aug_size = chosen_action_qvals.shape[-3], chosen_action_qvals.shape[-2]
                    chosen_action_qvals = jnp.reshape(chosen_action_qvals, chosen_action_qvals.shape[:-3]+(chosen_action_qvals.shape[-3]*chosen_action_qvals.shape[-2],chosen_action_qvals.shape[-1]))
                    target_max_qvals = jnp.stack(list(target_max_qvals.values()))
                    target_max_qvals = jnp.reshape(target_max_qvals, target_max_qvals.shape[:-3]+(target_max_qvals.shape[-3]*target_max_qvals.shape[-2],target_max_qvals.shape[-1]))
                    qmix = self.mixer.apply(params['mixer'], chosen_action_qvals, obs_all[:-1])
                    qmix_next = self.mixer.apply(target_mixer_param, target_max_qvals, obs_all[1:])
                    # get centralized reward vector along action types
                    rewards_vec=jnp.stack([jnp.stack([learn_traj.infos[f'reward_{x}'][...,i] for x in range(len(self.wrapped_env._env.act_type_idx))],axis=-1) for i,_ in enumerate(self.wrapped_env._env.agents)],axis=0).sum(0)
                    rewards_vec = jnp.reshape(rewards_vec, rewards_vec.shape[:-3]+(rewards_vec.shape[-3]*rewards_vec.shape[-2],rewards_vec.shape[-1]))
                    # compute the centralized targets using the "__all__" rewards and dones
                    dones=jnp.tile(learn_traj.dones['__all__'][...,None],(qmix_next.shape[-1],))
                    dones = jnp.reshape(dones, dones.shape[:-3]+(dones.shape[-3]*dones.shape[-2],dones.shape[-1]))
                    mean_axes=tuple(range(qmix.ndim-1))
                    targets = jax.tree_util.tree_map(
                        self.target_fn,
                        qmix_next,
                        rewards_vec-env.reward_min,  # ensure rewards are non-negative
                        dones,
                    )
                    err = qmix - jax.lax.stop_gradient(targets)
                    loss = (0.5 if self.config.get('TD_LAMBDA_LOSS', True) else 1)*(err**2)
                    err=jnp.abs(err)
                    # (time, batch*aug, act_dim)
                    # unmerge batch dim from aug dim to apply sample weights
                    loss = jnp.reshape(loss, loss.shape[:-2]+(batch_size,aug_size,loss.shape[-1]))
                    err = jnp.reshape(err, err.shape[:-2]+(batch_size,aug_size,err.shape[-1]))
                    # (time, batch, aug, act_dim)
                    err_axes=tuple(range(err.ndim))
                    importance_weights = importance_weights[(None, ...) + (None,) * (loss.ndim - importance_weights.ndim - 1)]
                    return (loss*importance_weights).mean(axis=mean_axes).mean(), err.mean(axis=err_axes[0:1]+err_axes[2:])/len(self.wrapped_env._env.agents)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                def _learn_phase(carry,_):
                    train_state, buffer_state, rng = carry
                    # sample a batched trajectory from the buffer and set the time step dim in first axis
                    rng, _rng = jax.random.split(rng)
                    learn_traj_batch = self.buffer.sample(buffer_state, _rng) # (batch_size, 1, max_time_steps, ...)
                    learn_traj = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(x[:, 0], 0, 1), # remove the dummy sequence dim (1) and swap batch and temporal dims
                        learn_traj_batch.experience
                    ) # (max_time_steps, batch_size, ...)  # (max_time_steps, batch_size, ...)
                    importance_weights = (
                        learn_traj_batch.priorities
                        ** (-self.importance_weights_sch(time_state["updates"]))
                        if self.config.get("PRIORITIZED_EXPERIENCE_REPLAY", False)
                        else jnp.ones((self.config['BUFFER_BATCH_SIZE'],))
                    )
                    importance_weights /= importance_weights.max()
                    init_hs = ScannedRNN.initialize_carry(self.config['AGENT_HIDDEN_DIM'], len(self.wrapped_env._env.agents)*self.config["BUFFER_BATCH_SIZE"], (self.wrapped_env._env.transform_no+1)) # (n_agents*batch_size, hs_size)

                    _, target_q_vals = self.foward_pass(target_agent_params['agent'], init_hs, learn_traj.obs, learn_traj.dones)
                    # merge aug dim with batch dim for dimension-agnostic mixing net
                    obs_all = self.wrapped_env.global_state(learn_traj.obs, None)
                    obs_all = jnp.reshape(obs_all, obs_all.shape[:-3]+(obs_all.shape[-3]*obs_all.shape[-2],obs_all.shape[-1]))
                    # compute loss and optimize grad
                    grad_results = grad_fn(train_state.params, target_q_vals, target_agent_params['mixer'], init_hs, learn_traj, obs_all, importance_weights)
                    loss, grads = grad_results
                    loss, priorities = loss
                    # rescale_factor = 1/np.sqrt(len(self.wrapped_env._env.act_type_idx)+1)
                    train_state = train_state.apply_gradients(grads=grads)
                    # update priorities of sampled batch
                    if self.config.get('PRIORITIZED_EXPERIENCE_REPLAY', False):
                        buffer_state = self.buffer.set_priorities(buffer_state, learn_traj_batch.indices, priorities)
                    return (train_state,buffer_state,rng),loss
                is_learn_time = (self.buffer.can_sample(buffer_state))
                rng,_rng=jax.random.split(rng)
                (train_state,buffer_state,rng),loss=jax.lax.cond(
                    is_learn_time,
                    lambda train_state,buffer_state,rng:jax.lax.scan(_learn_phase,(train_state,buffer_state,rng),None,self.config['NUM_EPOCHS']),
                    lambda train_state,buffer_state,rng:((train_state,buffer_state,rng),jnp.zeros(self.config["NUM_EPOCHS"])),
                    train_state,buffer_state,_rng
                )
                # UPDATE THE VARIABLES AND RETURN
                # reset the environment
                rng, _rng = jax.random.split(rng)
                init_obs, env_state = self.wrapped_env.batch_reset(_rng)
                init_dones = {agent:jnp.zeros((self.config["NUM_ENVS"]), dtype=bool) for agent in self.wrapped_env._env.agents+['__all__']}

                # update the states
                time_state['timesteps'] = step_state[-1]
                time_state['updates']   = time_state['updates'] + 1
                time_state['gradients'] = time_state['gradients'] + is_learn_time*self.config['NUM_EPOCHS']

                # update the target network if necessary
                target_agent_params = jax.lax.cond(
                    time_state['updates'] % self.config['TARGET_UPDATE_INTERVAL'] == 0,
                    lambda _: optax.incremental_update(train_state.params,target_agent_params,self.config["TAU"]),
                    lambda _: target_agent_params,
                    operand=None
                )

                # update the greedy rewards
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    time_state['updates'] % (self.config["TEST_INTERVAL"] // self.config["NUM_STEPS"] // self.config["NUM_ENVS"]) == 0,
                    lambda _: self._get_greedy_metrics(_rng, train_state.params['agent'], time_state),
                    lambda _: test_metrics,
                    operand=None
                )

                # update the returning metrics
                metrics = {
                    'timesteps': time_state['timesteps']*self.config['NUM_ENVS'],
                    'updates' : time_state['updates'],
                    'gradients' : time_state['gradients'],
                    'loss': jax.lax.select(is_learn_time,loss.mean(),np.nan),
                    'rewards': jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0).mean(), traj_batch.rewards),
                }
                metrics['test_metrics'] = test_metrics # add the test metrics dictionary

                if self.config.get('WANDB_ONLINE_REPORT', False):
                    jax.debug.callback(callback_wandb_report, metrics, traj_batch.infos)
                # reset param if necessary
                # if self.config.get('PARAM_RESET', False):
                #     rng, _rng = jax.random.split(rng)
                #     if self.config.get('PARAMETERS_SHARING', True):
                #         network_params = self.agent.init(_rng, init_hs, init_x)
                #     else:
                #         rngs = jax.random.split(_rng, len(self.wrapped_env._env.agents)) # a random init for each agent
                #         network_params = jax.vmap(self.agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)
                #     for sub_param in self.agent.reset_layers:
                #         layer_name=self.agent.layer_name[sub_param]
                #         param_to_copy=network_params['params'][layer_name]
                #         train_state.params['params'][layer_name]=jax.tree_util.tree_map(lambda x,y:jax.lax.select(time_state['updates']%self.config['PARAM_RESET_INTERVAL']==0,jnp.copy(x),y),param_to_copy,train_state.params['params'][layer_name])
                #         target_agent_params['params'][layer_name]=jax.tree_util.tree_map(lambda x,y:jax.lax.select(time_state['updates']%self.config['PARAM_RESET_INTERVAL']==0,jnp.copy(x),y),param_to_copy,target_agent_params['params'][layer_name])
                runner_state = (
                    train_state,
                    target_agent_params,
                    env_state,
                    buffer_state,
                    time_state,
                    init_obs,
                    init_dones,
                    test_metrics,
                    rng
                )

                return runner_state, metrics
            
            time_state = {
                'timesteps':jnp.array(0),
                'updates':  jnp.array(0),
                'gradients':jnp.array(0)
            }
            rng, _rng = jax.random.split(rng)
            test_metrics = self._get_greedy_metrics(_rng, train_state.params['agent'], time_state) # initial greedy metrics
            
            # train
            rng, _rng = jax.random.split(rng)
            runner_state = (
                train_state,
                target_agent_params,
                env_state,
                buffer_state,
                time_state,
                init_obs,
                init_dones,
                test_metrics,
                _rng
            )
            runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, self.num_updates)
            return {'runner_state':runner_state, 'metrics':metrics}
        
        self.train_fn = train

class SUNRISE(BaseQL): # (Dueling DDQN + Ensemble + Bellman reweighting + UCB exploration) https://arxiv.org/pdf/2007.04938
    def _greedy_env_step(self, step_state, unused):
        params, env_state, last_obs, last_dones, hstate, rng = step_state
        rng, key_s = jax.random.split(rng)
        obs_ = {a: last_obs[a] for a in self.test_env._env.agents}
        obs_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], obs_)
        dones_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], last_dones)
        hstate, q_vals = self.foward_pass(params, hstate, obs_, dones_)
        q_vals = q_vals.mean(-1)
        valid_q_vals = jax.tree_util.tree_map(
            lambda q, valid_idx: q.squeeze(0)[..., valid_idx],
            q_vals,
            self.test_env.valid_actions,
        )
        actions = jax.tree_util.tree_map(
            lambda q: jnp.stack(
                [jnp.argmax(q[..., x], axis=-1) for x in self.test_env._env.act_type_idx],
                axis=-1,
            ),
            valid_q_vals,
        )  # one argmax per action type
        obs, env_state, rewards, dones, infos = self.test_env.batch_step(key_s, env_state, actions)
        step_state = (params, env_state, obs, dones, hstate, rng)
        return step_state, (rewards, dones, infos)

    def __init__(self, config: dict, env):
        super().__init__(config, env)
        self.agent = AgentRNN(
            action_dim=self.wrapped_env.max_action_space,
            atom_dim=config["ENSEMBLE_NUM_PARAMS"],
            hidden_dim=self.config["AGENT_HIDDEN_DIM"],
            init_scale=self.config["AGENT_INIT_SCALE"],
            act_type_idx=self.wrapped_env._env.act_type_idx,
            use_rnn=self.config["RNN_LAYER"],
        )
        def train(rng):
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = self.wrapped_env.batch_reset(_rng)
            init_dones = {agent: jnp.zeros((self.config["NUM_ENVS"]), dtype=bool) for agent in self.wrapped_env._env.agents + ["__all__"]}

            # INIT BUFFER
            # to initalize the buffer is necessary to sample a trajectory to know its strucutre
            _, sample_traj = jax.lax.scan(self._env_sample_step, (env_state,_rng), None, self.config["NUM_STEPS"])
            sample_traj_unbatched = jax.tree_util.tree_map(lambda x: x[:, 0][:, np.newaxis], sample_traj)  # remove the NUM_ENV dim, add dummy dim 1
            sample_traj_unbatched = sample_traj_unbatched.augment_reward_invariant(
                self.wrapped_env._env.agents,
                self.wrapped_env._env.agents,
                self.wrapped_env._env.reward_invariant_transform_obs,
                self.wrapped_env._env.reward_invariant_transform_acts,
                self.wrapped_env._env.transform_no + 1,
                1,  # append augment data at dim 1
            )
            buffer_state = self.buffer.init(sample_traj_unbatched)

            # INIT NETWORK
            rng, _rng = jax.random.split(rng)
            if self.config.get("PARAMETERS_SHARING", True):
                init_x = (
                    jnp.zeros((1, 1, self.wrapped_env.obs_size)),  # (time_step, batch_size, obs_size)
                    jnp.zeros((1, 1)),  # (time_step, batch size)
                )
                init_hs = ScannedRNN.initialize_carry(self.config["AGENT_HIDDEN_DIM"], 1)  # (batch_size, hidden_dim)
                if self.init_param is None:
                    network_params = self.agent.init(_rng, init_hs, init_x)
                else:
                    network_params = self.init_param
            else:
                init_x = (
                    jnp.zeros((len(self.wrapped_env._env.agents), 1, 1, self.wrapped_env.obs_size)),  # (time_step, batch_size, obs_size)
                    jnp.zeros((len(self.wrapped_env._env.agents), 1, 1)),  # (time_step, batch size)
                )
                init_hs = ScannedRNN.initialize_carry(self.config["AGENT_HIDDEN_DIM"], len(self.wrapped_env._env.agents), 1)  # (n_agents, batch_size, hidden_dim)
                rngs = jax.random.split(_rng, len(self.wrapped_env._env.agents))  # a random init for each agent
                if self.init_param is None:
                    network_params = jax.vmap(self.agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)
                else:
                    network_params = jax.vmap(lambda x,y: y.copy(), in_axes=(0))(rngs, self.init_param)

            # INIT TRAIN STATE AND OPTIMIZER
            train_state = TrainState.create(
                apply_fn=self.agent.apply,
                params=network_params,
                tx=self.tx,
            )
            # target network params
            target_agent_params = jax.tree_util.tree_map(lambda x: jnp.copy(x), train_state.params)

            # TRAINING LOOP
            def _update_step(runner_state, unused):
                (
                    train_state,
                    target_agent_params,
                    env_state,
                    buffer_state,
                    time_state,
                    init_obs,
                    init_dones,
                    test_metrics,
                    rng,
                ) = runner_state

                # EPISODE STEP
                # prepare the step state and collect the episode trajectory
                rng, _rng = jax.random.split(rng)
                if self.config.get("PARAMETERS_SHARING", True):
                    hstate = ScannedRNN.initialize_carry(self.config["AGENT_HIDDEN_DIM"], len(self.wrapped_env._env.agents) * self.config["NUM_ENVS"])  # (n_agents*n_envs, hs_size)
                else:
                    hstate = ScannedRNN.initialize_carry(self.config["AGENT_HIDDEN_DIM"], len(self.wrapped_env._env.agents), self.config["NUM_ENVS"])  # (n_agents, n_envs, hs_size)

                step_state = (
                    train_state.params,
                    env_state,
                    init_obs,
                    init_dones,
                    hstate,
                    _rng,
                    time_state["timesteps"],  # t is needed to compute epsilon
                )

                step_state, traj_batch = jax.lax.scan(self._env_step, step_state, None, self.config["NUM_STEPS"])
                # BUFFER UPDATE: save the collected trajectory in the buffer
                buffer_traj_batch = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis, :, np.newaxis],  # put the batch dim first, add a dummy sequence dim, add dummy dim for augment
                    traj_batch,
                )  # (num_envs, 1, time_steps, ...)
                buffer_traj_batch = buffer_traj_batch.augment_reward_invariant(
                    self.wrapped_env._env.agents,
                    self.wrapped_env._env.agents,
                    self.wrapped_env._env.reward_invariant_transform_obs,
                    self.wrapped_env._env.reward_invariant_transform_acts,
                    self.wrapped_env._env.transform_no + 1,
                    3,  # append augment data at dim 3
                )
                buffer_state = self.buffer.add(buffer_state, buffer_traj_batch)

                if self.config.get("PARAMETERS_SHARING", True):

                    def _loss_fn(params, target_q_vals, init_hs, learn_traj, importance_weights):
                        # obs_={a:learn_traj.obs[a] for a in self.wrapped_env._env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                        _, q_vals = self.foward_pass(params, init_hs, learn_traj.obs, learn_traj.dones)
                        if not self.config.get("DISTRIBUTION_Q", False):
                            # get the q_vals of the taken actions (with exploration) for each agent
                            chosen_action_qvals = jax.tree_util.tree_map(
                                lambda q, u: q_of_action(q, u + jnp.broadcast_to(self.act_type_idx_offset, u.shape))[:-1],  # avoid last timestep
                                q_vals,
                                learn_traj.actions,
                            )
                            # get the target for each agent (assumes every agent has a reward)
                            valid_q_vals = jax.tree_util.tree_map(
                                lambda q, valid_idx: q[..., valid_idx],
                                q_vals,
                                self.wrapped_env.valid_actions,
                            )
                            target_max_qvals = jax.tree_util.tree_map(
                                lambda t_q, q: q_of_action(
                                    t_q,
                                    jnp.stack(
                                        [jnp.argmax(q[..., x], axis=-1) + self.act_type_idx_offset[i] for i, x in enumerate(self.wrapped_env._env.act_type_idx)],
                                        axis=-1,
                                    ),
                                )[1:],  # avoid first timestep
                                target_q_vals,
                                jax.lax.stop_gradient(valid_q_vals),
                            )
                            # get reward vector along action types
                            rewards_vec = {
                                a: jnp.stack(
                                    [learn_traj.infos[f"reward_{x}"][..., i] for x in range(len(self.wrapped_env._env.act_type_idx))],
                                    axis=-1,
                                )
                                for i, a in enumerate(self.wrapped_env._env.agents)
                            }
                            dones = {
                                a: jnp.tile(
                                    learn_traj.dones[a][..., None],
                                    (target_max_qvals[a].shape[-1],),
                                )
                                for a in self.wrapped_env._env.agents
                            }
                        else:
                            q_vals = q_softmax_from_dis(q_vals)
                            target_q_vals = q_softmax_from_dis(target_q_vals)
                            chosen_action_qvals = jax.tree_util.tree_map(
                                lambda q, u: q_of_action(
                                    q,
                                    (u + jnp.broadcast_to(self.act_type_idx_offset, u.shape))[..., None],
                                    -2,
                                )[:-1],  # avoid last timestep
                                q_vals,
                                learn_traj.actions,
                            )
                            # get the target for each agent (assumes every agent has a reward)
                            valid_q_vals = jax.tree_util.tree_map(
                                lambda q, valid_idx: (q * self.dis_support).sum(-1)[..., valid_idx],
                                q_vals,
                                self.wrapped_env.valid_actions,
                            )  # get expectation of q-value
                            target_max_qvals_dis = jax.tree_util.tree_map(
                                lambda t_q, q: q_of_action(
                                    t_q,
                                    jnp.stack(
                                        [jnp.argmax(q[..., x], axis=-1) + self.act_type_idx_offset[i] for i, x in enumerate(self.wrapped_env._env.act_type_idx)],
                                        axis=-1,
                                    )[..., None],
                                    -2,
                                )[1:],  # avoid first timestep
                                target_q_vals,
                                jax.lax.stop_gradient(valid_q_vals),
                            )
                            target_max_qvals = jax.tree_util.tree_map(
                                lambda x: jnp.tile(self.dis_support, x.shape[:-1] + (1,)),
                                target_max_qvals_dis,
                            )
                            # get reward vector along action types
                            rewards_vec = {
                                a: jnp.stack(
                                    [learn_traj.infos[f"reward_{x}"][..., i] for x in range(len(self.wrapped_env._env.act_type_idx))],
                                    axis=-1,
                                )[..., None]
                                for i, a in enumerate(self.wrapped_env._env.agents)
                            }
                            dones = {
                                a: jnp.tile(
                                    learn_traj.dones[a][..., None],
                                    (target_max_qvals[a].shape[-2],),
                                )[..., None]
                                for a in self.wrapped_env._env.agents
                            }
                        # compute a single loss for all the agents in one pass (parameter sharing)
                        targets = jax.tree_util.tree_map(
                            self.target_fn,
                            target_max_qvals,
                            rewards_vec,  # {agent:learn_traj.rewards[agent] for agent in self.wrapped_env._env.agents}, # rewards and agents could contain additional keys
                            dones,
                        )
                        chosen_action_qvals = jnp.concatenate(list(chosen_action_qvals.values()))
                        targets = jnp.concatenate(list(targets.values()))
                        if not self.config.get("DISTRIBUTION_Q", False):
                            importance_weights = importance_weights[(None, ...) + (None,) * (targets.ndim - importance_weights.ndim - 1)]
                            mean_axes = tuple(range(targets.ndim - 1))
                            err = chosen_action_qvals - jax.lax.stop_gradient(targets)
                            if self.config.get("TD_LAMBDA_LOSS", True):
                                loss = jnp.mean(0.5 * (err**2) * importance_weights, axis=mean_axes)
                            else:
                                loss = jnp.mean((err**2) * importance_weights, axis=mean_axes)
                            err = jnp.clip(jnp.abs(err), 1e-7, None)
                            err_axes = tuple(range(err.ndim))
                            return loss.mean(), err.mean(axis=err_axes[0:1] + err_axes[2:])  # maintain 1 abs error for each batch
                        else:

                            def dis_shift(dis, base_support, new_support, cell_radius):  # assuming dis and support span the last dimension
                                coef = jnp.clip(
                                    cell_radius - jnp.abs(base_support[..., None] - new_support),
                                    0.0,
                                    None,
                                )  # projection coefficients
                                return (dis[..., None] * coef).sum(-2) / cell_radius  # project back onto initial support

                            target_max_qvals_dis = jnp.concatenate(list(target_max_qvals_dis.values()))
                            targets = dis_shift(
                                target_max_qvals_dis,
                                jnp.clip(targets, self.config["DISTRIBUTION_RANGE"][0], self.config["DISTRIBUTION_RANGE"][1]),
                                self.dis_support,
                                self.dis_support_step,
                            )
                            mean_axes = tuple(range(targets.ndim - 2))
                            loss = -((jax.lax.stop_gradient(targets)) * jnp.log(chosen_action_qvals)).sum(-1)  # cross-entropy
                            importance_weights = importance_weights[(None, ...) + (None,) * (loss.ndim - importance_weights.ndim - 1)]
                            err = jnp.clip(loss, 1e-7, None)
                            err_axes = tuple(range(err.ndim))
                            return (loss * importance_weights).mean(axis=mean_axes).mean(), err.mean(axis=err_axes[0:1] + err_axes[2:])/len(self.wrapped_env._env.agents)  # maintain 1 abs error for each batch
                else:
                    # without parameters sharing, a different loss must be computed for each agent via vmap
                    def _loss_fn(
                        params,
                        target_q_vals,
                        init_hs,
                        obs,
                        dones,
                        actions,
                        valid_actions,
                        rewards,
                        type_idx,
                    ):
                        _, q_vals = self.agent.apply(params, init_hs, (obs, dones))
                        chosen_action_qvals = q_of_action(q_vals, actions)[:-1]
                        valid_actions = valid_actions.reshape(*[1] * len(q_vals.shape[:-1]), -1)  # reshape to match q_vals shape
                        valid_argmax = jnp.argmax(
                            jnp.where(
                                valid_actions.astype(bool),
                                jax.lax.stop_gradient(q_vals),
                                -1000000.0,
                            ),
                            axis=-1,
                        )
                        target_max_qvals = q_of_action(target_q_vals, valid_argmax)[1:]  # target q_vals of greedy actions, avoiding first time step
                        targets = self.target_fn(target_max_qvals, rewards, dones)
                        return jnp.mean((chosen_action_qvals - jax.lax.stop_gradient(targets)) ** 2).mean(), jnp.abs(chosen_action_qvals - targets)[..., type_idx].sum(-1)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                def _learn_phase(carry, _):
                    train_state, buffer_state, rng = carry
                    # sample a batched trajectory from the buffer and set the time step dim in first axis
                    rng, _rng = jax.random.split(rng)
                    learn_traj_batch = self.buffer.sample(buffer_state, _rng)  # (batch_size, 1, max_time_steps, ...)
                    learn_traj = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(x[:, 0], 0, 1),  # remove the dummy sequence dim (1) and swap batch and temporal dims
                        learn_traj_batch.experience,
                    )  # (max_time_steps, batch_size, ...)
                    importance_weights = (
                        learn_traj_batch.priorities ** (-self.importance_weights_sch(time_state["updates"]))
                        if self.config.get("PRIORITIZED_EXPERIENCE_REPLAY", False)
                        else jnp.ones((self.config['BUFFER_BATCH_SIZE'],))
                    )
                    importance_weights /= importance_weights.max()

                    # for iql the loss must be computed differently with or without parameters sharing
                    if self.config.get("PARAMETERS_SHARING", True):
                        init_hs = ScannedRNN.initialize_carry(
                            self.config["AGENT_HIDDEN_DIM"],
                            len(self.wrapped_env._env.agents) * self.config["BUFFER_BATCH_SIZE"],
                            (self.wrapped_env._env.transform_no + 1),
                        )  # (n_agents*batch_size, hs_size, augment_size)
                        _, target_q_vals = self.foward_pass(target_agent_params, init_hs, learn_traj.obs, learn_traj.dones)
                        # compute loss and optimize grad
                        grad_results = grad_fn(
                            train_state.params,
                            target_q_vals,
                            init_hs,
                            learn_traj,
                            importance_weights,
                        )
                        loss, grads = grad_results
                        loss, priorities = loss
                        # grads = jax.tree_util.tree_map(lambda *x: jnp.array(x).sum(0), *grads)
                    else:
                        obs_, dones_ = self.batchify(learn_traj.obs), self.batchify(learn_traj.dones)
                        init_hs = ScannedRNN.initialize_carry(
                            self.config["AGENT_HIDDEN_DIM"],
                            len(self.wrapped_env._env.agents),
                            self.config["BUFFER_BATCH_SIZE"],
                        )  # (n_agents, batch_size, hs_size)
                        _, target_q_vals = self.agent.apply(target_agent_params, init_hs, (obs_, dones_))

                        loss, grads = jax.vmap(grad_fn, in_axes=0)(
                            train_state.params,
                            target_q_vals,
                            init_hs,
                            obs_,
                            dones_,
                            self.batchify(learn_traj.actions),
                            self.batchify(self.wrapped_env.valid_actions_oh),
                            self.batchify(learn_traj.rewards),
                        )
                        loss, priorities = loss
                        loss, priorities = loss.mean(), priorities.mean(0)

                    # apply gradients
                    # rescale_factor = 1/np.sqrt(len(self.wrapped_env._env.act_type_idx)+1)
                    # for x in self.agent.common_layers:
                    #     if self.agent.layer_name[x] in grads['params'].keys():
                    #         grads['params'][self.agent.layer_name[x]]=jax.tree_util.tree_map(lambda z:z*rescale_factor,grads['params'][self.agent.layer_name[x]])
                    train_state = train_state.apply_gradients(grads=grads)
                    # rescale_factor = 1/np.sqrt(len(self.wrapped_env._env.act_type_idx)+1)
                    # for x in grads:
                    #     train_state = train_state.apply_gradients(grads=jax.tree_util.tree_map(lambda z:z*rescale_factor,x))
                    # update priorities of sampled batch
                    if self.config.get("PRIORITIZED_EXPERIENCE_REPLAY", False):
                        buffer_state = self.buffer.set_priorities(buffer_state, learn_traj_batch.indices, priorities)
                    return (train_state, buffer_state, rng), loss

                is_learn_time = self.buffer.can_sample(buffer_state)
                rng, _rng = jax.random.split(rng)
                (train_state, buffer_state, rng), loss = jax.lax.cond(
                    is_learn_time,
                    lambda train_state, buffer_state, rng: jax.lax.scan(
                        _learn_phase,
                        (train_state, buffer_state, rng),
                        None,
                        self.config["NUM_EPOCHS"],
                    ),
                    lambda train_state, buffer_state, rng: (
                        (train_state, buffer_state, rng),
                        jnp.zeros(self.config["NUM_EPOCHS"]),
                    ),
                    train_state,
                    buffer_state,
                    _rng,
                )
                # UPDATE THE VARIABLES AND RETURN
                # reset the environment
                rng, _rng = jax.random.split(rng)
                init_obs, env_state = self.wrapped_env.batch_reset(_rng)
                init_dones = {agent: jnp.zeros((self.config["NUM_ENVS"]), dtype=bool) for agent in self.wrapped_env._env.agents + ["__all__"]}

                # update the states
                time_state["timesteps"] = step_state[-1]
                time_state["updates"] = time_state["updates"] + 1
                time_state["gradients"] = time_state["gradients"] + is_learn_time * self.config["NUM_EPOCHS"]

                # update the target network if necessary
                target_agent_params = jax.lax.cond(
                    time_state["updates"] % self.config["TARGET_UPDATE_INTERVAL"] == 0,
                    lambda _: optax.incremental_update(train_state.params, target_agent_params, self.config["TAU"]),
                    lambda _: target_agent_params,
                    operand=None,
                )

                # update the greedy rewards
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    time_state["updates"] % (self.config["TEST_INTERVAL"] // self.config["NUM_STEPS"] // self.config["NUM_ENVS"]) == 0,
                    lambda _: self._get_greedy_metrics(_rng, train_state.params, time_state),
                    lambda _: test_metrics,
                    operand=None,
                )

                # update the returning metrics
                metrics = {
                    "timesteps": time_state["timesteps"] * self.config["NUM_ENVS"],
                    "updates": time_state["updates"],
                    "gradients": time_state["gradients"],
                    "loss": jax.lax.select(is_learn_time, loss.mean(), np.nan),
                    "rewards": jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0).mean(), traj_batch.rewards),
                }
                metrics["test_metrics"] = test_metrics  # add the test metrics dictionary

                if self.config.get("WANDB_ONLINE_REPORT", False):
                    jax.debug.callback(callback_wandb_report, metrics, traj_batch.infos)

                # reset param if necessary
                if self.config.get("PARAM_RESET", False):
                    rng, _rng = jax.random.split(rng)
                    if self.config.get("PARAMETERS_SHARING", True):
                        network_params = self.agent.init(_rng, init_hs, init_x)
                    else:
                        rngs = jax.random.split(_rng, len(self.wrapped_env._env.agents))  # a random init for each agent
                        network_params = jax.vmap(self.agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)
                    for sub_param in self.agent.reset_layers:
                        layer_name = self.agent.layer_name[sub_param]
                        param_to_copy = network_params["params"][layer_name]
                        train_state.params["params"][layer_name] = jax.tree_util.tree_map(
                            lambda x, y: jax.lax.select(
                                time_state["updates"] % self.config["PARAM_RESET_INTERVAL"] == 0,
                                jnp.copy(x),
                                y,
                            ),
                            param_to_copy,
                            train_state.params["params"][layer_name],
                        )
                        target_agent_params["params"][layer_name] = jax.tree_util.tree_map(
                            lambda x, y: jax.lax.select(
                                time_state["updates"] % self.config["PARAM_RESET_INTERVAL"] == 0,
                                jnp.copy(x),
                                y,
                            ),
                            param_to_copy,
                            target_agent_params["params"][layer_name],
                        )
                runner_state = (
                    train_state,
                    target_agent_params,
                    env_state,
                    buffer_state,
                    time_state,
                    init_obs,
                    init_dones,
                    test_metrics,
                    rng,
                )

                return runner_state, metrics

            time_state = {
                "timesteps": jnp.array(0),
                "updates": jnp.array(0),
                "gradients": jnp.array(0),
            }
            rng, _rng = jax.random.split(rng)
            test_metrics = self._get_greedy_metrics(_rng, train_state.params, time_state)  # initial greedy metrics

            # train
            rng, _rng = jax.random.split(rng)
            runner_state = (
                train_state,
                target_agent_params,
                env_state,
                buffer_state,
                time_state,
                init_obs,
                init_dones,
                test_metrics,
                _rng,
            )
            runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, self.num_updates)
            return {"runner_state": runner_state, "metrics": metrics}

        self.train_fn = train
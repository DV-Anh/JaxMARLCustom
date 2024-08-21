import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


import optax
from flax.training.train_state import TrainState
import flashbax as fbx
from jaxmarl.wrappers.baselines import CTRolloutManager

from custom.qlearning.common import ScannedRNN, AgentRNN, EpsilonGreedy, Transition, homogeneous_pass_ps, homogeneous_pass_nops, q_of_action, q_softmax_from_dis, q_expectation_from_dis, td_targets, callback_wandb_report

class IndependentQL:
    def make_train(config, env):
        config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        td_max_steps = min(config["NUM_STEPS"] - 1, config["TD_MAX_STEPS"])

        def train(rng):
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"], preprocess_obs=False)
            test_env = CTRolloutManager(env, batch_size=config["NUM_TEST_EPISODES"], preprocess_obs=False)  # batched env for testing (has different batch size)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents + ["__all__"]}

            # INIT BUFFER
            # to initalize the buffer is necessary to sample a trajectory to know its strucutre
            def _env_sample_step(env_state, unused):
                rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)  # use a dummy rng here
                key_a = jax.random.split(key_a, env.num_agents)
                actions = {agent: wrapped_env.batch_sample(key_a[i], agent) for i, agent in enumerate(env.agents)}
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(obs, actions, rewards, dones, infos)
                return env_state, transition

            _, sample_traj = jax.lax.scan(_env_sample_step, env_state, None, config["NUM_STEPS"])
            sample_traj_unbatched = jax.tree_util.tree_map(lambda x: x[:, 0][:, np.newaxis], sample_traj)  # remove the NUM_ENV dim, add dummy dim 1
            sample_traj_unbatched = sample_traj_unbatched.augment_reward_invariant(
                env.agents,
                env.agents,
                env.reward_invariant_transform_obs,
                env.reward_invariant_transform_acts,
                env.transform_no + 1,
                1,  # append augment data at dim 1
            )
            if config.get("PRIORITIZED_EXPERIENCE_REPLAY", False):
                buffer = fbx.make_prioritised_trajectory_buffer(
                    max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
                    min_length_time_axis=config["BUFFER_BATCH_SIZE"],
                    sample_batch_size=config["BUFFER_BATCH_SIZE"],
                    add_batch_size=config["NUM_ENVS"],
                    sample_sequence_length=1,
                    period=1,
                    priority_exponent=config["PRIORITY_EXPONENT"],
                )
            else:
                buffer = fbx.make_trajectory_buffer(
                    max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
                    min_length_time_axis=config["BUFFER_BATCH_SIZE"],
                    sample_batch_size=config["BUFFER_BATCH_SIZE"],
                    add_batch_size=config["NUM_ENVS"],
                    sample_sequence_length=1,
                    period=1,
                )
            buffer_state = buffer.init(sample_traj_unbatched)

            # INIT NETWORK
            act_type_idx_offset = jnp.array(env.act_type_idx_offset)
            if config.get("DISTRIBUTION_Q", False):
                agent = AgentRNN(
                    action_dim=wrapped_env.max_action_space,
                    atom_dim=config["DISTRIBUTION_ATOMS"],
                    hidden_dim=config["AGENT_HIDDEN_DIM"],
                    init_scale=config["AGENT_INIT_SCALE"],
                    act_type_idx=env.act_type_idx,
                    use_rnn=config["RNN_LAYER"],
                )
                dis_support_step = (config["DISTRIBUTION_RANGE"][1] - config["DISTRIBUTION_RANGE"][0]) / (config["DISTRIBUTION_ATOMS"] - 1)
                dis_support = jnp.arange(
                    config["DISTRIBUTION_RANGE"][0],
                    config["DISTRIBUTION_RANGE"][1] + dis_support_step,
                    dis_support_step,
                )
            else:
                agent = AgentRNN(
                    action_dim=wrapped_env.max_action_space,
                    atom_dim=1,
                    hidden_dim=config["AGENT_HIDDEN_DIM"],
                    init_scale=config["AGENT_INIT_SCALE"],
                    act_type_idx=env.act_type_idx,
                    use_rnn=config["RNN_LAYER"],
                )
            rng, _rng = jax.random.split(rng)
            if config.get("PARAMETERS_SHARING", True):
                init_x = (
                    jnp.zeros((1, 1, wrapped_env.obs_size)),  # (time_step, batch_size, obs_size)
                    jnp.zeros((1, 1)),  # (time_step, batch size)
                )
                init_hs = ScannedRNN.initialize_carry(config["AGENT_HIDDEN_DIM"], 1)  # (batch_size, hidden_dim)
                network_params = agent.init(_rng, init_hs, init_x)
            else:
                init_x = (
                    jnp.zeros((len(env.agents), 1, 1, wrapped_env.obs_size)),  # (time_step, batch_size, obs_size)
                    jnp.zeros((len(env.agents), 1, 1)),  # (time_step, batch size)
                )
                init_hs = ScannedRNN.initialize_carry(config["AGENT_HIDDEN_DIM"], len(env.agents), 1)  # (n_agents, batch_size, hidden_dim)
                rngs = jax.random.split(_rng, len(env.agents))  # a random init for each agent
                network_params = jax.vmap(agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)

            # INIT TRAIN STATE AND OPTIMIZER
            def linear_schedule(count):
                frac = 1.0 - (count / config["NUM_UPDATES"])
                return config["LR"] * frac

            lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr, eps=config["EPS_ADAM"]),
            )
            train_state = TrainState.create(
                apply_fn=agent.apply,
                params=network_params,
                tx=tx,
            )
            # INIT REWEIGHTING SCHEDULE
            importance_weights_sch = optax.linear_schedule(
                config["PRIORITY_IMPORTANCE_EXPONENT_START"],
                config["PRIORITY_IMPORTANCE_EXPONENT_END"],
                config["NUM_UPDATES"],
                config["BUFFER_BATCH_SIZE"],
            )
            # target network params
            target_agent_params = jax.tree_util.tree_map(lambda x: jnp.copy(x), train_state.params)

            # INIT EXPLORATION STRATEGY
            explorer = EpsilonGreedy(
                start_e=config["EPSILON_START"],
                end_e=config["EPSILON_FINISH"],
                duration=config["EPSILON_ANNEAL_TIME"],
                act_type_idx=env.act_type_idx,
            )

            # depending if using parameters sharing or not, q-values are computed using one or multiple parameters
            if config.get("PARAMETERS_SHARING", True):
                homogeneous_pass = partial(homogeneous_pass_ps, agent)
            else:
                homogeneous_pass = partial(homogeneous_pass_nops, agent)

            # preparing target function for loss calc
            target_fn = partial(td_targets, _lambda=config["TD_LAMBDA"], td_max_steps=td_max_steps, _gamma=config["GAMMA"], is_multistep=config.get("TD_LAMBDA_LOSS", True))

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
                def _env_step(step_state, unused):
                    params, env_state, last_obs, last_dones, hstate, rng, t = step_state

                    # prepare rngs for actions and step
                    rng, key_a, key_s = jax.random.split(rng, 3)

                    # SELECT ACTION
                    # add a dummy time_step dimension to the agent input
                    obs_ = {a: last_obs[a] for a in env.agents}  # ensure to not pass the global state (obs["__all__"]) to the network
                    obs_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], obs_)
                    dones_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], last_dones)
                    # get the q_values from the agent netwoek
                    hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
                    if config.get("DISTRIBUTION_Q", False):
                        q_vals = q_expectation_from_dis(q_vals, dis_support)
                    # remove the dummy time_step dimension and index qs by the valid actions of each agent
                    valid_q_vals = jax.tree_util.tree_map(
                        lambda q, valid_idx: q.squeeze(0)[..., valid_idx],
                        q_vals,
                        wrapped_env.valid_actions,
                    )
                    # explore with epsilon greedy_exploration
                    actions = explorer.choose_actions(valid_q_vals, t, key_a)

                    # STEP ENV
                    obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                    transition = Transition(last_obs, actions, rewards, dones, infos)

                    step_state = (params, env_state, obs, dones, hstate, rng, t + 1)
                    return step_state, transition

                # prepare the step state and collect the episode trajectory
                rng, _rng = jax.random.split(rng)
                if config.get("PARAMETERS_SHARING", True):
                    hstate = ScannedRNN.initialize_carry(config["AGENT_HIDDEN_DIM"], len(env.agents) * config["NUM_ENVS"])  # (n_agents*n_envs, hs_size)
                else:
                    hstate = ScannedRNN.initialize_carry(config["AGENT_HIDDEN_DIM"], len(env.agents), config["NUM_ENVS"])  # (n_agents, n_envs, hs_size)

                step_state = (
                    train_state.params,
                    env_state,
                    init_obs,
                    init_dones,
                    hstate,
                    _rng,
                    time_state["timesteps"],  # t is needed to compute epsilon
                )

                step_state, traj_batch = jax.lax.scan(_env_step, step_state, None, config["NUM_STEPS"])

                if config.get("PARAMETERS_SHARING", True):

                    def _loss_fn(params, target_agent_params, init_hs, learn_traj, importance_weights):
                        # obs_={a:learn_traj.obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                        _, q_vals = homogeneous_pass(params, init_hs, learn_traj.obs, learn_traj.dones)
                        _, target_q_vals = homogeneous_pass(target_agent_params, init_hs, learn_traj.obs, learn_traj.dones)
                        if not config.get("DISTRIBUTION_Q", False):
                            # get the q_vals of the taken actions (with exploration) for each agent
                            chosen_action_qvals = jax.tree_util.tree_map(
                                lambda q, u: q_of_action(q, u + jnp.broadcast_to(act_type_idx_offset, u.shape))[:-1],  # avoid last timestep
                                q_vals,
                                learn_traj.actions,
                            )
                            # get the target for each agent (assumes every agent has a reward)
                            valid_q_vals = jax.tree_util.tree_map(
                                lambda q, valid_idx: q[..., valid_idx],
                                q_vals,
                                wrapped_env.valid_actions,
                            )
                            target_max_qvals = jax.tree_util.tree_map(
                                lambda t_q, q: q_of_action(
                                    t_q,
                                    jnp.stack(
                                        [jnp.argmax(q[..., x], axis=-1) + act_type_idx_offset[i] for i, x in enumerate(env.act_type_idx)],
                                        axis=-1,
                                    ),
                                )[1:],  # avoid first timestep
                                target_q_vals,
                                jax.lax.stop_gradient(valid_q_vals),
                            )
                            # get reward vector along action types
                            rewards_vec = {
                                a: jnp.stack(
                                    [learn_traj.infos[f"reward_{x}"][..., i] for x in range(len(env.act_type_idx))],
                                    axis=-1,
                                )
                                for i, a in enumerate(env.agents)
                            }
                            dones = {
                                a: jnp.tile(
                                    learn_traj.dones[a][..., None],
                                    (target_max_qvals[a].shape[-1],),
                                )
                                for a in env.agents
                            }
                        else:
                            q_vals = q_softmax_from_dis(q_vals)
                            target_q_vals = q_softmax_from_dis(target_q_vals)
                            chosen_action_qvals = jax.tree_util.tree_map(
                                lambda q, u: q_of_action(
                                    q,
                                    (u + jnp.broadcast_to(act_type_idx_offset, u.shape))[..., None],
                                    -2,
                                )[:-1],  # avoid last timestep
                                q_vals,
                                learn_traj.actions,
                            )
                            # get the target for each agent (assumes every agent has a reward)
                            valid_q_vals = jax.tree_util.tree_map(
                                lambda q, valid_idx: (q * dis_support).sum(-1)[..., valid_idx],
                                q_vals,
                                wrapped_env.valid_actions,
                            )  # get expectation of q-value
                            target_max_qvals_dis = jax.tree_util.tree_map(
                                lambda t_q, q: q_of_action(
                                    t_q,
                                    jnp.stack(
                                        [jnp.argmax(q[..., x], axis=-1) + act_type_idx_offset[i] for i, x in enumerate(env.act_type_idx)],
                                        axis=-1,
                                    )[..., None],
                                    -2,
                                )[1:],  # avoid first timestep
                                target_q_vals,
                                jax.lax.stop_gradient(valid_q_vals),
                            )
                            target_max_qvals = jax.tree_util.tree_map(
                                lambda x: jnp.tile(dis_support, x.shape[:-1] + (1,)),
                                target_max_qvals_dis,
                            )
                            # get reward vector along action types
                            rewards_vec = {
                                a: jnp.stack(
                                    [learn_traj.infos[f"reward_{x}"][..., i] for x in range(len(env.act_type_idx))],
                                    axis=-1,
                                )[..., None]
                                for i, a in enumerate(env.agents)
                            }
                            dones = {
                                a: jnp.tile(
                                    learn_traj.dones[a][..., None],
                                    (target_max_qvals[a].shape[-2],),
                                )[..., None]
                                for a in env.agents
                            }
                        # compute a single loss for all the agents in one pass (parameter sharing)
                        targets = jax.tree_util.tree_map(
                            target_fn,
                            target_max_qvals,
                            rewards_vec,  # {agent:learn_traj.rewards[agent] for agent in env.agents}, # rewards and agents could contain additional keys
                            dones,
                        )
                        chosen_action_qvals = jnp.concatenate(list(chosen_action_qvals.values()))
                        targets = jnp.concatenate(list(targets.values()))
                        if not config.get("DISTRIBUTION_Q", False):
                            importance_weights = importance_weights[(None, ...) + (None,) * (targets.ndim - importance_weights.ndim - 1)]
                            mean_axes = tuple(range(targets.ndim - 1))
                            err = chosen_action_qvals - jax.lax.stop_gradient(targets)
                            if config.get("TD_LAMBDA_LOSS", True):
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
                                jnp.clip(targets, config["DISTRIBUTION_RANGE"][0], config["DISTRIBUTION_RANGE"][1]),
                                dis_support,
                                dis_support_step,
                            )
                            mean_axes = tuple(range(targets.ndim - 2))
                            loss = -((jax.lax.stop_gradient(targets)) * jnp.log(chosen_action_qvals)).sum(-1)  # cross-entropy
                            importance_weights = importance_weights[(None, ...) + (None,) * (loss.ndim - importance_weights.ndim - 1)]
                            err = jnp.clip(loss, 1e-7, None)
                            err_axes = tuple(range(err.ndim))
                            return (loss * importance_weights).mean(axis=mean_axes).mean(), err.mean(axis=err_axes[0:1] + err_axes[2:])  # maintain 1 abs error for each batch
                else:
                    # without parameters sharing, a different loss must be computed for each agent via vmap
                    def _loss_fn(
                        params,
                        target_params,
                        init_hs,
                        obs,
                        dones,
                        actions,
                        valid_actions,
                        rewards,
                        type_idx,
                    ):
                        _, q_vals = agent.apply(params, init_hs, (obs, dones))
                        _, target_q_vals = agent.apply(target_params, init_hs, (obs, dones))
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
                        targets = target_fn(target_max_qvals, rewards, dones)
                        return jnp.mean((chosen_action_qvals - jax.lax.stop_gradient(targets)) ** 2).mean(), jnp.abs(chosen_action_qvals - targets)[..., type_idx].sum(-1)

                # BUFFER UPDATE: save the collected trajectory in the buffer
                buffer_traj_batch = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis, :, np.newaxis],  # put the batch dim first, add a dummy sequence dim, add dummy dim for augment
                    traj_batch,
                )  # (num_envs, 1, time_steps, ...)
                buffer_traj_batch = buffer_traj_batch.augment_reward_invariant(
                    env.agents,
                    env.agents,
                    env.reward_invariant_transform_obs,
                    env.reward_invariant_transform_acts,
                    env.transform_no + 1,
                    3,  # append augment data at dim 3
                )
                buffer_state = buffer.add(buffer_state, buffer_traj_batch)
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                def _learn_phase(carry, _):
                    train_state, buffer_state, rng = carry
                    # sample a batched trajectory from the buffer and set the time step dim in first axis
                    rng, _rng = jax.random.split(rng)
                    learn_traj_batch = buffer.sample(buffer_state, _rng)  # (batch_size, 1, max_time_steps, ...)
                    learn_traj = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(x[:, 0], 0, 1),  # remove the dummy sequence dim (1) and swap batch and temporal dims
                        learn_traj_batch.experience,
                    )  # (max_time_steps, batch_size, ...)
                    importance_weights = (
                        learn_traj_batch.priorities ** (-importance_weights_sch(time_state["updates"]))
                        if config.get("PRIORITIZED_EXPERIENCE_REPLAY", False)
                        else jnp.ones_like(learn_traj_batch.indices)
                    )
                    importance_weights /= importance_weights.max()

                    # for iql the loss must be computed differently with or without parameters sharing
                    if config.get("PARAMETERS_SHARING", True):
                        init_hs = ScannedRNN.initialize_carry(
                            config["AGENT_HIDDEN_DIM"],
                            len(env.agents) * config["BUFFER_BATCH_SIZE"],
                            (env.transform_no + 1),
                        )  # (n_agents*batch_size, hs_size, augment_size)
                        # compute loss and optimize grad
                        grad_results = grad_fn(
                            train_state.params,
                            target_agent_params,
                            init_hs,
                            learn_traj,
                            importance_weights,
                        )
                        loss, grads = grad_results
                        loss, priorities = loss
                        # grads = jax.tree_util.tree_map(lambda *x: jnp.array(x).sum(0), *grads)
                    else:
                        init_hs = ScannedRNN.initialize_carry(
                            config["AGENT_HIDDEN_DIM"],
                            len(env.agents),
                            config["BUFFER_BATCH_SIZE"],
                        )  # (n_agents, batch_size, hs_size)

                        def batchify(x):
                            return jnp.stack([x[agent] for agent in env.agents], axis=0)

                        loss, grads = jax.vmap(grad_fn, in_axes=0)(
                            train_state.params,
                            target_agent_params,
                            init_hs,
                            batchify(learn_traj.obs),
                            batchify(learn_traj.dones),
                            batchify(learn_traj.actions),
                            batchify(wrapped_env.valid_actions_oh),
                            batchify(learn_traj.rewards),
                        )
                        loss, priorities = loss
                        loss, priorities = loss.mean(), priorities.mean(0)

                    # apply gradients
                    # rescale_factor = 1/np.sqrt(len(env.act_type_idx)+1)
                    # for x in agent.common_layers:
                    #     if agent.layer_name[x] in grads['params'].keys():
                    #         grads['params'][agent.layer_name[x]]=jax.tree_util.tree_map(lambda z:z*rescale_factor,grads['params'][agent.layer_name[x]])
                    train_state = train_state.apply_gradients(grads=grads)
                    # rescale_factor = 1/np.sqrt(len(env.act_type_idx)+1)
                    # for x in grads:
                    #     train_state = train_state.apply_gradients(grads=jax.tree_util.tree_map(lambda z:z*rescale_factor,x))
                    # update priorities of sampled batch
                    if config.get("PRIORITIZED_EXPERIENCE_REPLAY", False):
                        buffer_state = buffer.set_priorities(buffer_state, learn_traj_batch.indices, priorities)
                    return (train_state, buffer_state, rng), loss

                is_learn_time = buffer.can_sample(buffer_state)
                rng, _rng = jax.random.split(rng)
                (train_state, buffer_state, rng), loss = jax.lax.cond(
                    is_learn_time,
                    lambda train_state, buffer_state, rng: jax.lax.scan(
                        _learn_phase,
                        (train_state, buffer_state, rng),
                        None,
                        config["NUM_EPOCHS"],
                    ),
                    lambda train_state, buffer_state, rng: (
                        (train_state, buffer_state, rng),
                        jnp.zeros(config["NUM_EPOCHS"]),
                    ),
                    train_state,
                    buffer_state,
                    _rng,
                )
                # UPDATE THE VARIABLES AND RETURN
                # reset the environment
                rng, _rng = jax.random.split(rng)
                init_obs, env_state = wrapped_env.batch_reset(_rng)
                init_dones = {agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents + ["__all__"]}

                # update the states
                time_state["timesteps"] = step_state[-1]
                time_state["updates"] = time_state["updates"] + 1
                time_state["gradients"] = time_state["gradients"] + is_learn_time * config["NUM_EPOCHS"]

                # update the target network if necessary
                target_agent_params = jax.lax.cond(
                    time_state["updates"] % config["TARGET_UPDATE_INTERVAL"] == 0,
                    lambda _: optax.incremental_update(train_state.params, target_agent_params, config["TAU"]),
                    lambda _: target_agent_params,
                    operand=None,
                )

                # update the greedy rewards
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    time_state["updates"] % (config["TEST_INTERVAL"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0,
                    lambda _: get_greedy_metrics(_rng, train_state.params, time_state),
                    lambda _: test_metrics,
                    operand=None,
                )

                # update the returning metrics
                metrics = {
                    "timesteps": time_state["timesteps"] * config["NUM_ENVS"],
                    "updates": time_state["updates"],
                    "gradients": time_state["gradients"],
                    "loss": jax.lax.select(is_learn_time, loss.mean(), np.nan),
                    "rewards": jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0).mean(), traj_batch.rewards),
                }
                metrics["test_metrics"] = test_metrics  # add the test metrics dictionary

                if config.get("WANDB_ONLINE_REPORT", False):
                    jax.debug.callback(callback_wandb_report, metrics, traj_batch.infos)

                # reset param if necessary
                if config.get("PARAM_RESET", False):
                    rng, _rng = jax.random.split(rng)
                    if config.get("PARAMETERS_SHARING", True):
                        network_params = agent.init(_rng, init_hs, init_x)
                    else:
                        rngs = jax.random.split(_rng, len(env.agents))  # a random init for each agent
                        network_params = jax.vmap(agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)
                    for sub_param in agent.reset_layers:
                        layer_name = agent.layer_name[sub_param]
                        param_to_copy = network_params["params"][layer_name]
                        train_state.params["params"][layer_name] = jax.tree_util.tree_map(
                            lambda x, y: jax.lax.select(
                                time_state["updates"] % config["PARAM_RESET_INTERVAL"] == 0,
                                jnp.copy(x),
                                y,
                            ),
                            param_to_copy,
                            train_state.params["params"][layer_name],
                        )
                        target_agent_params["params"][layer_name] = jax.tree_util.tree_map(
                            lambda x, y: jax.lax.select(
                                time_state["updates"] % config["PARAM_RESET_INTERVAL"] == 0,
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

            def get_greedy_metrics(rng, params, time_state):
                """Help function to test greedy policy during training"""

                def _greedy_env_step(step_state, unused):
                    params, env_state, last_obs, last_dones, hstate, rng = step_state
                    rng, key_s = jax.random.split(rng)
                    obs_ = {a: last_obs[a] for a in env.agents}
                    obs_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], obs_)
                    dones_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], last_dones)
                    hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
                    if config.get("DISTRIBUTION_Q", False):
                        q_vals = q_expectation_from_dis(q_vals, dis_support)
                    valid_q_vals = jax.tree_util.tree_map(
                        lambda q, valid_idx: q.squeeze(0)[..., valid_idx],
                        q_vals,
                        test_env.valid_actions,
                    )
                    actions = jax.tree_util.tree_map(
                        lambda q: jnp.stack(
                            [jnp.argmax(q[..., x], axis=-1) for x in env.act_type_idx],
                            axis=-1,
                        ),
                        valid_q_vals,
                    )  # one argmax per action type
                    obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
                    step_state = (params, env_state, obs, dones, hstate, rng)
                    return step_state, (rewards, dones, infos)

                rng, _rng = jax.random.split(rng)
                init_obs, env_state = test_env.batch_reset(_rng)
                init_dones = {agent: jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool) for agent in env.agents + ["__all__"]}
                rng, _rng = jax.random.split(rng)
                if config["PARAMETERS_SHARING"]:
                    hstate = ScannedRNN.initialize_carry(
                        config["AGENT_HIDDEN_DIM"],
                        len(env.agents) * config["NUM_TEST_EPISODES"],
                    )  # (n_agents*n_envs, hs_size)
                else:
                    hstate = ScannedRNN.initialize_carry(
                        config["AGENT_HIDDEN_DIM"],
                        len(env.agents),
                        config["NUM_TEST_EPISODES"],
                    )  # (n_agents, n_envs, hs_size)
                step_state = (
                    params,
                    env_state,
                    init_obs,
                    init_dones,
                    hstate,
                    _rng,
                )
                step_state, (rewards, dones, infos) = jax.lax.scan(_greedy_env_step, step_state, None, config["NUM_STEPS"])

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
                if config.get("VERBOSE", False):

                    def callback(timestep, updates, gradients, val):
                        print(f"Timestep: {timestep}, updates: {updates}, gradients: {gradients}, return: {val}")

                    jax.debug.callback(
                        callback,
                        time_state["timesteps"] * config["NUM_ENVS"],
                        time_state["updates"],
                        time_state["gradients"],
                        first_returns["__all__"].mean() / len(env.agents),
                    )
                return metrics

            time_state = {
                "timesteps": jnp.array(0),
                "updates": jnp.array(0),
                "gradients": jnp.array(0),
            }
            rng, _rng = jax.random.split(rng)
            test_metrics = get_greedy_metrics(_rng, train_state.params, time_state)  # initial greedy metrics

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
            runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
            return {"runner_state": runner_state, "metrics": metrics}

        return train

class VDN:
    def make_train(config, env):
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
        td_max_steps = min(config["NUM_STEPS"] - 1, config["TD_MAX_STEPS"])
        
        def train(rng):
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"],preprocess_obs=False)
            test_env = CTRolloutManager(env, batch_size=config["NUM_TEST_EPISODES"],preprocess_obs=False) # batched env for testing (has different batch size)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

            # INIT BUFFER
            # to initalize the buffer is necessary to sample a trajectory to know its strucutre
            def _env_sample_step(env_state, unused):
                rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3) # use a dummy rng here
                key_a = jax.random.split(key_a, env.num_agents)
                actions = {agent: wrapped_env.batch_sample(key_a[i], agent) for i, agent in enumerate(env.agents)}
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(obs, actions, rewards, dones, infos)
                return env_state, transition
            _, sample_traj = jax.lax.scan(
                _env_sample_step, env_state, None, config["NUM_STEPS"]
            )
            sample_traj_unbatched = jax.tree_util.tree_map(lambda x: x[:, 0][:, np.newaxis], sample_traj) # remove the NUM_ENV dim, add dummy dim 1
            sample_traj_unbatched = sample_traj_unbatched.augment_reward_invariant(
                env.agents, None,
                env.reward_invariant_transform_obs,
                env.reward_invariant_transform_acts,
                env.transform_no+1,
                1 # append augment data at dim 1
            )
            if config.get('PRIORITIZED_EXPERIENCE_REPLAY', False):
                buffer = fbx.make_prioritised_trajectory_buffer(
                    max_length_time_axis=config['BUFFER_SIZE']//config['NUM_ENVS'],
                    min_length_time_axis=config['BUFFER_BATCH_SIZE'],
                    sample_batch_size=config['BUFFER_BATCH_SIZE'],
                    add_batch_size=config['NUM_ENVS'],
                    sample_sequence_length=1,
                    period=1,
                    priority_exponent=config['PRIORITY_EXPONENT']
                )
            else:
                buffer = fbx.make_trajectory_buffer(
                    max_length_time_axis=config['BUFFER_SIZE']//config['NUM_ENVS'],
                    min_length_time_axis=config['BUFFER_BATCH_SIZE'],
                    sample_batch_size=config['BUFFER_BATCH_SIZE'],
                    add_batch_size=config['NUM_ENVS'],
                    sample_sequence_length=1,
                    period=1,
                )
            buffer_state = buffer.init(sample_traj_unbatched) 

            # INIT NETWORK
            act_type_idx_offset=jnp.array(env.act_type_idx_offset)
            if config.get('DISTRIBUTION_Q', False):
                agent = AgentRNN(action_dim=wrapped_env.max_action_space,atom_dim=config['DISTRIBUTION_ATOMS'],hidden_dim=config["AGENT_HIDDEN_DIM"],init_scale=config['AGENT_INIT_SCALE'],act_type_idx=env.act_type_idx,use_rnn=config['RNN_LAYER'])
                dis_support_step=(config['DISTRIBUTION_RANGE'][1]-config['DISTRIBUTION_RANGE'][0])/(config['DISTRIBUTION_ATOMS']-1)
                dis_support=jnp.arange(config['DISTRIBUTION_RANGE'][0],config['DISTRIBUTION_RANGE'][1]+dis_support_step,dis_support_step)
            else:
                agent = AgentRNN(action_dim=wrapped_env.max_action_space,atom_dim=1,hidden_dim=config["AGENT_HIDDEN_DIM"],init_scale=config['AGENT_INIT_SCALE'],act_type_idx=env.act_type_idx,use_rnn=config['RNN_LAYER'])
            rng, _rng = jax.random.split(rng)
            if config.get('PARAMETERS_SHARING', True):
                init_x = (
                    jnp.zeros((1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
                    jnp.zeros((1, 1)) # (time_step, batch size)
                )
                init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], 1) # (batch_size, hidden_dim)
                network_params = agent.init(_rng, init_hs, init_x)
            else:
                init_x = (
                    jnp.zeros((len(env.agents), 1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
                    jnp.zeros((len(env.agents), 1, 1)) # (time_step, batch size)
                )
                init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents),  1) # (n_agents, batch_size, hidden_dim)
                rngs = jax.random.split(_rng, len(env.agents)) # a random init for each agent
                network_params = jax.vmap(agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)

            # INIT TRAIN STATE AND OPTIMIZER
            def linear_schedule(count):
                frac = 1.0 - (count / config["NUM_UPDATES"])
                return config["LR"] * frac
            lr = linear_schedule if config.get('LR_LINEAR_DECAY', False) else config['LR']
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr, eps=config['EPS_ADAM']),
            )
            train_state = TrainState.create(
                apply_fn=agent.apply,
                params=network_params,
                tx=tx,
            )
            # INIT REWEIGHTING SCHEDULE
            importance_weights_sch = optax.linear_schedule(
                config["PRIORITY_IMPORTANCE_EXPONENT_START"],
                config["PRIORITY_IMPORTANCE_EXPONENT_END"],
                config["NUM_UPDATES"],
                config["BUFFER_BATCH_SIZE"],
            )
            # target network params
            target_agent_params = jax.tree_util.tree_map(lambda x: jnp.copy(x), train_state.params)

            # INIT EXPLORATION STRATEGY
            explorer = EpsilonGreedy(
                start_e=config["EPSILON_START"],
                end_e=config["EPSILON_FINISH"],
                duration=config["EPSILON_ANNEAL_TIME"],
                act_type_idx=env.act_type_idx
            )

            # depending if using parameters sharing or not, q-values are computed using one or multiple parameters
            if config.get("PARAMETERS_SHARING", True):
                homogeneous_pass = partial(homogeneous_pass_ps, agent)
            else:
                homogeneous_pass = partial(homogeneous_pass_nops, agent)
            
            # prepare td-target function
            target_fn = partial(td_targets, _lambda=config["TD_LAMBDA"], td_max_steps=td_max_steps, _gamma=config["GAMMA"], is_multistep=config.get("TD_LAMBDA_LOSS", True))

            # TRAINING LOOP
            def _update_step(runner_state, unused):

                train_state, target_agent_params, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state

                # EPISODE STEP
                def _env_step(step_state, unused):

                    params, env_state, last_obs, last_dones, hstate, rng, t = step_state

                    # prepare rngs for actions and step
                    rng, key_a, key_s = jax.random.split(rng, 3)

                    # SELECT ACTION
                    # add a dummy time_step dimension to the agent input
                    obs_   = {a:last_obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                    obs_   = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], obs_)
                    dones_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], last_dones)
                    # get the q_values from the agent netwoek
                    hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
                    if config.get('DISTRIBUTION_Q',False):
                        q_vals=q_expectation_from_dis(q_vals,dis_support)
                    # remove the dummy time_step dimension and index qs by the valid actions of each agent 
                    valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q.squeeze(0)[..., valid_idx], q_vals, wrapped_env.valid_actions)
                    # explore with epsilon greedy_exploration
                    actions = explorer.choose_actions(valid_q_vals, t, key_a)

                    # STEP ENV
                    obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                    transition = Transition(last_obs, actions, rewards, dones, infos)

                    step_state = (params, env_state, obs, dones, hstate, rng, t+1)
                    return step_state, transition


                # prepare the step state and collect the episode trajectory
                rng, _rng = jax.random.split(rng)
                if config.get('PARAMETERS_SHARING', True):
                    hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["NUM_ENVS"]) # (n_agents*n_envs, hs_size)
                else:
                    hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents), config["NUM_ENVS"]) # (n_agents, n_envs, hs_size)

                step_state = (
                    train_state.params,
                    env_state,
                    init_obs,
                    init_dones,
                    hstate, 
                    _rng,
                    time_state['timesteps'] # t is needed to compute epsilon
                )

                step_state, traj_batch = jax.lax.scan(
                    _env_step, step_state, None, config["NUM_STEPS"]
                )
                # LEARN PHASE
                def _loss_fn(params, target_agent_params, init_hs, learn_traj, importance_weights):
                    _, q_vals = homogeneous_pass(params, init_hs, learn_traj.obs, learn_traj.dones)
                    _, target_q_vals = homogeneous_pass(target_agent_params, init_hs, learn_traj.obs, learn_traj.dones)
                    if not config.get('DISTRIBUTION_Q',False):
                        # get the q_vals of the taken actions (with exploration) for each agent
                        chosen_action_qvals = jax.tree_util.tree_map(
                            lambda q, u: q_of_action(q, u+jnp.broadcast_to(jnp.array(act_type_idx_offset),u.shape))[:-1], # avoid last timestep
                            q_vals,
                            learn_traj.actions
                        )
                        # get the target q values of the greedy actions
                        valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q[..., valid_idx], q_vals, wrapped_env.valid_actions)
                        target_max_qvals = jax.tree_util.tree_map(
                            lambda t_q, q: q_of_action(t_q, jnp.stack([jnp.argmax(q[...,x], axis=-1)+act_type_idx_offset[i] for i,x in enumerate(env.act_type_idx)],axis=-1))[1:], # get the greedy actions and avoid first timestep
                            target_q_vals,
                            valid_q_vals
                        )
                        # VDN: computes q_tot as the sum of the agents' individual q values
                        chosen_action_qvals_sum = jnp.stack(list(chosen_action_qvals.values())).sum(axis=0)
                        target_max_qvals_sum = jnp.stack(list(target_max_qvals.values())).sum(axis=0)
                        # get centralized reward vector along action types
                        rewards_vec=jnp.stack([jnp.stack([learn_traj.infos[f'reward_{x}'][...,i] for x in range(len(env.act_type_idx))],axis=-1) for i,_ in enumerate(env.agents)],axis=0).sum(0)
                        # compute the centralized targets using the "__all__" rewards and dones
                        dones=jnp.tile(learn_traj.dones['__all__'][...,None],(target_max_qvals_sum.shape[-1],))
                        # rewards_=jnp.tile(learn_traj.rewards['__all__'][...,None],(target_max_qvals_sum.shape[-1],))
                        mean_axes=tuple(range(chosen_action_qvals_sum.ndim-1))
                    else:
                        q_vals=q_softmax_from_dis(q_vals)
                        target_q_vals=q_softmax_from_dis(target_q_vals)
                        chosen_action_qvals = jax.tree_util.tree_map(
                            lambda q, u: q_of_action(q, (u+jnp.broadcast_to(act_type_idx_offset,u.shape))[...,None],-2)[:-1], # avoid last timestep
                            q_vals,
                            learn_traj.actions
                        )
                        # get the target for each agent (assumes every agent has a reward)
                        valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: (q*dis_support).sum(-1)[..., valid_idx], q_vals, wrapped_env.valid_actions)# get expectation of q-value
                        target_max_qvals_dis = jax.tree_util.tree_map(
                            lambda t_q, q: q_of_action(t_q, jnp.stack([jnp.argmax(q[...,x], axis=-1)+act_type_idx_offset[i] for i,x in enumerate(env.act_type_idx)],axis=-1)[...,None],-2)[1:], # avoid first timestep
                            target_q_vals,
                            jax.lax.stop_gradient(valid_q_vals)
                        )
                        # TODO: calculate sum of distributions along axis 0
                        chosen_action_qvals_sum = jnp.stack(list(chosen_action_qvals.values()))
                        target_max_qvals=jax.tree_util.tree_map(lambda x:jnp.tile(dis_support,x.shape[:-1]+(1,)),target_max_qvals_dis)
                        # get centralized reward vector along action types
                        rewards_vec=jnp.stack([jnp.stack([learn_traj.infos[f'reward_{x}'][...,i] for x in range(len(env.act_type_idx))],axis=-1)[...,None] for i,_ in enumerate(env.agents)],axis=0).sum(0)
                        # compute the centralized targets using the "__all__" rewards and dones
                        dones=jnp.tile(learn_traj.dones['__all__'][...,None],(target_max_qvals_sum.shape[-1],))[...,None]
                        # rewards_=jnp.tile(learn_traj.rewards['__all__'][...,None],(target_max_qvals_sum.shape[-1],))
                        mean_axes=tuple(range(chosen_action_qvals_sum.ndim-2))
                    targets = jax.tree_util.tree_map(
                        target_fn,
                        target_max_qvals_sum,
                        rewards_vec,  # {agent:learn_traj.rewards[agent] for agent in env.agents}, # rewards and agents could contain additional keys
                        dones,
                    )
                    if not config.get('DISTRIBUTION_Q',False):
                        err = chosen_action_qvals_sum - jax.lax.stop_gradient(targets)
                        loss = (0.5 if config.get('TD_LAMBDA_LOSS', True) else 1)*(err**2)
                        err=jnp.abs(err)
                    else:
                        def dis_shift(dis,base_support,new_support,cell_radius):# assuming dis and support span the last dimension
                            coef=jnp.clip(cell_radius-jnp.abs(base_support[...,None]-new_support),0.0,None)# projection coefficients
                            return (dis[...,None]*coef).sum(-2)/cell_radius# project back onto initial support
                        targets=dis_shift(target_max_qvals_dis,jnp.clip(targets,config['DISTRIBUTION_RANGE'][0],config['DISTRIBUTION_RANGE'][1]),dis_support,dis_support_step)
                        loss=-((jax.lax.stop_gradient(targets))*jnp.log(chosen_action_qvals_sum)).sum(-1)# cross-entropy
                        err=loss
                    err_axes=tuple(range(err.ndim))
                    importance_weights = importance_weights[(None, ...) + (None,) * (loss.ndim - importance_weights.ndim - 1)]
                    return (loss*importance_weights).mean(axis=mean_axes).mean(), err.mean(axis=err_axes[0:1]+err_axes[2:])

                # BUFFER UPDATE: save the collected trajectory in the buffer
                buffer_traj_batch = jax.tree_util.tree_map(
                    lambda x:jnp.swapaxes(x, 0, 1)[:, np.newaxis,:, np.newaxis], # put the batch dim first, add a dummy sequence dim, add dummy dim for augment
                    traj_batch
                ) # (num_envs, 1, time_steps, ...)
                buffer_traj_batch = buffer_traj_batch.augment_reward_invariant(
                    env.agents, None,
                    env.reward_invariant_transform_obs,
                    env.reward_invariant_transform_acts,
                    env.transform_no+1,
                    3 # append augment data at dim 3
                )
                buffer_state = buffer.add(buffer_state, buffer_traj_batch)
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                def _learn_phase(carry,_):
                    train_state, buffer_state, rng = carry
                    # sample a batched trajectory from the buffer and set the time step dim in first axis
                    rng, _rng = jax.random.split(rng)
                    learn_traj_batch = buffer.sample(buffer_state, _rng) # (batch_size, 1, max_time_steps, ...)
                    learn_traj = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(x[:, 0], 0, 1), # remove the dummy sequence dim (1) and swap batch and temporal dims
                        learn_traj_batch.experience
                    ) # (max_time_steps, batch_size, ...)  # (max_time_steps, batch_size, ...)
                    importance_weights = (
                        learn_traj_batch.priorities
                        ** (-importance_weights_sch(time_state["updates"]))
                        if config.get("PRIORITIZED_EXPERIENCE_REPLAY", False)
                        else jnp.ones_like(learn_traj_batch.indices)
                    )
                    importance_weights /= importance_weights.max()
                    if config.get('PARAMETERS_SHARING', True):
                        init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["BUFFER_BATCH_SIZE"], (env.transform_no+1)) # (n_agents*batch_size, hs_size)
                    else:
                        init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents), config["BUFFER_BATCH_SIZE"]) # (n_agents, batch_size, hs_size)

                    # compute loss and optimize grad
                    grad_results = grad_fn(train_state.params, target_agent_params, init_hs, learn_traj, importance_weights)
                    loss, grads = grad_results
                    loss, priorities = loss
                    # rescale_factor = 1/np.sqrt(len(env.act_type_idx)+1)
                    train_state = train_state.apply_gradients(grads=grads)
                    # update priorities of sampled batch
                    if config.get('PRIORITIZED_EXPERIENCE_REPLAY', False):
                        buffer_state = buffer.set_priorities(buffer_state, learn_traj_batch.indices, priorities)
                    return (train_state,buffer_state,rng),loss
                is_learn_time = (buffer.can_sample(buffer_state))
                rng,_rng=jax.random.split(rng)
                (train_state,buffer_state,rng),loss=jax.lax.cond(
                    is_learn_time,
                    lambda train_state,buffer_state,rng:jax.lax.scan(_learn_phase,(train_state,buffer_state,rng),None,config['NUM_EPOCHS']),
                    lambda train_state,buffer_state,rng:((train_state,buffer_state,rng),jnp.zeros(config["NUM_EPOCHS"])),
                    train_state,buffer_state,_rng
                )
                # UPDATE THE VARIABLES AND RETURN
                # reset the environment
                rng, _rng = jax.random.split(rng)
                init_obs, env_state = wrapped_env.batch_reset(_rng)
                init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

                # update the states
                time_state['timesteps'] = step_state[-1]
                time_state['updates']   = time_state['updates'] + 1
                time_state['gradients'] = time_state['gradients'] + is_learn_time*config['NUM_EPOCHS']

                # update the target network if necessary
                target_agent_params = jax.lax.cond(
                    time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                    lambda _: optax.incremental_update(train_state.params,target_agent_params,config["TAU"]),
                    lambda _: target_agent_params,
                    operand=None
                )

                # update the greedy rewards
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    time_state['updates'] % (config["TEST_INTERVAL"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0,
                    lambda _: get_greedy_metrics(_rng, train_state.params, time_state),
                    lambda _: test_metrics,
                    operand=None
                )

                # update the returning metrics
                metrics = {
                    'timesteps': time_state['timesteps']*config['NUM_ENVS'],
                    'updates' : time_state['updates'],
                    'gradients' : time_state['gradients'],
                    'loss': jax.lax.select(is_learn_time,loss.mean(),np.nan),
                    'rewards': jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0).mean(), traj_batch.rewards),
                }
                metrics['test_metrics'] = test_metrics # add the test metrics dictionary

                if config.get('WANDB_ONLINE_REPORT', False):
                    jax.debug.callback(callback_wandb_report, metrics, traj_batch.infos)
                # reset param if necessary
                if config.get('PARAM_RESET', False):
                    rng, _rng = jax.random.split(rng)
                    if config.get('PARAMETERS_SHARING', True):
                        network_params = agent.init(_rng, init_hs, init_x)
                    else:
                        rngs = jax.random.split(_rng, len(env.agents)) # a random init for each agent
                        network_params = jax.vmap(agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)
                    for sub_param in agent.reset_layers:
                        layer_name=agent.layer_name[sub_param]
                        param_to_copy=network_params['params'][layer_name]
                        train_state.params['params'][layer_name]=jax.tree_util.tree_map(lambda x,y:jax.lax.select(time_state['updates']%config['PARAM_RESET_INTERVAL']==0,jnp.copy(x),y),param_to_copy,train_state.params['params'][layer_name])
                        target_agent_params['params'][layer_name]=jax.tree_util.tree_map(lambda x,y:jax.lax.select(time_state['updates']%config['PARAM_RESET_INTERVAL']==0,jnp.copy(x),y),param_to_copy,target_agent_params['params'][layer_name])
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

            def get_greedy_metrics(rng, params, time_state):
                """Help function to test greedy policy during training"""
                def _greedy_env_step(step_state, unused):
                    params, env_state, last_obs, last_dones, hstate, rng = step_state
                    rng, key_s = jax.random.split(rng)
                    obs_   = {a:last_obs[a] for a in env.agents}
                    obs_   = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], obs_)
                    dones_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], last_dones)
                    hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
                    if config.get('DISTRIBUTION_Q',False):
                        q_vals=q_expectation_from_dis(q_vals,dis_support)
                    valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q.squeeze(0)[..., valid_idx], q_vals, test_env.valid_actions)
                    actions = jax.tree_util.tree_map(lambda q: jnp.stack([jnp.argmax(q[...,x], axis=-1) for x in env.act_type_idx],axis=-1), valid_q_vals)
                    obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
                    step_state = (params, env_state, obs, dones, hstate, rng)
                    return step_state, (rewards, dones, infos)
                rng, _rng = jax.random.split(rng)
                init_obs, env_state = test_env.batch_reset(_rng)
                init_dones = {agent:jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool) for agent in env.agents+['__all__']}
                rng, _rng = jax.random.split(rng)
                if config["PARAMETERS_SHARING"]:
                    hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["NUM_TEST_EPISODES"]) # (n_agents*n_envs, hs_size)
                else:
                    hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents), config["NUM_TEST_EPISODES"]) # (n_agents, n_envs, hs_size)
                step_state = (
                    params,
                    env_state,
                    init_obs,
                    init_dones,
                    hstate, 
                    _rng,
                )
                step_state, (rewards, dones, infos) = jax.lax.scan(
                    _greedy_env_step, step_state, None, config["NUM_STEPS"]
                )
                # compute the metrics of the first episode that is done for each parallel env
                def first_episode_returns(rewards, dones):
                    first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
                    first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
                    return jnp.where(first_episode_mask, rewards, 0.).sum()/(first_done+1)
                all_dones = dones['__all__']
                first_returns = jax.tree_util.tree_map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones), rewards)
                first_infos   = jax.tree_util.tree_map(lambda i: jax.vmap(first_episode_returns, in_axes=1)(i[..., 0], all_dones), infos)
                metrics = {
                    **{'test_returns_'+k:v.mean() for k, v in first_returns.items()},#'test_returns': first_returns['__all__'],# episode returns
                    **{'test_'+k:v for k,v in first_infos.items()}
                }
                if config.get('VERBOSE', False):
                    def callback(timestep, updates, gradients, val):
                        print(f"Timestep: {timestep}, updates: {updates}, gradients: {gradients}, return: {val}")
                    jax.debug.callback(callback, time_state['timesteps']*config['NUM_ENVS'], time_state['updates'], time_state['gradients'], first_returns['__all__'].mean()/len(env.agents))
                return metrics

            time_state = {
                'timesteps':jnp.array(0),
                'updates':  jnp.array(0),
                'gradients':jnp.array(0)
            }
            rng, _rng = jax.random.split(rng)
            test_metrics = get_greedy_metrics(_rng, train_state.params, time_state) # initial greedy metrics
            
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
            runner_state, metrics = jax.lax.scan(
                _update_step, runner_state, None, config["NUM_UPDATES"]
            )
            return {'runner_state':runner_state, 'metrics':metrics}
        
        return train
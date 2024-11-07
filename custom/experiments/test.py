import os
from datetime import datetime
import jax
import jax.numpy as jnp
import numpy as np
import hydra
from omegaconf import OmegaConf
from functools import partial

from jaxmarl.wrappers.baselines import (
    load_params,
)

from custom.registry import make_env
from custom.qlearning.common import ScannedRNN, AgentRNN, q_expectation_from_dis
from custom.utils.blackboxOptimizer import PermutationSolver
from custom.utils.mpe_visualizer import MPEVisualizer
from custom.environments.customMPE import init_obj_to_array

import json
import matplotlib.pyplot as plt
from custom.experiments.output_results import output_results

plot_colors = ("red", "blue", "green", "key")


def single_run(config, alg_name, env, env_name):
    os.makedirs(config["SAVE_PATH"], exist_ok=True)
    p = []
    for i in range(config["NUM_TRAIN_SEEDS"]):
        p.append(
            load_params(f'{config["MODEL_PATH"]}/{env_name}_{alg_name}_{i}.safetensors')
        )
    # get most recent training hyperparam setting, needed to initialize model container
    config_model = f'{config["MODEL_PATH"]}/{env_name}_{alg_name}_config.json'
    f = open(config_model)
    assert (not f.closed), f'Cannot load model config file: {config_model}'
    alg_config = (json.load(f))["alg"]
    f.close()
    # prepare test
    key = jax.random.PRNGKey(config["SEED"])
    key, key_r, key_a = jax.random.split(key, 3)
    max_st = config["ENV_KWARGS"]["max_steps"]
    cc_enabled = config["ENV_KWARGS"].get("central_controller", False)
    init_obs, init_state = env.reset(key_r)
    max_action_space = env.action_space(env.agents[0]).n
    valid_actions = {
        a: jnp.arange(env.action_space(env.agents[0]).n)
        for a, u in env.action_spaces.items()
    }
    for i in range(config["NUM_TRAIN_SEEDS"]):
        if "agent" in p[i].keys():
            p[i] = p[i]["agent"]  # qmix also have mixer params
    if alg_config["PARAMETERS_SHARING"]:
        if alg_config.get("DISTRIBUTION_Q", False):
            agent = AgentRNN(
                action_dim=max_action_space,
                atom_dim=alg_config["DISTRIBUTION_ATOMS"],
                hidden_dim=alg_config["AGENT_HIDDEN_DIM"],
                init_scale=alg_config["AGENT_INIT_SCALE"],
                act_type_idx=env.act_type_idx,
                use_rnn=alg_config["RNN_LAYER"],
            )
            dis_support = jnp.arange(alg_config.get("DISTRIBUTION_ATOMS", 1))
        else:
            agent = AgentRNN(
                action_dim=max_action_space,
                atom_dim=1,
                hidden_dim=alg_config["AGENT_HIDDEN_DIM"],
                init_scale=alg_config["AGENT_INIT_SCALE"],
                act_type_idx=env.act_type_idx,
                use_rnn=alg_config["RNN_LAYER"],
            )
            dis_support = None

    # explorer = EpsilonGreedy(
    #     start_e=0.1,
    #     end_e=0.1,
    #     duration=alg_config["EPSILON_ANNEAL_TIME"],
    #     act_type_idx=env.act_type_idx,
    # )

    def obs_to_act(hstate, obs, dones, params, env, state, key_a, dis_support=None):
        # obs = jax.tree_util.tree_map(_preprocess_obs, obs, agents_one_hot)

        # add a dummy temporal dimension
        obs_ = jax.tree_util.tree_map(
            lambda x: x[np.newaxis, np.newaxis, :], obs
        )  # add also a dummy batch dim to obs
        dones_ = jax.tree_util.tree_map(lambda x: x[np.newaxis, :], dones)
        agents, flatten_agents_obs = zip(*obs_.items())
        original_shape = flatten_agents_obs[0].shape
        batched_input = (
            jnp.concatenate(
                flatten_agents_obs, axis=1
            ),  # (time_step, n_agents*n_envs, obs_size)
            jnp.concatenate(
                [dones_[a] for a in agents], axis=1
            ),  # ensure to not pass other keys (like __all__)
        )
        # pass in one with homogeneous pass
        hstate, q_vals = agent.apply(params, hstate, batched_input)
        q_vals = jnp.reshape(
            q_vals,
            (original_shape[0], len(agents), *original_shape[1:-1])
            + q_vals.shape[(len(original_shape) - len(q_vals.shape) - 1) :],
        )  # (time_steps, n_agents, n_envs, action_dim)
        q_vals = {a: q_vals[:, i] for i, a in enumerate(agents)}
        # get actions from q vals
        if dis_support is None:  # scalar case
            valid_q_vals = jax.tree_util.tree_map(
                lambda q, valid_idx: q.squeeze(0)[..., valid_idx], q_vals, valid_actions
            )
        else:  # distribution case
            valid_q_vals = jax.tree_util.tree_map(
                lambda q, valid_idx: q_expectation_from_dis(q.squeeze(0), dis_support)[
                    ..., valid_idx
                ],
                q_vals,
                valid_actions,
            )
        # actions = jax.tree_util.tree_map(lambda q: jnp.argmax(q, axis=-1).squeeze(0), valid_q_vals) #Greedy
        actions = jax.tree_util.tree_map(
            lambda q: jnp.array(
                [jnp.argmax(q[..., x], axis=-1).squeeze(0) for x in env.act_type_idx]
            ),
            valid_q_vals,
        )  # Greedy segmented
        # actions = jax.tree_util.tree_map(lambda q: jax.random.choice(key_a,a=valid_actions[agents[0]],p=jnp.exp(q).squeeze(0)/jnp.sum(jnp.exp(q), axis=-1).squeeze(0)), valid_q_vals) #Sample
        # actions = {a:i[0] for a, i in explorer.choose_actions(valid_q_vals, state.step, key_a).items()} #ep-Greedy

        return actions, hstate

    # run test + collect results
    state_seq, act_seq, info_seq, done_run, act_id_seq = [], [], [], [], []
    init_dones = {agent: jnp.zeros(1, dtype=bool) for agent in env.agents + ["__all__"]}
    rew_tallys = np.zeros((config["NUM_TRAIN_SEEDS"], max_st, env.num_agents))
    for k in range(config["NUM_TRAIN_SEEDS"]):
        key, key_i = jax.random.split(key, 2)
        hstate = ScannedRNN.initialize_carry(
            alg_config["AGENT_HIDDEN_DIM"], env.num_agents
        )
        state, dones, obs, act, info, act_id = [init_state], init_dones, init_obs, [], [], []
        task_update_timer = 0
        for j in range(max_st):
            # Iterate random keys and sample actions
            key_i, key_s, key_a = jax.random.split(key_i, 3)
            acts, hstate = obs_to_act(
                hstate, obs, dones, p[k], env, state[-1], key_a, dis_support
            )
            act_id.append([acts[a].tolist() for a in env.agents])
            act_ar = partial(env.action_decoder, actions=acts)
            act.append(
                [
                    np.concatenate(
                        [
                            act_ar(acts[v])[0][i],
                            env.tar_resolve_rad[
                                i,
                                state[-1].tar_resolve_idx[i] : (
                                    state[-1].tar_resolve_idx[i] + 2
                                ),
                            ],
                        ]
                    )
                    for i, v in enumerate(acts.keys())
                ]
            )
            # Step environment
            if cc_enabled & state[-1].is_task_changed:
                task_pool = state[-1].get_task_indices().tolist()
                if len(task_pool) > 0:
                    cc_fit = partial(env.get_task_queue_cost, state[-1])
                    schedule_solver = PermutationSolver(
                        fitness=cc_fit,
                        opt_dir=-1,
                        sol_shape=(env.num_agents, env.task_queue_length),
                        value_pool=task_pool,
                    )
                    # new_task_queues=schedule_solver.solve_rls(1,(len(task_pool)*env.task_queue_length_total)**2)
                    new_task_queues = schedule_solver.solve_ls(1, 3)
                    # print(new_task_queues) # display task allocation
                    state[-1] = state[-1].replace(task_queues=new_task_queues)
                    task_update_timer = 0
                else:
                    task_update_timer += 1
            obs, s, rewards, dones, infos = env.step(key_s, state[-1], acts)
            dones = jax.tree_util.tree_map(lambda x: x[None], dones)
            rew_tallys[k, j] = np.array([rewards[a] for a in env.agents])
            state.append(s)
            info.append(infos)
            if dones["__all__"]:
                break
                # hstate=ScannedRNN.initialize_carry(alg_config["AGENT_HIDDEN_DIM"],env.num_agents)
        state_seq.append(state)
        info_seq.append(info)
        act_seq.append(act)
        done_run.append(dones["__all__"])
        act_id_seq.append(act_id)
    return state_seq, info_seq, act_seq, done_run, rew_tallys, act_id_seq


def bulk_run(config, alg_names):
    plt.ioff() # turn off matplotlib interactive plotting to allow programmatic behaviours
    if isinstance(alg_names, str):
        alg_names = [alg_names]
    os.makedirs(config["SAVE_PATH"], exist_ok=True)
    if config.get(
        "ENV_PATH", None
    ):  # read env arg list from path, TODO: exception check
        f = open(config["ENV_PATH"], "r")
        benchmark_dict = json.load(f)
        f.close
        if isinstance(benchmark_dict, list):
            benchmark_dict = benchmark_dict[0]  # TODO: loop through env args
        env_name = benchmark_dict["env_name"]
        config["ENV_KWARGS"] = benchmark_dict["args"]
    else:
        env_name = config["ENV_NAME"]
    plot_out = (
        f"{config['SAVE_PATH']}/plot_{env_name}.pdf"  # plot data from all algorithms
    )
    config_env = config["ENV_KWARGS"]
    if 'obj_list' in config_env.keys(): # if objs are stored in object-oriented dict, flatten positions and velocity
        init_p, init_v, num_obj_dict = init_obj_to_array(config_env['obj_list'])
        config_env.pop('obj_list', None)
        config_env|={'init_p':init_p, 'init_v':init_v, 'num_agents':num_obj_dict['agent'], 'num_obs':num_obj_dict['obstacle'], 'num_tar':num_obj_dict['target']}
    env = make_env(env_name, **config["ENV_KWARGS"])
    state_list, info_list, act_list, done_list, rew_list, f2r_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    fig, ax = plt.subplots(3)
    plot_title = "Mean stop time PAR2:"
    for alg_idx, alg_name in enumerate(alg_names):
        state_seq, info_seq, act_seq, done_run, rew_tallys, act_id_seq = single_run(
            config, alg_name, env, env_name
        )
        state_list.append(state_seq)
        info_list.append(info_seq)
        act_list.append(act_seq)
        done_list.append(done_run)
        
        run_data_out = f"{config['SAVE_PATH']}/data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{env_name}_{alg_name}.json"
        state_dict = [
            {"run_id": i, "states": [s._to_dict(True) for s in z]}
            for i, z in enumerate(state_seq)
        ]
        for i in range(len(act_seq)):
            state_dict[i]["actions"] = [
                [{"obj_id": j, "action": s.tolist()} for j, s in enumerate(z)]
                for z in act_seq[i]
            ]
            state_dict[i]["action_ids"] = act_id_seq[i]
        for i in range(len(info_seq)):
            state_dict[i]["infos"] = [
                [
                    {
                        "obj_id": j,
                        "target": z["targets"][j].tolist(),
                        "avoid": z["avoid"][j].tolist(),
                    }
                    for j, s in enumerate(z["targets"])
                ]
                for z in info_seq[i]
            ]
        output_results(config, run_data_out, env, state_dict)
        rew_score, f2r = [], []
        for i in range(config["NUM_TRAIN_SEEDS"]):
            state_seq[i] = state_seq[i][:-1]
            rew_score.append(rew_tallys[i, : len(state_seq[i])].tolist())
            f2r.extend([i] * len(state_seq[i]))
        rew_list.append(rew_score)
        f2r_list.append(f2r)
        mission_prog, collisions, diversity = (
            [[int(x.mission_prog) for x in z] for z in state_seq],
            [[float(jnp.mean(x.collision_count)) for x in z] for z in state_seq],
            [[float(jnp.mean(x.min_dist_to_furthest_tar)) for x in z] for z in state_seq],
        )
        # rew = np.mean(rew_tallys, axis=2)
        runtimes = np.array([len(x) for i, x in enumerate(state_seq)])
        time_max = np.max(runtimes)
        mission_prog_fill, collisions_fill, diversity_fill = (
            np.array([x + ([np.nan] * (time_max - len(x))) for x in mission_prog]),
            np.array([x + ([np.nan] * (time_max - len(x))) for x in collisions]),
            np.array([x + ([np.nan] * (time_max - len(x))) for x in diversity]),
        )
        mission_prog_mean, mission_prog_var = (
            np.nanmean(mission_prog_fill, axis=0),
            np.nanvar(mission_prog_fill, axis=0),
        )
        collisions_mean, collisions_var = (
            np.nanmean(collisions_fill, axis=0),
            np.nanvar(collisions_fill, axis=0),
        )
        diversity_mean, diversity_var = (
            np.nanmean(diversity_fill, axis=0),
            np.nanvar(diversity_fill, axis=0),
        )
        runtimespar2 = np.array(
            [len(x) * (1 if done_run[i] else 2) for i, x in enumerate(state_seq)]
        )
        # plot stats
        ax[0].plot(range(time_max), mission_prog_mean, color=plot_colors[alg_idx])
        ax[0].fill_between(
            range(time_max),
            mission_prog_mean - mission_prog_var,
            mission_prog_mean + mission_prog_var,
            facecolor=plot_colors[alg_idx],
            alpha=0.3,
        )
        ax[1].plot(range(time_max), collisions_mean, color=plot_colors[alg_idx])
        ax[1].fill_between(
            range(time_max),
            collisions_mean - collisions_var,
            collisions_mean + collisions_var,
            facecolor=plot_colors[alg_idx],
            alpha=0.3,
        )
        ax[2].plot(range(time_max), diversity_mean, color=plot_colors[alg_idx])
        ax[2].fill_between(
            range(time_max),
            diversity_mean - diversity_var,
            diversity_mean + diversity_var,
            facecolor=plot_colors[alg_idx],
            alpha=0.3,
        )
        for k in range(config["NUM_TRAIN_SEEDS"]):
            ax[0].axvline(x=runtimes[k] - 1, alpha=0.3, color=plot_colors[alg_idx])
            ax[1].axvline(x=runtimes[k] - 1, alpha=0.3, color=plot_colors[alg_idx])
            ax[2].axvline(x=runtimes[k] - 1, alpha=0.3, color=plot_colors[alg_idx])
        ax[0].plot([], [], color=plot_colors[alg_idx], label=alg_name)
        plot_title += f" {alg_name}:{runtimespar2.mean()},"
    ax[0].legend(title="Legends")
    ax[-1].set_xlabel("Step")
    ax[0].set_ylabel("Progress score")
    ax[1].set_ylabel("Cumulative avg collision steps")
    ax[2].set_ylabel("Min. dist. to furthest target")
    fig.suptitle(plot_title[:-1])
    fig.savefig(plot_out)
    if config.get("SHOW_STATS_PLOTS", False):
        plt.show()
    plt.close(fig)
    # render animation
    for alg_idx, alg_name in enumerate(alg_names):
        state_seq, rew_score, act_seq, info_seq, f2r = (
            state_list[alg_idx],
            rew_list[alg_idx],
            act_list[alg_idx],
            info_list[alg_idx],
            f2r_list[alg_idx],
        )
        gif_out = f"{config['SAVE_PATH']}/visual_{env_name}_{alg_name}.gif"
        plt.show()
        viz = MPEVisualizer(
            env,
            np.concatenate(state_seq),
            np.concatenate(rew_score),
            np.concatenate(act_seq),
            np.concatenate(info_seq),
            f2r,
            alg_name,
        )
        viz.animate(view=config.get("SHOW_RENDER", False), save_fname=gif_out)


from pathlib import Path


@hydra.main(version_base=None, config_path="./config", config_name="config_test")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    assert config.get("algname", None), "Must supply an algorithm name"
    bulk_run(config, config["algname"])


if __name__ == "__main__":
    main()

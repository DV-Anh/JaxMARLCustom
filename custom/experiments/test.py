import os
import json
from datetime import datetime
import jax
import jax.numpy as jnp
import numpy as np
import hydra
from omegaconf import OmegaConf
from functools import partial

from jaxmarl.wrappers.baselines import (
    load_params, CTRolloutManager
)
from custom.registry import make_env, make_alg_runner
from custom.qlearning.common import ScannedRNN
from custom.utils.blackboxOptimizer import PermutationSolver
from custom.utils.mpe_visualizer import MPEVisualizer
from custom.environments.customMPE import init_obj_to_array
from custom.firehandler.firehandler import Fire, FireHandler


def main(config_test, model_paths: dict[int, list[str]], model_config_paths: dict[int, str], model_ids: dict[int, list[str]], is_animation=True, is_plot=True) -> list[dict]:
    hydra_config = collect_hydra_config(config_test)
    out_results = bulk_run(hydra_config, model_paths, model_config_paths, model_ids, is_animation=is_animation, is_plot=is_plot)
    return out_results, hydra_config # returns a list of dict, one for each alg, serialisable


@hydra.main(version_base=None, config_path="./config", config_name="config_test")
def collect_hydra_config(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    # assert config.get("algname", None), "Must supply an algorithm name" - WHen used with the front end, this doesn't make sense!! 
    return config


def bulk_run(config_test, model_paths: dict[int, list[str]], model_config_paths: list[str], model_ids: dict[int, list[str]], is_animation: bool=True, is_plot=True):
    """
    Args:
        model_paths: A list of model_paths, where each model has N number of variations depending on the number of batches it was trained with.
    """
    env, env_name = _set_up_env(config_test)
    out_dicts, state_list, info_list, act_list, done_list, rew_list, f2r_list = _run_tests(model_paths, config_test, model_config_paths, env, env_name)
    if is_plot:
        _plot_tests(config_test, model_paths, model_ids, state_list, done_list, env_name)
    if is_animation:
        _render_animation(config_test, model_ids, env_name, env, state_list, rew_list, act_list, info_list, f2r_list)
    return out_dicts


def to_fire_list(p_pos, rad, tar_touch, tar_amounts, is_exist, init_time, config, prev_fire_list=None):
    num_tar = p_pos.shape[0]
    fires = []
    track_id = 0
    for i in range(num_tar):
        if (prev_fire_list is not None) and (track_id<len(prev_fire_list)) and (i==prev_fire_list[track_id].id):
            fires.append(prev_fire_list[track_id])
            track_id += 1
        else:
            fires.append(None)
    fire_list = [
            Fire(
            id=i,
            initialization_time=int(init_time) if fires[i] is None else fires[i].initialization_time,
            position=np.array((p_pos[i]).tolist()),
            intensity=(float((tar_amounts[i]-tar_touch[i])/tar_amounts[i])*(config['max_intensity']-config['min_intensity'])+config['min_intensity']) if (fires[i] is None) else (float((tar_amounts[i]-tar_touch[i])/tar_amounts[i])*(fires[i].max_intensity-fires[i].min_intensity)+fires[i].min_intensity),
            radius=float(rad[i]),
            max_intensity=config['max_intensity'] if fires[i] is None else fires[i].max_intensity,
            min_intensity=config['min_intensity'] if fires[i] is None else fires[i].min_intensity,
            is_spread=config['is_spread'] if fires[i] is None else fires[i].is_spread,
            spread_radius_multiplier=config['spread_radius_multiplier'] if fires[i] is None else fires[i].spread_radius_multiplier,
            spread_intensity_multiplier=config['spread_intensity_multiplier'] if fires[i] is None else fires[i].spread_intensity_multiplier,
            spread_min_radius=config['spread_min_radius'] if fires[i] is None else fires[i].spread_min_radius,
            spread_min_threshold_intensity=config['spread_min_threshold_intensity'] if fires[i] is None else fires[i].spread_min_threshold_intensity,
            is_grow=config['is_grow'] if fires[i] is None else fires[i].is_grow,
            grow_intensity_multiplier=config['grow_intensity_multiplier'] if fires[i] is None else fires[i].grow_intensity_multiplier,
            grow_probability=config['grow_probability'] if fires[i] is None else fires[i].grow_probability,
            grow_radius_multiplier=config['grow_radius_multiplier'] if fires[i] is None else fires[i].grow_radius_multiplier,
        ) for i in range(num_tar) if bool(is_exist[i])
    ]
    return fire_list

def to_fire_jax(fire_list, env):
    p_pos = jnp.zeros((env.num_tar,env.dim_p),dtype=float)
    rad = jnp.zeros((env.num_tar,),dtype=float)
    tar_touch = jnp.zeros((env.num_tar,),dtype=int)
    is_exist = jnp.full((env.num_tar,),False,dtype=bool)
    for i,x in enumerate(fire_list):
        p_pos = p_pos.at[x.id].set(x.position)
        rad = rad.at[x.id].set(x.radius)
        tar_touch = tar_touch.at[x.id].set(round((x.max_intensity-x.intensity)/(x.max_intensity-x.min_intensity)*env.tar_amounts[i]))
        is_exist = is_exist.at[x.id].set(True)
    return p_pos, rad, tar_touch, is_exist

def single_run(config, model_variation_paths, model_config_path, env, env_name):
    """
    Args:
        model_variation_paths: A list of model_paths, where each model was trained with the same training config, but has different parameters.
    """
    os.makedirs(config["SAVE_PATH"], exist_ok=True)
    p = []
    for model_path in model_variation_paths:
        p.append(
            load_params(model_path)
        )
    # get most recent training hyperparam setting, needed to initialize model container
    f = open(model_config_path)
    assert (not f.closed), f'Cannot load model config file: {model_config_path}'
    alg_config = (json.load(f))["alg"]
    f.close()
    runner = make_alg_runner(alg_config['NAME'], alg_config|{'NUM_TEST_EPISODES':1}, env)
    wrapped_env = CTRolloutManager(env, batch_size=1, preprocess_obs=False)
    # prepare test
    key = jax.random.PRNGKey(config["SEED"])
    key, key_r = jax.random.split(key)
    max_st = config["ENV_KWARGS"]["max_steps"]
    cc_enabled = config["ENV_KWARGS"].get("central_controller", False)
    init_obs, init_state = wrapped_env.batch_reset(key_r)
    for i in range(config["NUM_TRAIN_SEEDS"]):
        if "agent" in p[i].keys():
            p[i] = p[i]["agent"]  # qmix also have mixer params
    # run test + collect results
    state_seq, act_seq, info_seq, done_run, act_id_seq = [], [], [], [], []
    init_dones = {agent: jnp.zeros(1, dtype=bool) for agent in env.agents + ["__all__"]}
    rew_tallys = np.zeros((config["NUM_TRAIN_SEEDS"], max_st, env.num_agents))
    if config.get('FIRE_HANDLER', False):
        # fire handler init fire
        init_fire_list=to_fire_list(init_state.p_pos[-env.num_tar:],init_state.rad[-env.num_tar:],init_state.tar_touch,env.tar_amounts,init_state.is_exist[-env.num_tar:],init_state.step,config['FIRE_HANDLER_CONFIG'],None)
    for k in range(config["NUM_TRAIN_SEEDS"]):
        key, key_i = jax.random.split(key)
        hstate = ScannedRNN.initialize_carry(
            alg_config["AGENT_HIDDEN_DIM"], env.num_agents
        )
        state, dones, obs, act, info, act_id = [jax.tree_util.tree_map(lambda x:x.squeeze(0), init_state)], init_dones, init_obs, [], [], []
        s = init_state
        if config.get('FIRE_HANDLER', False):
            fire_list = init_fire_list
        task_update_timer = 0
        for j in range(max_st):
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
            if config.get('FIRE_HANDLER', False):
                # update fire list
                fire_list=to_fire_list(state[-1].p_pos[-env.num_tar:],state[-1].rad[-env.num_tar:],state[-1].tar_touch,env.tar_amounts,state[-1].is_exist[-env.num_tar:],state[-1].step,config['FIRE_HANDLER_CONFIG'],fire_list)
                fire_list=FireHandler.update_fires(
                    current_fires=fire_list,
                    agent_interactions=[False]*len(fire_list),
                    max_number_fires=env.num_tar,
                    timestep=int(state[-1].step),
                    wind=None
                )
                p_pos, rad, tar_touch, is_exist = to_fire_jax(fire_list, env)
                s = s.replace(
                    p_pos=jnp.concatenate([state[-1].p_pos[:-env.num_tar],p_pos])[None],
                    rad=jnp.concatenate([state[-1].rad[:-env.num_tar],rad])[None],
                    is_exist=jnp.concatenate([state[-1].is_exist[:-env.num_tar],is_exist])[None],
                    tar_touch=tar_touch[None],
                )
            # step
            key_i, key_a = jax.random.split(key_i)
            run_state, info_state = runner((p[k], s, obs, dones, hstate, key_a), None)
            _, s, obs, dones, hstate, key_a = run_state
            rewards, dones, infos, acts = info_state
            acts = jax.tree_util.tree_map(lambda x:x.squeeze(0), acts)
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
            # obs, s, rewards, dones, infos = env.step(key_s, state[-1], acts)
            # dones = jax.tree_util.tree_map(lambda x: x[None], dones)
            rew_tallys[k, j] = np.array([rewards[a].squeeze(0) for a in env.agents])
            state.append(jax.tree_util.tree_map(lambda x:x.squeeze(0), s))
            info.append(jax.tree_util.tree_map(lambda x:x.squeeze(0), infos))
            if dones["__all__"]:
                break
                # hstate=ScannedRNN.initialize_carry(alg_config["AGENT_HIDDEN_DIM"],env.num_agents)
        state_seq.append(state)
        info_seq.append(info)
        act_seq.append(act)
        done_run.append(dones["__all__"])
        act_id_seq.append(act_id)
    return state_seq, info_seq, act_seq, done_run, rew_tallys, act_id_seq


def _set_up_env(config):
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

    config_env = config["ENV_KWARGS"]
    if 'obj_list' in config_env.keys(): # if objs are stored in object-oriented dict, flatten positions and velocity
        init_p, init_v, num_obj_dict = init_obj_to_array(config_env['obj_list'])
        config_env.pop('obj_list', None)
        config_env|={'init_p':init_p, 'init_v':init_v, 'num_agents':num_obj_dict['agent'], 'num_obs':num_obj_dict['obstacle'], 'num_tar':num_obj_dict['target']}
    env = make_env(env_name, **config_env)

    return env, env_name


def _run_tests(model_paths, config_test, model_config_paths, env, env_name):
    state_list, info_list, act_list, done_list, rew_list, f2r_list, out_dict = (
        [], [], [], [], [], [], [],
    )
    out_dicts = [] 
    for model_idx, model_variation_paths in model_paths.items():
        state_seq, info_seq, act_seq, done_run, rew_tallys, act_id_seq = single_run(
            config_test, model_variation_paths, model_config_paths[model_idx], env, env_name
        )
        state_list.append(state_seq)
        info_list.append(info_seq)
        act_list.append(act_seq)
        done_list.append(done_run)
        
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

        out_dict = {"env": env._to_dict(), "runs": state_dict}
        out_dicts.append(out_dict)

        rew_score, f2r = [], []
        for i in range(config_test["NUM_TRAIN_SEEDS"]):
            state_seq[i] = state_seq[i][:-1]
            rew_score.append(rew_tallys[i, : len(state_seq[i])].tolist())
            f2r.extend([i] * len(state_seq[i]))
        rew_list.append(rew_score)
        f2r_list.append(f2r)
    return out_dicts, state_list, info_list, act_list, done_list, rew_list, f2r_list



def _render_animation(config, model_ids, env_name, env, state_list, rew_list, act_list, info_list, f2r_list):
    import matplotlib.pyplot as plt
    for model_idx, model_id in enumerate(model_ids):
        state_seq, rew_score, act_seq, info_seq, f2r = (
            state_list[model_idx],
            rew_list[model_idx],
            act_list[model_idx],
            info_list[model_idx],
            f2r_list[model_idx],
        )
        gif_out = f"{config['SAVE_PATH']}/visual_{env_name}_{model_id}.gif"
        plt.show()
        viz = MPEVisualizer(
            env,
            np.concatenate(state_seq),
            np.concatenate(rew_score),
            np.concatenate(act_seq),
            np.concatenate(info_seq),
            f2r,
            model_id,
        )
        viz.animate(view=config.get("SHOW_RENDER", False), save_fname=gif_out)


def _plot_tests(config, model_paths, model_names, state_list, done_list, env_name):
    import matplotlib
    matplotlib.use("Agg")  # Use non-GUI backend
    import matplotlib.pyplot as plt
    plt.ioff() # turn off matplotlib interactive plotting to allow programmatic behaviours
    plot_colors = ("red", "blue", "green", "black") # TODO: add more colors with good contrasts

    fig, ax = plt.subplots(3)
    plot_title = "Mean stop time PAR2:"
    for model_idx, model_path in enumerate(model_paths):
        state_seq = state_list[model_idx]
        done_run = done_list[model_idx]
        
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
        ax[0].plot(range(time_max), mission_prog_mean, color=plot_colors[model_idx])
        ax[0].fill_between(
            range(time_max),
            mission_prog_mean - mission_prog_var,
            mission_prog_mean + mission_prog_var,
            facecolor=plot_colors[model_idx],
            alpha=0.3,
        )
        ax[1].plot(range(time_max), collisions_mean, color=plot_colors[model_idx])
        ax[1].fill_between(
            range(time_max),
            collisions_mean - collisions_var,
            collisions_mean + collisions_var,
            facecolor=plot_colors[model_idx],
            alpha=0.3,
        )
        ax[2].plot(range(time_max), diversity_mean, color=plot_colors[model_idx])
        ax[2].fill_between(
            range(time_max),
            diversity_mean - diversity_var,
            diversity_mean + diversity_var,
            facecolor=plot_colors[model_idx],
            alpha=0.3,
        )
        for k in range(config["NUM_TRAIN_SEEDS"]):
            ax[0].axvline(x=runtimes[k] - 1, alpha=0.3, color=plot_colors[model_idx])
            ax[1].axvline(x=runtimes[k] - 1, alpha=0.3, color=plot_colors[model_idx])
            ax[2].axvline(x=runtimes[k] - 1, alpha=0.3, color=plot_colors[model_idx])
        ax[0].plot([], [], color=plot_colors[model_idx], label=model_names[model_idx])
        plot_title += f" {model_names[model_idx]}:{runtimespar2.mean()},"
    ax[0].legend(title="Legends")
    ax[-1].set_xlabel("Step")
    ax[0].set_ylabel("Progress score")
    ax[1].set_ylabel("Cumulative avg collision steps")
    ax[2].set_ylabel("Min. dist. to furthest target")
    fig.suptitle(plot_title[:-1])
    plot_out = (
        f"{config['SAVE_PATH']}/plot_{env_name}.pdf"  # plot data from all algorithms
    )   
    fig.savefig(plot_out)
    if config.get("SHOW_STATS_PLOTS", False):
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

from .TargetHandler import TargetHandler
import jax
import jax.numpy as jnp

class FireHandler(TargetHandler):
    def __init__(self, env, mission_rew=1.0, fire_growth_rate=0.1, fire_spread_threshold=5.0, initial_fire_size=1.0, spread_distance_std=1.0):
        super().__init__(env)
        self.mission_rew = mission_rew
        self.fire_growth_rate = fire_growth_rate
        self.fire_spread_threshold = fire_spread_threshold
        self.initial_fire_size = initial_fire_size
        self.spread_distance_std = spread_distance_std
        self.max_num_tar = env.max_num_tar

    def initialize_state(self, state, key):
        # Initialize fire_size and other fire-specific state variables
        fire_size = jnp.zeros((self.max_num_tar,))
        fire_size = fire_size.at[:self.env.num_tar].set(self.initial_fire_size)
        # Initialize other target-specific arrays
        tar_touch = jnp.zeros((self.max_num_tar,))
        is_exist_targets = jnp.full((self.max_num_tar,), True)
        is_exist = jnp.concatenate([state.is_exist[:self.env.num_agents + self.env.num_obs], is_exist_targets], axis=0)
        p_pos = jnp.concatenate([state.p_pos, jnp.zeros((self.max_num_tar - self.env.num_tar, self.env.dim_p))], axis=0)
        tar_touch_b = jnp.zeros((self.env.num_agents, self.max_num_tar))
        # Store target-specific state
        target_state = {'fire_size': fire_size, 'current_num_tar': self.env.num_tar}
        # Update state
        state = state.replace(
            tar_touch=tar_touch,
            is_exist=is_exist,
            p_pos=p_pos,
            tar_touch_b=tar_touch_b,
            target_state=target_state
        )
        return state

    def update_targets(self, state, key):
        # Fire-specific update logic (growth and spread)
        # Retrieve target-specific state
        fire_size = state.target_state['fire_size']
        current_num_tar = state.target_state['current_num_tar']
        # Update fire sizes
        updated_fire_size = fire_size.at[:current_num_tar].add(self.fire_growth_rate)
        # Determine which fires will spread
        fires_to_spread = updated_fire_size[:current_num_tar] >= self.fire_spread_threshold
        num_fires_to_spread = jnp.sum(fires_to_spread)
        new_total_fires = jnp.minimum(current_num_tar + num_fires_to_spread, self.max_num_tar)
        # Generate new fires
        key, key_spread = jax.random.split(key)
        spread_indices = jnp.where(fires_to_spread)[0]
        spread_positions = state.p_pos[-self.max_num_tar:][current_num_tar - self.env.num_tar:current_num_tar][fires_to_spread]
        num_new_fires = new_total_fires - current_num_tar
        num_spreading_fires = spread_positions.shape[0]
        num_new_fires = jnp.maximum(num_new_fires, 0)
        offsets = jax.random.normal(
            key_spread,
            shape=(num_new_fires, self.env.dim_p),
            mean=0.0,
            stddev=self.spread_distance_std
        )
        repeats = (num_new_fires + num_spreading_fires - 1) // num_spreading_fires
        spread_positions_repeated = jnp.tile(spread_positions, (repeats, 1))[:num_new_fires]
        new_fire_positions = spread_positions_repeated + offsets
        # Initialize new fires
        new_fire_sizes = jnp.full((num_new_fires,), self.initial_fire_size)
        new_fire_exists = jnp.full((num_new_fires,), True)
        new_fire_touch = jnp.zeros((num_new_fires,))
        # Update state arrays
        p_pos = state.p_pos.at[-self.max_num_tar:][current_num_tar:new_total_fires].set(new_fire_positions)
        updated_fire_size = updated_fire_size.at[current_num_tar:new_total_fires].set(new_fire_sizes)
        tar_touch = state.tar_touch[:current_num_tar]
        tar_touch = jnp.concatenate([tar_touch, new_fire_touch], axis=0)
        is_exist_targets = state.is_exist[-self.max_num_tar:]
        is_exist_targets = is_exist_targets.at[current_num_tar:new_total_fires].set(new_fire_exists)
        is_exist = jnp.concatenate([state.is_exist[:self.env.num_agents + self.env.num_obs], is_exist_targets], axis=0)
        current_num_tar = new_total_fires
        # Update existence flags for exhausted fires
        active_targets = tar_touch < self.env.tar_amounts
        is_exist_targets = is_exist_targets.at[:current_num_tar].set(active_targets[:current_num_tar])
        is_exist = jnp.concatenate([state.is_exist[:self.env.num_agents + self.env.num_obs], is_exist_targets], axis=0)
        # Update mission progress and scores
        num_new_exhausted = jnp.sum(jnp.logical_not(active_targets[:current_num_tar]))
        new_mission_prog = state.mission_prog + num_new_exhausted
        updated_last_score_timer = jax.lax.select(
            new_mission_prog > state.mission_prog,
            0,
            state.last_score_timer + 1
        )
        updated_mission_con = state.mission_con + jnp.any(state.tar_touch_b[:, :current_num_tar], axis=1)
        mission_score_increment = num_new_exhausted * (
            self.env.tar_amounts / (state.last_score_timer + 1) + 1 / self.mission_rew
        )
        updated_mission_score = state.mission_score + mission_score_increment
        # Update tar_touch_b
        updated_tar_touch_b = jnp.zeros_like(state.tar_touch_b)
        updated_tar_touch_b = updated_tar_touch_b.at[:, :current_num_tar].set(state.tar_touch_b[:, :current_num_tar])
        # Update target_state
        target_state = {'fire_size': updated_fire_size, 'current_num_tar': current_num_tar}
        # Replace state with updated values
        state = state.replace(
            tar_touch=tar_touch,
            is_exist=is_exist,
            p_pos=p_pos,
            last_score_timer=updated_last_score_timer,
            mission_prog=new_mission_prog,
            mission_con=updated_mission_con,
            mission_score=updated_mission_score,
            tar_touch_b=updated_tar_touch_b,
            target_state=target_state
        )
        return state
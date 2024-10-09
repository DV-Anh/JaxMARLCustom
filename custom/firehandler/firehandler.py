from .fire import Fire
import numpy as np
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import heapq
import math

def clamp(intensity, min_intensity, max_intensity):
    return min(max_intensity, max(intensity, min_intensity))

@dataclass
class Fire:
    position: np.ndarray
    intensity: float
    radius: float
    max_intensity: float
    min_intensity: float
    is_spread: bool = False
    spread_radius_multiplier: Optional[float]
    spread_intensity_multiplier: Optional[float]
    spread_min_radius: Optional[float]
    spread_min_intensity: Optional[float]
    is_grow: bool = False
    grow_new_fire_intensity: Optional[float]
    grow_probability: Optional[float]
    grow_radius_multiplier: Optional[float]
   
    def __post_init__(fire):
        if fire.is_spread:
            if (fire.spread_radius_multiplier is None or
                fire.spread_intensity_multiplier is None or
                fire.spread_min_radius is None or
                fire.spread_min_intensity is None):
                raise ValueError("When 'is_spread' is True, 'spread_radius_multiplier', "
                                    "'spread_intensity_multiplier', and 'spread_min_radius' must all be provided.")

        if fire.is_grow:
            if (fire.grow_new_fire_intensity is None or
                fire.grow_probability is None,
                fire.grow_radius_multiplier is None):
                raise ValueError("When 'is_grow' is True, 'grow_new_fire_intensity' and 'grow_probability' and 'grow_radius_multiplier' must be provided.")

        if (fire.intensity > fire.max_intensity 
            or fire.intensity < fire.min_intensity):
            raise ValueError("intensity it less than min_intensity, or greater than max_intensity")
    
    def set_intensity(cls, intensity: int):
        cls.intensity = clamp()
        


class FireHandler:
    @classmethod
    def update_fires( 
        cls,
        current_fires: List[Fire], 
        max_number_fires: int, 
        timestep: int, 
        wind: Optional[np.ndarray] = None
        ) -> Dict[int, Any]:
        
        n_possible_new_fires = len(current_fires) - max_number_fires
        updated_fires = []
        new_fires = []
        parent_fire_radius = {}
        for fire in current_fires:
            parent_radius = fire.radius
            updated_fire, new_fire = cls._update(fire, wind, timestep)
            updated_fires.append(updated_fire) 
            new_fires.append(new_fire)
            parent_fire_radius.setdefault(parent_radius, []).append(new_fire)
        
        if n_possible_new_fires > len(new_fires):
            updated_fires.extend(new_fires)
        else:
            # Sort parent_fire_radius by key (parent radius) in descending order
            # The intuition here is that since we have a maximum number of fires, new fires from parent fires
            # That had a massive radius are most likely to spread, so they should be prioritized as new fires.
            sorted_fires_by_radius = sorted(parent_fire_radius.items(), key=lambda item: item[0], reverse=True)
            selected_new_fires = []
            for _, fires in sorted_fires_by_radius:
                if len(selected_new_fires) < n_possible_new_fires:
                    selected_new_fires.extend(fires[:n_possible_new_fires - len(selected_new_fires)])
                else:
                    break
        
        updated_fires.extend(selected_new_fires)
    
    @classmethod 
    def _update(
        cls,
        fire: Fire, 
        timestep: int,
        wind: Optional[np.ndarray] = None
        ) -> Tuple[Fire, Fire|None]:
       
        if fire.is_spread:
            cls._spread(fire, timestep, wind)
        
        if fire.is_grow:
            cls._grow()
            
        if wind:
            cls._apply_wind(wind, fire)
            
    def _spread(fire: Fire, time_step: int, wind: Optional[np.ndarray] = None):
        """
        A fire may spread_ to create new fires. Fires that spread from another fire
        are considered as new fires.

        Args:
            - wind: 2D vector containing speed in N, E directions (m/s)
            - time_step: the time_step of the simulation (m/s)

        TODO: Should probably also be based on Nearby viable spread_ing direction (not a river, etc)

        Outputs:
            - Fire object
        """

        # Check that fire is above a certain size threshold
        if (fire.radius > fire.spread_min_radius) and (
            fire.spread_intensity_multiplier > 3
        ):
            new_fire_position = fire.position + wind * time_step
            new_fire_intensity = math.floor(
                fire.spread_intensity_multiplier * fire.intensity
            )
            new_fire_radius = fire.radius * fire.spread_radius_multiplier
            new_fire = Fire(
                **fire.get_state_for_copying()
            )
            print("New FIRE returned")
            return new_fire
        else:
            return None

    def _apply_wind(wind, fire):
        pass
    
    def _grow(fire: Fire,):
        if np.random.randint(0, 100) > fire.grow_probability:
            fire.intensity += fire.grow_new_fire_intensity
            fire.radius *= fire.grow_radius_multiplier
        
    def _new_fire():
        pass
        
    def _set_intensity(fire, new_intensity):
        fire.intensity = clamp(new_intensity, fire.min_intensity, fire.max_intensity)
    
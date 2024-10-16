from dataclasses import dataclass, field
from pydantic import field_validator
import numpy as np
from typing import Optional, List, Tuple
import math


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value between a minimum and maximum value."""
    return max(min(value, max_value), min_value)


@dataclass
class Fire:
    # Fields without default values
    id: int = field(metadata={"description": "Unique identifier for the fire instance."})
    initialization_time: int = field(
        metadata={"description": "Timestep for when the fire was initialized."}
    )
    position: np.ndarray = field(
        metadata={"description": "2D position coordinates of the fire's location."}
    )
    intensity: float = field(metadata={"description": "Current intensity level of the fire."})
    radius: float = field(metadata={"description": "Current radius of the fire."})
    max_intensity: float = field(
        metadata={"description": "Maximum possible intensity the fire can reach."}
    )
    min_intensity: float = field(
        metadata={"description": "Minimum intensity the fire must reach before it dies out."}
    )
    # Fields with default values
    is_exists: bool = field(
        default=True, metadata={"description": "Whether the fire currently exists or not."}
    )
    is_spread: bool = field(
        default=False,
        metadata={"description": "Flag indicating whether the fire is able to spread."}
    )
    spread_radius_multiplier: Optional[float] = field(
        default=None,
        metadata={"description": "Multiplier used to set the radius of the new fire due to spreading."}
    )
    spread_intensity_multiplier: Optional[float] = field(
        default=None,
        metadata={"description": "Multiplier used to set the intensity of the new fire due to spreading."}
    )
    spread_min_radius: Optional[float] = field(
        default=None,
        metadata={"description": "Minimum radius threshold required for the fire to spread."}
    )
    spread_min_threshold_intensity: Optional[float] = field(
        default=None,
        metadata={"description": "Minimum intensity threshold required for the fire to spread."}
    )
    is_grow: bool = field(
        default=False,
        metadata={"description": "Flag indicating whether the fire is able to grow new fires."}
    )
    grow_intensity_multiplier: Optional[float] = field(
        default=None,
        metadata={"description": "Multiplier applied to the radius of the newly grown fire."}
    )
    grow_probability: Optional[float] = field(
        default=None,
        metadata={"description": "Probability that the fire will grow."}
    )
    grow_radius_multiplier: Optional[float] = field(
        default=None,
        metadata={"description": "Multiplier applied to the radius of the newly grown fire."}
    )

    @field_validator('position', mode='before')
    def validate_position(cls, v):
        if isinstance(v, np.ndarray):
            return v
        elif isinstance(v, list):
            return np.array(v)
        else:
            raise ValueError("Position must be a numpy.ndarray or list")

    def __post_init__(self):
        if self.is_spread:
            if (
                self.spread_radius_multiplier is None
                or self.spread_intensity_multiplier is None
                or self.spread_min_radius is None
                or self.spread_min_threshold_intensity is None
            ):
                raise ValueError(
                    "When 'is_spread' is True, 'spread_radius_multiplier', "
                    "'spread_intensity_multiplier', 'spread_min_radius', and 'spread_min_intensity' must all be provided."
                )

        if self.is_grow:
            if (
                self.grow_intensity_multiplier is None
                or self.grow_probability is None
                or self.grow_radius_multiplier is None
            ):
                raise ValueError(
                    "When 'is_grow' is True, 'grow_new_fire_intensity', "
                    "'grow_probability', and 'grow_radius_multiplier' must all be provided."
                )

        self._set_intensity(self.intensity)

    def _set_intensity(self, intensity: float):
        """Set the intensity for a specific instance of Fire, clamping it to the valid range."""
        self.intensity = clamp(intensity, self.min_intensity, self.max_intensity)

    def _update(
        self, agent_interaction: bool, timestep: int, wind: Optional[np.ndarray] = None
    ) -> Tuple[Optional['Fire'], Optional['Fire']]:
        """
        Updates the fire's state based on agent interactions and environmental factors.
        """
        if agent_interaction:
            self.is_exists = False
            return (None, None)
        else:
            new_fire = None
            if self.is_spread:
                new_fire = self._spread(timestep, wind)

            if self.is_grow:
                self._grow()

            return (self, new_fire)

    def _spread(self, timestep: int, wind: Optional[np.ndarray] = None) -> Optional['Fire']:
        """A fire may spread to create new fires."""
        if wind is None:
            wind = np.array([0.0, 0.0])

        if (self.radius > self.spread_min_radius) and (self.intensity > self.spread_min_threshold_intensity):
            new_fire_position = self.position + wind * timestep
            new_fire_intensity = math.floor(
                self.spread_intensity_multiplier * self.intensity
            )
            new_fire_radius = self.radius * self.spread_radius_multiplier

            new_fire = Fire(
                id=-1,  # Placeholder, will be set by FireHandler
                initialization_time=timestep,
                position=new_fire_position,
                intensity=new_fire_intensity,
                radius=new_fire_radius,
                max_intensity=self.max_intensity,
                min_intensity=self.min_intensity,
                # Rest of the fields
                is_spread=self.is_spread,
                spread_radius_multiplier=self.spread_radius_multiplier,
                spread_intensity_multiplier=self.spread_intensity_multiplier,
                spread_min_radius=self.spread_min_radius,
                spread_min_threshold_intensity=self.spread_min_threshold_intensity,
                is_grow=self.is_grow,
                grow_intensity_multiplier=self.grow_intensity_multiplier,
                grow_probability=self.grow_probability,
                grow_radius_multiplier=self.grow_radius_multiplier
            )
            return new_fire
        else:
            return None

    def _grow(self):
        """Handles the growth of the fire based on probability."""
        if self.grow_probability is not None and np.random.random() < self.grow_probability:
            self._set_intensity(self.intensity*self.grow_intensity_multiplier)
            self.radius *= self.grow_radius_multiplier


class FireHandler:
    @classmethod
    def update_fires(
        cls,
        current_fires: List[Fire],
        agent_interactions: List[bool],
        max_number_fires: int,
        timestep: int,
        wind: Optional[np.ndarray] = None,
    ) -> List[Fire]:
        """
        Updates a list of fires based on agent interactions and environmental factors.
        Returns a single list of fires, combining updated and new fires.
        """
        # Ensure that current_fires length is less than or equal to max_number_fires
        assert len(current_fires) <= max_number_fires, "current_fires exceeds max_number_fires"

        used_ids = {fire.id for fire in current_fires if fire.is_exists}
        available_ids = set(range(max_number_fires)) - used_ids

        updated_fires: List[Fire] = []
        new_fires_with_parent_radius: List[Tuple[Fire, float]] = []

        for i, fire in enumerate(current_fires):
            if i >= len(agent_interactions):
                raise IndexError("Length of agent_interactions does not match current_fires.")

            agent_interaction = agent_interactions[i]
            updated_fire, spawned_fire = fire._update(agent_interaction, timestep, wind)

            if updated_fire:
                updated_fires.append(updated_fire)
            if spawned_fire:
                # Collect the new fire along with the parent fire's radius
                new_fires_with_parent_radius.append((spawned_fire, fire.radius))

        # Total number of fires we can have is max_number_fires
        # So the number of available slots is max_number_fires - len(updated_fires)
        num_available_slots = max_number_fires - len(updated_fires)

        if num_available_slots > 0 and new_fires_with_parent_radius:
            # Sort new fires based on parent fire radius in descending order
            new_fires_with_parent_radius.sort(key=lambda x: x[1], reverse=True)
            # Select up to num_available_slots new fires
            selected_new_fires = [fire for fire, _ in new_fires_with_parent_radius[:num_available_slots]]
            # Assign IDs to selected new fires
            for new_fire in selected_new_fires:
                if available_ids:
                    assigned_id = available_ids.pop()
                    new_fire.id = assigned_id
                    updated_fires.append(new_fire)
                else:
                    # This shouldn't happen as we've checked num_available_slots
                    break
            # Discard fires that couldn't be added due to capacity limits
        # Remove fires that no longer exist
        final_fires = [fire for fire in updated_fires if fire.is_exists]

        return final_fires

if __name__ == "__main__":
    fire1 = Fire(
        id=0,
        initialization_time=0,
        position=np.array([0.0, 0.0]),
        intensity=1.0,
        radius=1.0,
        max_intensity=1.0,
        min_intensity=0.0,
        is_spread=True,
        spread_radius_multiplier=1.2,
        spread_intensity_multiplier=0.8,
        spread_min_radius=0.5,
        spread_min_threshold_intensity=0.2,
        is_grow=True,
        grow_intensity_multiplier=0.05,
        grow_probability=0.3,
        grow_radius_multiplier=1.1
    )

    fire2 = Fire(
        id=1,
        initialization_time=0,
        position=np.array([2.0, 3.0]),
        intensity=0.8,
        radius=1.0,
        max_intensity=1.0,
        min_intensity=0.0,
        is_spread=True,
        spread_radius_multiplier=1.2,
        spread_intensity_multiplier=0.8,
        spread_min_radius=0.5,
        spread_min_threshold_intensity=0.2,
        is_grow=True,
        grow_intensity_multiplier=0.05,
        grow_probability=0.3,
        grow_radius_multiplier=1.1
    )

    current_fires = [fire1, fire2]
    agent_interactions = [False, False]
    max_number_fires = 5
    wind = np.array([0.1, 0.05])

    num_timesteps = 10 

    for timestep in range(1, num_timesteps + 1):
        print(f"\nTimestep {timestep}:")
        current_fires = FireHandler.update_fires(
            current_fires=current_fires,
            agent_interactions=agent_interactions,
            max_number_fires=max_number_fires,
            timestep=timestep,
            wind=wind
        )
        agent_interactions = [False for _ in current_fires]

        print("Fires after update:")
        for fire in current_fires:
            print(fire)
import math
import numpy as np

def clamp(intensity, min_intensity, max_intensity):
    return min(max_intensity, max(intensity, min_intensity))

class Fire:
    def __init__(self, position, intensity, radius, **kwargs):
        self.position = position
        self.intensity = intensity
        self.radius = radius
        print(self.radius)
        self.is_spread = kwargs.get("is_spread", False)
        self.is_grow = kwargs.get("is_grow", False)
        self.max_intensity = kwargs.get("max_intensity", 10)
        self.min_intensity = kwargs.get("min_intensity", 0)
        self.spread_radius_multiplier = kwargs.get("spread_radius_multiplier", 0)
        self.spread_intensity_multiplier = kwargs.get("spread_intensity_multiplier", 0)
        self.spread_min_radius = kwargs.get("spread_min_radius", 0)
        self.spread_min_intensity = kwargs.get("spread_min_intensity", 0)
        self.grow_new_fire_intensity = kwargs.get("grow_new_fire_intensity", 0)
        self.grow_probability = kwargs.get("grow_probability", 0)
        self.grow_radius_multiplier = kwargs.get("grow_radius_multiplier", 0)
        self.damage_per_time_step = kwargs.get("damage_per_time_step", 0)

        self.health = self.evaluate_health()
        self.set_intensity(intensity)

    def get_state_for_agents(self):
        return {
            "position": self.position,
            "radius": self.radius,
            "intensity": self.intensity,
            "health": self.health,
        }

    def get_state_for_copying(self):
        attributes = [
            "is_spread",
            "is_grow",
            "max_intensity",
            "min_intensity",
            "spread_radius_multiplier",
            "spread_intensity_multiplier",
            "spread_min_radius",
            "spread_min_intensity",
            "grow_intensity",
            "grow_probability",
            "grow_radius",
            "damage_per_time_step",
        ]
        return {attr: getattr(self, attr) for attr in attributes}

    def update(self, wind, time_step):
        self.evaluate_health()
        new_fire = self.spread(wind, time_step)
        self.grow()
        return new_fire

    def set_intensity(self, new_intensity):
        self.intensity = clamp(new_intensity, self.min_intensity, self.max_intensity)

    def spread(self, wind, time_step):
        """
        A fire may spread_ to create new fires. Fires that spread from another fire
        are considered as new fires.

        Inputs:
            - wind: 2D vector containing speed in N, E directions (m/s)
            - time_step: the time_step of the simulation (m/s)

        Should probably also be based on:
            - Nearby viable spread_ing direction (not a river, etc)

        Outputs:
            - Fire object
        """

        # Check that fire is above a certain size threshold
        if (self.radius > self.spread_min_radius) and (
            self.spread_intensity_multiplier > 3
        ):
            new_fire_position = self.position + wind * time_step
            new_fire_intensity = math.floor(
                self.spread_intensity_multiplier * self.intensity
            )
            new_fire_radius = self.radius * self.spread_radius_multiplier
            new_fire = Fire(
                new_fire_position,
                new_fire_intensity,
                new_fire_radius,
                **self.get_state_for_copying()
            )
            print("New FIRE returned")
            return new_fire
        else:
            return None

    def grow(self):
        """
        A fire has the chance to grow in radius and intensity
        """
        if np.random.randint(0, 100) > self.grow_probability:
            self.intensity += self.grow_new_fire_intensity
            self.radius *= self.grow_radius_multiplier

    def burn(self):
        return self.intensity * self.damage_per_time_step

    def evaluate_health(self):
        return self.radius * self.intensity
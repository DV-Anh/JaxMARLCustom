# tests/test_firehandler.py

import pytest
import numpy as np
from unittest.mock import patch
from firehandler import Fire, FireHandler, clamp

# ------------------------------
# Fixtures and Factory Functions
# ------------------------------

@pytest.fixture
def fire_factory():
    """
    Factory fixture to create Fire instances with default parameters.
    Override defaults by passing keyword arguments.
    """
    def _create_fire(**kwargs):
        defaults = {
            "id": 0,
            "initialization_time": 0,
            "position": np.array([0.0, 0.0]),
            "intensity": 1.0,
            "radius": 1.0,
            "max_intensity": 1.0,
            "min_intensity": 0.0,
            "is_exists": True,
            "is_spread": False,
            # Spread parameters (set to None by default)
            "spread_radius_multiplier": None,
            "spread_intensity_multiplier": None,
            "spread_min_radius": None,
            "spread_min_threshold_intensity": None,
            "is_grow": False,
            # Grow parameters (set to None by default)
            "grow_intensity_multiplier": None,
            "grow_probability": None,
            "grow_radius_multiplier": None
        }
        defaults.update(kwargs)
        return Fire(**defaults)
    return _create_fire

# ------------------------------
# Key Tests with Complex Functionality
# ------------------------------

def test_fire_grow_always(fire_factory):
    """
    Test that a fire grows when grow_probability conditions are met.
    """
    initial_intensity = 0.8
    fire = fire_factory(
        is_grow=True,
        grow_intensity_multiplier=1.05,
        grow_probability=0.3,
        grow_radius_multiplier=1.1,
        intensity=initial_intensity,
        radius=1.0
    )
    with patch('numpy.random.random', return_value=0.1):  # Less than grow_probability
        fire._grow()
        # Intensity should be clamped to max_intensity=1.0
        expected_intensity = clamp(initial_intensity*fire.grow_intensity_multiplier , fire.min_intensity, fire.max_intensity)
        assert fire.intensity == expected_intensity
        # Radius should increase by grow_radius_multiplier
        expected_radius = 1.0 * 1.1
        assert fire.radius == expected_radius

def test_fire_grow_never(fire_factory):
    """
    Test that a fire does not grow when grow_probability conditions are not met.
    """
    fire = fire_factory(
        is_grow=True,
        grow_intensity_multiplier=1.05,
        grow_probability=0.3,
        grow_radius_multiplier=1.1,
        intensity=0.95,
        radius=1.0
    )
    with patch('numpy.random.random', return_value=0.4):  # Greater than grow_probability
        fire._grow()
        # Intensity and radius should remain unchanged
        assert fire.intensity == 0.95
        assert fire.radius == 1.0

def test_fire_update_spread(fire_factory):
    """
    Test that a fire spreads when spread conditions are met.
    """
    fire = fire_factory(
        is_spread=True,
        spread_radius_multiplier=1.2,
        spread_intensity_multiplier=0.8,
        spread_min_radius=0.5,
        spread_min_threshold_intensity=0.2,
        intensity=0.3,
        radius=1.0,
        is_grow=False
    )
    # Expected new fire after spread
    new_fire = fire_factory(
        id=-1,
        initialization_time=1,
        position=fire.position + np.array([0.1, 0.05]),
        intensity=0,  # As per your Fire class logic
        radius=1.2,
        is_spread=True,
        spread_radius_multiplier=1.2,
        spread_intensity_multiplier=0.8,
        spread_min_radius=0.5,
        spread_min_threshold_intensity=0.2,
        is_grow=False
    )
    with patch.object(Fire, '_spread', return_value=new_fire):
        updated_fire, spawned_fire = fire._update(agent_interaction=False, timestep=1, wind=np.array([0.1, 0.05]))
        assert updated_fire == fire
        assert spawned_fire == new_fire

def test_firehandler_update_fires_with_spread_and_grow(fire_factory):
    """
    Test FireHandler.update_fires with fires that can spread and grow.
    """
    fire1 = fire_factory(
        id=0,
        is_spread=True,
        spread_radius_multiplier=1.2,
        spread_intensity_multiplier=0.8,
        spread_min_radius=0.5,
        spread_min_threshold_intensity=0.2,
        is_grow=True,
        grow_intensity_multiplier=1.5,
        grow_probability=0.3,
        grow_radius_multiplier=1.1,
        intensity=0.9,
        radius=1.0
    )
    current_fires = [fire1]
    agent_interactions = [False]
    max_number_fires = 5
    wind = np.array([0.1, 0.05])

    # Mock the _spread and _grow methods
    new_fire = fire_factory(
        id=-1,
        initialization_time=1,
        position=fire1.position + wind * 1,  # timestep=1
        intensity=0,  # As per your Fire class logic
        radius=fire1.radius * 1.2,
        is_spread=True,
        is_grow=True,
        spread_radius_multiplier=1.2,
        spread_intensity_multiplier=0.8,
        spread_min_radius=0.5,
        spread_min_threshold_intensity=0.2,
        grow_intensity_multiplier=1.5,
        grow_probability=0.3,
        grow_radius_multiplier=1.1
    )

    with patch.object(Fire, '_spread', return_value=new_fire), \
         patch.object(Fire, '_grow') as mock_grow:
        mock_grow.return_value = None  # Assume grow happens internally
        updated_fires = FireHandler.update_fires(
            current_fires=current_fires,
            agent_interactions=agent_interactions,
            max_number_fires=max_number_fires,
            timestep=1,
            wind=wind
        )
        assert len(updated_fires) == 2
        assert updated_fires[0].id == 0  # Original fire
        assert updated_fires[1].id == 1  # New fire assigned next available ID

def test_fire_extinguish_due_to_agent(fire_factory):
    """
    Test that a fire is extinguished when there is agent interaction.
    """
    fire = fire_factory(id=0)
    current_fires = [fire]
    agent_interactions = [True]  # Agent interacts with the fire
    max_number_fires = 5
    wind = np.array([0.0, 0.0])

    updated_fires = FireHandler.update_fires(
        current_fires=current_fires,
        agent_interactions=agent_interactions,
        max_number_fires=max_number_fires,
        timestep=1,
        wind=wind
    )
    # Fire should be extinguished
    assert len(updated_fires) == 0

def test_firehandler_prioritizes_fires_when_max_reached(fire_factory):
    """
    Test that FireHandler prioritizes fires correctly when max_number_fires is reached.
    """
    # Common parameters for spread fires
    spread_params = {
        'is_spread': True,
        'spread_radius_multiplier': 1.2,
        'spread_intensity_multiplier': 0.8,
        'spread_min_radius': 0.5,
        'spread_min_threshold_intensity': 0.2,
        'is_grow': False
    }
    
    # Create multiple fires with different radii
    fire_radii = [2.0, 1.5, 1.0]
    current_fires = []
    for idx, radius in enumerate(fire_radii):
        fire = fire_factory(
            id=idx,
            radius=radius,
            **spread_params
        )
        current_fires.append(fire)
    
    agent_interactions = [False] * len(current_fires)
    max_number_fires = 4  # Only one new fire can be added
    wind = np.array([0.1, 0.05])
    
    # Mock new fires spawned from each fire
    new_fires = []
    for fire in current_fires:
        new_fire = fire_factory(
            id=-1,
            initialization_time=1,
            position=fire.position + wind,
            intensity=0,
            radius=1.2,
            **spread_params
        )
        new_fires.append(new_fire)
    
    with patch.object(Fire, '_spread', side_effect=new_fires), \
         patch.object(Fire, '_grow'):
        updated_fires = FireHandler.update_fires(
            current_fires=current_fires,
            agent_interactions=agent_interactions,
            max_number_fires=max_number_fires,
            timestep=1,
            wind=wind
        )
        # Only the highest priority new fire (from fire1) should be added
        assert len(updated_fires) == 4
        ids = [fire.id for fire in updated_fires]
        assert 0 in ids
        assert 1 in ids
        assert 2 in ids
        assert 3 in ids  # New fire assigned id=3 (from fire1)
        # Ensure that only one new fire is added due to max_number_fires constraint

def test_intensity_clamping_in_growth(fire_factory):
    """
    Test that intensity is correctly clamped during growth.
    """
    fire = fire_factory(
        is_grow=True,
        grow_intensity_multiplier=1.1,
        grow_probability=1.0,  # Ensure growth happens
        grow_radius_multiplier=1.1,
        intensity=0.95,
        max_intensity=1.0,
        min_intensity=0.0
    )
    fire._grow()
    # Intensity should be clamped to max_intensity
    assert fire.intensity == 1.0  # Clamped to max_intensity

def test_fire_spread_conditions(fire_factory):
    """
    Test that a fire does not spread if conditions are not met.
    """
    fire = fire_factory(
        is_spread=True,
        spread_radius_multiplier=1.2,
        spread_intensity_multiplier=0.8,
        spread_min_radius=1.0,
        spread_min_threshold_intensity=0.5,
        intensity=0.4,  # Below threshold
        radius=0.9,     # Below threshold
        is_grow=False
    )
    with patch.object(Fire, '_spread') as mock_spread:
        mock_spread.return_value = None
        updated_fire, new_fire = fire._update(agent_interaction=False, timestep=1, wind=None)
        assert updated_fire == fire
        assert new_fire is None
        mock_spread.assert_called_once()

def test_fire_does_not_grow_when_disabled(fire_factory):
    """
    Test that a fire does not grow if is_grow is False.
    """
    fire = fire_factory(
        is_grow=False,
        intensity=0.5,
        radius=1.0
    )
    with patch('numpy.random.random') as mock_random:
        fire._grow()
        # Ensure random.random() was not called
        mock_random.assert_not_called()
        # Intensity and radius should remain unchanged
        assert fire.intensity == 0.5
        assert fire.radius == 1.0

# Copyright 2026 Christopher Newport University
# Capable Humanitarian Robotics and Intelligent Systems Lab (CHRISLAB)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the trajectory validation helpers."""

from types import SimpleNamespace

import pytest

from flexible_drones_tools.trajectories.planners import validation
from flexible_drones_tools.trajectories.planners.errors import TrajectoryPlanningError


def _point(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def test_continuity_difference_uses_local_real_time_coefficients():
    segments = [
        {'T': 2.0, 'c_x': [0.0, 0.0, 1.0]},
        {'T': 3.0, 'c_x': [4.0, 4.0, 0.0]},
    ]

    pos_diff, pos_left, pos_right = validation.continuity_difference(segments, 'c_x', 0, 0)
    vel_diff, vel_left, vel_right = validation.continuity_difference(segments, 'c_x', 0, 1)

    assert pos_left == pytest.approx(4.0)
    assert pos_right == pytest.approx(4.0)
    assert pos_diff == pytest.approx(0.0)
    assert vel_left == pytest.approx(4.0)
    assert vel_right == pytest.approx(4.0)
    assert vel_diff == pytest.approx(0.0)


def test_validate_boundary_constraints_accepts_exact_endpoints():
    segments = [
        {
            'T': 1.0,
            'c_x': [1.0, 0.0, 0.0],
            'c_y': [2.0, 0.0, 0.0],
            'c_z': [3.0, 0.0, 0.0],
        },
    ]
    start = _point(1.0, 2.0, 3.0)
    end = _point(1.0, 2.0, 3.0)
    zero_velocity = _point()

    validation.validate_boundary_constraints(segments, start, end, zero_velocity, zero_velocity)


def test_validate_boundary_constraints_rejects_endpoint_drift():
    segments = [
        {
            'T': 1.0,
            'c_x': [1.001, 0.0, 0.0],
            'c_y': [2.0, 0.0, 0.0],
            'c_z': [3.0, 0.0, 0.0],
        },
    ]
    start = _point(1.0, 2.0, 3.0)
    end = _point(1.0, 2.0, 3.0)
    zero_velocity = _point()

    with pytest.raises(TrajectoryPlanningError, match='start position.x'):
        validation.validate_boundary_constraints(segments, start, end, zero_velocity, zero_velocity)


def test_validate_obstacle_clearance_rejects_sampled_intrusion():
    segments = [
        {
            'T': 1.0,
            'c_x': [0.0],
            'c_y': [3.05],
            'c_z': [1.0],
        },
    ]
    obstacles = [{'type': 'cylinder', 'x': 0.0, 'y': 2.5, 'radius': 0.6, 'height': 3.0}]

    with pytest.raises(TrajectoryPlanningError, match='clearance=-0.0500 m') as exc_info:
        validation.validate_obstacle_clearance(segments, obstacles)
    message = str(exc_info.value)
    assert 'sample=0/204' in message
    assert 's=0.0000' in message
    assert 'local_t=0.0000s' in message
    assert 'global_t=0.0000s' in message
    assert 'point=(0.0000, 3.0500, 1.0000)' in message


def test_validate_obstacle_clearance_allows_finite_cylinder_overflight():
    segments = [
        {
            'T': 1.0,
            'c_x': [0.0],
            'c_y': [3.05],
            'c_z': [2.0],
        },
    ]
    obstacles = [{'type': 'cylinder', 'x': 0.0, 'y': 2.5, 'radius': 0.6, 'height': 1.0}]

    validation.validate_obstacle_clearance(segments, obstacles)


def test_validate_kinematic_limits_accepts_within_limits():
    segments = [
        {
            'T': 1.0,
            'c_x': [0.0, 0.5],
            'c_y': [0.0],
            'c_z': [1.0],
        },
    ]
    limits = {'v_max': 1.0, 'a_max': 1.0, 'j_max': 1.0, 's_max': 1.0}

    validation.validate_kinematic_limits(segments, limits)


def test_validate_kinematic_limits_rejects_sampled_velocity_violation():
    segments = [
        {
            'T': 1.0,
            'c_x': [0.0, 1.25],
            'c_y': [0.0],
            'c_z': [1.0],
        },
    ]
    limits = {'v_max': 1.0, 'a_max': 10.0, 'j_max': 10.0, 's_max': 10.0}

    with pytest.raises(TrajectoryPlanningError, match='velocity.x=1.25') as exc_info:
        validation.validate_kinematic_limits(segments, limits)
    message = str(exc_info.value)
    assert 'sample=0/204' in message
    assert 'global_t=0.0000s' in message
    assert 'point=(0.0000, 0.0000, 1.0000)' in message
    assert 'velocity_vector=(1.25, 0, 0)' in message


def test_validate_position_bounds_accepts_in_bounds_samples():
    segments = [
        {
            'T': 1.0,
            'c_x': [0.0, 0.5],
            'c_y': [0.0],
            'c_z': [1.0],
        },
    ]
    bounds = {'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0, 'z_min': 0.25, 'z_max': 2.5}

    validation.validate_position_bounds(segments, bounds)


def test_validate_position_bounds_rejects_sampled_position_violation():
    segments = [
        {
            'T': 1.0,
            'c_x': [0.0, 1.25],
            'c_y': [0.0],
            'c_z': [1.0],
        },
    ]
    bounds = {'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0, 'z_min': 0.25, 'z_max': 2.5}

    with pytest.raises(TrajectoryPlanningError, match='position.x=1.25') as exc_info:
        validation.validate_position_bounds(segments, bounds)
    message = str(exc_info.value)
    assert 'sample=204/204' in message
    assert 's=1.0000' in message
    assert 'local_t=1.0000s' in message
    assert 'global_t=1.0000s' in message
    assert 'point=(1.2500, 0.0000, 1.0000)' in message


def test_validate_continuity_rejects_junction_jump():
    segments = [
        {'T': 1.0, 'c_x': [0.0, 1.0], 'c_y': [0.0], 'c_z': [1.0]},
        {'T': 1.0, 'c_x': [2.0, 0.0], 'c_y': [0.0], 'c_z': [1.0]},
    ]

    with pytest.raises(TrajectoryPlanningError, match='Continuity validation failed') as exc_info:
        validation.validate_continuity(segments)
    message = str(exc_info.value)
    assert 'junction_t=1.0000s' in message
    assert 'point=(1.0000, 0.0000, 1.0000)' in message


def test_seed_continuity_tolerances_are_looser_than_shippable():
    # A small position-only junction jump (1e-3) that exceeds the shippable gate but
    # not the loose seed gate. This locks in the solver/validation tolerance coupling:
    # the warm-start seed (solved at looser IPOPT tolerances) must not be discarded
    # over solver-noise-level violations the constrained stage will re-impose anyway.
    segments = [
        {'T': 1.0, 'c_x': [0.0, 1.0], 'c_y': [0.0], 'c_z': [1.0]},
        {'T': 1.0, 'c_x': [1.001, 1.0], 'c_y': [0.0], 'c_z': [1.0]},
    ]

    assert validation.SEED_CONTINUITY_TOLERANCES[0] > validation.CONTINUITY_TOLERANCES[0]

    with pytest.raises(TrajectoryPlanningError, match='Continuity validation failed'):
        validation.validate_continuity(segments)

    # Same trajectory passes under the loose seed tolerances.
    validation.validate_continuity(
        segments, tolerances=validation.SEED_CONTINUITY_TOLERANCES)


def test_warn_on_discontinuity_logs_for_jumps():
    warnings = []
    logger = SimpleNamespace(warning=warnings.append)
    # Position discontinuous across the junction: seg0 ends at 1.0, seg1 starts at 5.0.
    segments = [
        {'T': 1.0, 'c_x': [0.0, 1.0], 'c_y': [0.0], 'c_z': [0.0]},
        {'T': 1.0, 'c_x': [5.0, 0.0], 'c_y': [0.0], 'c_z': [0.0]},
    ]
    validation.warn_on_discontinuity(segments, 0, 1e-6, logger)
    assert any('Discontinuity detected in c_x' in w for w in warnings)

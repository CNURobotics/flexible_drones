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

import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.modules.setdefault('casadi', SimpleNamespace())

import flexible_drones_tools.trajectories.planners.casadi_obstacle_planner as planner_module  # noqa: E402
from flexible_drones_tools.trajectories.planners.casadi_obstacle_planner import CasadiObstaclePlanner  # noqa: E402
from flexible_drones_tools.trajectories.utilities.io import load_trajectory_csv  # noqa: E402


def test_default_seed_strategy_is_obstacle_boundary_subdivision_with_six_segments():
    planner = CasadiObstaclePlanner()

    assert planner.seed_strategy == 'obstacle_boundary_subdivision'
    assert planner.n_segments == 6


def test_unknown_seed_strategy_is_rejected():
    with pytest.raises(ValueError, match='Unknown seed_strategy'):
        CasadiObstaclePlanner(seed_strategy='not_a_seed')


def test_dubins_seed_receives_radius_from_velocity_and_yaw_limit(monkeypatch):
    planner = CasadiObstaclePlanner(seed_strategy='dubins')
    start = {
        'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'velocity': {'x': 1.0, 'y': 0.0, 'z': 0.0},
    }
    goal = {
        'position': {'x': 4.0, 'y': 0.0, 'z': 0.0},
        'velocity': {'x': 1.0, 'y': 0.0, 'z': 0.0},
    }
    captured = {}

    def fake_seed(*args, **kwargs):
        captured.update(kwargs)
        return planner_module.seeding.straight_line_seed(*args[:6])

    monkeypatch.setitem(planner_module.seeding.SEED_STRATEGIES, 'dubins', fake_seed)

    planner.build_seed(
        start,
        goal,
        obstacles=[],
        limits={
            'v_max': 1.2,
            'a_max': 4.0,
            'j_max': 10.0,
            's_max': 50.0,
            'yaw_rate_max': 1.5,
            'yaw_accel_max': 4.0,
        },
        n_segments=4,
    )

    assert captured['turning_radius'] == pytest.approx(0.8)
    assert captured['return_diagnostics'] is True


def test_dubins_seed_receives_start_and_goal_up_axes(monkeypatch):
    planner = CasadiObstaclePlanner(seed_strategy='dubins')
    start = {
        'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'velocity': {'x': 1.0, 'y': 0.0, 'z': 0.0},
        'up_axis': np.asarray([0.0, 0.0, 1.0]),
    }
    goal = {
        'position': {'x': 4.0, 'y': 0.0, 'z': 0.0},
        'velocity': {'x': 1.0, 'y': 0.0, 'z': 0.0},
        'up_axis': np.asarray([0.0, 1.0, 1.0]),
    }
    captured = {}

    def fake_seed(*args, **kwargs):
        captured.update(kwargs)
        return planner_module.seeding.straight_line_seed(*args[:6])

    monkeypatch.setitem(planner_module.seeding.SEED_STRATEGIES, 'dubins', fake_seed)

    planner.build_seed(start, goal, obstacles=[], n_segments=4)

    assert captured['z_axis'] == pytest.approx([0.0, 0.0, 1.0])
    assert captured['goal_z_axis'] == pytest.approx([0.0, 1.0, 1.0])


def test_unconstrained_optimizer_seed_uses_obstacle_free_plan(monkeypatch):
    planner = CasadiObstaclePlanner(seed_strategy='unconstrained_optimizer')
    start = {
        'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    }
    goal = {
        'position': {'x': 3.0, 'y': 0.0, 'z': 0.0},
        'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    }
    captured = {}

    def fake_plan(start_arg, goal_arg, obstacles=None, initial_segments=None, **kwargs):
        captured['start'] = start_arg
        captured['goal'] = goal_arg
        captured['obstacles'] = obstacles
        captured['initial_segments'] = initial_segments
        captured['kwargs'] = kwargs
        return [
            {
                'T': 2.0,
                'c_x': np.asarray([1.0, 2.0, 3.0]),
                'c_y': np.asarray([4.0, 5.0, 6.0]),
                'c_z': np.asarray([7.0, 8.0, 9.0]),
            },
        ]

    monkeypatch.setattr(planner, 'plan', fake_plan)

    seed = planner.build_seed(
        start,
        goal,
        obstacles=[{'type': 'cylinder', 'x': 1.0, 'y': 0.0, 'radius': 0.25}],
        n_segments=6,
    )

    assert captured['obstacles'] == []
    assert len(captured['initial_segments']) == 3
    assert len(seed) == 1
    assert seed[0]['T'] == pytest.approx(2.0)
    assert seed[0]['c_x'][:3] == pytest.approx([1.0, 4.0, 12.0])
    assert planner.last_seed_diagnostics['strategy'] == 'unconstrained_optimizer'


def test_build_seed_returns_three_segments_when_no_obstacle_adjustment_is_needed():
    planner = CasadiObstaclePlanner()
    start = {
        'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    }
    goal = {
        'position': {'x': 6.0, 'y': 0.0, 'z': 0.0},
        'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    }

    seed = planner.build_seed(start, goal, obstacles=[], n_segments=6)

    assert len(seed) == 3
    assert planner.n_segments == 3
    assert sum(segment['T'] for segment in seed) == pytest.approx(8.0)


def test_plan_requires_precomputed_initial_segments():
    planner = CasadiObstaclePlanner()
    start = {
        'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    }
    goal = {
        'position': {'x': 1.0, 'y': 0.0, 'z': 0.0},
        'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    }

    with pytest.raises(ValueError, match='requires initial_segments'):
        planner.plan(start, goal, obstacles=[])


def test_plan_scales_optimizer_kinematic_caps_but_keeps_validation_limits(monkeypatch):
    planner = CasadiObstaclePlanner(
        kinematic_constraint_scale=0.8,
        limits={
            'v_max': 2.0,
            'a_max': 4.0,
            'j_max': 8.0,
            's_max': 16.0,
            'yaw_rate_max': 0.0,
            'yaw_accel_max': 0.0,
        },
    )
    start = {
        'position': {'x': 0.0, 'y': 0.0, 'z': 1.0},
        'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    }
    goal = {
        'position': {'x': 1.0, 'y': 0.0, 'z': 1.0},
        'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    }
    captured = {}

    def fake_poly_segment_time_optimal(
        _A_data, _B_data, _p_lim, v_lim, a_lim, j_lim, s_lim, *_args, **_kwargs,
    ):
        captured['v_lim'] = v_lim
        captured['a_lim'] = a_lim
        captured['j_lim'] = j_lim
        captured['s_lim'] = s_lim
        return [
            {
                'T': 1.0,
                'c_x': np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                'c_y': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                'c_z': np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
        ]

    monkeypatch.setattr(planner, 'poly_segment_time_optimal', fake_poly_segment_time_optimal)
    monkeypatch.setattr(planner, '_find_obstacle_refinements', lambda *args, **kwargs: {})
    monkeypatch.setattr(planner, '_find_yaw_rate_refinements', lambda *args, **kwargs: {})
    monkeypatch.setattr(planner, '_kinematic_overshoot_segments', lambda *args, **kwargs: set())

    planner.plan(
        start,
        goal,
        obstacles=[],
        initial_segments=[{'T': 1.0, 'c_x': np.array([0.0]), 'c_y': np.array([0.0]), 'c_z': np.array([1.0])}],
    )

    assert captured['v_lim'] == pytest.approx((-1.6, 1.6))
    assert captured['a_lim'] == pytest.approx((-3.2, 3.2))
    assert captured['j_lim'] == pytest.approx((-6.4, 6.4))
    assert captured['s_lim'] == pytest.approx((-12.8, 12.8))
    assert planner.v_max == pytest.approx(2.0)
    assert planner.s_max == pytest.approx(16.0)


def test_cylinder_clearance_constraint_respects_height(monkeypatch):
    monkeypatch.setattr(
        planner_module,
        'ca',
        SimpleNamespace(if_else=lambda condition, when_true, when_false: when_true if condition else when_false),
    )
    obstacle = {'type': 'cylinder', 'x': 0.0, 'y': 0.0, 'radius': 0.5, 'height': 1.0}

    below_top = CasadiObstaclePlanner.casadi_cylinder_clearance_constraint(0.0, 0.0, 0.5, obstacle)
    above_top = CasadiObstaclePlanner.casadi_cylinder_clearance_constraint(0.0, 0.0, 1.5, obstacle)

    assert below_top == pytest.approx(-0.25)
    assert above_top == pytest.approx(0.0)


def test_failed_solver_artifacts_include_stats_and_candidate_segments(tmp_path):
    segments = [
        {
            'T': 1.25,
            'c_x': np.array([0.0, 1.0]),
            'c_y': np.array([2.0, 3.0]),
            'c_z': np.array([4.0, 5.0]),
        },
    ]

    artifact_dir = CasadiObstaclePlanner._write_failed_solver_artifacts(
        return_status='Infeasible_Problem_Detected',
        solver_stats={
            'return_status': 'Infeasible_Problem_Detected',
            'iter_count': np.int64(7),
            'nested': {'residual': np.array([1.0, 2.0])},
        },
        segments=segments,
        timing={'solve_sec': 0.5},
        artifact_root=tmp_path,
    )

    stats = json.loads((artifact_dir / 'solver_stats.json').read_text(encoding='utf-8'))
    assert stats['return_status'] == 'Infeasible_Problem_Detected'
    assert stats['accepted_statuses'] == ['Solve_Succeeded']
    assert stats['solver_stats']['iter_count'] == 7
    assert stats['solver_stats']['nested']['residual'] == [1.0, 2.0]

    trajectory_path = artifact_dir / 'candidate_trajectory.csv'
    durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs = load_trajectory_csv(trajectory_path)
    assert durations == [1.25]
    assert x_coeffs[0] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert y_coeffs[0] == [2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert z_coeffs[0] == [4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert yaw_coeffs[0] == [0.0] * 8

    with (artifact_dir / 'candidate_segments_debug.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ['segment', 'duration', 'axis', 'c0', 'c1']
    assert rows[1] == ['0', '1.25', 'c_x', '0.0', '1.0']
    assert rows[2] == ['0', '1.25', 'c_y', '2.0', '3.0']
    assert rows[3] == ['0', '1.25', 'c_z', '4.0', '5.0']


def test_obstacle_refinement_targets_close_small_obstacle_without_far_oversampling():
    planner = CasadiObstaclePlanner(
        obstacle_refine_check_samples=101,
        obstacle_refine_margin=0.03,
        obstacle_refine_min_spacing=0.01,
        obstacle_refine_max_spacing=0.25,
    )
    segments = [
        {
            'T': 1.0,
            'c_x': np.array([-1.0, 2.0]),
            'c_y': np.array([0.0]),
            'c_z': np.array([1.0]),
        },
    ]
    obstacles = [
        {'type': 'cylinder', 'x': 0.0, 'y': 0.02, 'radius': 0.03, 'height': 2.0},
        {'type': 'cylinder', 'x': 0.0, 'y': 1.0, 'radius': 0.6, 'height': 2.0},
    ]

    refinements = planner._find_obstacle_refinements(segments, obstacles)

    assert set(refinements) == {(0, 0)}
    samples = refinements[(0, 0)]
    assert len(samples) > 3
    assert min(samples) < 0.5 < max(samples)
    assert all(0.0 <= sample <= 1.0 for sample in samples)


def test_obstacle_refinement_stays_empty_for_clear_path():
    planner = CasadiObstaclePlanner(obstacle_refine_check_samples=51)
    segments = [
        {
            'T': 1.0,
            'c_x': np.array([-1.0, 2.0]),
            'c_y': np.array([0.0]),
            'c_z': np.array([1.0]),
        },
    ]
    obstacles = [
        {'type': 'cylinder', 'x': 0.0, 'y': 1.0, 'radius': 0.03, 'height': 2.0},
    ]

    assert planner._find_obstacle_refinements(segments, obstacles) == {}


def test_kinematic_overshoot_flags_dense_snap_segment():
    planner = CasadiObstaclePlanner(obstacle_refine_check_samples=101)
    planner.s_max = 50.0
    # snap(s) = -50.5 has no position/velocity/accel/jerk boundary drama, but should
    # still flag the segment for a dense kinematic re-solve because it exceeds snap.
    overshoot_segment = {
        'T': 1.0,
        'c_x': np.array([0.0, 0.0, 0.0, 0.0, -50.5 / 24.0]),
        'c_y': np.array([0.0]),
        'c_z': np.array([1.0]),
    }
    # A flat, within-limit segment must not be flagged.
    clear_segment = {
        'T': 1.0,
        'c_x': np.array([0.0]),
        'c_y': np.array([0.0]),
        'c_z': np.array([1.0]),
    }

    flagged = planner._kinematic_overshoot_segments([clear_segment, overshoot_segment])

    assert flagged == {1}


def test_kinematic_time_dilation_factor_targets_snap_overshoot():
    planner = CasadiObstaclePlanner(obstacle_refine_check_samples=101)
    planner.v_max = 100.0
    planner.a_max = 100.0
    planner.j_max = 100.0
    planner.s_max = 50.0
    # snap(s) = 24*a is a constant 50.5 (> 50), while v/a/jerk stay well under their
    # generous caps, so only the snap limit drives the dilation factor.
    coeff = 50.5 / 24.0
    overshoot_segment = {
        'T': 1.0,
        'c_x': np.array([0.0, 0.0, 0.0, 0.0, coeff]),
        'c_y': np.array([0.0]),
        'c_z': np.array([1.0]),
    }
    clear_segment = {
        'T': 1.0,
        'c_x': np.array([0.0]),
        'c_y': np.array([0.0]),
        'c_z': np.array([1.0]),
    }
    segments = [clear_segment, overshoot_segment]

    factors = planner.kinematic_time_dilation_factors(segments, margin=1.0)

    assert set(factors) == {1}
    assert factors[1] == pytest.approx((50.5 / 50.0) ** 0.25, rel=1e-3)

    dilated = planner.dilate_segment_durations(segments, factors)

    # Only the offending segment's duration stretches; coefficients are untouched, so
    # endpoint positions (and thus position continuity) are preserved.
    assert dilated[0]['T'] == 1.0
    assert dilated[1]['T'] == pytest.approx(factors[1])
    assert np.array_equal(dilated[1]['c_x'], overshoot_segment['c_x'])


def test_boundary_obstacle_precheck_reports_closest_clearance(capsys):
    planner = CasadiObstaclePlanner()
    start = {'x': 0.0, 'y': -1.687, 'z': 1.5}
    target = {'x': -1.688, 'y': 0.058, 'z': 2.5}
    obstacles = [
        {'type': 'cylinder', 'x': 0.0, 'y': -2.5, 'radius': 0.6, 'height': 3.0},
        {'type': 'cylinder', 'x': -2.0, 'y': 0.0, 'radius': 0.04, 'height': 2.0},
    ]

    planner._report_boundary_obstacle_clearance(start, target, obstacles)

    captured = capsys.readouterr()
    assert 'start=(0.000, -1.687, 1.500)' in captured.out
    assert 'target=(-1.688, 0.058, 2.500)' in captured.out
    assert 'clearance=0.2130' in captured.out
    assert 'clearance=0.5000' in captured.out


def test_boundary_obstacle_precheck_rejects_boundary_inside_inflated_obstacle():
    planner = CasadiObstaclePlanner()
    start = {'x': 0.0, 'y': -1.687, 'z': 1.5}
    target = {'x': 1.0, 'y': 0.0, 'z': 1.0}
    obstacles = [
        {'type': 'cylinder', 'x': 0.0, 'y': -2.5, 'radius': 0.9, 'height': 3.0},
    ]

    with pytest.raises(ValueError, match='Boundary obstacle precheck failed'):
        planner._report_boundary_obstacle_clearance(start, target, obstacles)

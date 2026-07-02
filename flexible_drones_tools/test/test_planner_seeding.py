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

"""Unit tests for initial-trajectory seeding strategies."""

import numpy as np
import pytest

from flexible_drones_tools.trajectories.planners import seeding


def test_straight_line_seed_uses_total_time_and_matches_boundary_state():
    zero = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    A_data = (
        {'x': 0.0, 'y': -2.5, 'z': 1.5},
        {'x': 0.5, 'y': 0.0, 'z': 0.0},
        dict(zero),
        dict(zero),
        dict(zero),
    )
    B_data = (
        {'x': -2.5, 'y': 0.0, 'z': 2.5},
        {'x': 0.0, 'y': 0.5, 'z': 0.0},
        dict(zero),
        dict(zero),
        dict(zero),
    )

    segments = seeding.straight_line_seed(
        A_data, B_data, total_time=6.0, n_segments=2, n_coeff=8, time_min=0.4)

    assert [segment['T'] for segment in segments] == pytest.approx([3.0, 3.0])
    assert sum(segment['T'] for segment in segments) == pytest.approx(6.0)

    first_x = np.polynomial.Polynomial(segments[0]['c_x'])
    first_y = np.polynomial.Polynomial(segments[0]['c_y'])
    last_y = np.polynomial.Polynomial(segments[-1]['c_y'])
    last_z = np.polynomial.Polynomial(segments[-1]['c_z'])

    assert first_x(0.0) == pytest.approx(0.0)
    assert first_y(0.0) == pytest.approx(-2.5)
    assert first_x.deriv(1)(0.0) / segments[0]['T'] == pytest.approx(0.5)
    assert last_y(1.0) == pytest.approx(0.0)
    assert last_z(1.0) == pytest.approx(2.5)
    assert last_y.deriv(1)(1.0) / segments[-1]['T'] == pytest.approx(0.5)


def test_total_time_floored_by_time_min():
    zero = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    data = ({'x': 0.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))
    # Requested total_time below time_min * n_segments is floored.
    segments = seeding.straight_line_seed(
        data, data, total_time=0.1, n_segments=2, n_coeff=8, time_min=0.4)
    assert sum(segment['T'] for segment in segments) == pytest.approx(0.8)


def test_pad_coefficients_right_pads_with_zeros():
    assert seeding.pad_coefficients([1.0, 2.0], 4) == [1.0, 2.0, 0.0, 0.0]


def _poly_value(segment, axis, s, derivative_order=0):
    polynomial = np.polynomial.Polynomial(segment[f'c_{axis}'])
    value = polynomial.deriv(derivative_order)(s)
    if derivative_order:
        value /= float(segment['T']) ** derivative_order
    return value


def test_velocity_biased_chord_seed_blends_projected_velocity_points():
    zero = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    A_data = (
        {'x': 0.0, 'y': 0.0, 'z': 0.0},
        {'x': 0.0, 'y': 2.0, 'z': 0.0},
        dict(zero),
        dict(zero),
        dict(zero),
    )
    B_data = (
        {'x': 6.0, 'y': 0.0, 'z': 0.0},
        {'x': 0.0, 'y': -2.0, 'z': 0.0},
        dict(zero),
        dict(zero),
        dict(zero),
    )

    segments = seeding.velocity_biased_chord_seed(
        A_data, B_data, total_time=9.0, n_segments=3, n_coeff=8, time_min=0.4, vmax=2.0)

    assert [segment['T'] for segment in segments] == pytest.approx([3.0, 3.0, 3.0])
    # A=(2,0), A'=(0,2), B=(4,0), B'=(6,2); 50/50 blend gives pA=(1,1), pB=(5,1).
    assert _poly_value(segments[0], 'x', 1.0) == pytest.approx(1.0)
    assert _poly_value(segments[0], 'y', 1.0) == pytest.approx(1.0)
    assert _poly_value(segments[1], 'x', 0.0) == pytest.approx(1.0)
    assert _poly_value(segments[1], 'y', 0.0) == pytest.approx(1.0)
    assert _poly_value(segments[1], 'x', 1.0) == pytest.approx(5.0)
    assert _poly_value(segments[1], 'y', 1.0) == pytest.approx(1.0)
    assert _poly_value(segments[2], 'x', 0.0) == pytest.approx(5.0)
    assert _poly_value(segments[2], 'y', 0.0) == pytest.approx(1.0)

    unit_scale = 1.0 / np.sqrt(26.0)
    assert _poly_value(segments[0], 'x', 1.0, derivative_order=1) == pytest.approx(5.0 * unit_scale)
    assert _poly_value(segments[0], 'y', 1.0, derivative_order=1) == pytest.approx(1.0 * unit_scale)
    assert _poly_value(segments[1], 'x', 1.0, derivative_order=1) == pytest.approx(5.0 * unit_scale)
    assert _poly_value(segments[1], 'y', 1.0, derivative_order=1) == pytest.approx(-1.0 * unit_scale)


def test_velocity_biased_chord_seed_uses_straight_thirds_when_boundary_velocity_is_zero():
    zero = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    A_data = ({'x': 0.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))
    B_data = ({'x': 6.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))

    segments = seeding.velocity_biased_chord_seed(
        A_data, B_data, total_time=0.1, n_segments=3, n_coeff=8, time_min=0.4, vmax=2.0)

    assert sum(segment['T'] for segment in segments) == pytest.approx(1.2)
    assert _poly_value(segments[0], 'x', 1.0) == pytest.approx(2.0)
    assert _poly_value(segments[1], 'x', 1.0) == pytest.approx(4.0)
    assert _poly_value(segments[0], 'x', 1.0, derivative_order=1) == pytest.approx(1.0)
    assert _poly_value(segments[1], 'x', 1.0, derivative_order=1) == pytest.approx(1.0)


def test_velocity_biased_chord_seed_is_registered():
    assert seeding.SEED_STRATEGIES['velocity_biased_chord'] is seeding.velocity_biased_chord_seed


def test_obstacle_boundary_subdivision_seed_moves_intersecting_waypoint_and_subdivides():
    zero = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    A_data = ({'x': 0.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))
    B_data = ({'x': 6.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))
    obstacles = [
        {'type': 'cylinder', 'x': 2.0, 'y': 0.0, 'radius': 0.4, 'height': 2.0},
    ]

    segments = seeding.obstacle_boundary_subdivision_seed(
        A_data,
        B_data,
        total_time=6.0,
        n_segments=6,
        n_coeff=8,
        time_min=0.4,
        vmax=2.0,
        obstacles=obstacles,
        clearance=0.1,
    )

    assert len(segments) == 6
    assert [segment['T'] for segment in segments] == pytest.approx([1.0] * 6)
    # The original pA was exactly at the cylinder core (2, 0, 0), so the fallback
    # direction pushes toward the world origin to x=1.5.
    assert _poly_value(segments[1], 'x', 1.0) == pytest.approx(1.5)
    assert _poly_value(segments[1], 'y', 1.0) == pytest.approx(0.0)
    # Each original 3-leg segment is divided once.
    assert _poly_value(segments[0], 'x', 1.0) == pytest.approx(0.75)
    assert _poly_value(segments[2], 'x', 1.0) == pytest.approx(2.75)


def test_obstacle_boundary_subdivision_seed_stays_three_segments_without_moves():
    zero = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    A_data = ({'x': 0.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))
    B_data = ({'x': 6.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))

    segments, diagnostics = seeding.obstacle_boundary_subdivision_seed(
        A_data,
        B_data,
        total_time=6.0,
        n_segments=6,
        n_coeff=8,
        time_min=0.4,
        vmax=2.0,
        obstacles=[],
        return_diagnostics=True,
    )

    assert len(segments) == 3
    assert diagnostics['subdivided'] is False
    assert diagnostics['returned_segments'] == 3
    assert diagnostics['moves'] == []


def test_obstacle_boundary_subdivision_seed_reports_diagnostics():
    zero = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    A_data = ({'x': 0.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))
    B_data = ({'x': 6.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))
    obstacles = [
        {'type': 'cylinder', 'x': 2.0, 'y': 0.0, 'radius': 0.4, 'height': 2.0},
    ]

    segments, diagnostics = seeding.obstacle_boundary_subdivision_seed(
        A_data,
        B_data,
        total_time=6.0,
        n_segments=6,
        n_coeff=8,
        time_min=0.4,
        vmax=2.0,
        obstacles=obstacles,
        clearance=0.1,
        return_diagnostics=True,
    )

    assert len(segments) == 6
    assert diagnostics['subdivided'] is True
    assert diagnostics['returned_segments'] == 6
    assert diagnostics['moves'][0]['label'] == 'pA'
    assert diagnostics['moves'][0]['obstacle_index'] == 0
    assert diagnostics['moves'][0]['distance_before'] == pytest.approx(0.0)
    assert diagnostics['moves'][0]['distance_after'] == pytest.approx(0.5)


def test_obstacle_boundary_subdivision_seed_is_registered():
    assert (
        seeding.SEED_STRATEGIES['obstacle_boundary_subdivision']
        is seeding.obstacle_boundary_subdivision_seed
    )


def test_dubins_seed_uses_straight_candidate_when_tangents_align():
    zero = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    A_data = (
        {'x': 0.0, 'y': 0.0, 'z': 0.0},
        {'x': 1.0, 'y': 0.0, 'z': 0.0},
        dict(zero),
        dict(zero),
        dict(zero),
    )
    B_data = (
        {'x': 4.0, 'y': 0.0, 'z': 0.0},
        {'x': 1.0, 'y': 0.0, 'z': 0.0},
        dict(zero),
        dict(zero),
        dict(zero),
    )

    segments, diagnostics = seeding.dubins_seed(
        A_data,
        B_data,
        total_time=4.0,
        n_segments=4,
        n_coeff=8,
        time_min=0.4,
        vmax=1.0,
        turning_radius=1.0,
        return_diagnostics=True,
    )

    assert len(segments) == 4
    assert diagnostics['strategy'] == 'dubins'
    assert diagnostics['path_length'] == pytest.approx(4.0)
    assert [segment['T'] for segment in segments] == pytest.approx([1.0] * 4)
    for segment_index, segment in enumerate(segments):
        assert _poly_value(segment, 'x', 0.0) == pytest.approx(float(segment_index))
        assert _poly_value(segment, 'x', 1.0) == pytest.approx(float(segment_index + 1))
        assert _poly_value(segment, 'y', 0.0) == pytest.approx(0.0)
        assert _poly_value(segment, 'y', 1.0) == pytest.approx(0.0)


def test_dubins_path_supports_reverse_travel_direction():
    zero = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    A_data = (
        {'x': 0.0, 'y': 0.0, 'z': 0.0},
        {'x': -1.0, 'y': 0.0, 'z': 0.0},
        dict(zero),
        dict(zero),
        dict(zero),
    )
    B_data = (
        {'x': -4.0, 'y': 0.0, 'z': 0.0},
        {'x': -1.0, 'y': 0.0, 'z': 0.0},
        dict(zero),
        dict(zero),
        dict(zero),
    )

    sampled = seeding.sample_dubins_inspired_path(
        A_data, B_data, radius=1.0, sample_count=5)

    assert sampled is not None
    assert sampled['arc_length'][-1] == pytest.approx(4.0)
    assert sampled['tangents'][0] == pytest.approx([-1.0, 0.0, 0.0])
    assert sampled['tangents'][-1] == pytest.approx([-1.0, 0.0, 0.0])
    assert sampled['positions'][:, 1] == pytest.approx([0.0] * 5)


def test_dubins_path_uses_explicit_tangents_when_boundary_velocity_is_zero():
    zero = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    A_data = ({'x': 0.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))
    B_data = ({'x': 4.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))

    sampled = seeding.sample_dubins_inspired_path(
        A_data,
        B_data,
        radius=1.0,
        sample_count=7,
        start_tangent=np.asarray([0.0, 1.0, 0.0]),
        goal_tangent=np.asarray([0.0, 1.0, 0.0]),
    )

    assert sampled is not None
    assert sampled['tangents'][0] == pytest.approx([0.0, 1.0, 0.0])
    assert sampled['tangents'][-1] == pytest.approx([0.0, 1.0, 0.0])
    assert np.max(np.abs(sampled['positions'][:, 1])) > 0.1


def test_dubins_path_combines_start_and_goal_up_axes():
    zero = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    A_data = ({'x': 0.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))
    B_data = ({'x': 4.0, 'y': 0.0, 'z': 0.0}, dict(zero), dict(zero), dict(zero), dict(zero))

    sampled = seeding.sample_dubins_inspired_path(
        A_data,
        B_data,
        radius=1.0,
        sample_count=7,
        z_axis=np.asarray([0.0, 0.0, 1.0]),
        goal_z_axis=np.asarray([0.0, 1.0, 1.0]),
        start_tangent=np.asarray([1.0, 0.0, 0.0]),
        goal_tangent=np.asarray([1.0, 0.0, 0.0]),
    )

    assert sampled is not None
    assert sampled['z_axis'][1] > 0.1
    assert sampled['z_axis'][2] > 0.1


def test_dubins_seed_is_registered():
    assert seeding.SEED_STRATEGIES['dubins'] is seeding.dubins_seed

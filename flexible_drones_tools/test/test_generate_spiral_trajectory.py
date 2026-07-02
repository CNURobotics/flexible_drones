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

import importlib.util
from pathlib import Path

import numpy as np
import pytest


GENERATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / 'flexible_drones_tools'
    / 'trajectories'
    / 'generators'
    / 'generate_spiral_trajectory.py'
)
SPEC = importlib.util.spec_from_file_location('source_generate_spiral_trajectory', GENERATOR_PATH)
generate_spiral_trajectory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_spiral_trajectory)


def test_spiral_filename_encodes_radius_and_height_to_two_decimal_places():
    assert generate_spiral_trajectory.spiral_filename(
        3.0,
        0.75,
        2.5,
    ) == 'spiral_trajectory_r3_00m_z0_75m_to_2_50m.csv'
    assert generate_spiral_trajectory.spiral_filename(
        1.234,
        0.456,
        7.891,
    ) == 'spiral_trajectory_r1_23m_z0_46m_to_7_89m.csv'


def test_spiral_uses_configurable_radius_and_height_endpoints():
    ramp_duration = 2.0
    radius = 3.0
    z_floor = 0.75
    z_ceil = 2.5
    t, x, y, z, _ = generate_spiral_trajectory.GenerateSpiralTrajectory(
        ramp_duration=ramp_duration,
        radius=radius,
        z_floor=z_floor,
        z_ceil=z_ceil,
    ).generate_trajectory(num_points=501)

    top_indices = np.flatnonzero(z == np.max(z))
    top_start = top_indices[0]

    assert t[0] == pytest.approx(0.0)
    assert t[top_start] > ramp_duration

    assert x[0] == pytest.approx(0.0)
    assert y[0] == pytest.approx(0.0)
    assert z[0] == pytest.approx(z_floor)
    assert x[top_start] == pytest.approx(radius)
    assert y[top_start] == pytest.approx(0.0, abs=1e-12)
    assert z[top_start] == pytest.approx(z_ceil)
    assert x[-1] == pytest.approx(0.0)
    assert y[-1] == pytest.approx(0.0)
    assert z[-1] == pytest.approx(z_floor)


def test_spiral_accelerates_and_decelerates_xyz_on_both_legs():
    t, x, y, z, _ = generate_spiral_trajectory.GenerateSpiralTrajectory(
        max_vx=0.75,
        ramp_duration=2.0,
    ).generate_trajectory(num_points=1001)
    speed = np.linalg.norm(
        np.column_stack((
            np.gradient(x, t),
            np.gradient(y, t),
            np.gradient(z, t),
        )),
        axis=1,
    )
    top_indices = np.flatnonzero(z == np.max(z))
    top_start = top_indices[0]
    descent_start = top_indices[-1] + 1
    first_cruise = top_start // 2
    second_cruise = descent_start + (len(t) - descent_start) // 2

    assert speed[0] < speed[first_cruise]
    assert speed[top_start] < speed[first_cruise]
    assert speed[descent_start - 1] < speed[second_cruise]
    assert speed[-1] < speed[second_cruise]
    assert np.gradient(z, t)[0] < np.gradient(z, t)[first_cruise]
    assert abs(np.gradient(z, t)[top_start]) < abs(np.gradient(z, t)[first_cruise])


def test_spiral_rotates_in_place_at_top_before_descending():
    yaw_turn_duration = 1.0
    t, x, y, z, yaw = generate_spiral_trajectory.GenerateSpiralTrajectory(
    ).generate_trajectory(num_points=1001)
    top_indices = np.flatnonzero(z == np.max(z))
    top_start = top_indices[0]
    top_end = top_indices[-1]
    yaw_change = np.arctan2(
        np.sin(yaw[top_end] - yaw[top_start]),
        np.cos(yaw[top_end] - yaw[top_start]),
    )

    assert t[top_end] - t[top_start] == pytest.approx(yaw_turn_duration)
    assert x[top_indices] == pytest.approx(np.full_like(top_indices, x[top_start], dtype=float))
    assert y[top_indices] == pytest.approx(np.full_like(top_indices, y[top_start], dtype=float))
    assert z[top_indices] == pytest.approx(np.full_like(top_indices, z[top_start], dtype=float))
    assert abs(yaw_change) == pytest.approx(np.pi)


def test_spiral_limits_body_velocity_to_max_vx():
    max_vx = 0.75
    t, x, y, z, _ = generate_spiral_trajectory.GenerateSpiralTrajectory(
        max_vx=max_vx,
        ramp_duration=1.0,
    ).generate_trajectory(num_points=4000)
    speed = np.linalg.norm(
        np.column_stack((
            np.gradient(x, t),
            np.gradient(y, t),
            np.gradient(z, t),
        )),
        axis=1,
    )

    assert np.max(speed) == pytest.approx(max_vx, rel=0.03)
    assert np.max(speed) <= max_vx * 1.04


def test_spiral_limits_angular_rate_to_max_wz():
    max_wz = np.pi / 2
    z_floor = 0.75
    z_ceil = 2.5
    t, _, _, z, _ = generate_spiral_trajectory.GenerateSpiralTrajectory(
        max_vx=2.0,
        max_wz=max_wz,
        z_floor=z_floor,
        z_ceil=z_ceil,
        ramp_duration=0.0,
        yaw_turn_duration=0.0,
    ).generate_trajectory(num_points=4000)
    top_start = np.flatnonzero(z == np.max(z))[0]
    theta = (
        (z[:top_start + 1] - z_floor)
        / (z_ceil - z_floor)
        * generate_spiral_trajectory.DEFAULT_MAX_ANGLE
    )
    theta_dot = np.gradient(theta, t[:top_start + 1])

    assert np.max(theta_dot) == pytest.approx(max_wz, rel=0.03)
    assert np.max(theta_dot) <= max_wz * 1.04


def test_spiral_yaw_follows_xy_tangent():
    radius = 3.0
    z_floor = 0.75
    z_ceil = 2.5
    _, x, y, z, yaw = generate_spiral_trajectory.GenerateSpiralTrajectory(
        radius=radius,
        z_floor=z_floor,
        z_ceil=z_ceil,
        ramp_duration=1.0,
    ).generate_trajectory(num_points=1001)

    outbound_index = 250
    top_indices = np.flatnonzero(z == np.max(z))
    inbound_indices = np.arange(top_indices[-1] + 1, len(yaw))
    inbound_index = inbound_indices[np.argmin(np.abs(z[inbound_indices] - z[outbound_index]))]
    progress = (z[outbound_index] - z_floor) / (z_ceil - z_floor)
    theta = progress * generate_spiral_trajectory.DEFAULT_MAX_ANGLE
    radius_at_theta = np.hypot(x[outbound_index], y[outbound_index])
    dr_dtheta = radius / generate_spiral_trajectory.DEFAULT_MAX_ANGLE
    expected_outbound_yaw = np.arctan2(
        dr_dtheta * np.sin(theta) + radius_at_theta * np.cos(theta),
        dr_dtheta * np.cos(theta) - radius_at_theta * np.sin(theta),
    )

    outbound_error = np.arctan2(
        np.sin(yaw[outbound_index] - expected_outbound_yaw),
        np.cos(yaw[outbound_index] - expected_outbound_yaw),
    )
    inbound_error = np.arctan2(
        np.sin(yaw[inbound_index] - expected_outbound_yaw - np.pi),
        np.cos(yaw[inbound_index] - expected_outbound_yaw - np.pi),
    )

    assert outbound_error == pytest.approx(0.0, abs=1e-3)
    assert inbound_error == pytest.approx(0.0, abs=1e-3)
    assert np.ptp(yaw) > np.pi


def test_spiral_rejects_invalid_timing():
    with pytest.raises(ValueError, match='max_vx must be greater than 0.0'):
        generate_spiral_trajectory.GenerateSpiralTrajectory(max_vx=0.0)

    with pytest.raises(ValueError, match='max_wz must be greater than 0.0'):
        generate_spiral_trajectory.GenerateSpiralTrajectory(max_wz=0.0)

    with pytest.raises(ValueError, match='radius must be greater than 0.0'):
        generate_spiral_trajectory.GenerateSpiralTrajectory(radius=0.0)

    with pytest.raises(ValueError, match='ramp_duration must be greater than or equal'):
        generate_spiral_trajectory.GenerateSpiralTrajectory(ramp_duration=-0.1)

    with pytest.raises(ValueError, match='yaw_turn_duration must be greater than or equal'):
        generate_spiral_trajectory.GenerateSpiralTrajectory(yaw_turn_duration=-0.1)

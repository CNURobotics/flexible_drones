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

"""Unit tests for shared geometry helpers."""

import math

from flexible_drones_core.geometry import (
    enu_flu_rpy_to_ned_frd,
    enu_xyz_to_ned_xyz,
    enu_yaw_to_ned,
    euler_from_quaternion_xyzw,
    ned_frd_angular_rates_to_enu_flu,
    ned_frd_rpy_to_enu_flu,
    ned_xyz_to_enu_xyz,
    normalize_angle,
    quaternion_from_euler_xyzw,
    yaw_from_quaternion_xyzw,
)

import pytest


def test_quaternion_from_euler_identity():
    """Convert zero RPY to identity quaternion."""
    assert quaternion_from_euler_xyzw(0.0, 0.0, 0.0) == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_euler_quaternion_round_trip():
    """Round-trip representative RPY values through ROS xyzw quaternion order."""
    roll, pitch, yaw = 0.2, -0.3, 1.1
    q = quaternion_from_euler_xyzw(roll, pitch, yaw)

    result = euler_from_quaternion_xyzw(q)

    assert result == pytest.approx((roll, pitch, yaw))


def test_yaw_from_quaternion_xyzw_extracts_ros_yaw():
    """Extract yaw from a ROS xyzw quaternion."""
    q = quaternion_from_euler_xyzw(0.0, 0.0, math.pi / 2.0)

    assert yaw_from_quaternion_xyzw(q) == pytest.approx(math.pi / 2.0)


def test_enu_yaw_to_ned_rotates_origin_and_flips_sign():
    """Convert ENU yaw from east/CCW to NED yaw from north/clockwise."""
    assert enu_yaw_to_ned(0.0) == pytest.approx(math.pi / 2.0)
    assert enu_yaw_to_ned(math.pi / 2.0) == pytest.approx(0.0)
    assert enu_yaw_to_ned(math.pi) == pytest.approx(-math.pi / 2.0)


def test_enu_ned_xyz_conversion_swaps_xy_and_flips_z():
    """Convert vectors between ROS ENU and MAVLink NED axis conventions."""
    assert enu_xyz_to_ned_xyz(1.0, 2.0, 3.0) == pytest.approx((2.0, 1.0, -3.0))
    assert ned_xyz_to_enu_xyz(2.0, 1.0, -3.0) == pytest.approx((1.0, 2.0, 3.0))


def test_flu_frd_rpy_conversion_flips_pitch_and_converts_yaw():
    """Convert body attitude between ROS FLU/ENU and MAVLink FRD/NED conventions."""
    ned_rpy = enu_flu_rpy_to_ned_frd(0.2, -0.3, math.pi / 2.0)

    assert ned_rpy == pytest.approx((0.2, 0.3, 0.0))
    assert ned_frd_rpy_to_enu_flu(*ned_rpy) == pytest.approx((0.2, -0.3, math.pi / 2.0))


def test_ned_frd_angular_rates_to_enu_flu_flips_pitch_and_yaw_rates():
    """Convert angular-rate signs from MAVLink FRD/NED to ROS FLU/ENU."""
    assert ned_frd_angular_rates_to_enu_flu(0.1, -0.2, 0.3) == pytest.approx((0.1, 0.2, -0.3))


def test_normalize_angle_uses_half_open_interval():
    """Normalize angles to the [-pi, pi) interval."""
    assert normalize_angle(math.pi) == pytest.approx(-math.pi)
    assert normalize_angle(3.0 * math.pi) == pytest.approx(-math.pi)
    assert normalize_angle(-3.0 * math.pi) == pytest.approx(-math.pi)
    assert normalize_angle(0.5 * math.pi) == pytest.approx(0.5 * math.pi)

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

"""Small dependency-free geometry helpers shared across Flexible Drones."""

import math
from collections.abc import Sequence


def normalize_angle(angle: float) -> float:
    """Normalize an angle in radians to the [-pi, pi) interval."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def euler_from_quaternion_xyzw(q: Sequence[float]) -> tuple[float, float, float]:
    """Convert a ROS-style quaternion [x, y, z, w] to roll, pitch, yaw."""
    x, y, z, w = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def yaw_from_quaternion_xyzw(q: Sequence[float]) -> float:
    """Extract yaw from a ROS-style quaternion [x, y, z, w]."""
    return euler_from_quaternion_xyzw(q)[2]


def enu_yaw_to_ned(yaw_enu: float) -> float:
    """Convert ROS ENU yaw to ArduPilot/MAVLink LOCAL_NED yaw."""
    return normalize_angle(math.pi / 2.0 - float(yaw_enu))


def enu_xyz_to_ned_xyz(x_enu: float, y_enu: float, z_enu: float) -> tuple[float, float, float]:
    """Convert ENU xyz components to NED xyz components."""
    return (float(y_enu), float(x_enu), -float(z_enu))


def ned_xyz_to_enu_xyz(x_ned: float, y_ned: float, z_ned: float) -> tuple[float, float, float]:
    """Convert NED xyz components to ENU xyz components."""
    return (float(y_ned), float(x_ned), -float(z_ned))


def enu_flu_rpy_to_ned_frd(
    roll_enu: float,
    pitch_enu: float,
    yaw_enu: float,
) -> tuple[float, float, float]:
    """Convert ROS ENU/base_link FLU roll, pitch, yaw to MAVLink NED/aircraft FRD."""
    return (float(roll_enu), -float(pitch_enu), enu_yaw_to_ned(yaw_enu))


def ned_frd_rpy_to_enu_flu(
    roll_ned: float,
    pitch_ned: float,
    yaw_ned: float,
) -> tuple[float, float, float]:
    """Convert MAVLink NED/aircraft FRD roll, pitch, yaw to ROS ENU/base_link FLU."""
    return (float(roll_ned), -float(pitch_ned), enu_yaw_to_ned(yaw_ned))


def ned_frd_angular_rates_to_enu_flu(
    rollspeed_ned: float,
    pitchspeed_ned: float,
    yawspeed_ned: float,
) -> tuple[float, float, float]:
    """Convert MAVLink NED/aircraft FRD angular rates to ROS ENU/base_link FLU."""
    return (float(rollspeed_ned), -float(pitchspeed_ned), -float(yawspeed_ned))


def quaternion_from_euler_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert roll, pitch, yaw to a ROS-style quaternion [x, y, z, w]."""
    half_roll = float(roll) * 0.5
    half_pitch = float(pitch) * 0.5
    half_yaw = float(yaw) * 0.5

    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )

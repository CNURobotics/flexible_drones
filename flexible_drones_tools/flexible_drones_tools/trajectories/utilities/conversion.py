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

from copy import deepcopy

import numpy as np
from builtin_interfaces.msg import Duration

from flexible_drones_msgs.msg import SampledTrajectory, Trajectory, TrajectoryPoint, TrajectoryPolynomialPiece


def duration_from_seconds(seconds):
    whole_seconds = int(seconds)
    nanoseconds = int(round((float(seconds) - whole_seconds) * 1_000_000_000))
    if nanoseconds >= 1_000_000_000:
        whole_seconds += 1
        nanoseconds -= 1_000_000_000
    return Duration(sec=whole_seconds, nanosec=nanoseconds)


def duration_to_seconds(duration):
    return float(duration.sec) + float(duration.nanosec) / 1_000_000_000.0


def coefficients_to_drone_order(coefficients, coefficient_order='drone'):
    values = [float(value) for value in coefficients]
    if coefficient_order == 'drone':
        return values
    if coefficient_order == 'numpy':
        return list(reversed(values))
    raise ValueError(f"Unknown coefficient_order '{coefficient_order}'.")


def coefficients_from_drone_order(coefficients, coefficient_order='drone'):
    values = [float(value) for value in coefficients]
    if coefficient_order == 'drone':
        return values
    if coefficient_order == 'numpy':
        return list(reversed(values))
    raise ValueError(f"Unknown coefficient_order '{coefficient_order}'.")


def polynomial_piece_from_coefficients(
    duration,
    poly_x,
    poly_y,
    poly_z,
    poly_yaw,
    coefficient_order='drone',
):
    piece = TrajectoryPolynomialPiece()
    piece.duration = duration_from_seconds(duration)
    piece.poly_x = coefficients_to_drone_order(poly_x, coefficient_order)
    piece.poly_y = coefficients_to_drone_order(poly_y, coefficient_order)
    piece.poly_z = coefficients_to_drone_order(poly_z, coefficient_order)
    piece.poly_yaw = coefficients_to_drone_order(poly_yaw, coefficient_order)
    return piece


def pieces_from_coefficients(
    durations,
    x_coeffs,
    y_coeffs,
    z_coeffs,
    yaw_coeffs,
    coefficient_order='drone',
):
    return [
        polynomial_piece_from_coefficients(
            duration,
            x_coeff,
            y_coeff,
            z_coeff,
            yaw_coeff,
            coefficient_order=coefficient_order,
        )
        for duration, x_coeff, y_coeff, z_coeff, yaw_coeff in zip(
            durations,
            x_coeffs,
            y_coeffs,
            z_coeffs,
            yaw_coeffs,
        )
    ]


def coefficients_from_pieces(pieces, coefficient_order='drone'):
    durations = []
    x_coeffs = []
    y_coeffs = []
    z_coeffs = []
    yaw_coeffs = []
    for piece in pieces:
        durations.append(duration_to_seconds(piece.duration))
        x_coeffs.append(coefficients_from_drone_order(piece.poly_x, coefficient_order))
        y_coeffs.append(coefficients_from_drone_order(piece.poly_y, coefficient_order))
        z_coeffs.append(coefficients_from_drone_order(piece.poly_z, coefficient_order))
        yaw_coeffs.append(coefficients_from_drone_order(piece.poly_yaw, coefficient_order))
    return durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs


def trajectory_from_pieces(trajectory_id, pieces):
    trajectory = Trajectory()
    trajectory.trajectory_id = int(trajectory_id)
    trajectory.pieces = [deepcopy(piece) for piece in pieces]
    return trajectory


def evaluate_piece_axis(coefficients, t):
    return float(np.polyval(list(reversed(coefficients)), t))


def sampled_trajectory_from_pieces(trajectory_id, name, pieces, sample_dt):
    sample_dt = float(sample_dt) if sample_dt and sample_dt > 0.0 else 0.05
    trajectory = SampledTrajectory()
    trajectory.trajectory_id = int(trajectory_id)
    trajectory.name = name
    trajectory.points = []

    t_offset = 0.0
    for piece in pieces:
        duration = duration_to_seconds(piece.duration)
        if duration <= 0.0:
            continue

        local_ts = np.arange(0.0, duration, sample_dt, dtype=np.float64)
        if local_ts.size == 0 or local_ts[-1] < duration:
            local_ts = np.append(local_ts, duration)

        for t_local in local_ts:
            point = TrajectoryPoint()
            point.x = evaluate_piece_axis(piece.poly_x, t_local)
            point.y = evaluate_piece_axis(piece.poly_y, t_local)
            point.z = evaluate_piece_axis(piece.poly_z, t_local)
            point.yaw = evaluate_piece_axis(piece.poly_yaw, t_local)
            point.time_from_start = float(t_offset + t_local)
            trajectory.points.append(point)
        t_offset += duration
    return trajectory

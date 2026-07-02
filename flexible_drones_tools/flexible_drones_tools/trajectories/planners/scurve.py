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

"""
Mode B: in-place / short-move S-curve generator (hover-and-rotate).

When a request barely translates and starts/ends near hover, the obstacle NLP is
overkill -- there is no meaningful path tangent to look ahead along, so the move is
essentially a smooth reorientation. This module builds a single rest-to-rest degree-7
polynomial per position axis (x/y/z) directly from the boundary conditions, with the
segment duration sized so the realized velocity/accel/jerk/snap stay within the
kinematic limits and the *yaw slew* (produced afterwards by the Concern B yaw stage)
stays within yaw_rate_max / yaw_accel_max.

There is no optimization here -- the position polynomial is a closed-form boundary
value solve -- so hover-and-rotate is produced cheaply. The output yaw is filled in by
the normal output-yaw stage, which (with the speed gate near zero) simply slews from
the start yaw pin to the target yaw pin.
"""

import math

import numpy as np

from flexible_drones_tools.trajectories.planners.errors import TrajectoryPlanningError

DEFAULT_COEFFICIENT_COUNT = 8  # degree-7, matching x/y/z and the rest of the pipeline

# Peak derivative factors for the unit rest-to-rest septic ("smootherstep7"),
# s(u) = 35u^4 - 84u^5 + 70u^6 - 20u^7, used to size the yaw-slew duration in closed
# form: peak |s'| ~= 2.1875, peak |s''| ~= 7.5 over u in [0, 1].
_SEPTIC_PEAK_RATE_FACTOR = 35.0 / 16.0
_SEPTIC_PEAK_ACCEL_FACTOR = 7.5


def _falling_factorial(k, m):
    result = 1.0
    for offset in range(m):
        result *= (k - offset)
    return result


def _derivative_row(n, t, order):
    row = np.zeros(n)
    for k in range(order, n):
        row[k] = _falling_factorial(k, order) * (t ** (k - order))
    return row


def boundary_value_septic(p0, v0, p1, v1, duration, n=DEFAULT_COEFFICIENT_COUNT):
    """
    Solve a degree-7 real-time polynomial from position+velocity at both ends.

    Boundary conditions: position and velocity from the request, with acceleration and
    jerk pinned to zero at both ends (smooth rest-to-rest in the higher derivatives).
    Returns the 8 real-time coefficients ``c`` with ``p(t) = sum_k c_k t^k``.
    """
    T = float(duration)
    rows = [
        _derivative_row(n, 0.0, 0), _derivative_row(n, 0.0, 1),
        _derivative_row(n, 0.0, 2), _derivative_row(n, 0.0, 3),
        _derivative_row(n, T, 0), _derivative_row(n, T, 1),
        _derivative_row(n, T, 2), _derivative_row(n, T, 3),
    ]
    targets = [p0, v0, 0.0, 0.0, p1, v1, 0.0, 0.0]
    return np.linalg.solve(np.vstack(rows), np.asarray(targets, dtype=np.float64))


def _peak_derivatives(coeffs, duration, sample_count=200):
    """Return (peak|v|, peak|a|, peak|j|, peak|s|) of a real-time polynomial over [0, T]."""
    from numpy.polynomial.polynomial import Polynomial

    poly = Polynomial(np.asarray(coeffs, dtype=np.float64))
    derivs = [poly.deriv(order) for order in (1, 2, 3, 4)]
    times = np.linspace(0.0, float(duration), sample_count)
    return tuple(float(np.max(np.abs(d(times)))) for d in derivs)


def _translation_duration(deltas, velocities, limits, t_min, t_max):
    """Grow the duration until every axis honours the v/a/j/s limits (iterative sizing)."""
    limit_values = [
        float(limits['v_max']), float(limits['a_max']),
        float(limits['j_max']), float(limits['s_max']),
    ]
    t_max = float(t_max)
    duration = max(float(t_min), 1e-3)
    worst_ratio = 0.0
    for _ in range(60):
        worst_ratio = 0.0
        for axis in range(3):
            coeffs = boundary_value_septic(
                0.0, velocities[axis], deltas[axis], velocities[axis], duration
            )
            peaks = _peak_derivatives(coeffs, duration)
            for order, (peak, limit) in enumerate(zip(peaks, limit_values), start=1):
                if limit > 0.0 and peak > 0.0:
                    # peak of a derivative scales ~ 1/T^order, so this maps a ratio
                    # back to the duration scaling that would satisfy it.
                    worst_ratio = max(worst_ratio, (peak / limit) ** (1.0 / order))
        if worst_ratio <= 1.0 + 1e-3:
            return duration
        if duration >= t_max:
            break
        duration = min(t_max, duration * max(worst_ratio, 1.01))
    # Growing the duration to t_max still didn't bring every axis under its v/a/j/s
    # limit -- returning it anyway would silently ship a trajectory known to violate
    # a hard kinematic limit. Raise so the caller can fall back (e.g. to the full
    # obstacle optimizer) instead of shipping it.
    raise TrajectoryPlanningError(
        'In-place S-curve translation cannot satisfy v/a/j/s limits within '
        f't_max={t_max:g}s (worst limit ratio={worst_ratio:.3f} at duration={duration:.3f}s).'
    )


def _yaw_slew_duration(delta_yaw, yaw_rate_max, yaw_accel_max):
    """Smallest duration so a rest-to-rest yaw slew of delta_yaw honours the yaw limits."""
    magnitude = abs(float(delta_yaw))
    if magnitude < 1e-9:
        return 0.0
    duration = 0.0
    if yaw_rate_max and float(yaw_rate_max) > 0.0:
        duration = max(duration, _SEPTIC_PEAK_RATE_FACTOR * magnitude / float(yaw_rate_max))
    if yaw_accel_max and float(yaw_accel_max) > 0.0:
        duration = max(
            duration,
            math.sqrt(_SEPTIC_PEAK_ACCEL_FACTOR * magnitude / float(yaw_accel_max)),
        )
    return duration


def wrap_to_pi(angle):
    """Wrap an angle (radians) to the (-pi, pi] interval."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def plan_in_place_scurve(
    start_position,
    target_position,
    start_velocity,
    target_velocity,
    start_yaw,
    target_yaw,
    limits,
    t_min=0.4,
    t_max=60.0,
):
    """
    Build the Mode B x/y/z S-curve segment for a short / in-place move.

    ``*_position`` / ``*_velocity`` are ``(x, y, z)`` sequences; ``*_yaw`` are radians
    (or ``None`` to skip the yaw-slew duration term). Returns a one-element list of
    ``{'T', 'c_x', 'c_y', 'c_z'}`` with real-time coefficients, matching the planner
    output convention. The output yaw is produced separately by the yaw stage.
    """
    deltas = [float(target_position[i]) - float(start_position[i]) for i in range(3)]
    velocities = [0.5 * (float(start_velocity[i]) + float(target_velocity[i])) for i in range(3)]

    duration = _translation_duration(deltas, velocities, limits, t_min, t_max)

    if start_yaw is not None and target_yaw is not None:
        delta_yaw = wrap_to_pi(float(target_yaw) - float(start_yaw))
        yaw_duration = _yaw_slew_duration(
            delta_yaw, limits.get('yaw_rate_max'), limits.get('yaw_accel_max')
        )
        duration = max(duration, yaw_duration)

    duration = min(max(duration, float(t_min)), float(t_max))

    segment = {'T': duration}
    for axis, key in ((0, 'c_x'), (1, 'c_y'), (2, 'c_z')):
        segment[key] = boundary_value_septic(
            float(start_position[axis]), float(start_velocity[axis]),
            float(target_position[axis]), float(target_velocity[axis]),
            duration,
        )
    return [segment]

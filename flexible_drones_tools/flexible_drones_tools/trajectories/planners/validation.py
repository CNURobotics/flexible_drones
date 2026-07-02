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

"""Post-optimization checks on planned trajectories (boundary, clearance, continuity)."""

import math

import numpy as np
from geometry_msgs.msg import Vector3

from flexible_drones_tools.trajectories.planners.errors import TrajectoryPlanningError
from flexible_drones_tools.trajectories.planners.geometry import point_capsule_distance

# Hard-gate tolerances for *shippable* plans. These are deliberately looser than the
# constrained stage's acceptable_constr_viol_tol (~1e-5) so a converged candidate --
# or one accepted at the iteration limit -- is not rejected on solver-noise-level
# violations. See the orchestration plan's solver/validation tolerance-coupling note.
# Per-derivative tolerance for endpoint boundary checks (metres / m·s^-1 / m·s^-2).
BOUNDARY_TOLERANCES = {
    0: 1e-4,   # position, metres
    1: 1e-4,   # velocity, metres/second
    2: 1e-3,   # acceleration, metres/second^2
}
# Dense-check sample counts are a multiple of a base count rather than an
# independent magic number, mirroring casadi_obstacle_planner's
# obstacle_base_samples/obstacle_refine_multiplier: this structurally keeps
# every *_SAMPLE_COUNT >= 2 (guarding the t_eval = duration * i / (count - 1)
# divisions below) and keeps this module's default dense grid in lockstep with
# the planner's default dense re-solve grid (obstacle_base_samples *
# obstacle_refine_multiplier), so a converged dense re-solve actually satisfies
# the limits at the points re-checked here.
_DENSE_SAMPLE_BASE_COUNT = 41
_DENSE_SAMPLE_MULTIPLIER = 5
if _DENSE_SAMPLE_MULTIPLIER < 1:
    raise ValueError(
        f'_DENSE_SAMPLE_MULTIPLIER must be >= 1; got {_DENSE_SAMPLE_MULTIPLIER}.')
_DENSE_SAMPLE_COUNT = _DENSE_SAMPLE_BASE_COUNT * _DENSE_SAMPLE_MULTIPLIER

OBSTACLE_CLEARANCE_SAMPLE_COUNT = _DENSE_SAMPLE_COUNT
OBSTACLE_CLEARANCE_TOLERANCE = 1e-4
KINEMATIC_LIMIT_SAMPLE_COUNT = _DENSE_SAMPLE_COUNT
KINEMATIC_LIMIT_TOLERANCE = 1e-4
POSITION_BOUND_SAMPLE_COUNT = _DENSE_SAMPLE_COUNT
POSITION_BOUND_TOLERANCE = 1e-4
YAW_LIMIT_SAMPLE_COUNT = _DENSE_SAMPLE_COUNT
YAW_LIMIT_TOLERANCE = 1e-3
KINEMATIC_LIMITS = (
    (1, 'velocity', 'v_max'),
    (2, 'acceleration', 'a_max'),
    (3, 'jerk', 'j_max'),
    (4, 'snap', 's_max'),
)
CONTINUITY_TOLERANCES = {
    0: 1e-4,
    1: 4e-4,
    2: 1e-3,
    3: 5e-3,
    4: 5e-3,
}

# Looser tolerances for gating a *non-shippable* warm-start seed, whose producing
# solve runs at acceptable_constr_viol_tol ~1e-3. The seed only needs gross sanity --
# the constrained stage re-imposes every constraint -- so reject it only on large
# violations; otherwise the seed is needlessly discarded and the warm start wasted.
SEED_BOUNDARY_TOLERANCES = {0: 1e-2, 1: 1e-2, 2: 1e-2}
SEED_KINEMATIC_LIMIT_TOLERANCE = 1e-2
SEED_POSITION_BOUND_TOLERANCE = 1e-2
SEED_CONTINUITY_TOLERANCES = {0: 1e-2, 1: 4e-2, 2: 1e-1, 3: 5e-1, 4: 5e-1}

# Tight, informational continuity tolerances for post-solve *warnings* (diagnostics
# only; the hard gate uses the looser CONTINUITY_TOLERANCES above).
DIAGNOSTIC_CONTINUITY_TOLERANCES = {1: 1e-6, 2: 4e-6, 3: 1e-5, 4: 5e-5}


def evaluate_segment_axis(segment, axis, derivative_order, t):
    """Evaluate a local real-time polynomial axis (or a derivative) at time ``t``."""
    from numpy.polynomial.polynomial import Polynomial

    coeffs = np.asarray(segment[axis], dtype=np.float64)
    return float(Polynomial(coeffs).deriv(derivative_order)(float(t)))


def segment_start_times(segments):
    """Return cumulative start times for each segment."""
    starts = []
    total = 0.0
    for segment in segments:
        starts.append(total)
        total += float(segment['T'])
    return starts


def evaluate_position(segments, segment_index, t):
    """Evaluate xyz position at a segment-local time."""
    segment = segments[segment_index]
    return (
        evaluate_segment_axis(segment, 'c_x', 0, t),
        evaluate_segment_axis(segment, 'c_y', 0, t),
        evaluate_segment_axis(segment, 'c_z', 0, t),
    )


def evaluate_derivative_vector(segments, segment_index, derivative_order, t):
    """Evaluate xyz derivative vector at a segment-local time."""
    segment = segments[segment_index]
    return (
        evaluate_segment_axis(segment, 'c_x', derivative_order, t),
        evaluate_segment_axis(segment, 'c_y', derivative_order, t),
        evaluate_segment_axis(segment, 'c_z', derivative_order, t),
    )


def _sample_location_text(sample):
    sample_count = max(1, int(sample['sample_count']))
    sample_index = int(sample['sample_index'])
    s_value = 0.0 if sample_count <= 1 else sample_index / (sample_count - 1)
    point = sample['point']
    return (
        f'segment={sample["segment_index"]}, '
        f'sample={sample_index}/{sample_count - 1}, '
        f's={s_value:.4f}, '
        f'local_t={sample["time"]:.4f}s, '
        f'global_t={sample["global_time"]:.4f}s, '
        f'point=({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})'
    )


def continuity_difference(segments, axis, segment_index, derivative_order):
    """Return (diff, left, right) of the ``derivative_order`` value across a junction."""
    from numpy.polynomial.polynomial import Polynomial

    seg1 = segments[segment_index]
    seg2 = segments[segment_index + 1]

    dt1 = float(seg1['T'])
    coeffs1 = np.asarray(seg1[axis], dtype=np.float64)
    coeffs2 = np.asarray(seg2[axis], dtype=np.float64)

    p1 = Polynomial(coeffs1)
    p2 = Polynomial(coeffs2)

    # Planner coefficients are local real-time polynomials:
    # p_i(t) = sum_k c_k t^k, with t in seconds from the segment start.
    d1 = p1.deriv(derivative_order)(dt1)
    d2 = p2.deriv(derivative_order)(0.0)
    return float(d1 - d2), float(d1), float(d2)


def validate_boundary_constraints(
    segments,
    start_pose,
    target_pose,
    start_linearv,
    target_linearv,
    tolerances=None,
):
    """Raise if the trajectory endpoints miss the requested position/velocity/accel."""
    if not segments:
        raise TrajectoryPlanningError('Planner returned no segments.')

    tolerances = BOUNDARY_TOLERANCES if tolerances is None else tolerances

    zero_acc = Vector3()
    endpoints = (
        ('start', segments[0], 0.0, start_pose, start_linearv, zero_acc),
        ('target', segments[-1], float(segments[-1]['T']), target_pose, target_linearv, zero_acc),
    )
    derivative_expectations = (
        (0, 'position'),
        (1, 'velocity'),
        (2, 'acceleration'),
    )

    violations = []
    for label, segment, t_eval, position, velocity, acceleration in endpoints:
        expected_by_order = {
            0: position,
            1: velocity,
            2: acceleration,
        }
        for derivative_order, derivative_label in derivative_expectations:
            expected_vector = expected_by_order[derivative_order]
            tolerance = tolerances[derivative_order]
            for axis_name, segment_axis in (('x', 'c_x'), ('y', 'c_y'), ('z', 'c_z')):
                actual = evaluate_segment_axis(
                    segment,
                    segment_axis,
                    derivative_order,
                    t_eval,
                )
                expected = float(getattr(expected_vector, axis_name))
                error = abs(actual - expected)
                if error > tolerance:
                    violations.append(
                        f'{label} {derivative_label}.{axis_name}: '
                        f'actual={actual:.9g}, expected={expected:.9g}, '
                        f'error={error:.3g}, tol={tolerance:.3g}'
                    )

    if violations:
        raise TrajectoryPlanningError(
            'Boundary constraint validation failed: ' + '; '.join(violations)
        )


def validate_obstacle_clearance(segments, obstacles):
    """Raise if any densely-sampled trajectory point penetrates an obstacle."""
    if not obstacles:
        return

    worst = None
    starts = segment_start_times(segments)
    for segment_index, segment in enumerate(segments):
        duration = float(segment['T'])
        for sample_index in range(OBSTACLE_CLEARANCE_SAMPLE_COUNT):
            t_eval = duration * sample_index / max(OBSTACLE_CLEARANCE_SAMPLE_COUNT - 1, 1)
            x_val = evaluate_segment_axis(segment, 'c_x', 0, t_eval)
            y_val = evaluate_segment_axis(segment, 'c_y', 0, t_eval)
            z_val = evaluate_segment_axis(segment, 'c_z', 0, t_eval)
            for obstacle_index, obstacle in enumerate(obstacles):
                radius = float(obstacle['radius'])
                if obstacle.get('type') == 'capsule':
                    distance = point_capsule_distance((x_val, y_val, z_val), obstacle)
                else:
                    if z_val > float(obstacle.get('height', math.inf)):
                        continue
                    distance = math.hypot(
                        x_val - float(obstacle['x']),
                        y_val - float(obstacle['y']),
                    )
                clearance = distance - radius
                if worst is None or clearance < worst['clearance']:
                    worst = {
                        'clearance': clearance,
                        'segment_index': segment_index,
                        'sample_index': sample_index,
                        'sample_count': OBSTACLE_CLEARANCE_SAMPLE_COUNT,
                        'obstacle_index': obstacle_index,
                        'obstacle': obstacle,
                        'time': t_eval,
                        'global_time': starts[segment_index] + t_eval,
                        'point': (x_val, y_val, z_val),
                    }

    if worst and worst['clearance'] < -OBSTACLE_CLEARANCE_TOLERANCE:
        obstacle = worst['obstacle']
        if obstacle.get('type') == 'capsule':
            obstacle_text = (
                f"a=({float(obstacle['ax']):.3f}, {float(obstacle['ay']):.3f}, {float(obstacle['az']):.3f}), "
                f"b=({float(obstacle['bx']):.3f}, {float(obstacle['by']):.3f}, {float(obstacle['bz']):.3f})"
            )
        else:
            obstacle_text = (
                f"center=({float(obstacle['x']):.3f}, {float(obstacle['y']):.3f})"
            )
        raise TrajectoryPlanningError(
            'Obstacle clearance validation failed: '
            f'{_sample_location_text(worst)}, '
            f'obstacle={worst["obstacle_index"]}, '
            f'{obstacle_text}, '
            f'radius={float(obstacle["radius"]):.3f}, '
            f'clearance={worst["clearance"]:.4f} m'
        )


def validate_kinematic_limits(segments, limits, tolerance=None):
    """Raise if any densely-sampled axis derivative exceeds its configured limit."""
    if not segments:
        raise TrajectoryPlanningError('Planner returned no segments.')

    tolerance = KINEMATIC_LIMIT_TOLERANCE if tolerance is None else float(tolerance)

    worst = None
    starts = segment_start_times(segments)
    for segment_index, segment in enumerate(segments):
        duration = float(segment['T'])
        for sample_index in range(KINEMATIC_LIMIT_SAMPLE_COUNT):
            t_eval = duration * sample_index / max(KINEMATIC_LIMIT_SAMPLE_COUNT - 1, 1)
            for derivative_order, derivative_label, limit_key in KINEMATIC_LIMITS:
                limit = float(limits[limit_key])
                for axis_name, segment_axis in (('x', 'c_x'), ('y', 'c_y'), ('z', 'c_z')):
                    actual = evaluate_segment_axis(
                        segment,
                        segment_axis,
                        derivative_order,
                        t_eval,
                    )
                    excess = abs(actual) - limit
                    if worst is None or excess > worst['excess']:
                        worst = {
                            'excess': excess,
                            'actual': actual,
                            'limit': limit,
                            'limit_key': limit_key,
                            'axis_name': axis_name,
                            'derivative_label': derivative_label,
                            'derivative_order': derivative_order,
                            'segment_index': segment_index,
                            'sample_index': sample_index,
                            'sample_count': KINEMATIC_LIMIT_SAMPLE_COUNT,
                            'time': t_eval,
                            'global_time': starts[segment_index] + t_eval,
                            'point': evaluate_position(segments, segment_index, t_eval),
                            'derivative_vector': evaluate_derivative_vector(
                                segments, segment_index, derivative_order, t_eval),
                        }

    if worst and worst['excess'] > tolerance:
        vector = worst['derivative_vector']
        raise TrajectoryPlanningError(
            'Kinematic limit validation failed: '
            f'{_sample_location_text(worst)}, '
            f'{worst["derivative_label"]}.{worst["axis_name"]}={worst["actual"]:.9g}, '
            f'{worst["derivative_label"]}_vector='
            f'({vector[0]:.6g}, {vector[1]:.6g}, {vector[2]:.6g}), '
            f'limit {worst["limit_key"]}={worst["limit"]:.9g}, '
            f'excess={worst["excess"]:.4g}'
        )


def find_yaw_limit_warnings(
    durations,
    yaw_coefficients,
    yaw_rate_max,
    yaw_accel_max,
    sample_count=YAW_LIMIT_SAMPLE_COUNT,
    tolerance=YAW_LIMIT_TOLERANCE,
):
    """
    Dense-sample the output yaw and report rate/accel overshoot (advisory, warn-only).

    The XYZ NLP caps and refines path-tangent yaw rate; output yaw acceleration is not
    constrained inside any solve by design, so callers may pass ``yaw_rate_max=None`` to
    keep parent-side warnings focused on acceleration only. Returns one human-readable
    string per violated limit kind (empty when within limits).
    """
    from numpy.polynomial.polynomial import Polynomial

    warnings = []
    checks = []
    if yaw_rate_max and float(yaw_rate_max) > 0.0:
        checks.append((1, 'yaw rate', float(yaw_rate_max), 'yaw_rate_max'))
    if yaw_accel_max and float(yaw_accel_max) > 0.0:
        checks.append((2, 'yaw acceleration', float(yaw_accel_max), 'yaw_accel_max'))
    if not yaw_coefficients or not checks:
        return warnings

    for order, label, limit, key in checks:
        worst = None
        for seg_index, (duration, coeffs) in enumerate(zip(durations, yaw_coefficients)):
            poly = Polynomial(np.asarray(coeffs, dtype=np.float64)).deriv(order)
            for sample_index in range(sample_count):
                t = float(duration) * sample_index / max(sample_count - 1, 1)
                value = abs(float(poly(t)))
                if worst is None or value > worst['value']:
                    worst = {'value': value, 'seg': seg_index, 't': t}
        if worst and worst['value'] > limit + tolerance:
            warnings.append(
                f'{label} {worst["value"]:.3f} exceeds {key}={limit:.3f} '
                f'(segment {worst["seg"]}, t={worst["t"]:.3f}s)'
            )
    return warnings


def validate_position_bounds(segments, bounds, tolerance=None):
    """Raise if any densely-sampled position leaves the configured flight bounds."""
    if not segments:
        raise TrajectoryPlanningError('Planner returned no segments.')

    tolerance = POSITION_BOUND_TOLERANCE if tolerance is None else float(tolerance)
    bound_keys = {
        'x': ('x_min', 'x_max'),
        'y': ('y_min', 'y_max'),
        'z': ('z_min', 'z_max'),
    }
    worst = None
    starts = segment_start_times(segments)
    for segment_index, segment in enumerate(segments):
        duration = float(segment['T'])
        for sample_index in range(POSITION_BOUND_SAMPLE_COUNT):
            t_eval = duration * sample_index / max(POSITION_BOUND_SAMPLE_COUNT - 1, 1)
            for axis_name, segment_axis in (('x', 'c_x'), ('y', 'c_y'), ('z', 'c_z')):
                min_key, max_key = bound_keys[axis_name]
                actual = evaluate_segment_axis(segment, segment_axis, 0, t_eval)
                lower = float(bounds[min_key])
                upper = float(bounds[max_key])
                excess = max(lower - actual, actual - upper)
                if worst is None or excess > worst['excess']:
                    worst = {
                        'excess': excess,
                        'actual': actual,
                        'lower': lower,
                        'upper': upper,
                        'axis_name': axis_name,
                        'segment_index': segment_index,
                        'sample_index': sample_index,
                        'sample_count': POSITION_BOUND_SAMPLE_COUNT,
                        'time': t_eval,
                        'global_time': starts[segment_index] + t_eval,
                        'point': evaluate_position(segments, segment_index, t_eval),
                    }

    if worst and worst['excess'] > tolerance:
        raise TrajectoryPlanningError(
            'Position bounds validation failed: '
            f'{_sample_location_text(worst)}, '
            f'position.{worst["axis_name"]}={worst["actual"]:.9g}, '
            f'bounds=[{worst["lower"]:.9g}, {worst["upper"]:.9g}], '
            f'excess={worst["excess"]:.4g}'
        )


def validate_continuity(segments, tolerances=None):
    """Raise if any polynomial junction has a derivative jump above tolerance."""
    if not segments:
        raise TrajectoryPlanningError('Planner returned no segments.')
    tolerances = dict(CONTINUITY_TOLERANCES if tolerances is None else tolerances)
    violations = []
    starts = segment_start_times(segments)
    for derivative_order, tolerance in sorted(tolerances.items()):
        for axis in ('c_x', 'c_y', 'c_z'):
            for segment_index in range(len(segments) - 1):
                diff, left, right = continuity_difference(
                    segments,
                    axis,
                    segment_index,
                    derivative_order,
                )
                if abs(diff) > float(tolerance):
                    junction_time = starts[segment_index] + float(segments[segment_index]['T'])
                    point = evaluate_position(
                        segments,
                        segment_index,
                        float(segments[segment_index]['T']),
                    )
                    violations.append(
                        f'{axis} segment {segment_index}->{segment_index + 1} '
                        f'order {derivative_order}: '
                        f'junction_t={junction_time:.4f}s, '
                        f'point=({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}), '
                        f'left={left:.9g}, right={right:.9g}, '
                        f'diff={abs(diff):.4g}, tol={float(tolerance):.4g}'
                    )
    if violations:
        raise TrajectoryPlanningError(
            'Continuity validation failed: ' + '; '.join(violations)
        )


def warn_on_discontinuity(segments, derivative_order, tolerance, logger):
    """
    Log a warning for any junction whose ``derivative_order`` value jumps > tolerance.

    Assumes local real-time polynomial arrays (e.g. c_x); derivative_order=4 checks
    up to snap.
    """
    for axis in ('c_x', 'c_y', 'c_z'):
        for i in range(len(segments) - 1):
            diff, d1, d2 = continuity_difference(segments, axis, i, derivative_order)
            if abs(diff) > tolerance:
                logger.warning(
                    f'Discontinuity detected in {axis} between segments {i}–{i + 1} '
                    f'for order {derivative_order}: '
                    f'{d1:.4f} vs {d2:.4f} (diff = {abs(diff):.4e})'
                )

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
Initial-trajectory seeding strategies for the polynomial planner.

A seed is the optimizer's initial guess; for obstacle problems it decides which
homotopy class (around vs. through) the solve converges to. See README.md
(Segments and initialization).
"""

import numpy as np

AXES = ('x', 'y', 'z')


def pad_coefficients(coefficients, coeff_count):
    """Right-pad a coefficient list with zeros to ``coeff_count`` entries."""
    values = [float(value) for value in coefficients]
    return values + [0.0] * (coeff_count - len(values))


def quintic_boundary_coefficients(p0, v0, a0, p1, v1, a1, T):
    """Local real-time quintic matching position/velocity/accel at both ends over ``T``."""
    T = float(T)
    c0 = float(p0)
    c1 = float(v0) * T
    c2 = 0.5 * float(a0) * T**2
    rhs = np.array([
        float(p1) - (c0 + c1 + c2),
        float(v1) * T - (c1 + 2.0 * c2),
        float(a1) * T**2 - (2.0 * c2),
    ])
    matrix = np.array([
        [1.0, 1.0, 1.0],
        [3.0, 4.0, 5.0],
        [6.0, 12.0, 20.0],
    ])
    c3, c4, c5 = np.linalg.solve(matrix, rhs)
    return [c0, c1, c2, c3, c4, c5]


def _dict_to_vector(values):
    return np.asarray([float(values[axis]) for axis in AXES], dtype=np.float64)


def _unit_or_fallback(vector, fallback):
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return np.asarray(vector, dtype=np.float64) / norm

    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm > 1e-9:
        return np.asarray(fallback, dtype=np.float64) / fallback_norm
    return np.zeros(3, dtype=np.float64)


def _mod_2pi(angle):
    return float(angle) % (2.0 * np.pi)


def _safe_plane_z(tangent, fallback=np.asarray([0.0, 0.0, 1.0], dtype=np.float64)):
    tangent = _unit_or_fallback(tangent, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    fallback = _unit_or_fallback(fallback, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    normal = fallback - tangent * float(np.dot(fallback, tangent))
    if float(np.linalg.norm(normal)) <= 1e-9:
        normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        normal = normal - tangent * float(np.dot(normal, tangent))
    return _unit_or_fallback(normal, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))


def _combined_plane_z(tangent, start_z_axis=None, goal_z_axis=None):
    start_z = (
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if start_z_axis is None else np.asarray(start_z_axis, dtype=np.float64)
    )
    if goal_z_axis is None:
        return _safe_plane_z(tangent, start_z)

    start_z = _unit_or_fallback(start_z, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    goal_z = _unit_or_fallback(
        np.asarray(goal_z_axis, dtype=np.float64),
        start_z,
    )
    if float(np.dot(start_z, goal_z)) < 0.0:
        goal_z = -goal_z
    combined = start_z + goal_z
    if float(np.linalg.norm(combined)) <= 1e-9:
        combined = start_z
    return _safe_plane_z(tangent, combined)


def _left_of_travel(tangent, z_axis):
    return _unit_or_fallback(np.cross(z_axis, tangent), np.asarray([0.0, 1.0, 0.0]))


def _signed_angle_about_z(start, end, z_axis):
    start = _unit_or_fallback(start, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    end = _unit_or_fallback(end, start)
    z_axis = _unit_or_fallback(z_axis, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    return float(np.arctan2(np.dot(z_axis, np.cross(start, end)), np.dot(start, end)))


def _turn_angle(start_radial, end_radial, z_axis, curvature):
    signed = _signed_angle_about_z(start_radial, end_radial, z_axis)
    return _mod_2pi(signed if curvature > 0 else -signed)


def _circle_point(center, radius, z_axis, tangent, curvature):
    left = _left_of_travel(tangent, z_axis)
    return center - float(curvature) * float(radius) * left


def _circle_tangent(point, center, z_axis, curvature):
    radial = _unit_or_fallback(point - center, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    return _unit_or_fallback(
        float(curvature) * np.cross(z_axis, radial),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    )


def _sample_arc(center, radius, z_axis, start_radial, angle, curvature, length_values):
    points = []
    tangents = []
    z_axis = _unit_or_fallback(z_axis, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    start_radial = _unit_or_fallback(start_radial, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    for length in length_values:
        theta = float(curvature) * float(length) / max(float(radius), 1e-9)
        radial = (
            start_radial * np.cos(theta)
            + np.cross(z_axis, start_radial) * np.sin(theta)
        )
        point = center + float(radius) * radial
        points.append(point)
        tangents.append(_circle_tangent(point, center, z_axis, curvature))
    return points, tangents


def _sample_line(start, end, length_values):
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    tangent = _unit_or_fallback(delta, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    points = []
    tangents = []
    for value in length_values:
        alpha = min(1.0, max(0.0, float(value) / max(length, 1e-9)))
        points.append(np.asarray(start, dtype=np.float64) + alpha * delta)
        tangents.append(tangent)
    return points, tangents


def _dubins_candidate(p0, t0, p1, t1, radius, z_axis, start_k, goal_k):
    radius = float(radius)
    z_axis = _safe_plane_z(t0 + t1, z_axis)
    t0 = _unit_or_fallback(t0, p1 - p0)
    t1 = _unit_or_fallback(t1, p1 - p0)
    left0 = _left_of_travel(t0, z_axis)
    left1 = _left_of_travel(t1, z_axis)
    c0 = p0 + float(start_k) * radius * left0
    c1 = p1 + float(goal_k) * radius * left1
    center_delta = c1 - c0
    center_distance = float(np.linalg.norm(center_delta))
    if center_distance <= 1e-9:
        return []

    tangents = []
    if start_k == goal_k:
        tangents.append(center_delta / center_distance)
    else:
        delta = radius * float(start_k - goal_k)
        if center_distance < abs(delta) - 1e-9:
            return []
        e = center_delta / center_distance
        left_e = _left_of_travel(e, z_axis)
        projection = -delta / center_distance
        side = float(np.sqrt(max(0.0, 1.0 - projection**2)))
        for sign in (-1.0, 1.0):
            normal = projection * e + sign * side * left_e
            tangents.append(_unit_or_fallback(-np.cross(z_axis, normal), e))

    candidates = []
    for tangent in tangents:
        q0 = _circle_point(c0, radius, z_axis, tangent, start_k)
        q1 = _circle_point(c1, radius, z_axis, tangent, goal_k)
        line_delta = q1 - q0
        straight_length = float(np.linalg.norm(line_delta))
        if straight_length <= 1e-9 or float(np.dot(line_delta, tangent)) <= 0.0:
            continue
        start_angle = _turn_angle(p0 - c0, q0 - c0, z_axis, start_k)
        goal_angle = _turn_angle(q1 - c1, p1 - c1, z_axis, goal_k)
        arc0 = radius * start_angle
        arc1 = radius * goal_angle
        candidates.append({
            'curvatures': (int(start_k), int(goal_k)),
            'centers': (c0, c1),
            'tangent_points': (q0, q1),
            'arc_lengths': (arc0, arc1),
            'straight_length': straight_length,
            'total_length': arc0 + straight_length + arc1,
            'z_axis': z_axis,
        })
    return candidates


def dubins_inspired_path(
    A_data,
    B_data,
    radius,
    z_axis=None,
    start_tangent=None,
    goal_tangent=None,
    goal_z_axis=None,
):
    """Return the shortest planar Dubins-style path metadata from boundary velocities."""
    A_pos, A_vel, _A_acc, _A_jerk, _A_snap = A_data
    B_pos, B_vel, _B_acc, _B_jerk, _B_snap = B_data
    p0 = _dict_to_vector(A_pos)
    p1 = _dict_to_vector(B_pos)
    chord = p1 - p0
    t0 = _unit_or_fallback(
        start_tangent if start_tangent is not None else _dict_to_vector(A_vel),
        chord,
    )
    t1 = _unit_or_fallback(
        goal_tangent if goal_tangent is not None else _dict_to_vector(B_vel),
        chord,
    )
    z_axis = _combined_plane_z(t0 + t1, z_axis, goal_z_axis)

    candidates = []
    for start_k in (1, -1):
        for goal_k in (1, -1):
            candidates.extend(_dubins_candidate(p0, t0, p1, t1, radius, z_axis, start_k, goal_k))
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate['total_length'])


def sample_dubins_inspired_path(
    A_data,
    B_data,
    radius,
    sample_count,
    z_axis=None,
    start_tangent=None,
    goal_tangent=None,
    goal_z_axis=None,
):
    """Sample xyz/tangent/arc-length arrays from the shortest Dubins-style candidate."""
    path = dubins_inspired_path(
        A_data,
        B_data,
        radius,
        z_axis=z_axis,
        start_tangent=start_tangent,
        goal_tangent=goal_tangent,
        goal_z_axis=goal_z_axis,
    )
    if path is None:
        return None

    A_pos, A_vel, _A_acc, _A_jerk, _A_snap = A_data
    B_pos, B_vel, _B_acc, _B_jerk, _B_snap = B_data
    p0 = _dict_to_vector(A_pos)
    p1 = _dict_to_vector(B_pos)
    t0 = _unit_or_fallback(
        start_tangent if start_tangent is not None else _dict_to_vector(A_vel),
        p1 - p0,
    )
    z_axis = path['z_axis']
    start_k, goal_k = path['curvatures']
    c0, c1 = path['centers']
    q0, q1 = path['tangent_points']
    arc0, arc1 = path['arc_lengths']
    straight = path['straight_length']
    total = path['total_length']

    sample_count = max(2, int(sample_count))
    distances = np.linspace(0.0, total, sample_count)
    positions = []
    tangents = []
    for distance in distances:
        if distance <= arc0:
            points, dirs = _sample_arc(
                c0, radius, z_axis, p0 - c0, arc0 / max(radius, 1e-9),
                start_k, [distance])
        elif distance <= arc0 + straight:
            points, dirs = _sample_line(q0, q1, [distance - arc0])
        else:
            points, dirs = _sample_arc(
                c1, radius, z_axis, q1 - c1, arc1 / max(radius, 1e-9),
                goal_k, [distance - arc0 - straight])
        positions.append(points[0])
        tangents.append(dirs[0])

    positions[0] = p0
    positions[-1] = p1
    tangents[0] = t0
    tangents[-1] = _unit_or_fallback(
        goal_tangent if goal_tangent is not None else _dict_to_vector(B_vel),
        p1 - p0,
    )
    return {
        'positions': np.asarray(positions, dtype=np.float64),
        'tangents': np.asarray(tangents, dtype=np.float64),
        'arc_length': distances,
        **path,
    }


def _closest_point_on_segment(point, start, end):
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return start
    alpha = float(np.dot(point - start, segment) / length_sq)
    return start + min(1.0, max(0.0, alpha)) * segment


def _closest_obstacle_core_point(point, obstacle):
    if obstacle.get('type') == 'capsule':
        start = np.asarray([obstacle['ax'], obstacle['ay'], obstacle['az']], dtype=np.float64)
        end = np.asarray([obstacle['bx'], obstacle['by'], obstacle['bz']], dtype=np.float64)
        return _closest_point_on_segment(point, start, end)

    z_top = float(obstacle.get('height', point[2]))
    z_min = min(0.0, z_top)
    z_max = max(0.0, z_top)
    return np.asarray([
        float(obstacle['x']),
        float(obstacle['y']),
        min(z_max, max(z_min, float(point[2]))),
    ], dtype=np.float64)


def _move_point_out_of_obstacles(point, obstacles, clearance, diagnostics=None, label='point'):
    moved = np.asarray(point, dtype=np.float64)
    for obstacle_index, obstacle in enumerate(obstacles or []):
        radius = float(obstacle['radius'])
        target_distance = radius + max(0.0, float(clearance))
        core_point = _closest_obstacle_core_point(moved, obstacle)
        offset = moved - core_point
        distance = float(np.linalg.norm(offset))
        if distance >= target_distance:
            continue

        # If the point is on the obstacle core, bias toward the world origin.
        # That matches the default gate world better than choosing a random side.
        direction = _unit_or_fallback(offset, -core_point)
        if float(np.linalg.norm(direction)) <= 1e-9:
            direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        before = moved.copy()
        moved = core_point + direction * target_distance
        if diagnostics is not None:
            diagnostics.setdefault('moves', []).append({
                'label': label,
                'obstacle_index': obstacle_index,
                'obstacle_type': obstacle.get('type', 'cylinder'),
                'radius': radius,
                'target_clearance': max(0.0, float(clearance)),
                'distance_before': distance,
                'distance_after': float(np.linalg.norm(moved - core_point)),
                'point_before': tuple(float(value) for value in before),
                'point_after': tuple(float(value) for value in moved),
            })
    return moved


def _subdivide_once(waypoints):
    subdivided = [waypoints[0]]
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        subdivided.append(0.5 * (start + end))
        subdivided.append(end)
    return subdivided


def _interior_tangent_velocities(waypoints, start_velocity, target_velocity, vmax):
    speed = 0.5 * max(0.0, float(vmax))
    velocities = [np.asarray(start_velocity, dtype=np.float64)]
    for index in range(1, len(waypoints) - 1):
        tangent = _unit_or_fallback(
            waypoints[index + 1] - waypoints[index - 1],
            waypoints[index] - waypoints[index - 1],
        )
        velocities.append(speed * tangent)
    velocities.append(np.asarray(target_velocity, dtype=np.float64))
    return velocities


def _segments_from_waypoints(waypoints, velocities, accelerations, total_time, n_coeff):
    segment_count = len(waypoints) - 1
    segment_time = float(total_time) / segment_count
    segments = []

    for segment_index in range(segment_count):
        segment = {'T': segment_time}
        for axis_index, axis in enumerate(AXES):
            coeffs = quintic_boundary_coefficients(
                waypoints[segment_index][axis_index],
                velocities[segment_index][axis_index],
                accelerations[segment_index][axis_index],
                waypoints[segment_index + 1][axis_index],
                velocities[segment_index + 1][axis_index],
                accelerations[segment_index + 1][axis_index],
                segment_time,
            )
            segment[f'c_{axis}'] = pad_coefficients(coeffs[:n_coeff], n_coeff)
        segments.append(segment)

    return segments


def straight_line_seed(A_data, B_data, total_time, n_segments, n_coeff, time_min):
    """
    Straight-line seed: equal waypoints with zero v/a at interior junctions.

    Obstacle-blind and stop-and-go -- a baseline initial guess. ``A_data``/``B_data``
    are (pos, vel, acc, jerk, snap) dicts; only pos/vel/acc seed the quintic.
    """
    A_pos, A_vel, A_acc, _A_jerk, _A_snap = A_data
    B_pos, B_vel, B_acc, _B_jerk, _B_snap = B_data
    total_time = max(float(total_time), float(time_min) * int(n_segments))
    segment_time = total_time / int(n_segments)
    segments = []

    for segment_index in range(int(n_segments)):
        alpha0 = segment_index / int(n_segments)
        alpha1 = (segment_index + 1) / int(n_segments)
        segment = {'T': segment_time}
        for axis in 'xyz':
            p0 = (1.0 - alpha0) * A_pos[axis] + alpha0 * B_pos[axis]
            p1 = (1.0 - alpha1) * A_pos[axis] + alpha1 * B_pos[axis]
            v0 = A_vel[axis] if segment_index == 0 else 0.0
            v1 = B_vel[axis] if segment_index == int(n_segments) - 1 else 0.0
            a0 = A_acc[axis] if segment_index == 0 else 0.0
            a1 = B_acc[axis] if segment_index == int(n_segments) - 1 else 0.0
            coeffs = quintic_boundary_coefficients(p0, v0, a0, p1, v1, a1, segment_time)
            segment[f'c_{axis}'] = pad_coefficients(coeffs[:n_coeff], n_coeff)
        segments.append(segment)

    return segments


def velocity_biased_chord_seed(
    A_data,
    B_data,
    total_time,
    n_segments,
    n_coeff,
    time_min,
    vmax=1.0,
    projection_time=1.0,
    blend=0.5,
    eps=1e-9,
):
    """
    Three-segment chord seed whose interior waypoints lean into boundary velocity.

    The straight start-to-goal chord is divided into thirds, then those third
    points are blended with one-second projections from the start/end velocities.
    Interior waypoint velocities follow broad chord tangents at half ``vmax``.
    """
    if int(n_segments) != 3:
        raise ValueError('velocity_biased_chord_seed requires n_segments=3.')

    A_pos, A_vel, A_acc, _A_jerk, _A_snap = A_data
    B_pos, B_vel, B_acc, _B_jerk, _B_snap = B_data
    p0 = _dict_to_vector(A_pos)
    p1 = _dict_to_vector(B_pos)
    v0 = _dict_to_vector(A_vel)
    v1 = _dict_to_vector(B_vel)
    a0 = _dict_to_vector(A_acc)
    a1 = _dict_to_vector(B_acc)

    chord = p1 - p0
    point_a = p0 + chord / 3.0
    point_b = p0 + 2.0 * chord / 3.0

    if float(np.linalg.norm(v0)) > float(eps):
        projected_a = p0 + v0 * float(projection_time)
    else:
        projected_a = point_a
    if float(np.linalg.norm(v1)) > float(eps):
        projected_b = p1 - v1 * float(projection_time)
    else:
        projected_b = point_b

    blend = min(1.0, max(0.0, float(blend)))
    waypoint_a = (1.0 - blend) * point_a + blend * projected_a
    waypoint_b = (1.0 - blend) * point_b + blend * projected_b

    tangent_a = _unit_or_fallback(waypoint_b - p0, waypoint_a - p0)
    tangent_b = _unit_or_fallback(p1 - waypoint_a, p1 - waypoint_b)
    interior_speed = 0.5 * max(0.0, float(vmax))
    velocity_a = interior_speed * tangent_a
    velocity_b = interior_speed * tangent_b

    total_time = max(float(total_time), float(time_min) * int(n_segments))
    return _segments_from_waypoints(
        [p0, waypoint_a, waypoint_b, p1],
        [v0, velocity_a, velocity_b, v1],
        [a0, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), a1],
        total_time,
        n_coeff,
    )


def obstacle_boundary_subdivision_seed(
    A_data,
    B_data,
    total_time,
    n_segments,
    n_coeff,
    time_min,
    vmax=1.0,
    obstacles=None,
    clearance=0.05,
    projection_time=1.0,
    blend=0.5,
    eps=1e-9,
    return_diagnostics=False,
):
    """
    Obstacle-aware seed from a nudged 3-leg chord.

    Starts with the velocity-biased chord guide points pA/pB, moves those guide
    points outside any intersecting obstacle boundary, then subdivides each of the
    three resulting legs once only when a guide point moved and ``n_segments=6``.
    ``n_segments=3`` always returns the unsubdivided version.
    """
    n_segments = int(n_segments)
    if n_segments not in (3, 6):
        raise ValueError('obstacle_boundary_subdivision_seed requires n_segments=3 or n_segments=6.')
    diagnostics = {
        'strategy': 'obstacle_boundary_subdivision',
        'requested_segments': n_segments,
        'base_segments': 3,
        'subdivided': False,
        'moves': [],
    }

    A_pos, A_vel, A_acc, _A_jerk, _A_snap = A_data
    B_pos, B_vel, B_acc, _B_jerk, _B_snap = B_data
    p0 = _dict_to_vector(A_pos)
    p1 = _dict_to_vector(B_pos)
    v0 = _dict_to_vector(A_vel)
    v1 = _dict_to_vector(B_vel)
    a0 = _dict_to_vector(A_acc)
    a1 = _dict_to_vector(B_acc)

    chord = p1 - p0
    point_a = p0 + chord / 3.0
    point_b = p0 + 2.0 * chord / 3.0
    projected_a = p0 + v0 * float(projection_time) if float(np.linalg.norm(v0)) > float(eps) else point_a
    projected_b = p1 - v1 * float(projection_time) if float(np.linalg.norm(v1)) > float(eps) else point_b
    blend = min(1.0, max(0.0, float(blend)))
    waypoint_a = (1.0 - blend) * point_a + blend * projected_a
    waypoint_b = (1.0 - blend) * point_b + blend * projected_b

    waypoint_a = _move_point_out_of_obstacles(
        waypoint_a, obstacles, clearance, diagnostics=diagnostics, label='pA')
    waypoint_b = _move_point_out_of_obstacles(
        waypoint_b, obstacles, clearance, diagnostics=diagnostics, label='pB')
    waypoints = [p0, waypoint_a, waypoint_b, p1]
    if n_segments == 6 and diagnostics['moves']:
        waypoints = _subdivide_once(waypoints)
        diagnostics['subdivided'] = True
    diagnostics['returned_segments'] = len(waypoints) - 1

    velocities = _interior_tangent_velocities(waypoints, v0, v1, vmax)
    accelerations = (
        [a0]
        + [np.zeros(3, dtype=np.float64) for _index in range(len(waypoints) - 2)]
        + [a1]
    )
    total_time = max(float(total_time), float(time_min) * diagnostics['returned_segments'])
    segments = _segments_from_waypoints(waypoints, velocities, accelerations, total_time, n_coeff)
    if return_diagnostics:
        return segments, diagnostics
    return segments


def dubins_seed(
    A_data,
    B_data,
    total_time,
    n_segments,
    n_coeff,
    time_min,
    vmax=1.0,
    turning_radius=0.5,
    start_tangent=None,
    goal_tangent=None,
    z_axis=None,
    goal_z_axis=None,
    return_diagnostics=False,
):
    """
    Dubins-inspired geometric seed following boundary travel tangents.

    The construction uses the start/end velocity vectors as desired travel
    tangents, builds the four left/right circle candidates in the horizontal
    turning plane, samples the shortest feasible arc-line-arc path, and fits one
    quintic segment between each adjacent sample. It is still only an optimizer
    initializer; dense obstacle validation and the obstacle solve remain the
    authority for the final trajectory.
    """
    n_segments = int(n_segments)
    if n_segments <= 0:
        raise ValueError('dubins_seed requires n_segments > 0.')

    radius = max(1e-6, float(turning_radius))
    sampled = sample_dubins_inspired_path(
        A_data,
        B_data,
        radius=radius,
        sample_count=n_segments + 1,
        z_axis=z_axis,
        start_tangent=start_tangent,
        goal_tangent=goal_tangent,
        goal_z_axis=goal_z_axis,
    )
    if sampled is None:
        result = straight_line_seed(
            A_data, B_data, total_time, n_segments, n_coeff, time_min)
        diagnostics = {
            'strategy': 'dubins',
            'fallback': 'straight_line',
            'turning_radius': radius,
            'returned_segments': len(result),
        }
        return (result, diagnostics) if return_diagnostics else result

    A_pos, A_vel, A_acc, _A_jerk, _A_snap = A_data
    B_pos, B_vel, B_acc, _B_jerk, _B_snap = B_data
    p0 = _dict_to_vector(A_pos)
    p1 = _dict_to_vector(B_pos)
    v0 = _dict_to_vector(A_vel)
    v1 = _dict_to_vector(B_vel)
    a0 = _dict_to_vector(A_acc)
    a1 = _dict_to_vector(B_acc)
    path_length = float(sampled['total_length'])
    total_time = max(
        float(total_time),
        path_length / max(float(vmax), 1e-6),
        float(time_min) * n_segments,
    )
    interior_speed = min(max(float(vmax), 0.0), path_length / max(total_time, 1e-6))
    waypoints = [np.asarray(point, dtype=np.float64) for point in sampled['positions']]
    tangents = [np.asarray(tangent, dtype=np.float64) for tangent in sampled['tangents']]
    velocities = (
        [v0]
        + [interior_speed * tangent for tangent in tangents[1:-1]]
        + [v1]
    )
    accelerations = (
        [a0]
        + [np.zeros(3, dtype=np.float64) for _index in range(len(waypoints) - 2)]
        + [a1]
    )
    segments = _segments_from_waypoints(
        [p0] + waypoints[1:-1] + [p1],
        velocities,
        accelerations,
        total_time,
        n_coeff,
    )
    diagnostics = {
        'strategy': 'dubins',
        'turning_radius': radius,
        'curvatures': sampled['curvatures'],
        'arc_lengths': sampled['arc_lengths'],
        'straight_length': sampled['straight_length'],
        'path_length': path_length,
        'returned_segments': len(segments),
    }
    return (segments, diagnostics) if return_diagnostics else segments


# Registry of available seeding strategies, keyed by name. New strategies (e.g.
# obstacle-aware) register here.
SEED_STRATEGIES = {
    'straight_line': straight_line_seed,
    'velocity_biased_chord': velocity_biased_chord_seed,
    'obstacle_boundary_subdivision': obstacle_boundary_subdivision_seed,
    'dubins': dubins_seed,
}

# Allowed n_segments per seed strategy; strategies omitted here accept any positive
# count. Single source of truth for the per-strategy requirements enforced inside the
# seed functions (velocity_biased_chord_seed / obstacle_boundary_subdivision_seed) so a
# request can be validated up front instead of failing deep in build_seed.
SEED_STRATEGY_SEGMENT_COUNTS = {
    'velocity_biased_chord': (3,),
    'obstacle_boundary_subdivision': (3, 6),
}

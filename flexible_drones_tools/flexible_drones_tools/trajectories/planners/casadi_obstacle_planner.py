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
import math
from pathlib import Path
import re
import time
import uuid

try:
    import casadi as ca
except ImportError as exc:
    raise ImportError('casadi not found; run: pip3 install casadi') from exc
import numpy as np
from numpy.polynomial import Polynomial

from flexible_drones_tools.trajectories.planners import seeding
from flexible_drones_tools.trajectories.planners.base_planner import (
    TrajectoryPlanner,
    point_xyz,
)
from flexible_drones_tools.trajectories.utilities.io import save_trajectory_csv


class CasadiObstaclePlanner(TrajectoryPlanner):
    """
    CasADi/IPOPT time-optimal planner with soft cylinder-obstacle avoidance.

    Plans a start->goal trajectory of ``n_segments`` 7th-order polynomial pieces,
    minimizing total time subject to the flight-volume box (``bounds``), the
    kinematic ``limits`` (v/a/j/s), and a smooth penalty for each cylinder
    obstacle. An empty obstacle list yields a clean minimum-time path.
    """

    SUCCESS_STATUSES = {'Solve_Succeeded'}
    FAILURE_ARTIFACT_ROOT = Path('/tmp')
    UNCONSTRAINED_OPTIMIZER_SEED = 'unconstrained_optimizer'
    UNCONSTRAINED_OPTIMIZER_BASE_SEED = 'obstacle_boundary_subdivision'
    DUBINS_MIN_TURN_RADIUS = 0.5
    IPOPT_OPTIONS = {
        'ipopt.print_level': 0,
        'ipopt.tol': 1e-8,
        'ipopt.constr_viol_tol': 1e-8,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_constr_viol_tol': 1e-8,
    }

    def __init__(
        self,
        limits=None,
        bounds=None,
        n_segments=6,
        n_eval=10,
        n_coeff=8,
        time_bounds=(0.5, 60.0),
        epsilon=1e-3,
        obstacle_base_samples=41,
        obstacle_refine_multiplier=5,
        obstacle_refine_check_samples=None,
        obstacle_refine_margin=0.03,
        obstacle_refine_radius_fraction=0.25,
        obstacle_refine_min_spacing=0.01,
        obstacle_refine_max_spacing=0.25,
        boundary_bias_margin=0.5,
        boundary_bias_weight=50.0,
        obstacle_penalty_margin=0.1,
        obstacle_penalty_weight=50.0,
        w_time=10.0,
        w_acc=1.0,
        w_jerk=0.0,
        w_snap=1.0,
        kinematic_constraint_scale=0.975,
        w_yaw_rate=0.0,
        w_yaw_acc=0.0,
        backtrack_objective_weight=0.0,
        behind_start_objective_weight=0.0,
        gate_transition_distance=0.5,
        gate_transition_position_weight=10.0,
        gate_transition_velocity_weight=1.0,
        gate_transition_velocity_threshold=0.01,
        seed_strategy='obstacle_boundary_subdivision',
        ipopt_options=None,
    ):
        super().__init__(limits=limits, bounds=bounds, n_segments=n_segments)
        self.n_eval = int(n_eval)
        self.n_coeff = int(n_coeff)
        self.time_min = float(time_bounds[0])
        self.time_max = float(time_bounds[1])
        self.epsilon = float(epsilon)
        # Soft inward bias keeping the path off the (hard-constrained) flight-volume
        # walls; vanishes 'boundary_bias_margin' metres inside each wall. Weight 0
        # disables it.
        self.boundary_bias_margin = float(boundary_bias_margin)
        self.boundary_bias_weight = float(boundary_bias_weight)
        # Soft one-sided quadratic hinge steering the path off each (already
        # inflated) obstacle capsule/cylinder. Clearance is measured to the inflated
        # surface, so the penalty vanishes 'obstacle_penalty_margin' metres clear of
        # it (zero value and gradient) and rises quadratically as the path approaches
        # or enters. Replaces the old hard clearance constraint + stiff tanh wall;
        # dense validation is the actual safety guarantee. Weight 0 disables it.
        self.obstacle_penalty_margin = float(obstacle_penalty_margin)
        self.obstacle_penalty_weight = float(obstacle_penalty_weight)
        # Objective weights: minimum-time (w_time) traded off against integrated
        # control-effort costs on acceleration/jerk/snap. Raising w_acc/w_jerk buys
        # gentler, lower-acceleration paths (e.g. sweeping turns instead of
        # brake-and-reverse) at the cost of some flight time.
        self.w_time = float(w_time)
        self.w_acc = float(w_acc)
        self.w_jerk = float(w_jerk)
        self.w_snap = float(w_snap)
        self.kinematic_constraint_scale = float(kinematic_constraint_scale)
        if self.kinematic_constraint_scale <= 0.0 or self.kinematic_constraint_scale > 1.0:
            raise ValueError(
                'kinematic_constraint_scale must be in (0, 1]; '
                f'got {self.kinematic_constraint_scale}.'
            )
        # w_yaw_rate is the primary "don't demand sharp turns" knob (penalizes the
        # path-tangent yaw rate); w_yaw_acc keeps the heading motion from getting jerky.
        self.w_yaw_rate = float(w_yaw_rate)
        self.w_yaw_acc = float(w_yaw_acc)
        self.backtrack_objective_weight = float(backtrack_objective_weight)
        self.behind_start_objective_weight = float(behind_start_objective_weight)
        self.gate_transition_distance = float(gate_transition_distance)
        self.gate_transition_position_weight = float(gate_transition_position_weight)
        self.gate_transition_velocity_weight = float(gate_transition_velocity_weight)
        self.gate_transition_velocity_threshold = float(gate_transition_velocity_threshold)
        self.seed_strategy = str(seed_strategy).strip() or 'obstacle_boundary_subdivision'
        self.ipopt_options = dict(ipopt_options or {})
        valid_seed_strategies = set(seeding.SEED_STRATEGIES)
        valid_seed_strategies.add(self.UNCONSTRAINED_OPTIMIZER_SEED)
        if self.seed_strategy not in valid_seed_strategies:
            raise ValueError(
                f"Unknown seed_strategy '{self.seed_strategy}'. "
                f'Available: {sorted(valid_seed_strategies)}.'
            )
        self.obstacle_base_samples = int(obstacle_base_samples)
        # Dense re-check/re-solve grids default to a multiple of obstacle_base_samples
        # rather than an unrelated magic number: this guarantees the dense sample
        # count can never collapse to 1 (which would divide-by-zero downstream in
        # validation.py's t_eval sampling) without needing a separate guard at every
        # call site, and keeps it in lockstep with validation.KINEMATIC_LIMIT_SAMPLE_COUNT
        # (see the dense-kinematic-resolve note below). Pass obstacle_refine_check_samples
        # explicitly to override the derived value (e.g. in tests).
        self.obstacle_refine_multiplier = int(obstacle_refine_multiplier)
        if self.obstacle_refine_multiplier < 1:
            raise ValueError(
                'obstacle_refine_multiplier must be >= 1; '
                f'got {self.obstacle_refine_multiplier}.'
            )
        self.obstacle_refine_check_samples = (
            int(obstacle_refine_check_samples)
            if obstacle_refine_check_samples is not None
            else self.obstacle_base_samples * self.obstacle_refine_multiplier
        )
        self.obstacle_refine_margin = float(obstacle_refine_margin)
        self.obstacle_refine_radius_fraction = float(obstacle_refine_radius_fraction)
        self.obstacle_refine_min_spacing = float(obstacle_refine_min_spacing)
        self.obstacle_refine_max_spacing = float(obstacle_refine_max_spacing)

        # Populated per-plan from the effective limits/bounds; seeded with the
        # instance defaults so the standalone solver/plot helpers stay callable.
        self.v_max = self.limits['v_max']
        self.a_max = self.limits['a_max']
        self.j_max = self.limits['j_max']
        self.s_max = self.limits['s_max']
        self.yaw_rate_max = self.limits['yaw_rate_max']
        self.yaw_accel_max = self.limits['yaw_accel_max']
        self.z_height = self.bounds['z_max']
        self.obstacles = []
        self.last_seed_diagnostics = None
        self.last_solve_status = None
        self.last_objective_value = None
        self.last_solve_sec = None
        self.last_overall_sec = None
        self._last_planner_output_time = 0.0

    def _ipopt_options(self, overrides=None):
        options = dict(self.IPOPT_OPTIONS)
        options.update(self.ipopt_options)
        if overrides:
            options.update(overrides)
        options['print_time'] = False
        return options

    def _planner_output(self, message, *, force=False):
        now = time.monotonic()
        if not force and now - self._last_planner_output_time < 1.0:
            return
        self._last_planner_output_time = now
        print(message, flush=True)

    @staticmethod
    def _normalize_obstacles(obstacles, z_max):
        """Validate obstacles and resolve optional height to a concrete cylinder top."""
        normalized = []
        for obs in obstacles or []:
            if isinstance(obs, dict):
                otype = (obs.get('type') or 'cylinder').lower()
                radius = float(obs['radius'])
            else:
                otype = (getattr(obs, 'type', 'cylinder') or 'cylinder').lower()
                radius = float(obs.radius)
            if radius <= 0.0:
                raise ValueError(f'Obstacle radius must be positive; got {radius}.')

            if otype == 'capsule':
                getter = obs.get if isinstance(obs, dict) else lambda key: getattr(obs, key)
                normalized.append({
                    'type': 'capsule',
                    'ax': float(getter('ax')),
                    'ay': float(getter('ay')),
                    'az': float(getter('az')),
                    'bx': float(getter('bx')),
                    'by': float(getter('by')),
                    'bz': float(getter('bz')),
                    'radius': radius,
                })
                continue

            if otype != 'cylinder':
                raise ValueError(
                    f"Unsupported obstacle type '{otype}'; only 'cylinder' and 'capsule' are supported."
                )
            if isinstance(obs, dict):
                x, y = float(obs['x']), float(obs['y'])
                height = obs.get('height')
            else:
                x, y = float(obs.x), float(obs.y)
                height = getattr(obs, 'height', None)
            height = float(height) if height is not None else 0.0
            top = height if (height > 0.0 and not np.isnan(height)) else float(z_max)
            normalized.append({'type': 'cylinder', 'x': x, 'y': y, 'radius': radius, 'height': top})
        return normalized

    @staticmethod
    def _boundary_data(start, goal):
        sx, sy, sz = point_xyz(start['position'] if isinstance(start, dict) else start.position)
        svx, svy, svz = point_xyz(start['velocity'] if isinstance(start, dict) else start.velocity)
        ex, ey, ez = point_xyz(goal['position'] if isinstance(goal, dict) else goal.position)
        evx, evy, evz = point_xyz(goal['velocity'] if isinstance(goal, dict) else goal.velocity)

        zero = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        A_data = ({'x': sx, 'y': sy, 'z': sz}, {'x': svx, 'y': svy, 'z': svz},
                  dict(zero), dict(zero), dict(zero))
        B_data = ({'x': ex, 'y': ey, 'z': ez}, {'x': evx, 'y': evy, 'z': evz},
                  dict(zero), dict(zero), dict(zero))
        return A_data, B_data

    @staticmethod
    def _segment_sample_from_global_fraction(fraction, n_segments):
        bounded = min(1.0, max(0.0, float(fraction)))
        scaled = bounded * int(n_segments)
        nearest = round(scaled)
        if abs(scaled - nearest) <= 1e-9 and 0 < nearest <= int(n_segments):
            return int(nearest) - 1, 1.0
        segment = min(int(n_segments) - 1, max(0, int(math.floor(scaled))))
        return segment, scaled - segment

    def _estimate_seed_total_time(self, A_data, B_data, n_segments):
        start = np.asarray([A_data[0][axis] for axis in 'xyz'], dtype=np.float64)
        target = np.asarray([B_data[0][axis] for axis in 'xyz'], dtype=np.float64)
        distance = float(np.linalg.norm(target - start))
        speed = max(1e-6, float(self.v_max))
        return max(distance / speed, self.time_min * int(n_segments))

    def _seed_segments_from_data(self, A_data, B_data, total_time, n_segments, obstacles):
        seed_fn = seeding.SEED_STRATEGIES[self.seed_strategy]
        seed_kwargs = {}
        if self.seed_strategy in ('velocity_biased_chord', 'obstacle_boundary_subdivision', 'dubins'):
            seed_kwargs['vmax'] = self.v_max
        if self.seed_strategy == 'dubins':
            yaw_radius = (
                self.v_max / self.yaw_rate_max
                if float(self.yaw_rate_max) > 0.0 else 0.0
            )
            seed_kwargs['turning_radius'] = max(self.DUBINS_MIN_TURN_RADIUS, yaw_radius)
            seed_kwargs['start_tangent'] = getattr(self, '_seed_start_tangent', None)
            seed_kwargs['goal_tangent'] = getattr(self, '_seed_goal_tangent', None)
            seed_kwargs['z_axis'] = getattr(self, '_seed_z_axis', None)
            seed_kwargs['goal_z_axis'] = getattr(self, '_seed_goal_z_axis', None)
            seed_kwargs['return_diagnostics'] = True
        if self.seed_strategy == 'obstacle_boundary_subdivision':
            seed_kwargs['obstacles'] = obstacles
            seed_kwargs['return_diagnostics'] = True
        result = seed_fn(
            A_data,
            B_data,
            total_time,
            n_segments,
            self.n_coeff,
            self.time_min,
            **seed_kwargs,
        )
        if isinstance(result, tuple):
            segments, diagnostics = result
            self.last_seed_diagnostics = diagnostics
            return segments
        self.last_seed_diagnostics = None
        return result

    def _geometric_seed_segments_from_data(
        self,
        A_data,
        B_data,
        total_time,
        n_segments,
        obstacles,
        seed_strategy,
    ):
        previous_strategy = self.seed_strategy
        try:
            self.seed_strategy = seed_strategy
            return self._seed_segments_from_data(
                A_data,
                B_data,
                total_time,
                n_segments,
                obstacles,
            )
        finally:
            self.seed_strategy = previous_strategy

    def _unconstrained_optimizer_seed(self, start, goal, bounds, limits, n_segments):
        A_data, B_data = self._boundary_data(start, goal)
        total_time = self._estimate_seed_total_time(A_data, B_data, n_segments)
        geometric_seed = self._geometric_seed_segments_from_data(
            A_data,
            B_data,
            total_time,
            n_segments,
            [],
            self.UNCONSTRAINED_OPTIMIZER_BASE_SEED,
        )
        free_space_segments = self.plan(
            start,
            goal,
            obstacles=[],
            bounds=bounds,
            limits=limits,
            n_segments=len(geometric_seed),
            initial_segments=geometric_seed,
        )
        self.last_seed_diagnostics = {
            'strategy': self.UNCONSTRAINED_OPTIMIZER_SEED,
            'base_strategy': self.UNCONSTRAINED_OPTIMIZER_BASE_SEED,
            'base_segments': len(geometric_seed),
            'returned_segments': len(free_space_segments),
        }
        return [
            self._rescale_segment_to_normalized_time(segment)
            for segment in free_space_segments
        ]

    def build_seed(self, start, goal, obstacles=None, bounds=None, limits=None, n_segments=None):
        """Build the deterministic initial seed without constructing the NLP solver."""
        eff_bounds, eff_limits, eff_segments = self.resolve(bounds, limits, n_segments)
        self.v_max = eff_limits['v_max']
        self.a_max = eff_limits['a_max']
        self.j_max = eff_limits['j_max']
        self.s_max = eff_limits['s_max']
        self.yaw_rate_max = eff_limits['yaw_rate_max']
        self.yaw_accel_max = eff_limits['yaw_accel_max']
        self.z_height = eff_bounds['z_max']
        self.n_segments = eff_segments
        if self.seed_strategy == self.UNCONSTRAINED_OPTIMIZER_SEED:
            seed = self._unconstrained_optimizer_seed(
                start,
                goal,
                eff_bounds,
                eff_limits,
                eff_segments,
            )
            self.n_segments = len(seed)
            return seed
        A_data, B_data = self._boundary_data(start, goal)
        self._seed_start_tangent = (
            start.get('travel_tangent') if isinstance(start, dict) else None
        )
        self._seed_goal_tangent = (
            goal.get('travel_tangent') if isinstance(goal, dict) else None
        )
        self._seed_z_axis = (
            start.get('up_axis') if isinstance(start, dict) else None
        )
        self._seed_goal_z_axis = (
            goal.get('up_axis') if isinstance(goal, dict) else None
        )
        normalized_obstacles = self._normalize_obstacles(obstacles, eff_bounds['z_max'])
        total_time = self._estimate_seed_total_time(A_data, B_data, eff_segments)
        seed = self._seed_segments_from_data(
            A_data,
            B_data,
            total_time,
            eff_segments,
            normalized_obstacles,
        )
        self.n_segments = len(seed)
        return seed

    def build_base_seed(self, start, goal, obstacles=None, bounds=None, limits=None, n_segments=None):
        """Build the configured geometric seed without running the unconstrained optimizer."""
        if self.seed_strategy != self.UNCONSTRAINED_OPTIMIZER_SEED:
            return self.build_seed(start, goal, obstacles, bounds, limits, n_segments)
        previous_strategy = self.seed_strategy
        try:
            self.seed_strategy = self.UNCONSTRAINED_OPTIMIZER_BASE_SEED
            return self.build_seed(start, goal, obstacles, bounds, limits, n_segments)
        finally:
            self.seed_strategy = previous_strategy

    def plan(
        self,
        start,
        goal,
        obstacles=None,
        bounds=None,
        limits=None,
        n_segments=None,
        initial_segments=None,
        gate_transition_targets=None,
        ipopt_options=None,
        accepted_statuses=None,
    ):
        eff_bounds, eff_limits, eff_segments = self.resolve(bounds, limits, n_segments)
        self.v_max = eff_limits['v_max']
        self.a_max = eff_limits['a_max']
        self.j_max = eff_limits['j_max']
        self.s_max = eff_limits['s_max']
        self.yaw_rate_max = eff_limits['yaw_rate_max']
        self.yaw_accel_max = eff_limits['yaw_accel_max']
        self.z_height = eff_bounds['z_max']
        self.n_segments = eff_segments

        A_data, B_data = self._boundary_data(start, goal)
        gate_transition_targets = list(gate_transition_targets or [])

        p_lim = {
            'x': (eff_bounds['x_min'], eff_bounds['x_max']),
            'y': (eff_bounds['y_min'], eff_bounds['y_max']),
            'z': (eff_bounds['z_min'], eff_bounds['z_max']),
        }
        constraint_scale = self.kinematic_constraint_scale
        v_lim = (-self.v_max * constraint_scale, self.v_max * constraint_scale)
        a_lim = (-self.a_max * constraint_scale, self.a_max * constraint_scale)
        j_lim = (-self.j_max * constraint_scale, self.j_max * constraint_scale)
        s_lim = (-self.s_max * constraint_scale, self.s_max * constraint_scale)

        normalized_obstacles = self._normalize_obstacles(obstacles, eff_bounds['z_max'])
        self.obstacles = normalized_obstacles
        self._report_boundary_obstacle_clearance(A_data[0], B_data[0], normalized_obstacles)
        if initial_segments is None:
            raise ValueError('CasadiObstaclePlanner.plan requires initial_segments; call build_seed first.')

        segments = self.poly_segment_time_optimal(
            A_data, B_data, p_lim, v_lim, a_lim, j_lim, s_lim,
            self.n_eval, self.n_coeff, self.n_segments, normalized_obstacles,
            gate_transition_targets=gate_transition_targets,
            initial_segments=initial_segments,
            ipopt_options=ipopt_options,
            accepted_statuses=accepted_statuses)
        refinements = self._find_obstacle_refinements(segments, normalized_obstacles)
        yaw_rate_refinements = self._find_yaw_rate_refinements(segments)
        dense_kinematic_segments = self._kinematic_overshoot_segments(segments)
        if refinements or yaw_rate_refinements or dense_kinematic_segments:
            sample_count = sum(len(samples) for samples in refinements.values())
            yaw_sample_count = sum(len(samples) for samples in yaw_rate_refinements.values())
            self._planner_output(
                'planner refinement: '
                f'obstacle_pairs={len(refinements)} extra_samples={sample_count} '
                f'yaw_rate_segments={len(yaw_rate_refinements)} '
                f'yaw_extra_samples={yaw_sample_count} '
                f'kinematic_dense_segments={len(dense_kinematic_segments)} '
                f'kinematic_dense_samples={max(int(self.obstacle_refine_check_samples), self.n_eval)}',
                force=True,
            )
            segments = self.poly_segment_time_optimal(
                A_data, B_data, p_lim, v_lim, a_lim, j_lim, s_lim,
                self.n_eval, self.n_coeff, self.n_segments, normalized_obstacles,
                gate_transition_targets=gate_transition_targets,
                obstacle_sample_overrides=refinements,
                yaw_rate_sample_overrides=yaw_rate_refinements,
                dense_kinematic_segments=dense_kinematic_segments,
                initial_segments=segments,
                ipopt_options=ipopt_options,
                accepted_statuses=accepted_statuses,
            )

        # The optimizer works in normalized segment time s in [0, 1]; convert the
        # coefficients to real time t in [0, T] so every planner returns the same
        # convention x(t) = sum_k c_k t^k. (Previously done in the action server.)
        return [self._rescale_segment_to_real_time(seg) for seg in segments]

    @staticmethod
    def _format_point(point):
        return f"({float(point['x']):.3f}, {float(point['y']):.3f}, {float(point['z']):.3f})"

    @staticmethod
    def _boundary_clearance(point, obstacle):
        if obstacle.get('type') == 'capsule':
            distance = CasadiObstaclePlanner._point_to_capsule_distance(
                (point['x'], point['y'], point['z']),
                obstacle,
            )
            return distance, distance - float(obstacle['radius'])
        if float(point['z']) > float(obstacle['height']):
            return math.inf, float(point['z']) - float(obstacle['height'])
        distance_xy = math.hypot(
            float(point['x']) - float(obstacle['x']),
            float(point['y']) - float(obstacle['y']),
        )
        clearance_xy = distance_xy - float(obstacle['radius'])
        return distance_xy, clearance_xy

    @classmethod
    def _closest_boundary_obstacle(cls, point, obstacles):
        # 'distance'/'clearance' are 2D (xy) for cylinders and full 3D for
        # capsules; the output labels below disambiguate which model produced them.
        closest = None
        for index, obstacle in enumerate(obstacles):
            distance, clearance = cls._boundary_clearance(point, obstacle)
            candidate = {
                'index': index,
                'obstacle': obstacle,
                'distance': distance,
                'clearance': clearance,
            }
            if closest is None or clearance < closest['clearance']:
                closest = candidate
        return closest

    def _report_boundary_obstacle_clearance(self, start_pos, target_pos, obstacles):
        if not obstacles:
            self._planner_output(
                'planner obstacle precheck: no obstacles; '
                f'start={self._format_point(start_pos)}, target={self._format_point(target_pos)}',
                force=True,
            )
            return

        violations = []
        for label, point in (('start', start_pos), ('target', target_pos)):
            closest = self._closest_boundary_obstacle(point, obstacles)
            obstacle = closest['obstacle']
            if obstacle.get('type') == 'capsule':
                obstacle_text = (
                    f"a=({float(obstacle['ax']):.3f}, {float(obstacle['ay']):.3f}, {float(obstacle['az']):.3f}), "
                    f"b=({float(obstacle['bx']):.3f}, {float(obstacle['by']):.3f}, {float(obstacle['bz']):.3f})"
                )
                distance_label = 'distance'
                clearance_label = 'clearance'
            else:
                obstacle_text = (
                    f"center=({float(obstacle['x']):.3f}, {float(obstacle['y']):.3f}), "
                    f"height={float(obstacle['height']):.3f}"
                )
                distance_label = 'xy_distance'
                clearance_label = 'clearance'
            self._planner_output(
                'planner obstacle precheck: '
                f'{label}={self._format_point(point)}, '
                f'closest_obstacle={closest["index"]}, '
                f'{obstacle_text}, '
                f'radius={float(obstacle["radius"]):.3f}, '
                f'{distance_label}={closest["distance"]:.4f}, '
                f'{clearance_label}={closest["clearance"]:.4f}',
                force=True,
            )
            if closest['clearance'] < -1e-9:
                if obstacle.get('type') == 'capsule':
                    location_text = (
                        f"a=({float(obstacle['ax']):.3f}, {float(obstacle['ay']):.3f}, {float(obstacle['az']):.3f}), "
                        f"b=({float(obstacle['bx']):.3f}, {float(obstacle['by']):.3f}, {float(obstacle['bz']):.3f})"
                    )
                else:
                    location_text = (
                        f"center=({float(obstacle['x']):.3f}, {float(obstacle['y']):.3f})"
                    )
                violations.append(
                    f'{label} {self._format_point(point)} is inside obstacle '
                    f'{closest["index"]}: clearance={closest["clearance"]:.4f} m '
                    f'({location_text}, radius={float(obstacle["radius"]):.3f})'
                )

        if violations:
            raise ValueError(
                'Boundary obstacle precheck failed under the current hard obstacle '
                'clearance model: ' + '; '.join(violations)
            )

    @staticmethod
    def _evaluate_normalized_axis(segment, axis, s):
        coeffs = segment[f'c_{axis}']
        return sum(float(coeffs[k]) * float(s) ** k for k in range(len(coeffs)))

    @staticmethod
    def _point_to_capsule_distance(point, obstacle):
        point_vec = np.asarray(point, dtype=np.float64)
        a_vec = np.asarray(
            [obstacle['ax'], obstacle['ay'], obstacle['az']],
            dtype=np.float64,
        )
        b_vec = np.asarray(
            [obstacle['bx'], obstacle['by'], obstacle['bz']],
            dtype=np.float64,
        )
        segment = b_vec - a_vec
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            closest = a_vec
        else:
            t = float(np.dot(point_vec - a_vec, segment) / length_sq)
            closest = a_vec + max(0.0, min(1.0, t)) * segment
        return float(np.linalg.norm(point_vec - closest))

    @classmethod
    def _obstacle_distance(cls, point, obstacle):
        if obstacle.get('type') == 'capsule':
            return cls._point_to_capsule_distance(point, obstacle)
        if float(point[2]) > float(obstacle['height']):
            return math.inf
        return math.hypot(
            float(point[0]) - float(obstacle['x']),
            float(point[1]) - float(obstacle['y']),
        )

    @classmethod
    def _estimate_xy_path_length(cls, segment, sample_count=80):
        count = max(int(sample_count), 2)
        s_vals = np.linspace(0.0, 1.0, count)
        points = np.array([
            [
                cls._evaluate_normalized_axis(segment, 'x', s),
                cls._evaluate_normalized_axis(segment, 'y', s),
            ]
            for s in s_vals
        ])
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    @staticmethod
    def _merge_close_sample_windows(s_vals, close_mask, pad_s):
        windows = []
        start = None
        end = None
        for s, is_close in zip(s_vals, close_mask):
            if is_close:
                if start is None:
                    start = float(s)
                end = float(s)
            elif start is not None:
                windows.append((max(0.0, start - pad_s), min(1.0, end + pad_s)))
                start = None
                end = None
        if start is not None:
            windows.append((max(0.0, start - pad_s), min(1.0, end + pad_s)))
        return windows

    def _refinement_spacing(self, radius):
        spacing = float(radius) * self.obstacle_refine_radius_fraction
        return min(
            self.obstacle_refine_max_spacing,
            max(self.obstacle_refine_min_spacing, spacing),
        )

    def _find_obstacle_refinements(self, segments, obstacles):
        if not obstacles:
            return {}

        check_count = max(int(self.obstacle_refine_check_samples), 3)
        s_vals = np.linspace(0.0, 1.0, check_count)
        refinements = {}
        for segment_index, segment in enumerate(segments):
            path_length = self._estimate_xy_path_length(segment)
            for obstacle_index, obstacle in enumerate(obstacles):
                radius = float(obstacle['radius'])
                margin = max(0.0, self.obstacle_refine_margin)
                threshold = radius + margin
                distances = []
                for s in s_vals:
                    x_val = self._evaluate_normalized_axis(segment, 'x', s)
                    y_val = self._evaluate_normalized_axis(segment, 'y', s)
                    z_val = self._evaluate_normalized_axis(segment, 'z', s)
                    distances.append(self._obstacle_distance((x_val, y_val, z_val), obstacle))
                close_mask = np.asarray(distances) <= threshold
                if not np.any(close_mask):
                    continue

                spacing = self._refinement_spacing(radius)
                if path_length <= 1e-9:
                    samples_per_s = check_count
                else:
                    samples_per_s = max(int(math.ceil(path_length / spacing)), check_count)
                pad_s = max(2.0 / (check_count - 1), spacing / max(path_length, spacing))
                samples = set()
                for start_s, end_s in self._merge_close_sample_windows(s_vals, close_mask, pad_s):
                    count = max(3, int(math.ceil((end_s - start_s) * samples_per_s)) + 1)
                    samples.update(float(s) for s in np.linspace(start_s, end_s, count))
                refinements[(segment_index, obstacle_index)] = sorted(samples)

        return refinements

    @staticmethod
    def _evaluate_normalized_axis_derivative(segment, axis, s, order):
        """Evaluate the order-th derivative w.r.t. normalized time u of one axis at u=s."""
        coeffs = segment[f'c_{axis}']
        s = float(s)
        total = 0.0
        for k in range(order, len(coeffs)):
            falling = 1.0
            for offset in range(order):
                falling *= (k - offset)
            total += float(coeffs[k]) * falling * s ** (k - order)
        return total

    def _tangent_yaw_rate(self, segment, s):
        """Regularized path-tangent yaw rate at normalized parameter s, in real-time units."""
        duration = max(float(segment['T']), 1e-9)
        vx = self._evaluate_normalized_axis_derivative(segment, 'x', s, 1) / duration
        vy = self._evaluate_normalized_axis_derivative(segment, 'y', s, 1) / duration
        ax = self._evaluate_normalized_axis_derivative(segment, 'x', s, 2) / duration ** 2
        ay = self._evaluate_normalized_axis_derivative(segment, 'y', s, 2) / duration ** 2
        speed_eps_sq = (0.05 * max(float(self.v_max), 1e-3)) ** 2
        denom = vx * vx + vy * vy + speed_eps_sq
        return (vx * ay - vy * ax) / denom

    def _find_yaw_rate_refinements(self, segments):
        """
        Find segments whose tangent yaw rate overshoots the cap between collocation points.

        The hard yaw-rate constraint is imposed only at the coarse ``n_eval`` samples, so
        a dense re-check can find overshoot in between. Return, per segment, extra
        normalized-time points to add as collocation points on the re-solve so the
        in-solve cap is re-imposed there (mirrors obstacle refinement).
        """
        yaw_rate_max = float(self.yaw_rate_max)
        if yaw_rate_max <= 0.0:
            return {}
        check_count = max(int(self.obstacle_refine_check_samples), 3)
        s_vals = np.linspace(0.0, 1.0, check_count)
        threshold = yaw_rate_max * 1.05  # only refine meaningful (>5%) overshoot
        refinements = {}
        for segment_index, segment in enumerate(segments):
            rates = np.array([abs(self._tangent_yaw_rate(segment, s)) for s in s_vals])
            mask = rates > threshold
            if not np.any(mask):
                continue
            extra = set()
            for i in np.where(mask)[0]:
                lo = max(0, i - 1)
                hi = min(len(s_vals) - 1, i + 1)
                extra.update(float(s_vals[j]) for j in range(lo, hi + 1))
            refinements[segment_index] = sorted(extra)
        return refinements

    def _kinematic_overshoot_segments(self, segments):
        """
        Return the indices of segments whose dense XYZ kinematic scan exceeds a limit.

        The NLP constrains velocity/acceleration/jerk/snap only at the coarse
        ``n_eval`` collocation points, so a peak can hide between them. A dense
        re-solve constrains the whole offending segment (see
        ``dense_kinematic_segments``); we deliberately do not pin the exact peak,
        because constraining a single spot just pushes the violation to the nearest
        un-sampled point. This pass only flags *which* segments need that dense
        re-solve.
        """
        limits_by_order = {
            1: float(self.v_max),
            2: float(self.a_max),
            3: float(self.j_max),
            4: float(self.s_max),
        }
        check_count = max(int(self.obstacle_refine_check_samples), 3)
        s_vals = np.linspace(0.0, 1.0, check_count)
        overshoot_segments = set()
        for segment_index, segment in enumerate(segments):
            duration = max(float(segment['T']), 1e-9)
            for derivative_order, limit in limits_by_order.items():
                if limit <= 0.0:
                    continue
                threshold = limit * 1.001
                scale = duration ** derivative_order
                for axis in 'xyz':
                    values = np.array([
                        abs(
                            self._evaluate_normalized_axis_derivative(
                                segment, axis, s, derivative_order)
                            / scale
                        )
                        for s in s_vals
                    ])
                    if np.any(values > threshold):
                        overshoot_segments.add(segment_index)
                        break
                if segment_index in overshoot_segments:
                    break
        return overshoot_segments

    def kinematic_time_dilation_factors(self, segments, overshoot_threshold=1.001, margin=1.02):
        """
        Per-segment time-stretch factors that pull dense kinematic peaks within limits.

        Operates on normalized segments (normalized coeffs + duration ``T``). For each
        segment, scans the dense grid for the worst v/a/j/s overshoot; since
        ``|deriv| ~ 1/T**order`` for a fixed normalized shape, stretching the segment
        duration by ``(peak / limit)**(1/order)`` brings that peak to the limit. The
        returned factor (>1) is the max such stretch across orders/axes, with a small
        ``margin`` of headroom. Segments within limits are omitted. Intended as a cheap
        seed conditioner: it preserves segment-endpoint positions (only the normalized
        ``T`` changes), so the only seam discontinuity is in velocity and higher, which
        the downstream optimizer re-imposes as hard continuity constraints.
        """
        limits_by_order = {
            1: float(self.v_max),
            2: float(self.a_max),
            3: float(self.j_max),
            4: float(self.s_max),
        }
        check_count = max(int(self.obstacle_refine_check_samples), 3)
        s_vals = np.linspace(0.0, 1.0, check_count)
        factors = {}
        for segment_index, segment in enumerate(segments):
            duration = max(float(segment['T']), 1e-9)
            factor = 1.0
            for derivative_order, limit in limits_by_order.items():
                if limit <= 0.0:
                    continue
                scale = duration ** derivative_order
                peak = 0.0
                for axis in 'xyz':
                    peak = max(peak, max(
                        abs(
                            self._evaluate_normalized_axis_derivative(
                                segment, axis, s, derivative_order)
                            / scale
                        )
                        for s in s_vals
                    ))
                if peak > limit * overshoot_threshold:
                    factor = max(factor, (peak / limit) ** (1.0 / derivative_order))
            if factor > 1.0:
                factors[segment_index] = factor * float(margin)
        return factors

    def dilate_segment_durations(self, segments, factors):
        """
        Return normalized segments with the given per-index durations time-stretched.

        Only ``T`` is scaled (normalized coefficients are untouched), so each segment
        traces the same geometric path more slowly; endpoint positions are preserved.
        Durations are clamped to ``[time_min, time_max]``.
        """
        dilated = []
        for index, segment in enumerate(segments):
            new_segment = dict(segment)
            factor = float(factors.get(index, 1.0))
            new_T = float(segment['T']) * factor
            new_segment['T'] = min(self.time_max, max(self.time_min, new_T))
            dilated.append(new_segment)
        return dilated

    def _rescale_segment_to_real_time(self, segment):
        seconds = float(segment['T'])
        scale = 1.0 / np.array([seconds ** i for i in range(self.n_coeff)])
        rescaled = {'T': seconds}
        for axis in ('c_x', 'c_y', 'c_z'):
            rescaled[axis] = np.asarray(segment[axis], dtype=np.float64) * scale
        return rescaled

    def _rescale_segment_to_normalized_time(self, segment):
        seconds = float(segment['T'])
        scale = np.array([seconds ** i for i in range(self.n_coeff)])
        rescaled = {'T': seconds}
        for axis in ('c_x', 'c_y', 'c_z'):
            coeffs = self._pad_coefficients(segment[axis], self.n_coeff)[:self.n_coeff]
            rescaled[axis] = np.asarray(coeffs, dtype=np.float64) * scale
        return rescaled

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): CasadiObstaclePlanner._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [CasadiObstaclePlanner._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return [CasadiObstaclePlanner._json_safe(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @classmethod
    def _write_failed_solver_artifacts(
        cls,
        *,
        return_status,
        solver_stats,
        segments,
        timing,
        artifact_root=None,
    ):
        artifact_root = Path(artifact_root or cls.FAILURE_ARTIFACT_ROOT)
        artifact_dir = artifact_root / f'flexible_drones_planner_failed_{int(time.time())}_{uuid.uuid4().hex[:8]}'
        artifact_dir.mkdir(parents=True, exist_ok=True)

        stats_path = artifact_dir / 'solver_stats.json'
        stats_payload = {
            'return_status': return_status,
            'accepted_statuses': sorted(cls.SUCCESS_STATUSES),
            'timing': cls._json_safe(timing),
            'solver_stats': cls._json_safe(solver_stats),
        }
        stats_path.write_text(json.dumps(stats_payload, indent=2, sort_keys=True), encoding='utf-8')

        trajectory_path = artifact_dir / 'candidate_trajectory.csv'
        cls._write_standard_trajectory_csv(trajectory_path, segments)

        debug_path = artifact_dir / 'candidate_segments_debug.csv'
        with debug_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['segment', 'duration', 'axis', *[f'c{i}' for i in range(cls._max_coeff_count(segments))]])
            for segment_index, segment in enumerate(segments):
                for axis in ('c_x', 'c_y', 'c_z'):
                    coeffs = [float(value) for value in segment.get(axis, [])]
                    writer.writerow([segment_index, float(segment['T']), axis, *coeffs])

        return artifact_dir

    @staticmethod
    def _write_standard_trajectory_csv(path, segments):
        durations = [float(segment['T']) for segment in segments]
        x_coeffs = []
        y_coeffs = []
        z_coeffs = []
        yaw_coeffs = []

        for segment in segments:
            coeff_count = max(
                8,
                len(segment.get('c_x', [])),
                len(segment.get('c_y', [])),
                len(segment.get('c_z', [])),
            )
            x_coeffs.append(CasadiObstaclePlanner._pad_coefficients(segment.get('c_x', []), coeff_count))
            y_coeffs.append(CasadiObstaclePlanner._pad_coefficients(segment.get('c_y', []), coeff_count))
            z_coeffs.append(CasadiObstaclePlanner._pad_coefficients(segment.get('c_z', []), coeff_count))
            yaw_coeffs.append([0.0] * coeff_count)

        save_trajectory_csv(
            path,
            durations,
            x_coeffs,
            y_coeffs,
            z_coeffs,
            yaw_coeffs,
            coefficient_order='drone',
        )

    @classmethod
    def write_stage_artifact(cls, stage, segments, artifact_root=None):
        artifact_root = Path(artifact_root or cls.FAILURE_ARTIFACT_ROOT)
        artifact_root.mkdir(parents=True, exist_ok=True)
        safe_stage = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(stage)).strip('_') or 'stage'
        path = artifact_root / (
            f'flexible_drones_planner_{safe_stage}_{int(time.time())}_{uuid.uuid4().hex[:8]}.csv'
        )
        cls._write_standard_trajectory_csv(path, segments)
        return path

    @staticmethod
    def _pad_coefficients(coefficients, coeff_count):
        values = [float(value) for value in coefficients]
        return values + [0.0] * (coeff_count - len(values))

    @staticmethod
    def _max_coeff_count(segments):
        max_count = 0
        for segment in segments:
            for axis in ('c_x', 'c_y', 'c_z'):
                max_count = max(max_count, len(segment.get(axis, [])))
        return max_count

    # Assume x_sym and y_sym are symbolic CasADi expressions for your drone's trajectory
    def casadi_cylinder_with_top(self, x_sym, y_sym, z_sym,
                                 xc0, yc0, z_max, rc,
                                 scale_out=10_000, scale_in=500,
                                 slope_out=1000.0):

        # Distances
        d_cyl = ca.sqrt((x_sym - xc0)**2 + (y_sym - yc0)**2 + self.epsilon**2)
        # d_sphere = ca.sqrt((x_sym - xc0)**2 + (y_sym - yc0)**2 + (z_sym - z_max)**2)

        # Distance from the obstacle's surface. neg:inside, zero:on surface, pos:outside
        # Delta for tanh shaping
        delta_cyl = d_cyl - rc

        # Inflict huge penalty in cylinder to discourage flight pathing there
        # Penalty for cylindrical region
        inside_cyl = scale_out * scale_in * ca.tanh(-slope_out * delta_cyl / scale_in) + scale_out
        outside_cyl = scale_out * (ca.tanh(-slope_out * delta_cyl) + 1)
        penalty_cyl = ca.if_else(d_cyl < rc, inside_cyl, outside_cyl)

        # Case handling by z height
        use_cylinder = z_sym <= z_max
        use_sphere_cap = ca.logic_and(z_sym > z_max, z_sym <= z_max + rc)

        # Just decrease the impact above cylinder linearly
        penalty = ca.if_else(use_cylinder, penalty_cyl,
                             ca.if_else(use_sphere_cap, penalty_cyl * (z_max + rc - z_sym) / rc, 0.0))
        return penalty

    @staticmethod
    def casadi_cylinder_clearance_constraint(x_sym, y_sym, z_sym, obstacle):
        distance_sq = (x_sym - obstacle['x'])**2 + (y_sym - obstacle['y'])**2
        radial_clearance = distance_sq - obstacle['radius']**2
        return ca.if_else(z_sym <= obstacle['height'], radial_clearance, 0.0)

    def casadi_capsule_distance_sq(self, x_sym, y_sym, z_sym, obstacle):
        point = ca.vertcat(x_sym, y_sym, z_sym)
        a_vec = ca.vertcat(obstacle['ax'], obstacle['ay'], obstacle['az'])
        b_vec = ca.vertcat(obstacle['bx'], obstacle['by'], obstacle['bz'])
        segment = b_vec - a_vec
        length_sq = ca.dot(segment, segment)
        t_raw = ca.if_else(
            length_sq > 1e-12,
            ca.dot(point - a_vec, segment) / length_sq,
            0.0,
        )
        t = ca.fmin(1.0, ca.fmax(0.0, t_raw))
        closest = a_vec + t * segment
        delta = point - closest
        return ca.dot(delta, delta)

    def casadi_capsule_penalty(self, x_sym, y_sym, z_sym, obstacle,
                               scale_out=10_000, scale_in=500,
                               slope_out=1000.0):
        # Stiff smooth tanh penalty that shapes the path away from the capsule;
        # paired with a hard distance constraint elsewhere. See README.md
        # (Obstacles and boundaries).
        radius = obstacle['radius']
        distance = ca.sqrt(self.casadi_capsule_distance_sq(x_sym, y_sym, z_sym, obstacle) + self.epsilon**2)
        delta = distance - radius
        inside = scale_out * scale_in * ca.tanh(-slope_out * delta / scale_in) + scale_out
        outside = scale_out * (ca.tanh(-slope_out * delta) + 1)
        return ca.if_else(distance < radius, inside, outside)

    @staticmethod
    def _trapezoid_weights(sorted_s_vals):
        """
        Composite-trapezoid quadrature weights over sorted_s_vals, normalized to sum to 1.

        Unlike a flat 1/N weight, this makes the weighted sum approximate the same
        integral regardless of how many (non-uniformly spaced) samples are in the
        array: adding samples near one region increases the local resolution of
        the estimate there without diluting the weight carried by samples
        elsewhere, and without inflating the total once the extra samples land in
        already-covered territory.
        """
        n = len(sorted_s_vals)
        if n == 1:
            return np.array([1.0])
        weights = np.empty(n, dtype=np.float64)
        weights[1:-1] = (sorted_s_vals[2:] - sorted_s_vals[:-2]) / 2.0
        weights[0] = (sorted_s_vals[1] - sorted_s_vals[0]) / 2.0
        weights[-1] = (sorted_s_vals[-1] - sorted_s_vals[-2]) / 2.0
        total = weights.sum()
        if total <= 1e-12:
            return np.full(n, 1.0 / n)
        return weights / total

    def casadi_obstacle_clearance_penalty(self, x_sym, y_sym, z_sym, obstacle, weight=None):
        """
        Soft one-sided quadratic hinge keeping the path off an inflated obstacle.

        The obstacle radius already encodes the safety inflation, so 'clearance' is
        the signed distance to that inflated surface. The penalty is zero (value and
        gradient) once the path is 'obstacle_penalty_margin' metres clear of the
        surface, and rises quadratically as it approaches or crosses it -- the same
        margin-band hinge as casadi_boundary_bias. Unlike the previous stiff tanh
        wall this leaves the free-space geometry untouched away from obstacles and
        degrades gracefully to the obstacle-free solve when nothing is close; dense
        validation remains the hard guarantee. Weight 0 disables it.
        """
        margin = self.obstacle_penalty_margin
        weight = self.obstacle_penalty_weight if weight is None else float(weight)
        if weight <= 0.0 or margin <= 0.0:
            return 0.0
        if obstacle.get('type') == 'capsule':
            distance = ca.sqrt(
                self.casadi_capsule_distance_sq(x_sym, y_sym, z_sym, obstacle)
                + self.epsilon**2)
            clearance = distance - obstacle['radius']
            encroach = ca.fmax(0.0, margin - clearance) / margin
            return weight * encroach**2
        # Cylinder-with-top: penalize the radial encroachment below the top and taper
        # it over the spherical cap, matching casadi_cylinder_with_top's z handling.
        rc = obstacle['radius']
        z_top = obstacle.get('height', self.z_height)
        distance = ca.sqrt(
            (x_sym - obstacle['x'])**2 + (y_sym - obstacle['y'])**2 + self.epsilon**2)
        clearance = distance - rc
        encroach = ca.fmax(0.0, margin - clearance) / margin
        penalty_xy = weight * encroach**2
        use_cyl = z_sym <= z_top
        use_cap = ca.logic_and(z_sym > z_top, z_sym <= z_top + rc)
        return ca.if_else(
            use_cyl,
            penalty_xy,
            ca.if_else(use_cap, penalty_xy * (z_top + rc - z_sym) / rc, 0.0))

    def casadi_boundary_bias(self, x_sym, y_sym, z_sym, p_lim):
        """
        Soft inward bias from the flight-volume walls.

        Normalized one-sided quadratic ramp per wall: zero (with zero slope)
        beyond 'boundary_bias_margin' inside a wall, rising to 'boundary_bias_weight'
        at the wall. C1 so IPOPT stays happy; bounded because the walls are hard
        constraints, so distance never goes negative.
        See README.md (Obstacles and boundaries).
        """
        margin = self.boundary_bias_margin
        weight = self.boundary_bias_weight
        if weight <= 0.0 or margin <= 0.0:
            return 0.0
        cost = 0.0
        for value, axis in ((x_sym, 'x'), (y_sym, 'y'), (z_sym, 'z')):
            lo, hi = p_lim[axis][0], p_lim[axis][1]
            for dist in (value - lo, hi - value):
                encroach = ca.fmax(0.0, margin - dist) / margin
                cost += weight * encroach**2
        return cost

    def casadi_gate_transition_target_cost(
        self,
        x_coeffs,
        y_coeffs,
        z_coeffs,
        T,
        s,
        target,
        t_ref,
    ):
        """Softly pull a segment sample toward a gate transition point and velocity."""
        B = self.poly_basis(float(s), self.n_coeff)
        dB = self.dpoly_basis(float(s), self.n_coeff)
        position_scale = max(float(self.gate_transition_distance), 0.1)
        velocity_scale = max(float(self.v_max), 0.1)

        x_i = ca.dot(B, x_coeffs)
        y_i = ca.dot(B, y_coeffs)
        z_i = ca.dot(B, z_coeffs)
        target_position = target['position']
        cost = 0.0
        if self.gate_transition_position_weight > 0.0:
            position_error_sq = (
                (x_i - float(target_position[0]))**2
                + (y_i - float(target_position[1]))**2
                + (z_i - float(target_position[2]))**2
            ) / (position_scale**2)
            cost += self.gate_transition_position_weight * position_error_sq

        target_velocity = target.get('velocity')
        if target_velocity is not None and self.gate_transition_velocity_weight > 0.0:
            vx_i = ca.dot(dB, x_coeffs) / T
            vy_i = ca.dot(dB, y_coeffs) / T
            vz_i = ca.dot(dB, z_coeffs) / T
            velocity_error_sq = (
                (vx_i - float(target_velocity[0]))**2
                + (vy_i - float(target_velocity[1]))**2
                + (vz_i - float(target_velocity[2]))**2
            ) / (velocity_scale**2)
            # Match integrated effort terms' time normalization loosely: longer
            # trajectories should not make a fixed gate guide effectively stronger.
            cost += self.gate_transition_velocity_weight * velocity_error_sq * T / t_ref
        return cost

    def _reference_time(self, p_lim):
        """
        Characteristic time used to normalize the objective.

        Time to traverse the flight-volume diagonal (its longest straight path) at
        half of v_max -- a constant per (bounds, limits) configuration, so the
        objective weights stay dimensionless and the kinematic-vs-penalty balance
        is predictable. See README.md (Choosing T_ref).
        """
        diagonal = math.sqrt(sum(
            (float(p_lim[axis][1]) - float(p_lim[axis][0]))**2 for axis in 'xyz'
        ))
        half_v = 0.5 * max(float(self.v_max), 1e-6)
        return max(diagonal / half_v, 1e-3)

    def casadi_effort_terms(self, x_coeffs, y_coeffs, z_coeffs, T, t_ref, s_vals, ds):
        """
        Integrated, limit-normalized control-effort cost for one segment.

        For each active weight, adds ``w * integral( (|deriv| / limit)^2 ) dt / t_ref``
        discretized as ``w * (|deriv|/limit)^2 * (T*ds) / t_ref``. Dividing each
        derivative by its kinematic limit makes the term dimensionless and -- since
        the limits are enforced constraints -- bounded, removing the 1/T^k blow-up.
        See README.md (Weight normalization) for the full rationale.
        """
        specs = []
        if self.w_acc > 0.0:
            specs.append((self.w_acc, self.ddpoly_basis, 2, self.a_max))
        if self.w_jerk > 0.0:
            specs.append((self.w_jerk, self.dddpoly_basis, 3, self.j_max))
        if self.w_snap > 0.0:
            specs.append((self.w_snap, self.ddddpoly_basis, 4, self.s_max))

        cost = 0.0
        for weight, basis_fn, order, limit in specs:
            limit_sq = max(float(limit), 1e-9)**2
            for s in s_vals:
                Bd = basis_fn(float(s), self.n_coeff)
                dx = ca.dot(Bd, x_coeffs) / T**order
                dy = ca.dot(Bd, y_coeffs) / T**order
                dz = ca.dot(Bd, z_coeffs) / T**order
                normalized = (dx**2 + dy**2 + dz**2) / limit_sq
                cost += weight * normalized * (T * ds) / t_ref
        return cost

    def casadi_yaw_rate_terms(self, x_coeffs, y_coeffs, T, t_ref, s_vals, ds):
        """
        Penalize the regularized path-tangent yaw rate implied by x/y motion.

        The tangent heading is ``yaw = atan2(vy, vx)`` and its rate is
        ``(vx*ay - vy*ax) / (vx^2 + vy^2)``. A small ``speed_eps^2`` is added to the
        denominator so the expression stays well-conditioned near hover. This is the
        primary "don't demand sharp turns" knob: unlike the acceleration term, it also
        penalizes a *sustained* fast turn (which has zero yaw acceleration). The speed
        gate keeps near-hover samples, where tangent yaw is ill-defined, from dominating.
        """
        if self.w_yaw_rate <= 0.0:
            return 0.0

        speed_eps = 0.05 * max(float(self.v_max), 1e-3)
        speed_eps_sq = speed_eps**2
        yaw_rate_ref = max(float(self.a_max) / max(float(self.v_max), 1e-3), 1e-3)
        yaw_rate_ref_sq = yaw_rate_ref**2

        cost = 0.0
        for s in s_vals:
            dB = self.dpoly_basis(float(s), self.n_coeff)
            ddB = self.ddpoly_basis(float(s), self.n_coeff)

            vx = ca.dot(dB, x_coeffs) / T
            vy = ca.dot(dB, y_coeffs) / T
            ax = ca.dot(ddB, x_coeffs) / T**2
            ay = ca.dot(ddB, y_coeffs) / T**2

            speed_sq = vx**2 + vy**2
            denom = speed_sq + speed_eps_sq
            cross_va = vx * ay - vy * ax
            yaw_rate = cross_va / denom
            speed_gate = speed_sq / denom
            cost += (
                self.w_yaw_rate
                * speed_gate
                * yaw_rate**2
                / yaw_rate_ref_sq
                * (T * ds)
                / t_ref
            )
        return cost

    def casadi_yaw_acceleration_terms(self, x_coeffs, y_coeffs, T, t_ref, s_vals, ds):
        """
        Penalize acceleration of the path-tangent yaw implied by x/y motion.

        ``yaw = atan2(vy, vx)`` is not a decision variable in this planner, so the
        objective works directly from the x/y derivatives. A smooth speed gate
        keeps near-hover samples from dominating when tangent yaw is ill-defined.
        """
        if self.w_yaw_acc <= 0.0:
            return 0.0

        speed_eps = 0.05 * max(float(self.v_max), 1e-3)
        speed_eps_sq = speed_eps**2
        yaw_acc_ref = max(float(self.a_max) / max(float(self.v_max), 1e-3), 1e-3)
        yaw_acc_ref_sq = yaw_acc_ref**2

        cost = 0.0
        for s in s_vals:
            dB = self.dpoly_basis(float(s), self.n_coeff)
            ddB = self.ddpoly_basis(float(s), self.n_coeff)
            dddB = self.dddpoly_basis(float(s), self.n_coeff)

            vx = ca.dot(dB, x_coeffs) / T
            vy = ca.dot(dB, y_coeffs) / T
            ax = ca.dot(ddB, x_coeffs) / T**2
            ay = ca.dot(ddB, y_coeffs) / T**2
            jx = ca.dot(dddB, x_coeffs) / T**3
            jy = ca.dot(dddB, y_coeffs) / T**3

            speed_sq = vx**2 + vy**2
            denom = speed_sq + speed_eps_sq
            cross_va = vx * ay - vy * ax
            dot_va = vx * ax + vy * ay
            cross_vj = vx * jy - vy * jx
            yaw_acc = (cross_vj * denom - 2.0 * cross_va * dot_va) / denom**2
            speed_gate = speed_sq / denom
            cost += (
                self.w_yaw_acc
                * speed_gate
                * yaw_acc**2
                / yaw_acc_ref_sq
                * (T * ds)
                / t_ref
            )
        return cost

    # Polynomial basis and derivatives
    def poly_basis(self, s, order):
        return ca.vertcat(*[s**i for i in range(order)])

    def dpoly_basis(self, s, order):
        return ca.vertcat(*[i * s**(i - 1) if i > 0 else 0 for i in range(order)])

    def ddpoly_basis(self, s, order):
        return ca.vertcat(*[i * (i - 1) * s**(i - 2) if i > 1 else 0 for i in range(order)])

    def dddpoly_basis(self, s, order):
        return ca.vertcat(*[i * (i - 1) * (i - 2) * s**(i - 3) if i > 2 else 0 for i in range(order)])

    def ddddpoly_basis(self, s, order):
        return ca.vertcat(*[i * (i - 1) * (i - 2) * (i - 3) * s**(i - 4) if i > 3 else 0 for i in range(order)])

    def time_optimal(self, A_data, B_data, v_max, a_max, N):
        # Unpack initial and final states
        self._planner_output(f'time optimal initialization: start={A_data}, goal={B_data}')
        A_pos, A_vel = A_data
        B_pos, B_vel = B_data

        # Declare the symbolic variable for final time T
        # This is a key feature of CasADi: symbolic computation with automatic differentiation
        T = ca.MX.sym('T')
        dt = T / N  # Time step is symbolic (depends on T)

        # Define symbolic state and control variables for each time step
        # CasADi uses symbolic lists instead of large symbolic matrices
        # Each X[k] is a 6D vector: [px, py, pz, vx, vy, vz]
        # Each U[k] is a 3D vector: [ax, ay, az]
        X = [ca.MX.sym(f'X_{i}', 6) for i in range(N + 1)]
        U = [ca.MX.sym(f'U_{i}', 3) for i in range(N)]

        # Objective: minimize the final time T
        J = T

        # Initialize constraint and bound lists
        g = []    # All constraints go here
        lbg = []  # Lower bounds for constraints
        ubg = []  # Upper bounds for constraints

        # Initial condition constraints: position and velocity
        for ndx, axis in enumerate('xyz'):
            g += [X[0][ndx] - A_pos[axis]]   # Initial position must match
            g += [X[0][ndx + 3] - A_vel[axis]]   # Initial velocity must match
            g += [X[-1][ndx] - B_pos[axis]]  # Final position
            g += [X[-1][ndx + 3] - B_vel[axis]]  # Final velocity
            lbg += [0.0] * 4
            ubg += [0.0] * 4

        # Loop through each time step to impose dynamics and bounds
        for k in range(N):
            xk = X[k]
            x_next = X[k + 1]
            uk = U[k]

            # Forward Euler dynamics: x_{k+1} = x_k + dt * f(x_k, u_k)
            xk_est = ca.vertcat(
                xk[0:3] + dt * xk[3:6],  # Position update
                xk[3:6] + dt * uk       # Velocity update
            )
            g += [x_next - xk_est]
            lbg += [0.0] * 6
            ubg += [0.0] * 6

            # Velocity bounds: enforce vx, vy, vz in [-v_max, v_max]
            g += [xk[3:]]
            lbg += [-v_max] * 3
            ubg += [v_max] * 3

            # Acceleration bounds: enforce ax, ay, az in [-a_max, a_max]
            g += [uk]
            lbg += [-a_max] * 3
            ubg += [a_max] * 3

        # Flatten decision variables into a single vector
        w = [T] + X + U
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)

        # Provide an initial guess for the optimizer
        w0 = [1.0] + [0.0] * (6 * (N + 1)) + [0.0] * (3 * N)

        # Set bounds on decision variables
        lbw = [0.1] + [-ca.inf] * (6 * (N + 1)) + [-ca.inf] * (3 * N)
        ubw = [10.0] + [ca.inf] * (6 * (N + 1)) + [ca.inf] * (3 * N)

        # Define the nonlinear program
        nlp = {'x': w, 'f': J, 'g': g}  # CasADi NLP: decision vars, cost function, constraints
        opts = self._ipopt_options()
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)  # Create NLP solver

        # Solve the optimization problem
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()
        T_opt = w_opt[0]  # Optimal time

        # Extract the state and control trajectories
        x_start = 1
        x_end = x_start + 6 * (N + 1)
        X_opt = w_opt[x_start:x_end].reshape((N + 1, 6)).T

        u_start = x_end
        U_opt = w_opt[u_start:].reshape((N, 3)).T

        # Construct time vectors
        t_vals = np.linspace(0, T_opt, N + 1)
        t_ctrl = np.linspace(0, T_opt, N)

        self._planner_output(f'time optimal initialization result: T={T_opt:.3f}s')
        return t_vals, X_opt, t_ctrl, U_opt

    def poly_time_optimal(self, A_data, B_data, v_max, a_max, n_eval, n_coeff):
        A_pos, A_vel = A_data
        B_pos, B_vel = B_data

        # Create symbolic variables for total time and polynomial coefficients for each axis
        T = ca.MX.sym('T')
        c_x = ca.MX.sym('c_x', n_coeff)  # Coefficients for x(s)
        c_y = ca.MX.sym('c_y', n_coeff)  # Coefficients for y(s)
        c_z = ca.MX.sym('c_z', n_coeff)  # Coefficients for z(s)

        # Define basis functions and their derivatives for polynomial evaluation
        def poly_basis(s, order):
            return ca.vertcat(*[s**i for i in range(order)])

        def dpoly_basis(s, order):
            return ca.vertcat(*[i * s**(i - 1) if i > 0 else 0 for i in range(order)])

        def ddpoly_basis(s, order):
            return ca.vertcat(*[i * (i - 1) * s**(i - 2) if i > 1 else 0 for i in range(order)])

        # Constraints and bounds
        g, lbg, ubg = [], [], []
        s_vals = np.linspace(0, 1, n_eval)  # Normalized time samples

        # Enforce position and velocity constraints at s=0 and s=1
        for s, pos, vel in zip([0, 1], [A_pos, B_pos], [A_vel, B_vel]):
            B = poly_basis(s, n_coeff)
            dB = dpoly_basis(s, n_coeff)

            # Position match and velocity match (note: divide by T due to chain rule)
            g += [ca.dot(B, c_x) - pos['x'],
                  ca.dot(B, c_y) - pos['y'],
                  ca.dot(B, c_z) - pos['z'],
                  ca.dot(dB, c_x) / T - vel['x'],
                  ca.dot(dB, c_y) / T - vel['y'],
                  ca.dot(dB, c_z) / T - vel['z']]
            lbg += [0] * 6
            ubg += [0] * 6

        # Velocity and acceleration bounds over the trajectory
        for s in s_vals:
            dB = dpoly_basis(s, n_coeff)
            ddB = ddpoly_basis(s, n_coeff)

            # First and second derivatives of position with respect to time
            v = ca.vertcat(ca.dot(dB, c_x),
                           ca.dot(dB, c_y),
                           ca.dot(dB, c_z)) / T
            a = ca.vertcat(ca.dot(ddB, c_x),
                           ca.dot(ddB, c_y),
                           ca.dot(ddB, c_z)) / (T**2)

            # Apply element-wise box constraints
            g += [v, a]
            lbg += [-v_max] * 3 + [-a_max] * 3
            ubg += [v_max] * 3 + [a_max] * 3

        # Combine decision variables
        w = ca.vertcat(T, c_x, c_y, c_z)
        w0 = [1.0] + [0.0] * (3 * n_coeff)
        lbw = [0.001] + [-ca.inf] * (3 * n_coeff)
        ubw = [100.0] + [ca.inf] * (3 * n_coeff)

        # Define NLP problem to minimize T subject to constraints
        nlp = {'x': w, 'f': T, 'g': ca.vertcat(*g)}
        opts = self._ipopt_options()
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Solve the problem
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()
        T_opt = w_opt[0]
        c_x_opt = w_opt[1:1 + n_coeff]
        c_y_opt = w_opt[1 + n_coeff:1 + 2 * n_coeff]
        c_z_opt = w_opt[1 + 2 * n_coeff:]

        return T_opt, c_x_opt, c_y_opt, c_z_opt

    # Polynomial trajectory optimization with snap/jerk constraints using two segments
    # to allow for over-constrained boundary conditions

    def poly_segment_time_optimal(self, A_data, B_data,
                                  p_lim, v_lim, a_lim, j_lim, s_lim,
                                  n_eval, n_coeff, n_segments, obstacles,
                                  obstacle_sample_overrides=None,
                                  yaw_rate_sample_overrides=None,
                                  dense_kinematic_segments=None,
                                  gate_transition_targets=None,
                                  initial_segments=None,
                                  ipopt_options=None,
                                  accepted_statuses=None):

        main_start_time = time.time()

        A_pos, A_vel, A_acc, A_jerk, A_snap = A_data
        B_pos, B_vel, B_acc, B_jerk, B_snap = B_data

        # Define symbolic variables for each segment
        times = []
        coefficients = {'x': [], 'y': [], 'z': []}
        for ndx in range(n_segments):
            times.append(ca.MX.sym(f'T{ndx}'))
            for axis in 'xyz':
                coefficients[axis].append(ca.MX.sym(f'c_{axis}{ndx}', n_coeff))

        # Set up constraints for each axis
        constraints = {}
        for axis in 'xyz':
            constraints[axis] = {'g': [], 'lbg': [], 'ubg': []}

        # Add boundary constraints for each axis
        def add_endpoint_constraints(s, T, axis, coeff, pos, vel, acc, jerk, snap):
            B = self.poly_basis(s, n_coeff)
            dB = self.dpoly_basis(s, n_coeff)
            ddB = self.ddpoly_basis(s, n_coeff)
            dddB = self.dddpoly_basis(s, n_coeff)
            ddddB = self.ddddpoly_basis(s, n_coeff)
            cons = [
                ca.dot(B, coeff) - pos[axis],
                ca.dot(dB, coeff) / T - vel[axis],
                ca.dot(ddB, coeff) / T**2 - acc[axis],
                ca.dot(dddB, coeff) / T**3 - jerk[axis],
                ca.dot(ddddB, coeff) / T**4 - snap[axis],
            ]
            constraints[axis]['g'].extend(cons)
            constraints[axis]['lbg'].extend([0.] * len(cons))
            constraints[axis]['ubg'].extend([0.] * len(cons))

        for axis in 'xyz':
            # s = 0 at start
            add_endpoint_constraints(0, times[0], axis,
                                     coefficients[axis][0],
                                     A_pos, A_vel, A_acc, A_jerk, A_snap)
            # s = 1 at end
            add_endpoint_constraints(1, times[-1], axis,
                                     coefficients[axis][-1],
                                     B_pos, B_vel, B_acc, B_jerk, B_snap)

        self._planner_output(
            f"endpoint constraints per axis: x={len(constraints['x']['g'])}"
        )
        # Continuity constraints at internal junctions
        # continuous position, velocity, acceleration, jerk, and snap
        for B_fn in [self.poly_basis, self.dpoly_basis, self.ddpoly_basis, self.dddpoly_basis, self.ddddpoly_basis]:
            B1 = B_fn(1.0, n_coeff)
            B2 = B_fn(0.0, n_coeff)
            deg = B_fn.__name__.count('d')
            for ndx in range(1, n_segments):
                scale1 = times[ndx - 1]**deg
                scale2 = times[ndx]**deg
                for axis in 'xyz':
                    c1 = coefficients[axis][ndx - 1]
                    c2 = coefficients[axis][ndx]
                    expr = ca.dot(B1, c1) / scale1 - ca.dot(B2, c2) / scale2
                    constraints[axis]['g'] += [expr]
                    constraints[axis]['lbg'].append(0.)
                    constraints[axis]['ubg'].append(0.)
        self._planner_output(
            'continuity constraints per axis: '
            f"x={len(constraints['x']['g'])}, "
            f"y={len(constraints['y']['g'])}, "
            f"z={len(constraints['z']['g'])}"
        )

        # Per-axis, per-sample kinematic limit constraints. See README.md
        # (Kinematic limits): each sampled derivative is bounded directly by its
        # symmetric limit, -limit <= val(s_i) <= limit.
        s_vals = np.linspace(0, 1, n_eval)
        # Segments flagged by the dense kinematic scan are re-solved with a uniform
        # dense collocation grid (not targeted at the peak); pinning a single spot
        # just relocates the violation to the nearest un-sampled point. The grid
        # matches the dense-validation sample count (obstacle_refine_check_samples,
        # default obstacle_base_samples * obstacle_refine_multiplier
        # == validation.KINEMATIC_LIMIT_SAMPLE_COUNT's default derivation), so a
        # converged re-solve satisfies the kinematic limits at the very points
        # validation re-checks.
        dense_kinematic_segments = set(dense_kinematic_segments or ())
        dense_s_vals = np.linspace(
            0.0, 1.0, max(int(self.obstacle_refine_check_samples), n_eval))
        for seg in range(n_segments):
            T = times[seg]
            seg_s_vals = dense_s_vals if seg in dense_kinematic_segments else s_vals
            for axis in 'xyz':
                coeffs = coefficients[axis][seg]

                for s in seg_s_vals:
                    B = self.poly_basis(s, n_coeff)
                    dB = self.dpoly_basis(s, n_coeff)
                    ddB = self.ddpoly_basis(s, n_coeff)
                    dddB = self.dddpoly_basis(s, n_coeff)
                    ddddB = self.ddddpoly_basis(s, n_coeff)

                    p = ca.dot(B, coeffs)
                    v = ca.dot(dB, coeffs) / T
                    a = ca.dot(ddB, coeffs) / T**2
                    j = ca.dot(dddB, coeffs) / T**3
                    sn = ca.dot(ddddB, coeffs) / T**4

                    # Position bounds are hard and may be asymmetric (p_lim).
                    constraints[axis]['g'] += [p]
                    constraints[axis]['lbg'] += [p_lim[axis][0]]
                    constraints[axis]['ubg'] += [p_lim[axis][1]]

                    # Kinematic limits are symmetric, so bound each derivative
                    # directly: -limit <= val(s_i) <= limit. (An earlier version used
                    # a per-segment slack reformulation, slack**2 - val**2 >= 0 with
                    # slack <= limit; that enforces the same feasible set but adds
                    # 4*3*n_segments nonconvex auxiliary variables and converges far
                    # more slowly -- ~3x slower IPOPT solve on the gate scenarios for
                    # identical trajectories.)
                    for val, bound in zip(
                        [v, a, j, sn], [v_lim[1], a_lim[1], j_lim[1], s_lim[1]]
                    ):
                        constraints[axis]['g'] += [val]
                        constraints[axis]['lbg'] += [-bound]
                        constraints[axis]['ubg'] += [bound]

        constraints_time = time.time()
        self._planner_output(
            f'planner constraints ready in {constraints_time - main_start_time:.3f}s'
        )

        # Decision vector
        w = ca.vertcat(*times, *coefficients['x'], *coefficients['y'], *coefficients['z'])

        # Initial values
        if len(initial_segments) != n_segments:
            raise ValueError(
                f'initial_segments length must match n_segments '
                f'({len(initial_segments)} != {n_segments}).'
            )
        w0 = [float(segment['T']) for segment in initial_segments]
        for axis in 'xyz':
            for segment in initial_segments:
                coeffs = list(np.asarray(segment[f'c_{axis}'], dtype=np.float64))
                w0 += self._pad_coefficients(coeffs[:n_coeff], n_coeff)

        self._planner_output(
            f'planner initial guess prepared: variables={len(w0)}'
        )

        #     time bounds      + all other bounds for coefficients
        lbw = [self.time_min] * n_segments + [-ca.inf] * n_coeff * 3 * n_segments
        ubw = [self.time_max] * n_segments + [ca.inf] * n_coeff * 3 * n_segments

        g = constraints['x']['g'] + constraints['y']['g'] + constraints['z']['g']
        lbg = constraints['x']['lbg'] + constraints['y']['lbg'] + constraints['z']['lbg']
        ubg = constraints['x']['ubg'] + constraints['y']['ubg'] + constraints['z']['ubg']
        # Obstacle avoidance is a soft one-sided quadratic hinge on clearance to each
        # inflated capsule/cylinder surface (casadi_obstacle_clearance_penalty),
        # accumulated here over the dense obstacle sample set (plus any refinement
        # overrides) and folded into the objective below. This replaces the former
        # hard clearance constraints: IPOPT no longer pays barrier/feasibility-
        # restoration cost for obstacles it isn't near, and dense validation is the
        # safety guarantee. NOTE: not yet gated by distance -- every (segment,
        # obstacle, sample) contributes a (mostly zero) term for this first test.
        # Per-sample weights are composite-trapezoid quadrature weights over the
        # (sorted, deduped) sample set, normalized to sum to 1 -- i.e. each
        # segment-obstacle pair's total penalty approximates a fixed integral
        # obstacle_penalty_weight * mean(clearance_violation over [0, 1]),
        # matching how casadi_effort_terms/casadi_yaw_rate_terms use a ds-weighted
        # Riemann sum. This keeps refinement from diluting the base grid's push
        # (dividing by a growing sample count shrank every sample's weight) while
        # also keeping a densely-refined segment-obstacle pair from ballooning in
        # total magnitude and swamping the time/yaw objective terms -- both
        # failure modes of weighting by a plain sample count.
        obstacle_penalty = 0
        obstacle_penalty_sample_count = 0
        base_obstacle_s_vals = np.linspace(
            0,
            1,
            max(int(n_eval), max(2, int(self.obstacle_base_samples))),
        )
        obstacle_sample_overrides = obstacle_sample_overrides or {}
        for seg in range(n_segments):
            x_coeffs = coefficients['x'][seg]
            y_coeffs = coefficients['y'][seg]
            z_coeffs = coefficients['z'][seg]
            for obstacle_index, obs in enumerate(obstacles):
                extra_samples = obstacle_sample_overrides.get((seg, obstacle_index), [])
                obstacle_s_vals = np.unique(np.concatenate((
                    base_obstacle_s_vals,
                    np.asarray(extra_samples, dtype=np.float64),
                )))
                sample_weights = (
                    self._trapezoid_weights(obstacle_s_vals) * self.obstacle_penalty_weight)
                for s, sample_weight in zip(obstacle_s_vals, sample_weights):
                    B = self.poly_basis(float(s), n_coeff)
                    x_i = ca.dot(B, x_coeffs)
                    y_i = ca.dot(B, y_coeffs)
                    z_i = ca.dot(B, z_coeffs)
                    obstacle_penalty += self.casadi_obstacle_clearance_penalty(
                        x_i, y_i, z_i, obs, weight=float(sample_weight))
                    obstacle_penalty_sample_count += 1

        # Hard yaw-rate cap (policy limit). The path-tangent yaw rate is
        # cross_va / (speed_sq + speed_eps_sq). Multiplying the limit through the
        # non-negative denominator turns |yaw_rate| <= yaw_rate_max into the
        # polynomial, hover-tolerant two-sided constraint
        # |cross_va| <= yaw_rate_max * denom, sampled at the same collocation points
        # as the kinematic limits. yaw_rate_max <= 0 means "uncapped", so skip it.
        yaw_rate_max = float(self.yaw_rate_max)
        yaw_rate_constraint_count = 0
        yaw_rate_overrides = yaw_rate_sample_overrides or {}
        if yaw_rate_max > 0.0:
            speed_eps_sq = (0.05 * max(float(self.v_max), 1e-3))**2
            for seg in range(n_segments):
                T = times[seg]
                x_coeffs = coefficients['x'][seg]
                y_coeffs = coefficients['y'][seg]
                seg_s_vals = s_vals
                extra_yaw_samples = yaw_rate_overrides.get(seg)
                if extra_yaw_samples:
                    seg_s_vals = np.unique(np.concatenate((
                        s_vals, np.asarray(extra_yaw_samples, dtype=np.float64),
                    )))
                for s in seg_s_vals:
                    dB = self.dpoly_basis(float(s), n_coeff)
                    ddB = self.ddpoly_basis(float(s), n_coeff)
                    vx = ca.dot(dB, x_coeffs) / T
                    vy = ca.dot(dB, y_coeffs) / T
                    ax = ca.dot(ddB, x_coeffs) / T**2
                    ay = ca.dot(ddB, y_coeffs) / T**2
                    denom = vx**2 + vy**2 + speed_eps_sq
                    cross_va = vx * ay - vy * ax
                    bound = yaw_rate_max * denom
                    # |cross_va| <= bound  <=>  bound - cross_va >= 0 and bound + cross_va >= 0
                    g += [bound - cross_va, bound + cross_va]
                    lbg += [0.0, 0.0]
                    ubg += [ca.inf, ca.inf]
                    yaw_rate_constraint_count += 2

        self._planner_output(
            f'planner NLP size: variables={w.numel()} constraints={len(g)} '
            f'obstacle_penalty_samples={obstacle_penalty_sample_count} '
            f'yaw_rate_constraints={yaw_rate_constraint_count} '
            f'gate_transition_targets={len(gate_transition_targets or [])}'
        )

        # Objective: minimize total time (w_time) plus control-effort terms, all
        # normalized by a reference time so the weights are dimensionless and
        # transferable across problem scales. See README.md (Weight normalization)
        # and the brake-and-reverse rationale; helpers _reference_time / casadi_effort_terms.
        t_ref = self._reference_time(p_lim)
        f = 0
        for T in times:
            f += self.w_time * T / t_ref
        for ndx, T in enumerate(times):
            if ndx > 0:
                f += 0.25 * ((T - times[ndx - 1]) / t_ref)**2

        total_cost = 0
        ds = 1.0 / max(1, len(s_vals) - 1)
        start_vec = np.asarray([A_pos['x'], A_pos['y'], A_pos['z']], dtype=np.float64)
        goal_vec = np.asarray([B_pos['x'], B_pos['y'], B_pos['z']], dtype=np.float64)
        progress_vec = goal_vec - start_vec
        progress_norm = np.linalg.norm(progress_vec)
        use_progress_cost = (
            progress_norm > 1e-9
            and (
                self.backtrack_objective_weight > 0.0
                or self.behind_start_objective_weight > 0.0
            )
        )
        if use_progress_cost:
            progress_dir = progress_vec / progress_norm
            prev_progress = None
        transition_targets_by_segment = {}
        for target in gate_transition_targets or []:
            if int(n_segments) >= 3:
                fraction = (
                    1.0 / float(n_segments)
                    if target.get('label') == 'start_exit'
                    else (float(n_segments) - 1.0) / float(n_segments)
                )
            else:
                fraction = float(target['fraction'])
            segment_index, sample_s = self._segment_sample_from_global_fraction(
                fraction, n_segments)
            transition_targets_by_segment.setdefault(segment_index, []).append(
                (sample_s, target))
        for seg in range(n_segments):
            T = times[seg]
            x_coeffs = coefficients['x'][seg]
            y_coeffs = coefficients['y'][seg]
            z_coeffs = coefficients['z'][seg]
            # Integrated, limit-normalized control-effort cost for this segment.
            total_cost += self.casadi_effort_terms(
                x_coeffs, y_coeffs, z_coeffs, T, t_ref, s_vals, ds)
            total_cost += self.casadi_yaw_rate_terms(
                x_coeffs, y_coeffs, T, t_ref, s_vals, ds)
            total_cost += self.casadi_yaw_acceleration_terms(
                x_coeffs, y_coeffs, T, t_ref, s_vals, ds)
            for s in s_vals:
                B = self.poly_basis(s, n_coeff)
                x_i = ca.dot(B, x_coeffs)
                y_i = ca.dot(B, y_coeffs)
                z_i = ca.dot(B, z_coeffs)
                if use_progress_cost:
                    progress = (
                        (x_i - A_pos['x']) * progress_dir[0]
                        + (y_i - A_pos['y']) * progress_dir[1]
                        + (z_i - A_pos['z']) * progress_dir[2]
                    )
                    if self.behind_start_objective_weight > 0.0:
                        behind_start = ca.fmax(0.0, -progress)
                        total_cost += self.behind_start_objective_weight * behind_start**2
                    if prev_progress is not None and self.backtrack_objective_weight > 0.0:
                        backtrack = ca.fmax(0.0, prev_progress - progress)
                        total_cost += self.backtrack_objective_weight * backtrack**2
                    prev_progress = progress
                total_cost += self.casadi_boundary_bias(x_i, y_i, z_i, p_lim)
            for target_s, transition_target in transition_targets_by_segment.get(seg, []):
                total_cost += self.casadi_gate_transition_target_cost(
                    x_coeffs,
                    y_coeffs,
                    z_coeffs,
                    T,
                    target_s,
                    transition_target,
                    t_ref,
                )
        f = f + total_cost + obstacle_penalty
        self._planner_output('planner objective assembled')

        nlp = {'x': w, 'f': f, 'g': ca.vertcat(*g)}
        solver_setup_time = time.time()
        self._planner_output(
            f'planner model ready in {solver_setup_time - constraints_time:.3f}s '
            '(includes time initialization)'
        )
        solver = ca.nlpsol('solver', 'ipopt', nlp, self._ipopt_options(ipopt_options))
        solver_start_time = time.time()
        self._planner_output(
            f'planner solver ready in {solver_start_time - solver_setup_time:.3f}s'
        )

        #  w_opt = [
        #     T1,                  # Total time for segment 1
        #     T2,                  # Total time for segment 2
        #     ...
        #     TN
        #     c_x1[0:N],           # Coefficients for x in segment 1
        #     c_x2[0:N],           # Coefficients for x in segment 2
        #       ...
        #     c_xN[0:N],           # Coefficients for x in segment N
        #     c_y1[0:N],           # Coefficients for y in segment 1
        #     c_y2[0:N],           # Coefficients for y in segment 2
        #       ...
        #     c_yN[0:N],           # Coefficients for y in segment N
        #     c_z1[0:N],           # Coefficients for z in segment 1
        #     c_z2[0:N]            # Coefficients for z in segment 2
        #       ...
        #     c_zN[0:N]            # Coefficients for z in segment N
        # ]
        def parse_w_opt(w_opt, n_coeff):
            # Decision vector layout: [T0..T{n-1}, c_x blocks, c_y blocks, c_z
            # blocks]. The kinematic limits are imposed as direct two-sided
            # constraints, so there are no trailing auxiliary variables to skip.
            params = []
            for seg in range(n_segments):
                segment = {'T': w_opt[seg]}
                for n_ax, axis in enumerate('xyz'):
                    ndx_blk = n_segments + n_ax * n_segments * n_coeff + seg * n_coeff
                    segment[f'c_{axis}'] = w_opt[ndx_blk:(ndx_blk + n_coeff)]
                params.append(segment)
            return params

        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        solution_time = time.time()

        solve_sec = solution_time - solver_start_time
        overall_sec = solution_time - main_start_time

        solver_stats = solver.stats()
        return_status = str(solver_stats.get('return_status', '<unknown>'))
        self.last_solve_status = return_status
        self.last_objective_value = float(sol['f'])
        self.last_solve_sec = solve_sec
        self.last_overall_sec = overall_sec
        self._planner_output(
            f'planner solve status={return_status}, solve={solve_sec:.3f}s, '
            f'overall={overall_sec:.3f}s',
            force=True,
        )
        w_opt = sol['x'].full().flatten()
        segments = parse_w_opt(w_opt, n_coeff)

        accepted_statuses = set(accepted_statuses or self.SUCCESS_STATUSES)
        if return_status not in accepted_statuses:
            artifact_segments = [self._rescale_segment_to_real_time(segment) for segment in segments]
            artifact_dir = self._write_failed_solver_artifacts(
                return_status=return_status,
                solver_stats=solver_stats,
                segments=artifact_segments,
                timing={
                    'constraint_setup_sec': constraints_time - main_start_time,
                    'model_setup_sec': solver_setup_time - constraints_time,
                    'solver_setup_sec': solver_start_time - solver_setup_time,
                    'solve_sec': solution_time - solver_start_time,
                    'overall_sec': solution_time - main_start_time,
                },
            )
            self._planner_output(
                f'WARNING: IPOPT returned {return_status}; rejected candidate trajectory. '
                f'Wrote solver artifacts to {artifact_dir}',
                force=True,
            )
            raise RuntimeError(
                f'IPOPT failed with return_status={return_status}; '
                f'wrote candidate trajectory and solver stats to {artifact_dir}'
            )
        if return_status not in self.SUCCESS_STATUSES:
            self._planner_output(
                f'WARNING: IPOPT returned {return_status}; accepting candidate for validation.',
                force=True,
            )

        return segments

    def eval_segmented_polynomials(self, segments, n_points=100):

        results = {'t': [], 'x': [], 'y': [], 'z': [],
                   'vx': [], 'vy': [], 'vz': [],
                   'ax': [], 'ay': [], 'az': [],
                   'jx': [], 'jy': [], 'jz': [],
                   'sx': [], 'sy': [], 'sz': []}

        t_offset = 0.0
        for ndx, seg in enumerate(segments):
            T = seg['T']
            s_vals = np.linspace(0, 1, n_points)
            t_vals = t_offset + s_vals * T
            results['t'].extend(t_vals)
            t_offset += T

            for axis in ['x', 'y', 'z']:
                c = seg[f'c_{axis}']

                # Polynomial uses standard monomial form with
                # coefficients ordered from lowest to highest degree.
                # This matches our optimization, but is opposite polyfit !
                p = Polynomial(c)

                pos = p(s_vals)
                vel = p.deriv(1)(s_vals) / T
                acc = p.deriv(2)(s_vals) / T**2
                jrk = p.deriv(3)(s_vals) / T**3
                snp = p.deriv(4)(s_vals) / T**4

                if ndx != 0:
                    self._planner_output(
                        f'Continuity check {axis}: '
                        f'pos={results[axis][-1] - pos[0]:+.3e}, '
                        f"vel={results['v' + axis][-1] - vel[0]:+.3e}, "
                        f"acc={results['a' + axis][-1] - acc[0]:+.3e}, "
                        f"jerk={results['j' + axis][-1] - jrk[0]:+.3e}, "
                        f"snap={results['s' + axis][-1] - snp[0]:+.3e}"
                    )
                results[axis].extend(pos)
                results['v' + axis].extend(vel)
                results['a' + axis].extend(acc)
                results['j' + axis].extend(jrk)
                results['s' + axis].extend(snp)

        for k in results:
            results[k] = np.array(results[k])

        return results

    def compute_total_energy(self, vectors):
        g = np.array([0.0, 0.0, 9.81])  # Gravity vector
        a = np.vstack([vectors['ax'], vectors['ay'], vectors['az']]).T  # shape (N, 3)
        a_with_g = a + g  # broadcasts to (N, 3)

        energy_density = np.sum(a_with_g**2, axis=1)  # ||a + g||^2 at each timestep

        # Integrate using trapezoidal rule over time. np.trapezoid replaced the
        # removed-in-NumPy-2.0 np.trapz; fall back for older NumPy.
        t = vectors['t']
        trapezoid = getattr(np, 'trapezoid', None) or np.trapz
        energy = trapezoid(energy_density, x=t)

        return energy

    def plot_capped_cylinder(self, ax, xc=0.0, yc=0.4, z0=0.0, height=1.5, radius=0.3, cap=True, resolution=50):
        # Cylinder body
        theta = np.linspace(0, 2 * np.pi, resolution)
        z_cyl = np.linspace(z0, z0 + height, resolution)
        theta_grid, z_grid = np.meshgrid(theta, z_cyl)
        x_cyl = xc + radius * np.cos(theta_grid)
        y_cyl = yc + radius * np.sin(theta_grid)
        z_cyl = z_grid

        ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.3, color='red', linewidth=0, shade=True)

        # Hemispherical cap on top (optional)
        if cap:
            phi = np.linspace(0, np.pi / 2, resolution)
            theta_cap, phi_cap = np.meshgrid(theta, phi)
            x_cap = xc + radius * np.cos(theta_cap) * np.sin(phi_cap)
            y_cap = yc + radius * np.sin(theta_cap) * np.sin(phi_cap)
            z_cap = z0 + height + radius * np.cos(phi_cap)
            ax.plot_surface(x_cap, y_cap, z_cap, alpha=0.3, color='blue', linewidth=0, shade=True)

    def show_all_traj_plots(self, all_xyz, all_xyz_t, A_pos, B_pos, p_lim, t_dense, t_vals, t_ctrl):
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        # Plot 3D path
        fig = plt.figure()
        ax3d = fig.add_subplot(111, projection='3d')
        for obs in self.obstacles:
            self.plot_capped_cylinder(ax3d, xc=obs['x'], yc=obs['y'], radius=obs['radius'], cap=True, resolution=50)

        ax3d.plot(all_xyz[0][0], all_xyz[0][1], all_xyz[0][2], label='7th-order path', linewidth=2)
        ax3d.plot(all_xyz_t[0][0], all_xyz_t[0][1], all_xyz_t[0][2], label='Time optimal path', linestyle=':', linewidth=2)
        ax3d.scatter([A_pos['x']], [A_pos['y']], [A_pos['z']], c='green', label='Start')
        ax3d.scatter([B_pos['x']], [B_pos['y']], [B_pos['z']], c='red', label='Goal')
        ax3d.set_xlabel('X [m]')
        ax3d.set_ylabel('Y [m]')
        ax3d.set_zlabel('Z [m]')
        ax3d.set_title('3D Polynomial Trajectory')
        ax3d.legend()
        ax3d.grid(True)

        # Plot x-y position projection
        plt.figure()
        # Add red circular patch
        plt_ax = plt.gca()
        for obs in self.obstacles:
            xc, yc, rc = obs['x'], obs['y'], obs['radius']
            self._planner_output(f'plot obstacle: x={xc}, y={yc}, radius={rc}')
            circle = patches.Circle((xc, yc), rc, color='red', alpha=0.5, label='circular patch')
            plt_ax.add_patch(circle)
        plt.plot(all_xyz[0][0], all_xyz[0][1], '.', label='x 7th')
        plt.title('x-y Position')
        plt.xlabel('x position [m]')
        plt.ylabel('y position [m]')
        plt.legend()
        plt.axis([p_lim['x'][0], p_lim['x'][1], p_lim['y'][0], p_lim['y'][1]])
        plt.axis('equal')
        plt.grid(True)

        # Plot position vs. time
        plt.figure()
        plt.plot(t_dense, all_xyz[0][0], '.', label='x 7th')
        plt.plot(t_dense, all_xyz[0][1], '.', label='y 7th')
        plt.plot(t_dense, all_xyz[0][2], '.', label='z 7th')
        plt.plot(t_vals, all_xyz_t[0][0], linestyle=':', label='x t')
        plt.plot(t_vals, all_xyz_t[0][1], linestyle=':', label='y t')
        plt.plot(t_vals, all_xyz_t[0][2], linestyle=':', label='z t')
        plt.title('Position vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.legend()
        plt.grid(True)

        # Plot velocity
        plt.figure()
        plt.plot(t_dense, all_xyz[1][0], '.', label='vx 7th')
        plt.plot(t_dense, all_xyz[1][1], '.', label='vy 7th')
        plt.plot(t_dense, all_xyz[1][2], '.', label='vz 7th')
        plt.plot(t_vals, all_xyz_t[1][0], linestyle=':', label='vx t')
        plt.plot(t_vals, all_xyz_t[1][1], linestyle=':', label='vy t')
        plt.plot(t_vals, all_xyz_t[1][2], linestyle=':', label='vz t')
        plt.title('Velocity vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend()
        plt.grid(True)

        # Plot acceleration
        plt.figure()
        plt.plot(t_dense, all_xyz[2][0], '.', label='ax 7th')
        plt.plot(t_dense, all_xyz[2][1], '.', label='ay 7th')
        plt.plot(t_dense, all_xyz[2][2], '.', label='az 7th')
        plt.plot(t_ctrl, all_xyz_t[2][0], linestyle=':', label='ax t')
        plt.plot(t_ctrl, all_xyz_t[2][1], linestyle=':', label='ay t')
        plt.plot(t_ctrl, all_xyz_t[2][2], linestyle=':', label='az t')
        plt.title('Acceleration vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [m/s²]')
        plt.legend()
        plt.grid(True)

        # Plot jerk
        plt.figure()
        plt.plot(t_dense, all_xyz[3][0], '.', label='jx 7th')
        plt.plot(t_dense, all_xyz[3][1], '.', label='jy 7th')
        plt.plot(t_dense, all_xyz[3][2], '.', label='jz 7th')
        plt.title('Jerk vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Jerk [m/s³]')
        plt.legend()
        plt.grid(True)

        # Plot snap
        plt.figure()
        plt.plot(t_dense, all_xyz[4][0], '.', label='sx 7th')
        plt.plot(t_dense, all_xyz[4][1], '.', label='sy 7th')
        plt.plot(t_dense, all_xyz[4][2], '.', label='sz 7th')
        plt.title('Snap vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel(r'Snap [m/s$^4$]')
        plt.legend()
        plt.grid(True)

        # plt.show(block=False)

    def demo(self, start, goal, obstacles=None):
        """Plan a trajectory and plot it (matplotlib). Used by the __main__ demo."""
        import matplotlib.pyplot as plt

        start_time = time.time()
        segments = self.plan(start, goal, obstacles=obstacles)
        self._planner_output(
            f'optimization complete in {time.time() - start_time:.3f}s',
            force=True,
        )

        total_time = 0.0
        xs, ys = [], []
        for index, seg in enumerate(segments):
            self._planner_output(
                f"segment {index}: T={float(seg['T']):.3f}s",
            )
            total_time += seg['T']
            # Coefficients are real-time, ascending power: x(t) = sum_k c_k t^k.
            t_local = np.linspace(0.0, seg['T'], 200)
            xs.append(Polynomial(seg['c_x'])(t_local))
            ys.append(Polynomial(seg['c_y'])(t_local))
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        self._planner_output(
            f'Optimal time: {total_time:.3f} seconds for {self.n_segments} segment(s)',
            force=True,
        )

        plt.figure()
        plt_ax = plt.gca()
        for obs in self.obstacles:
            plt_ax.add_patch(plt.Circle((obs['x'], obs['y']), obs['radius'], color='red', alpha=0.3))
        plt.plot(xs, ys, '-', linewidth=2, label='planned path')
        plt_ax.set_aspect('equal', 'box')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Planned trajectory (xy)')
        plt.legend()
        plt.show()
        return segments


if __name__ == '__main__':
    planner = CasadiObstaclePlanner()
    demo_start = {'position': {'x': 1.5, 'y': 0.0, 'z': 1.25},
                  'velocity': {'x': 0.0, 'y': 0.5, 'z': 0.0}}
    demo_goal = {'position': {'x': -1.5, 'y': 0.0, 'z': 0.75},
                 'velocity': {'x': 0.0, 'y': 0.5, 'z': 0.0}}
    demo_obstacles = [
        {'type': 'cylinder', 'x': 0.0, 'y': 1.75, 'radius': 0.3},
        {'type': 'cylinder', 'x': 0.0, 'y': -1.75, 'radius': 0.3},
    ]
    planner.demo(demo_start, demo_goal, demo_obstacles)

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

import hashlib
import json
import math
import multiprocessing as mp
import time
import rclpy
import numpy as np
import casadi as ca
import re
from pathlib import Path
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import os
import threading
import uuid
import yaml

from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Path as NavPath
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs
from flexible_drones_tools.trajectories.planners.base_planner import (
    DEFAULT_BOUNDS,
    DEFAULT_LIMITS,
)
from flexible_drones_tools.ros_shutdown import BENIGN_SHUTDOWN_EXCEPTIONS
from flexible_drones_tools.trajectories.planners.errors import (
    TrajectoryPlanningCanceled,
    TrajectoryPlanningError,
    TrajectoryPlanningTimeout,
)
from flexible_drones_tools.trajectories.planners import (
    obstacles, optimization, scurve, seeding, validation, yaw_planning,
)
from flexible_drones_tools.trajectories.planners.obstacles import ObstacleManager
from flexible_drones_tools.trajectories.planners.optimization import PLANNERS
from flexible_drones_tools.trajectories.utilities.conversion import (
    coefficients_from_pieces,
    duration_to_seconds,
    polynomial_piece_from_coefficients,
    sampled_trajectory_from_pieces,
    trajectory_from_pieces,
)
from flexible_drones_tools.trajectories.utilities.io import (
    load_trajectory_pieces,
    normalize_path_value,
    save_trajectory_pieces,
)
from flexible_drones_tools.trajectories.utilities.model import PolyOrderTrajectory
from flexible_drones_tools.trajectories.visualize_trajectory import trajectory_to_path_msg
from flexible_drones_msgs.msg import Trajectory, SampledTrajectory
from flexible_drones_msgs.action import GetSampledTrajectory, GetTrajectory
from flexible_drones_msgs.srv import SetTrajectoryBounds, SetTrajectoryObstacles


class PlanTrajectoryActionServer(Node):
    MIN_SAMPLE_DT = 0.01
    MAX_SAMPLE_DT = 1.0
    # Cache filenames intentionally quantize numeric inputs to millimeter scale.
    # The optimizer still receives full-precision transformed values.
    CACHE_SIGNATURE_DECIMALS = 3

    def __init__(self):
        super().__init__('plan_trajectory_action_server')
        self.get_logger().info(
            f"Planner dependency versions: CasADi {getattr(ca, '__version__', 'unknown')}, "
            f'NumPy {np.__version__}'
        )
        # Generated/cached trajectories default to the user ROS directory so they
        # never pollute a package source tree. Override 'output_dir' to redirect
        # them (e.g. into a deployment package's trajectories folder).
        default_output_dir = str(Path.home() / '.ros' / 'traj_planner_gen')
        self.declare_parameter('output_dir', default_output_dir)
        self.declare_parameter('preferred_frame', 'map')
        self.declare_parameter('planning_timeout_sec', 300.0)
        self.declare_parameter('planning_feedback_period_sec', 5.0)
        self.declare_parameter(
            'obstacle_description_topics',
            [''],
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY),
        )
        self.declare_parameter('obstacle_safety_margin', 0.1)
        self.declare_parameter('obstacle_planning_frame', '')
        self.declare_parameter('planner_obstacles_marker_topic', '/planner_obstacles')
        self.declare_parameter('planned_trajectory_path_topic', '/planned_trajectory')
        self.declare_parameter('optimizing_trajectory_path_topic', '/optimizing_trajectory')
        self.declare_parameter('trajectory_visualization_samples', 160)

        # Planner selection and configuration.
        self.declare_parameter('planner', 'casadi_obstacle')
        self.declare_parameter('n_segments', 4)
        self.declare_parameter('min_segment_time', 0.5)
        self.declare_parameter('max_duration', 60.0)
        self.declare_parameter('gate_transition_distance', 0.5)
        self.declare_parameter('gate_transition_position_weight', 10.0)
        self.declare_parameter('gate_transition_velocity_weight', 1.0)
        self.declare_parameter('gate_transition_velocity_threshold', 0.01)
        # In the staged CasADi pipeline this selects the geometric/base initializer;
        # the child still runs the obstacle-constrained IPOPT stage afterward.
        self.declare_parameter('seed_strategy', 'dubins')
        self.declare_parameter('n_eval', 10)
        self.declare_parameter('obstacle_base_samples', 41)
        # Dense re-check/re-solve grid size is derived as
        # obstacle_base_samples * obstacle_refine_multiplier (see
        # CasadiObstaclePlanner.__init__), not set directly, so it can never
        # collapse to a degenerate 1-sample grid.
        self.declare_parameter('obstacle_refine_multiplier', 5)
        self.declare_parameter('obstacle_refine_margin', 0.03)
        self.declare_parameter('obstacle_refine_radius_fraction', 0.25)
        self.declare_parameter('obstacle_refine_min_spacing', 0.01)
        self.declare_parameter('obstacle_refine_max_spacing', 0.25)
        # Soft inward bias off the flight-volume walls (weight 0 disables it).
        self.declare_parameter('boundary_bias_margin', 0.5)
        self.declare_parameter('boundary_bias_weight', 50.0)
        # Objective weights: minimum-time vs integrated accel/jerk/snap effort.
        self.declare_parameter('w_time', 13.0)
        self.declare_parameter('w_acc', 1.0)
        self.declare_parameter('w_jerk', 0.0)
        self.declare_parameter('w_snap', 2.0)
        self.declare_parameter('kinematic_constraint_scale', 0.975)
        self.declare_parameter('backtrack_objective_weight', 0.0)
        self.declare_parameter('behind_start_objective_weight', 0.0)
        self.declare_parameter('planner_artifact_dir', '/tmp')
        self.declare_parameter('initial_ipopt_max_iter', 800)
        self.declare_parameter('initial_ipopt_tol', 1e-4)
        self.declare_parameter('initial_ipopt_constr_viol_tol', 1e-4)
        self.declare_parameter('initial_ipopt_acceptable_tol', 1e-3)
        self.declare_parameter('initial_ipopt_acceptable_constr_viol_tol', 1e-3)
        self.declare_parameter('initial_ipopt_acceptable_iter', 5)
        self.declare_parameter('constrained_ipopt_max_iter', 800)
        self.declare_parameter('constrained_ipopt_tol', 1e-6)
        self.declare_parameter('constrained_ipopt_constr_viol_tol', 1e-6)
        self.declare_parameter('constrained_ipopt_acceptable_tol', 1e-5)
        self.declare_parameter('constrained_ipopt_acceptable_constr_viol_tol', 1e-5)
        self.declare_parameter('constrained_ipopt_acceptable_iter', 5)
        self.declare_parameter('use_cache', False)
        self.declare_parameter('save_cache', True)
        self.declare_parameter('yaw_follow_tangent', True)
        # Primary path-shaping yaw knob: penalizes the path-tangent yaw *rate* so the
        # x/y curves avoid demanding sharp turns. w_yaw_acc penalizes yaw acceleration.
        self.declare_parameter('w_yaw_rate', 0.0)
        self.declare_parameter('w_yaw_acc', 5.0)
        # Output yaw (Concern B) weights: how hard to track the path tangent
        # (w_yaw_align) vs. smooth the heading (w_yaw_smooth = yaw acceleration,
        # w_yaw_accel_limit = acceleration normalized by limits, w_yaw_snap = yaw
        # snap). Boundary yaw is pinned from the request orientation.
        self.declare_parameter('w_yaw_align', 1.0)
        self.declare_parameter('w_yaw_smooth', 0.01)
        self.declare_parameter('w_yaw_accel_limit', 1.0)
        self.declare_parameter('w_yaw_snap', 0.0)
        # Two-mode dispatch: a request whose displacement and boundary speeds are both
        # below these thresholds is a hover-and-rotate / short move, handled by the
        # cheap in-place S-curve (Mode B) instead of the obstacle optimizer (Mode A).
        self.declare_parameter('mode_b_distance_threshold', 0.2)
        self.declare_parameter('mode_b_speed_threshold', 0.05)
        # Kinematic limits as a YAML string -> dict (defaults preserve historic values).
        self.declare_parameter('limits', yaml.safe_dump(DEFAULT_LIMITS))

        self._planner_type = self.get_parameter('planner').get_parameter_value().string_value.strip()
        if self._planner_type not in PLANNERS:
            raise ValueError(
                f"Unknown planner '{self._planner_type}'. Available: {sorted(PLANNERS)}.")
        self._limits = self._parse_limits(
            self.get_parameter('limits').get_parameter_value().string_value)

        # Default flight-space bounds (axis-aligned); overridable via SetTrajectoryBounds.
        self.declare_parameter('bounds.x_min', DEFAULT_BOUNDS['x_min'])
        self.declare_parameter('bounds.x_max', DEFAULT_BOUNDS['x_max'])
        self.declare_parameter('bounds.y_min', DEFAULT_BOUNDS['y_min'])
        self.declare_parameter('bounds.y_max', DEFAULT_BOUNDS['y_max'])
        self.declare_parameter('bounds.z_min', DEFAULT_BOUNDS['z_min'])
        self.declare_parameter('bounds.z_max', DEFAULT_BOUNDS['z_max'])
        self._state_lock = threading.Lock()

        self._planning_worker_lock = threading.Lock()
        self._planning_worker = None
        self._planning_reserved = False
        self._planning_callback_group = ReentrantCallbackGroup()
        self._mp_context = mp.get_context('spawn')

        # TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Bounds + obstacle handling (state, TF resolution, markers) live in the manager.
        self.obstacles = ObstacleManager(self)
        path_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._planned_path_topic = self.get_parameter(
            'planned_trajectory_path_topic').get_parameter_value().string_value.strip()
        self._optimizing_path_topic = self.get_parameter(
            'optimizing_trajectory_path_topic').get_parameter_value().string_value.strip()
        self._planned_path_publisher = (
            self.create_publisher(NavPath, self._planned_path_topic, path_qos)
            if self._planned_path_topic else None
        )
        self._optimizing_path_publisher = (
            self.create_publisher(NavPath, self._optimizing_path_topic, path_qos)
            if self._optimizing_path_topic else None
        )
        if self._planned_path_publisher is not None:
            self.get_logger().info(
                f"Planned trajectory Path visualization enabled on '{self._planned_path_topic}'."
            )
        if self._optimizing_path_publisher is not None:
            self.get_logger().info(
                f"Optimizing Path visualization enabled on '{self._optimizing_path_topic}'."
            )
        self.get_logger().info('Waiting to receive trajectory planning requests...')
        self._sampled_action_server = ActionServer(
            self,
            GetSampledTrajectory,
            'plan_sampled_trajectory',
            self.execute_sampled_trajectory_callback,
            goal_callback=self._planning_goal_callback,
            cancel_callback=self._planning_cancel_callback,
            callback_group=self._planning_callback_group,
        )
        self._trajectory_action_server = ActionServer(
            self,
            GetTrajectory,
            'plan_trajectory',
            self.execute_trajectory_callback,
            goal_callback=self._planning_goal_callback,
            cancel_callback=self._planning_cancel_callback,
            callback_group=self._planning_callback_group,
        )

        self._set_bounds_service = self.create_service(
            SetTrajectoryBounds, 'set_trajectory_bounds', self.obstacles.set_bounds_callback,
            callback_group=self._planning_callback_group)
        self._set_obstacles_service = self.create_service(
            SetTrajectoryObstacles, 'set_trajectory_obstacles', self.obstacles.set_obstacles_callback,
            callback_group=self._planning_callback_group)

        # Validate runtime parameter updates (the built-in SetParameters service) before
        # they are accepted; weights/thresholds must be non-negative and 'limits' must
        # parse with non-negative kinematic limits. Registered after every
        # declare_parameter so it governs only runtime sets, not the initial declarations.
        self.add_on_set_parameters_callback(self._validate_parameter_update)

    @staticmethod
    def _parse_limits(limits_yaml):
        parsed = dict(DEFAULT_LIMITS)
        if limits_yaml and limits_yaml.strip():
            loaded = yaml.safe_load(limits_yaml)
            if not isinstance(loaded, dict):
                raise ValueError("Parameter 'limits' must be a YAML mapping of limit names to values.")
            parsed.update({k: float(v) for k, v in loaded.items()})
        return parsed

    # Tunable weights/thresholds that must never be negative.
    NON_NEGATIVE_PARAMETERS = frozenset({
        'w_time', 'w_acc', 'w_jerk', 'w_snap',
        'w_yaw_rate', 'w_yaw_acc', 'w_yaw_align', 'w_yaw_smooth',
        'w_yaw_accel_limit', 'w_yaw_snap',
        'boundary_bias_weight', 'boundary_bias_margin',
        'backtrack_objective_weight', 'behind_start_objective_weight',
        'mode_b_distance_threshold', 'mode_b_speed_threshold',
        'gate_transition_distance', 'gate_transition_position_weight',
        'gate_transition_velocity_weight', 'gate_transition_velocity_threshold',
    })

    # Counts/spacings that would break the solve if set to zero or negative.
    POSITIVE_PARAMETERS = frozenset({
        'n_segments', 'n_eval', 'obstacle_base_samples', 'obstacle_refine_multiplier',
        'obstacle_refine_min_spacing', 'obstacle_refine_max_spacing',
        'trajectory_visualization_samples', 'min_segment_time', 'max_duration',
    })
    UNIT_INTERVAL_PARAMETERS = frozenset({
        'kinematic_constraint_scale',
    })

    def _validate_parameter_update(self, parameters):
        """Validate a runtime SetParameters request before the values are accepted."""
        for parameter in parameters:
            if parameter.name in self.NON_NEGATIVE_PARAMETERS:
                try:
                    value = float(parameter.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{parameter.name} must be a number')
                if value < 0.0:
                    return SetParametersResult(
                        successful=False,
                        reason=f'{parameter.name} must be >= 0 (got {value})')
            elif parameter.name in self.POSITIVE_PARAMETERS:
                try:
                    value = float(parameter.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{parameter.name} must be a number')
                if value <= 0.0:
                    return SetParametersResult(
                        successful=False,
                        reason=f'{parameter.name} must be > 0 (got {value})')
            elif parameter.name in self.UNIT_INTERVAL_PARAMETERS:
                try:
                    value = float(parameter.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{parameter.name} must be a number')
                if value <= 0.0 or value > 1.0:
                    return SetParametersResult(
                        successful=False,
                        reason=f'{parameter.name} must be in (0, 1] (got {value})')
            elif parameter.name == 'limits':
                try:
                    parsed = self._parse_limits(parameter.value)
                except Exception as exc:  # noqa: B902 -- surface any YAML/parse error
                    return SetParametersResult(
                        successful=False,
                        reason=f'invalid limits YAML: {exc}')
                for key in ('v_max', 'a_max', 'j_max', 's_max',
                            'yaw_rate_max', 'yaw_accel_max'):
                    if parsed[key] < 0.0:
                        return SetParametersResult(
                            successful=False,
                            reason=f'limits.{key} must be >= 0 (got {parsed[key]})')

        # Cross-parameter: n_segments must be compatible with the active seed strategy.
        # Use the updated value for whichever of the pair is in this request and the
        # current value for the other, so changing either is validated against both.
        updated = {parameter.name: parameter.value for parameter in parameters}
        if 'n_segments' in updated or 'seed_strategy' in updated:
            strategy = str(updated.get(
                'seed_strategy',
                self.get_parameter('seed_strategy').value)).strip()
            n_segments = int(updated.get(
                'n_segments',
                self.get_parameter('n_segments').value))
            allowed = seeding.SEED_STRATEGY_SEGMENT_COUNTS.get(strategy)
            if allowed is not None and n_segments not in allowed:
                return SetParametersResult(
                    successful=False,
                    reason=(f'n_segments={n_segments} is invalid for seed_strategy '
                            f'{strategy} (allowed: {sorted(allowed)})'))

        if 'min_segment_time' in updated or 'max_duration' in updated:
            min_segment_time = float(updated.get(
                'min_segment_time',
                self.get_parameter('min_segment_time').value))
            max_duration = float(updated.get(
                'max_duration',
                self.get_parameter('max_duration').value))
            if min_segment_time > max_duration:
                return SetParametersResult(
                    successful=False,
                    reason=(
                        'min_segment_time must be <= max_duration '
                        f'(got {min_segment_time} > {max_duration})'))

        return SetParametersResult(successful=True)

    @staticmethod
    def _default_trajectory_name(trajectory_id):
        return f'Trajectory {int(trajectory_id)}'

    @staticmethod
    def _is_gate_frame(frame_id):
        return 'gate' in str(frame_id or '').lower()

    @staticmethod
    def _unit_vector_or_none(vector, eps=1e-9):
        array = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(array))
        if norm <= float(eps):
            return None
        return array / norm

    @staticmethod
    def _point_to_vector(point):
        return np.asarray([float(point.x), float(point.y), float(point.z)], dtype=np.float64)

    @staticmethod
    def _linear_to_vector(linear):
        return np.asarray([float(linear.x), float(linear.y), float(linear.z)], dtype=np.float64)

    def _gate_transition_targets_from_goal(
        self,
        goal,
        base_start_pose,
        base_start_twist,
        base_target_pose,
        base_target_twist,
    ):
        distance = max(0.0, float(self.get_parameter('gate_transition_distance').value))
        position_weight = float(self.get_parameter('gate_transition_position_weight').value)
        velocity_weight = float(self.get_parameter('gate_transition_velocity_weight').value)
        if distance <= 0.0 or (position_weight <= 0.0 and velocity_weight <= 0.0):
            return []

        threshold = max(
            0.0,
            float(self.get_parameter('gate_transition_velocity_threshold').value),
        )

        def build_target(label, frame_id, pose_stamped, twist_stamped, fraction, sign):
            if not self._is_gate_frame(frame_id):
                return None
            position = self._point_to_vector(pose_stamped.pose.position)
            velocity = self._linear_to_vector(twist_stamped.twist.linear)
            speed = float(np.linalg.norm(velocity))
            direction = velocity / speed if speed > threshold else None
            if direction is None:
                tangent, _up = self._travel_tangent_from_pose_and_velocity(
                    pose_stamped.pose.orientation,
                    twist_stamped.twist.linear,
                )
                direction = self._unit_vector_or_none(tangent)
            if direction is None:
                return None
            return {
                'label': label,
                'fraction': float(fraction),
                'position': [float(value) for value in position + sign * distance * direction],
                'velocity': (
                    [float(value) for value in velocity]
                    if speed > threshold else None
                ),
            }

        targets = []
        for target in (
            build_target(
                'start_exit',
                goal.start.frame_id,
                base_start_pose,
                base_start_twist,
                1.0 / 3.0,
                1.0,
            ),
            build_target(
                'target_approach',
                goal.target.frame_id,
                base_target_pose,
                base_target_twist,
                2.0 / 3.0,
                -1.0,
            ),
        ):
            if target is not None:
                targets.append(target)
        return targets

    @staticmethod
    def _build_polynomial_piece(duration, poly_x, poly_y, poly_z, poly_yaw):
        return polynomial_piece_from_coefficients(
            duration,
            poly_x,
            poly_y,
            poly_z,
            poly_yaw,
            coefficient_order='drone',
        )

    @staticmethod
    def _poly_order_trajectory_from_pieces(pieces):
        durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs = coefficients_from_pieces(
            pieces,
            coefficient_order='numpy',
        )
        return PolyOrderTrajectory(durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs)

    def _visualization_samples(self):
        return max(2, int(self.get_parameter('trajectory_visualization_samples').value))

    def _publish_path_from_pieces(self, pieces, frame_id, publisher, topic, label):
        if publisher is None or not pieces:
            return
        try:
            trajectory = self._poly_order_trajectory_from_pieces(pieces)
            path_msg = trajectory_to_path_msg(
                trajectory,
                frame_id=frame_id,
                samples=self._visualization_samples(),
                stamp=self.get_clock().now().to_msg(),
            )
            publisher.publish(path_msg)
            self.get_logger().info(
                f"Published {label} trajectory visualization on '{topic}' "
                f'with {len(path_msg.poses)} poses.'
            )
        except Exception as exc:  # noqa: B902
            self.get_logger().warning(
                f"Failed to publish {label} trajectory visualization on '{topic}': {exc}"
            )

    def _pieces_from_segments_for_visualization(self, segments):
        if not segments:
            return []
        real_time_segments = [
            self._segment_coefficients_to_real_time(segment)
            for segment in segments
        ]
        return self._pieces_from_real_time_segments_for_visualization(real_time_segments)

    def _pieces_from_real_time_segments_for_visualization(self, real_time_segments):
        if not real_time_segments:
            return []
        yaw_coefficients = self._fit_output_yaw(real_time_segments)
        return [
            self._build_polynomial_piece(
                segment['T'],
                segment['c_x'],
                segment['c_y'],
                segment['c_z'],
                yaw_coeffs,
            )
            for segment, yaw_coeffs in zip(real_time_segments, yaw_coefficients)
        ]

    @staticmethod
    def _segment_coefficients_to_real_time(segment):
        duration = float(segment['T'])
        real_time_segment = {'T': duration}
        for axis in ('c_x', 'c_y', 'c_z'):
            coeffs = np.asarray(segment[axis], dtype=np.float64)
            scales = np.asarray([duration ** index for index in range(len(coeffs))], dtype=np.float64)
            real_time_segment[axis] = coeffs / scales
        return real_time_segment

    def _publish_optimizing_path(self, segments, frame_id, normalized_time=True):
        publisher = getattr(self, '_optimizing_path_publisher', None)
        if publisher is None:
            return
        if normalized_time:
            pieces = self._pieces_from_segments_for_visualization(segments)
        else:
            pieces = self._pieces_from_real_time_segments_for_visualization(segments)
        self._publish_path_from_pieces(
            pieces,
            frame_id,
            publisher,
            getattr(self, '_optimizing_path_topic', ''),
            'optimizing',
        )

    def _publish_empty_optimizing_path(self, frame_id):
        publisher = getattr(self, '_optimizing_path_publisher', None)
        if publisher is None:
            return
        path_msg = NavPath()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()
        publisher.publish(path_msg)

    def _publish_planned_path(self, pieces, frame_id):
        publisher = getattr(self, '_planned_path_publisher', None)
        if publisher is None:
            return
        self._publish_path_from_pieces(
            pieces,
            frame_id,
            publisher,
            getattr(self, '_planned_path_topic', ''),
            'planned',
        )

    @staticmethod
    def _empty_sampled_result(trajectory_id, name, message):
        result = GetSampledTrajectory.Result()
        result.success = False
        result.message = message
        result.trajectory = SampledTrajectory()
        result.trajectory.trajectory_id = int(trajectory_id)
        result.trajectory.name = name
        result.trajectory.points = []
        return result

    @staticmethod
    def _empty_trajectory_result(message):
        result = GetTrajectory.Result()
        result.success = False
        result.message = message
        result.trajectory = Trajectory()
        result.trajectory.trajectory_id = 255
        result.trajectory.pieces = []
        return result

    @staticmethod
    def _trajectory_summary(trajectory, planning_duration_sec=None):
        pieces = list(getattr(trajectory, 'pieces', []) or [])
        total_duration = sum(duration_to_seconds(piece.duration) for piece in pieces)
        segment_label = 'segment' if len(pieces) == 1 else 'segments'
        parts = [
            f'{len(pieces)} {segment_label}',
            f'duration={total_duration:.3f}s',
        ]
        if planning_duration_sec is not None:
            parts.append(f'planning_time={max(0.0, float(planning_duration_sec)):.3f}s')
        return ', '.join(parts)

    @classmethod
    def _trajectory_ready_message(
        cls,
        trajectory,
        prefix='Polynomial trajectory ready',
        planning_duration_sec=None,
    ):
        return f'{prefix}: {cls._trajectory_summary(trajectory, planning_duration_sec)}.'

    def _abort_sampled_with_result(self, goal_handle, trajectory_id, name, message):
        self.get_logger().error(message)
        goal_handle.abort()
        return self._empty_sampled_result(trajectory_id, name, message)

    def _abort_trajectory_with_result(self, goal_handle, message):
        self.get_logger().error(message)
        goal_handle.abort()
        return self._empty_trajectory_result(message)

    @staticmethod
    def _yaw_from_quaternion(orientation):
        """Extract the yaw (rotation about +z) from a geometry_msgs Quaternion."""
        x = float(orientation.x)
        y = float(orientation.y)
        z = float(orientation.z)
        w = float(orientation.w)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _frame_axes_from_quaternion(orientation):
        """Return vehicle forward and up axes from a geometry_msgs Quaternion."""
        rotation = R.from_quat([
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        ])
        return (
            rotation.apply([1.0, 0.0, 0.0]),
            rotation.apply([0.0, 0.0, 1.0]),
        )

    @classmethod
    def _travel_tangent_from_pose_and_velocity(cls, orientation, linear_velocity):
        """Return the Dubins travel tangent implied by pose forward axis and speed sign."""
        forward, up = cls._frame_axes_from_quaternion(orientation)
        velocity = np.asarray([
            float(linear_velocity.x),
            float(linear_velocity.y),
            float(linear_velocity.z),
        ], dtype=np.float64)
        sigma = -1.0 if float(np.dot(velocity, forward)) < -1e-6 else 1.0
        return sigma * forward, up

    # ANSI bright orange (256-colour) so the advisory yaw-accel warnings stand out.
    _YAW_WARN_ORANGE = '\033[1;38;5;208m'
    _YAW_WARN_RESET = '\033[0m'

    def _warn_on_output_yaw_accel_limit(self, durations, yaw_coefficients):
        """
        Advisory bright-orange warning when output yaw acceleration exceeds policy.

        Yaw rate is handled as an actionable path-tangent cap/refinement in the CasADi
        planner. Output yaw acceleration is soft-penalized in the KKT yaw fit; this
        warning remains advisory for residual overshoot.
        """
        messages = validation.find_yaw_limit_warnings(
            durations,
            yaw_coefficients,
            None,
            self._limits['yaw_accel_max'],
        )
        for message in messages:
            self.get_logger().warning(
                f'{self._YAW_WARN_ORANGE}YAW ACCEL LIMIT (advisory): {message}{self._YAW_WARN_RESET}'
            )

    def _should_use_in_place_scurve(self, start_pose, target_pose, start_linearv, target_linearv):
        """Mode B applies when the move is short and starts/ends near hover."""
        distance = math.sqrt(
            (target_pose.x - start_pose.x) ** 2
            + (target_pose.y - start_pose.y) ** 2
            + (target_pose.z - start_pose.z) ** 2
        )
        speed_start = math.sqrt(start_linearv.x ** 2 + start_linearv.y ** 2 + start_linearv.z ** 2)
        speed_target = math.sqrt(target_linearv.x ** 2 + target_linearv.y ** 2 + target_linearv.z ** 2)
        distance_threshold = float(self.get_parameter('mode_b_distance_threshold').value)
        speed_threshold = float(self.get_parameter('mode_b_speed_threshold').value)
        return (
            distance <= distance_threshold
            and speed_start <= speed_threshold
            and speed_target <= speed_threshold
        )

    def _build_in_place_scurve_pieces(
        self, start_pose, target_pose, start_linearv, target_linearv,
        base_start_pose, base_target_pose,
    ):
        """Build the Mode B x/y/z S-curve segments (yaw is filled in by the yaw stage)."""
        return scurve.plan_in_place_scurve(
            (start_pose.x, start_pose.y, start_pose.z),
            (target_pose.x, target_pose.y, target_pose.z),
            (start_linearv.x, start_linearv.y, start_linearv.z),
            (target_linearv.x, target_linearv.y, target_linearv.z),
            self._yaw_from_quaternion(base_start_pose.pose.orientation),
            self._yaw_from_quaternion(base_target_pose.pose.orientation),
            self._limits,
        )

    def _pieces_pass_validation(
        self, segments, start_pose, target_pose, start_linearv, target_linearv,
        bounds, obstacle_poses,
    ):
        """
        Run the dense trust-boundary validators; return False on failure.

        Used to decide whether a Mode B candidate is feasible in the current world (a
        blocked direct path fails ``validate_obstacle_clearance``) before committing to
        it; a failure falls back to the Mode A optimizer.
        """
        try:
            validation.validate_boundary_constraints(
                segments, start_pose, target_pose, start_linearv, target_linearv)
            validation.validate_kinematic_limits(segments, self._limits)
            validation.validate_position_bounds(segments, bounds)
            validation.validate_obstacle_clearance(segments, obstacle_poses)
        except TrajectoryPlanningError as exc:
            self.get_logger().info(f'Mode B in-place S-curve candidate rejected: {exc}')
            return False
        return True

    @staticmethod
    def _segments_from_pieces(pieces):
        durations, x_coeffs, y_coeffs, z_coeffs, _yaw_coeffs = coefficients_from_pieces(pieces)
        return [
            {
                'T': float(duration),
                'c_x': np.asarray(x_coeff, dtype=np.float64),
                'c_y': np.asarray(y_coeff, dtype=np.float64),
                'c_z': np.asarray(z_coeff, dtype=np.float64),
            }
            for duration, x_coeff, y_coeff, z_coeff in zip(
                durations, x_coeffs, y_coeffs, z_coeffs
            )
        ]

    def _cached_pieces_pass_validation(
        self, pieces, start_pose, target_pose, start_linearv, target_linearv,
        bounds, obstacle_poses,
    ):
        segments = self._segments_from_pieces(pieces)
        if not self._pieces_pass_validation(
            segments, start_pose, target_pose, start_linearv, target_linearv,
            bounds, obstacle_poses,
        ):
            return None
        validation.validate_continuity(segments)
        return segments

    def _fit_output_yaw(self, segments, start_yaw=None, target_yaw=None):
        """
        Produce the output yaw coefficients (Concern B) for the solved x/y/z path.

        Tracks the path tangent in the interior (look-ahead), pins the boundary yaw
        from the request orientation (``start_yaw``/``target_yaw``; ``None`` follows
        the tangent), and enforces C4 continuity across joins via a single KKT solve.
        """
        if not segments:
            return []
        return yaw_planning.fit_yaw_coefficients(
            segments,
            start_yaw=start_yaw,
            target_yaw=target_yaw,
            w_align=float(self.get_parameter('w_yaw_align').value),
            w_smooth=float(self.get_parameter('w_yaw_smooth').value),
            w_snap=float(self.get_parameter('w_yaw_snap').value),
            w_accel_limit=float(self.get_parameter('w_yaw_accel_limit').value),
            yaw_accel_max=self._limits.get('yaw_accel_max', DEFAULT_LIMITS['yaw_accel_max']),
            yaw_rate_max=self._limits.get('yaw_rate_max', DEFAULT_LIMITS['yaw_rate_max']),
            enforce_yaw_limits=True,
        )

    def _planning_goal_callback(self, _goal_request):
        if not self.obstacles.obstacles_configured():
            self.get_logger().warning(
                'Planning goal accepted with no service obstacle update yet. '
                'Default URDF obstacles will be used if available.'
            )
        with self._planning_worker_lock:
            worker_alive = self._planning_worker is not None and self._planning_worker.is_alive()
            if self._planning_reserved or worker_alive:
                self.obstacles.warn_rejected('Rejecting planning goal: another planning request is active.')
                return GoalResponse.REJECT
            self._planning_reserved = True
        return GoalResponse.ACCEPT

    def _planning_cancel_callback(self, _cancel_request):
        return CancelResponse.ACCEPT

    def _clear_planning_worker(self, worker=None):
        with self._planning_worker_lock:
            if worker is not None and self._planning_worker is not worker:
                return
            self._planning_worker = None
            self._planning_reserved = False

    def _set_planning_worker(self, worker):
        with self._planning_worker_lock:
            self._planning_worker = worker
            self._planning_reserved = True

    def _planning_timeout_sec(self, goal=None):
        server_max = max(
            0.1,
            float(self.get_parameter('planning_timeout_sec').get_parameter_value().double_value),
        )
        requested_timeout = float(getattr(goal, 'timeout_sec', 0.0)) if goal is not None else 0.0
        if requested_timeout > 0.0:
            return max(0.1, min(requested_timeout, server_max))
        return server_max

    def _planning_feedback_period_sec(self):
        parameter = self.get_parameter('planning_feedback_period_sec')
        value = parameter.get_parameter_value()
        period = float(value.double_value if value.double_value > 0.0 else parameter.value)
        return max(0.1, period)

    def _save_cache_enabled(self):
        return (
            bool(self.get_parameter('save_cache').value)
            or bool(self.get_parameter('use_cache').value)
        )

    @staticmethod
    def _publish_planning_feedback(goal_handle, feedback_type, start_time, timeout_sec, status=''):
        elapsed = max(0.0, time.monotonic() - start_time)
        feedback = feedback_type()
        feedback.remaining_sec = float(max(0.0, timeout_sec - elapsed))
        feedback.status = str(status or '')
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _safe_filename_part(value):
        cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value).strip())
        return cleaned.strip('_') or 'frame'

    @staticmethod
    def _float_signature(value):
        return round(float(value), PlanTrajectoryActionServer.CACHE_SIGNATURE_DECIMALS)

    @classmethod
    def _pose_signature(cls, pose):
        return {
            'position': {
                'x': cls._float_signature(pose.position.x),
                'y': cls._float_signature(pose.position.y),
                'z': cls._float_signature(pose.position.z),
            },
            'orientation': {
                'x': cls._float_signature(pose.orientation.x),
                'y': cls._float_signature(pose.orientation.y),
                'z': cls._float_signature(pose.orientation.z),
                'w': cls._float_signature(pose.orientation.w),
            },
        }

    @classmethod
    def _twist_signature(cls, twist):
        return {
            'linear': {
                'x': cls._float_signature(twist.linear.x),
                'y': cls._float_signature(twist.linear.y),
                'z': cls._float_signature(twist.linear.z),
            },
            'angular': {
                'x': cls._float_signature(twist.angular.x),
                'y': cls._float_signature(twist.angular.y),
                'z': cls._float_signature(twist.angular.z),
            },
        }

    @classmethod
    def _point_signature(cls, point):
        return {
            'x': cls._float_signature(point.x),
            'y': cls._float_signature(point.y),
            'z': cls._float_signature(point.z),
        }

    @classmethod
    def _vector_signature(cls, vector):
        return {
            'x': cls._float_signature(vector.x),
            'y': cls._float_signature(vector.y),
            'z': cls._float_signature(vector.z),
        }

    @classmethod
    def _obstacle_signature(cls, obstacles):
        signatures = []
        for obstacle in obstacles:
            sig = {}
            for key, value in sorted(obstacle.items()):
                if key in ('a_endpoint_contained', 'b_endpoint_contained'):
                    continue
                if isinstance(value, str):
                    sig[key] = value
                else:
                    sig[key] = cls._float_signature(value)
            signatures.append(sig)
        return signatures

    def _casadi_planner_kwargs(self):
        seed_strategy = str(self.get_parameter('seed_strategy').value).strip()
        return {
            'seed_strategy': seed_strategy,
            'time_bounds': (
                float(self.get_parameter('min_segment_time').value),
                float(self.get_parameter('max_duration').value),
            ),
            'gate_transition_distance': float(
                self.get_parameter('gate_transition_distance').value),
            'gate_transition_position_weight': float(
                self.get_parameter('gate_transition_position_weight').value),
            'gate_transition_velocity_weight': float(
                self.get_parameter('gate_transition_velocity_weight').value),
            'gate_transition_velocity_threshold': float(
                self.get_parameter('gate_transition_velocity_threshold').value),
            'n_eval': int(self.get_parameter('n_eval').value),
            'obstacle_base_samples': int(
                self.get_parameter('obstacle_base_samples').value),
            'obstacle_refine_multiplier': int(
                self.get_parameter('obstacle_refine_multiplier').value),
            'obstacle_refine_margin': float(
                self.get_parameter('obstacle_refine_margin').value),
            'obstacle_refine_radius_fraction': float(
                self.get_parameter('obstacle_refine_radius_fraction').value),
            'obstacle_refine_min_spacing': float(
                self.get_parameter('obstacle_refine_min_spacing').value),
            'obstacle_refine_max_spacing': float(
                self.get_parameter('obstacle_refine_max_spacing').value),
            'boundary_bias_margin': float(
                self.get_parameter('boundary_bias_margin').value),
            'boundary_bias_weight': float(
                self.get_parameter('boundary_bias_weight').value),
            'w_time': float(self.get_parameter('w_time').value),
            'w_acc': float(self.get_parameter('w_acc').value),
            'w_jerk': float(self.get_parameter('w_jerk').value),
            'w_snap': float(self.get_parameter('w_snap').value),
            'kinematic_constraint_scale': float(
                self.get_parameter('kinematic_constraint_scale').value),
            'w_yaw_rate': float(self.get_parameter('w_yaw_rate').value),
            'w_yaw_acc': float(self.get_parameter('w_yaw_acc').value),
            'backtrack_objective_weight': float(
                self.get_parameter('backtrack_objective_weight').value),
            'behind_start_objective_weight': float(
                self.get_parameter('behind_start_objective_weight').value),
        }

    def _ipopt_stage_options(self, prefix):
        return {
            'ipopt.max_iter': int(self.get_parameter(f'{prefix}_ipopt_max_iter').value),
            'ipopt.tol': float(self.get_parameter(f'{prefix}_ipopt_tol').value),
            'ipopt.constr_viol_tol': float(
                self.get_parameter(f'{prefix}_ipopt_constr_viol_tol').value),
            'ipopt.acceptable_tol': float(
                self.get_parameter(f'{prefix}_ipopt_acceptable_tol').value),
            'ipopt.acceptable_constr_viol_tol': float(
                self.get_parameter(f'{prefix}_ipopt_acceptable_constr_viol_tol').value),
            'ipopt.acceptable_iter': int(
                self.get_parameter(f'{prefix}_ipopt_acceptable_iter').value),
        }

    def _planner_cache_signature(self):
        return {
            'cache_schema': 4,
            'min_segment_time': self._float_signature(
                self.get_parameter('min_segment_time').value),
            'max_duration': self._float_signature(
                self.get_parameter('max_duration').value),
            'gate_transition_distance': self._float_signature(
                self.get_parameter('gate_transition_distance').value),
            'gate_transition_position_weight': self._float_signature(
                self.get_parameter('gate_transition_position_weight').value),
            'gate_transition_velocity_weight': self._float_signature(
                self.get_parameter('gate_transition_velocity_weight').value),
            'gate_transition_velocity_threshold': self._float_signature(
                self.get_parameter('gate_transition_velocity_threshold').value),
        }

    def _cache_filename(
        self,
        goal,
        start_label,
        target_label,
        preferred_frame,
        start_pose,
        target_pose,
        start_velocity,
        target_velocity,
        obstacle_poses,
        bounds,
    ):
        signature = {
            'planner_config': self._planner_cache_signature(),
            'preferred_frame': preferred_frame,
            'start_frame_id': goal.start.frame_id,
            'target_frame_id': goal.target.frame_id,
            'start_pose': self._pose_signature(goal.start.pose),
            'target_pose': self._pose_signature(goal.target.pose),
            'start_twist': self._twist_signature(goal.start.twist),
            'target_twist': self._twist_signature(goal.target.twist),
            'base_start_position': self._point_signature(start_pose),
            'base_target_position': self._point_signature(target_pose),
            'base_start_velocity': self._vector_signature(start_velocity),
            'base_target_velocity': self._vector_signature(target_velocity),
        }
        encoded = json.dumps(signature, sort_keys=True, separators=(',', ':')).encode()
        digest = hashlib.sha256(encoded).hexdigest()[:12]
        return f'{start_label}_{target_label}_{digest}.csv'

    async def execute_sampled_trajectory_callback(self, goal_handle):
        self.get_logger().info('GetSampledTrajectory goal request received.')
        start_time = time.monotonic()
        goal = goal_handle.request  # get goal
        timeout_sec = self._planning_timeout_sec(goal)
        traj_id = goal.trajectory_id
        traj_name = goal.name if goal.name else self._default_trajectory_name(traj_id)
        if not self.MIN_SAMPLE_DT <= float(goal.sample_dt) <= self.MAX_SAMPLE_DT:
            message = f'sample_dt must be between {self.MIN_SAMPLE_DT} and {self.MAX_SAMPLE_DT} seconds.'
            self._clear_planning_worker()
            return self._abort_sampled_with_result(goal_handle, traj_id, traj_name, message)

        status, trajectory, message = self._run_planning_worker(
            goal_handle,
            goal,
            traj_id,
            GetSampledTrajectory.Feedback,
            start_time,
            timeout_sec,
        )
        if status == 'canceled':
            goal_handle.canceled()
            return self._empty_sampled_result(traj_id, traj_name, message)
        if status == 'timeout' or status == 'error':
            self.get_logger().error(message)
            goal_handle.abort()
            return self._empty_sampled_result(traj_id, traj_name, message)

        sampled_trajectory = sampled_trajectory_from_pieces(
            traj_id,
            traj_name,
            trajectory.pieces,
            goal.sample_dt,
        )
        self._publish_planning_feedback(
            goal_handle,
            GetSampledTrajectory.Feedback,
            start_time,
            timeout_sec,
            'Sampling planned trajectory.',
        )
        self.get_logger().info(f'Gathered {len(sampled_trajectory.points)} trajectory points')

        goal_handle.succeed()

        result = GetSampledTrajectory.Result()
        result.success = True
        result.message = message
        result.trajectory = sampled_trajectory
        self._publish_planning_feedback(
            goal_handle,
            GetSampledTrajectory.Feedback,
            start_time,
            timeout_sec,
            'Sampled trajectory ready.',
        )
        self.get_logger().info(f'Sending sampled trajectory result: {result.trajectory}')
        return result

    async def execute_trajectory_callback(self, goal_handle):
        self.get_logger().info('GetTrajectory goal request received.')
        start_time = time.monotonic()
        goal = goal_handle.request
        timeout_sec = self._planning_timeout_sec(goal)
        traj_name = goal.name if goal.name else self._default_trajectory_name(goal.trajectory_id)
        status, trajectory, message = self._run_planning_worker(
            goal_handle,
            goal,
            goal.trajectory_id,
            GetTrajectory.Feedback,
            start_time,
            timeout_sec,
        )
        if status == 'canceled':
            goal_handle.canceled()
            return self._empty_trajectory_result(message)
        if status == 'timeout' or status == 'error':
            self.get_logger().error(message)
            goal_handle.abort()
            return self._empty_trajectory_result(message)

        goal_handle.succeed()

        result = GetTrajectory.Result()
        result.success = True
        result.trajectory = trajectory
        result.message = self._trajectory_ready_message(
            trajectory,
            message.rstrip('.'),
            planning_duration_sec=time.monotonic() - start_time,
        )
        self._publish_planning_feedback(
            goal_handle,
            GetTrajectory.Feedback,
            start_time,
            timeout_sec,
            result.message,
        )
        self.get_logger().info(
            'Sending polynomial trajectory result: '
            f'trajectory_id={trajectory.trajectory_id}, '
            f"name='{traj_name}', "
            f"message='{message}', "
            f'pieces={len(trajectory.pieces)}'
        )
        return result

    def _run_planning_worker(self, goal_handle, goal, traj_id, feedback_type, start_time, timeout_sec):
        status_box = ['Starting trajectory planning.']
        result_box = []
        error_box = []
        cancel_event = threading.Event()
        deadline = start_time + timeout_sec

        worker = threading.Thread(
            target=self._planning_worker_target,
            args=(goal, traj_id, status_box, result_box, error_box, cancel_event, deadline),
            daemon=True,
        )
        self._set_planning_worker(worker)
        worker.start()
        self._publish_planning_feedback(
            goal_handle, feedback_type, start_time, timeout_sec, status_box[0])
        feedback_period_sec = self._planning_feedback_period_sec()
        last_feedback_time = time.monotonic()

        while worker.is_alive():
            if goal_handle.is_cancel_requested:
                cancel_event.set()
                worker.join(timeout=2.0)
                self._clear_planning_worker(worker)
                return 'canceled', None, 'Trajectory planning canceled.'

            if time.monotonic() > deadline:
                worker.join(timeout=5.0)
                if result_box:
                    self._clear_planning_worker(worker)
                    trajectory, message = result_box[0]
                    return 'success', trajectory, message
                if error_box:
                    self._clear_planning_worker(worker)
                    status, message = error_box[0]
                    return status, None, message
                cancel_event.set()
                if not worker.is_alive():
                    self._clear_planning_worker(worker)
                return 'timeout', None, f'Trajectory planning timed out after {timeout_sec:.1f}s.'

            time.sleep(0.05)
            now = time.monotonic()
            if now - last_feedback_time >= feedback_period_sec:
                self._publish_planning_feedback(
                    goal_handle,
                    feedback_type,
                    start_time,
                    timeout_sec,
                    status_box[0],
                )
                last_feedback_time = now

        self._clear_planning_worker(worker)

        if error_box:
            status, message = error_box[0]
            return status, None, message
        if not result_box:
            return 'error', None, 'Trajectory planning worker exited without result.'
        trajectory, message = result_box[0]
        return 'success', trajectory, message

    def _planning_worker_target(
        self, goal, traj_id, status_box, result_box, error_box, cancel_event, deadline
    ):
        try:
            def status(message): return self._set_planning_status(status_box, message)
            trajectory, message = self._plan_polynomial_trajectory(
                goal,
                traj_id,
                status_callback=status,
                cancel_event=cancel_event,
                deadline=deadline,
            )
            result_box.append((trajectory, message))
        except TrajectoryPlanningCanceled as exc:
            error_box.append(('canceled', str(exc)))
        except TrajectoryPlanningTimeout as exc:
            error_box.append(('timeout', str(exc)))
        except TrajectoryPlanningError as exc:
            error_box.append(('error', str(exc)))
        finally:
            current_worker = threading.current_thread()
            with self._planning_worker_lock:
                if self._planning_worker is current_worker and cancel_event.is_set():
                    self._planning_worker = None
                    self._planning_reserved = False

    @staticmethod
    def _set_planning_status(status_box, value):
        status_box[0] = str(value or '')

    @staticmethod
    def _check_cancel_or_timeout(cancel_event, deadline):
        if cancel_event is not None and cancel_event.is_set():
            raise TrajectoryPlanningCanceled('Trajectory planning canceled.')
        if deadline is not None and time.monotonic() > deadline:
            raise TrajectoryPlanningTimeout('Trajectory planning timed out.')

    # Major-step labels keyed by the explicit ``step`` the optimizer child stamps on
    # each queue payload, so step numbering never depends on message wording.
    _PIPELINE_STEP_LABELS = {
        optimization.PIPELINE_STEP_SEED: 'Seed',
        optimization.PIPELINE_STEP_OBSTACLE_FREE: 'Geometric seed',
        optimization.PIPELINE_STEP_OBSTACLE_SOLVE: 'Obstacle solve',
        optimization.PIPELINE_STEP_DENSE_CHECK: 'Dense check',
        optimization.PIPELINE_STEP_DENSE_REFINE: 'Dense refinement',
    }
    # Fallback for payloads without an explicit step; keyed by stage only.
    _PIPELINE_STAGE_FALLBACK_LABELS = {
        'seed': 'Seed',
        'unconstrained': 'Geometric seed',
        'constrained': 'Obstacle solve',
    }

    @classmethod
    def _pipeline_step_label(cls, stage, step):
        label = cls._PIPELINE_STEP_LABELS.get(step)
        if label is not None:
            return label
        stage = str(stage or 'planner').lower()
        return cls._PIPELINE_STAGE_FALLBACK_LABELS.get(stage, f'Planner {stage}')

    @classmethod
    def _format_pipeline_status(cls, stage, message, step=None):
        """Clean single-line status for action feedback (no decoration)."""
        return f'{cls._pipeline_step_label(stage, step)} | {message}'

    @classmethod
    def _compact_pipeline_feedback(cls, payload):
        """Short status for action feedback; detailed text stays in node logs."""
        stage = str(payload.get('stage', '') or '').lower()
        step = payload.get('step')
        message = str(payload.get('msg', '') or '')
        lower = message.lower()
        label = cls._pipeline_step_label(stage, step)

        if step == optimization.PIPELINE_STEP_SEED:
            summary = 'seed ready'
        elif 'started' in lower or lower.startswith('starting'):
            summary = 'solving'
        elif 'dense-valid' in lower:
            summary = 'dense-valid'
        elif 'accepted as seed' in lower:
            summary = 'seed accepted'
        elif 'unavailable' in lower:
            summary = 'using base seed'
        elif 'failed dense validation' in lower:
            summary = 'dense validation failed'
        elif 'time-dilated' in lower:
            summary = 'time-scaled seed'
        elif 'finished' in lower and 'running dense validation' in lower:
            summary = 'checking dense samples'
        elif payload.get('valid'):
            summary = 'valid candidate'
        else:
            summary = 'running'

        detail_parts = []
        duration_sec = payload.get('duration_sec')
        cost = payload.get('cost')
        if duration_sec is not None and summary not in ('solving', 'running'):
            detail_parts.append(f'T={float(duration_sec):.2f}s')
        if cost is not None and summary not in ('solving', 'running'):
            detail_parts.append(f'cost={float(cost):.3g}')
        if detail_parts:
            summary = f'{summary} ({", ".join(detail_parts)})'
        if summary in ('solving', 'running'):
            return label
        return f'{label} | {summary}'

    @classmethod
    def _format_pipeline_log(cls, stage, message, step=None):
        """Console form: blank lines + marker separate steps amid terminal spam."""
        return f'\n\n\n>>> {cls._format_pipeline_status(stage, message, step)}'

    def _plan_polynomial_trajectory(
        self,
        goal,
        traj_id,
        progress_callback=None,
        status_callback=None,
        cancel_event=None,
        deadline=None,
    ):
        # Snapshot the kinematic limits from the (runtime-settable) 'limits' parameter
        # at the start of the request, so a live change takes effect on the next plan
        # without mutating any in-flight solve.
        self._limits = self._parse_limits(
            self.get_parameter('limits').get_parameter_value().string_value)

        preferred_frame = self.get_parameter('preferred_frame').get_parameter_value().string_value.strip()
        if not preferred_frame:
            raise TrajectoryPlanningError("Parameter 'preferred_frame' must not be empty.")
        preferred_frame = self.obstacles.obstacle_planning_frame(preferred_frame)
        if not preferred_frame:
            raise TrajectoryPlanningError('Planning frame must not be empty.')
        artifact_root = str(self.get_parameter('planner_artifact_dir').value).strip()
        artifact_request_id = f'request_{time.time_ns()}'
        self._check_cancel_or_timeout(cancel_event, deadline)

        # ~~~transform and calculate~~~~
        if progress_callback:
            progress_callback(0.1)
        if status_callback:
            status_callback('Transforming planning boundaries.')
        base_start_pose = self.transform_pose_to_frame(goal.start.pose, goal.start.frame_id, preferred_frame)
        self._check_cancel_or_timeout(cancel_event, deadline)
        base_start_twist = self.transform_twist_to_frame(goal.start.twist, goal.start.frame_id, preferred_frame)
        self._check_cancel_or_timeout(cancel_event, deadline)
        base_target_pose = self.transform_pose_to_frame(goal.target.pose, goal.target.frame_id, preferred_frame)
        self._check_cancel_or_timeout(cancel_event, deadline)
        base_target_twist = self.transform_twist_to_frame(goal.target.twist, goal.target.frame_id, preferred_frame)
        self._check_cancel_or_timeout(cancel_event, deadline)

        # Abort if any transform failed
        if not all([base_start_pose, base_start_twist, base_target_pose, base_target_twist]):
            raise TrajectoryPlanningError(
                f"Failed to transform start/target pose or twist to '{preferred_frame}' frame.",
            )

        self.get_logger().info('Completed transforms')
        if progress_callback:
            progress_callback(0.25)
        if status_callback:
            status_callback('Resolving bounds and obstacles.')
        self._log_planning_boundaries(
            goal,
            preferred_frame,
            base_start_pose,
            base_start_twist,
            base_target_pose,
            base_target_twist,
        )

        # get poses to pass into optimizer
        #
        # NOTE: only the boundary positions are used by the x/y/z optimizer here.
        # Boundary pose orientation is consumed later by the output-yaw fitting
        # stage when yaw_follow_tangent is enabled.
        start_pose = base_start_pose.pose.position
        target_pose = base_target_pose.pose.position
        start_linearv = base_start_twist.twist.linear
        target_linearv = base_target_twist.twist.linear

        self._publish_empty_optimizing_path(preferred_frame)
        self.obstacles.clear_markers(preferred_frame)

        # Snapshot runtime-configurable bounds and service obstacles, then merge
        # them with the latest default URDF capsules for this planning request.
        bounds, service_obstacles = self.obstacles.snapshot()
        obstacle_poses = self.obstacles.effective_obstacles(service_obstacles, bounds, preferred_frame)
        if not obstacle_poses:
            self.get_logger().warning(
                'Planning with no active obstacles. '
                'Configure set_trajectory_obstacles or obstacle_description_topics to enable avoidance.'
            )
        self.obstacles.publish_markers_if_subscribed(obstacle_poses, preferred_frame)
        self.get_logger().debug(f'Planning with bounds={bounds}, obstacles={obstacle_poses}')

        def check_existance():
            """Find the cache path for an existing or newly generated trajectory."""
            start_label = self._safe_filename_part(goal.start.frame_id or 'start')
            target_label = self._safe_filename_part(goal.target.frame_id or 'target')

            filename = self._cache_filename(
                goal,
                start_label,
                target_label,
                preferred_frame,
                start_pose,
                target_pose,
                start_linearv,
                target_linearv,
                obstacle_poses,
                bounds,
            )
            output_dir_param = self.get_parameter('output_dir').get_parameter_value().string_value
            if not output_dir_param.strip():
                raise ValueError(
                    "Parameter 'output_dir' is required for plan_trajectory_action_server output."
                )

            output_root = normalize_path_value(output_dir_param)

            file_path = output_root / filename

            # To make sure everything exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # print(f"Filename: {filename}")
            return file_path, start_label, target_label

        # Check if this trajectory already exists. If not, return the generated file path
        try:
            file_path, start_label, target_label = check_existance()
        except Exception as e:
            raise TrajectoryPlanningError(
                f'Trajectory cache setup failed: {e}',
            )
        self._check_cancel_or_timeout(cancel_event, deadline)
        if progress_callback:
            progress_callback(0.45)

        use_cache = bool(self.get_parameter('use_cache').value)
        cached_seed_segments = None
        if use_cache and os.path.isfile(file_path):
            self.get_logger().info(f'Trajectory {start_label} to {target_label} already exists.' ' Retrieving...')

            try:
                pieces = load_trajectory_pieces(file_path)
            except Exception as e:
                self.get_logger().warning(
                    f"Failed to load cached trajectory '{file_path}'; regenerating: {e}"
                )
            else:
                cache_valid_segments = None
                try:
                    cache_valid_segments = self._cached_pieces_pass_validation(
                        pieces,
                        start_pose,
                        target_pose,
                        start_linearv,
                        target_linearv,
                        bounds,
                        obstacle_poses,
                    )
                except TrajectoryPlanningError as exc:
                    self.get_logger().info(
                        f'Cached trajectory failed dense validation; regenerating with it as seed: {exc}'
                    )
                    cached_seed_segments = self._segments_from_pieces(pieces)

                if cache_valid_segments is not None:
                    trajectory = trajectory_from_pieces(traj_id, pieces)
                    self._publish_planned_path(pieces, preferred_frame)
                    self.get_logger().info(
                        f'Gathered {len(trajectory.pieces)} dense-valid cached trajectory pieces')
                    if progress_callback:
                        progress_callback(0.9)
                    return trajectory, 'Loaded cached trajectory.'

                if cached_seed_segments is None:
                    cached_seed_segments = self._segments_from_pieces(pieces)
                self.get_logger().info(
                    'Cached trajectory is not feasible under current limits/bounds/obstacles; '
                    'using it as the optimizer seed.')

        else:  # if the trajectory doesn't already exist, continue with generation
            if os.path.isfile(file_path):
                self.get_logger().info(
                    f'Cache disabled; regenerating trajectory {start_label} to {target_label}'
                )
            else:
                self.get_logger().info(f'Generating trajectory {start_label} to {target_label}')
            if progress_callback:
                progress_callback(optimization.OPTIMIZER_PROGRESS_START)
        self._check_cancel_or_timeout(cancel_event, deadline)

        # Two-mode dispatch. A short, near-hover request is a hover-and-rotate / small
        # move, handled by the cheap in-place S-curve (Mode B). If its direct path is
        # blocked or it violates a limit it fails validation and we fall back to the
        # Mode A obstacle optimizer, which is forced for that request.
        my_traj_pieces = None
        if (
            cached_seed_segments is None
            and self._should_use_in_place_scurve(start_pose, target_pose, start_linearv, target_linearv)
        ):
            try:
                candidate = self._build_in_place_scurve_pieces(
                    start_pose, target_pose, start_linearv, target_linearv,
                    base_start_pose, base_target_pose,
                )
            except TrajectoryPlanningError as exc:
                candidate = None
                self.get_logger().warning(
                    f'Mode B in-place S-curve could not be sized within limits ({exc}); '
                    'falling back to Mode A obstacle optimizer.')
            if candidate is not None and self._pieces_pass_validation(
                candidate, start_pose, target_pose, start_linearv, target_linearv,
                bounds, obstacle_poses,
            ):
                self.get_logger().info('Using Mode B in-place S-curve (hover-and-rotate).')
                if status_callback:
                    status_callback('Planning in-place S-curve (hover-and-rotate).')
                my_traj_pieces = candidate
            elif candidate is not None:
                self.get_logger().warning(
                    'Mode B in-place S-curve failed validation; '
                    'falling back to Mode A obstacle optimizer.')

        # Attempt to run the selected planner with the given parameters
        try:
            if my_traj_pieces is None:
                if status_callback:
                    status_callback('Building seed and launching planner worker.')
                gate_transition_targets = self._gate_transition_targets_from_goal(
                    goal,
                    base_start_pose,
                    base_start_twist,
                    base_target_pose,
                    base_target_twist,
                )
                my_traj_pieces = self._run_optimizer_process(
                    start_pose,
                    target_pose,
                    start_linearv,
                    target_linearv,
                    obstacle_poses,
                    bounds,
                    cancel_event,
                    deadline,
                    planning_frame=preferred_frame,
                    progress_callback=progress_callback,
                    status_callback=status_callback,
                    seed_segments=cached_seed_segments,
                    seed_segments_are_real_time=cached_seed_segments is not None,
                    artifact_root=artifact_root,
                    artifact_request_id=artifact_request_id,
                    start_orientation=base_start_pose.pose.orientation,
                    target_orientation=base_target_pose.pose.orientation,
                    gate_transition_targets=gate_transition_targets,
                )
                self._publish_empty_optimizing_path(preferred_frame)
            self._check_cancel_or_timeout(cancel_event, deadline)
            if progress_callback:
                progress_callback(0.8)
            # The CasADi staged child already dense-validates successful candidates.
            # Keep this parent-side gate as a planner-agnostic trust boundary after
            # process serialization; it duplicates normal CasADi work by design.
            validation.validate_boundary_constraints(
                my_traj_pieces,
                start_pose,
                target_pose,
                start_linearv,
                target_linearv,
            )
            if progress_callback:
                progress_callback(0.83)
            validation.validate_kinematic_limits(my_traj_pieces, self._limits)
            validation.validate_position_bounds(my_traj_pieces, bounds)
            if progress_callback:
                progress_callback(0.85)
            validation.validate_obstacle_clearance(my_traj_pieces, obstacle_poses)
            if progress_callback:
                progress_callback(0.86)

            # Informational continuity diagnostics at tight tolerances. The hard
            # continuity gate (looser, shippable) already ran inside the child via
            # validation.validate_continuity; these warnings only surface borderline
            # junctions for logs.
            for degree, tolerance in sorted(validation.DIAGNOSTIC_CONTINUITY_TOLERANCES.items()):
                self._check_cancel_or_timeout(cancel_event, deadline)
                validation.warn_on_discontinuity(
                    my_traj_pieces, degree, tolerance, self.get_logger())

        except (TrajectoryPlanningCanceled, TrajectoryPlanningTimeout):
            raise
        except Exception as e:
            raise TrajectoryPlanningError(
                f'Trajectory generation failed: {e}',
            )

        self.get_logger().info('Completed trajectory generation')
        self._check_cancel_or_timeout(cancel_event, deadline)
        if progress_callback:
            progress_callback(0.88)

        yaw_follow_tangent = bool(self.get_parameter('yaw_follow_tangent').value)
        if yaw_follow_tangent:
            # Pin the boundary yaw from the request orientation (honored as a valid
            # unit quaternion by contract); the interior tracks the path tangent.
            yaw_coefficients = self._fit_output_yaw(
                my_traj_pieces,
                start_yaw=self._yaw_from_quaternion(base_start_pose.pose.orientation),
                target_yaw=self._yaw_from_quaternion(base_target_pose.pose.orientation),
            )
        else:
            yaw_coefficients = [
                np.zeros_like(seg_dict['c_x'])
                for seg_dict in my_traj_pieces
            ]

        self._warn_on_output_yaw_accel_limit(
            [seg_dict['T'] for seg_dict in my_traj_pieces],
            yaw_coefficients,
        )

        pieces = []

        # Planners return real-time coefficients: x(t) = sum_k c_k t^k.
        for seg_dict, yaw_coeffs in zip(my_traj_pieces, yaw_coefficients):
            self._check_cancel_or_timeout(cancel_event, deadline)
            current_piece = self._build_polynomial_piece(
                seg_dict['T'],
                seg_dict['c_x'],
                seg_dict['c_y'],
                seg_dict['c_z'],
                yaw_coeffs,
            )
            pieces.append(current_piece)

        trajectory = trajectory_from_pieces(traj_id, pieces)

        self._check_cancel_or_timeout(cancel_event, None)
        optional_outputs_allowed = deadline is None or time.monotonic() <= deadline
        if optional_outputs_allowed:
            final_artifact = self._write_final_trajectory_artifact(
                pieces,
                artifact_root,
                artifact_request_id,
            )
            if final_artifact is not None:
                self.get_logger().info(
                    f'Wrote final yaw-processed trajectory artifact to {final_artifact}')
        else:
            self.get_logger().info(
                'Skipping final trajectory artifact because action result is ready at timeout.')

        if self._save_cache_enabled():
            if optional_outputs_allowed:
                try:
                    self._check_cancel_or_timeout(cancel_event, None)
                    self.save_to_csv(pieces, file_path)
                except TrajectoryPlanningCanceled:
                    raise
                except Exception as e:
                    raise TrajectoryPlanningError(
                        f"Failed to save trajectory cache '{file_path}': {e}",
                    )
            else:
                self.get_logger().info(
                    'Skipping trajectory cache write because action result is ready at timeout.')
        else:
            self.get_logger().info('Trajectory cache saving disabled; generated trajectory not written.')

        self.get_logger().info(f'Gathered {len(pieces)} trajectory pieces')
        if optional_outputs_allowed:
            self._publish_planned_path(pieces, preferred_frame)
        else:
            self.get_logger().info(
                'Skipping planned trajectory visualization because action result is ready at timeout.')
        if progress_callback:
            progress_callback(0.9)

        self._check_cancel_or_timeout(cancel_event, None)
        return trajectory, 'Generated trajectory.'

    def _log_planning_boundaries(
        self,
        goal,
        preferred_frame,
        base_start_pose,
        base_start_twist,
        base_target_pose,
        base_target_twist,
    ):
        def point_text(point):
            return f'({float(point.x):.3f}, {float(point.y):.3f}, {float(point.z):.3f})'

        requested_start = goal.start.pose.position
        requested_target = goal.target.pose.position
        requested_start_twist = goal.start.twist.linear
        requested_target_twist = goal.target.twist.linear
        transformed_start = base_start_pose.pose.position
        transformed_target = base_target_pose.pose.position
        transformed_start_twist = base_start_twist.twist.linear
        transformed_target_twist = base_target_twist.twist.linear

        self.get_logger().info(
            'Planning boundary request: '
            f"start_frame='{goal.start.frame_id}', start_pos={point_text(requested_start)}, "
            f'start_vel={point_text(requested_start_twist)}, '
            f"target_frame='{goal.target.frame_id}', target_pos={point_text(requested_target)}, "
            f'target_vel={point_text(requested_target_twist)}'
        )
        self.get_logger().info(
            'Planning boundary transformed: '
            f"preferred_frame='{preferred_frame}', "
            f'start_pos={point_text(transformed_start)}, start_vel={point_text(transformed_start_twist)}, '
            f'target_pos={point_text(transformed_target)}, target_vel={point_text(transformed_target_twist)}'
        )

    def _run_optimizer_process(
        self,
        start_pose,
        target_pose,
        start_linearv,
        target_linearv,
        obstacle_poses,
        bounds,
        cancel_event,
        deadline,
        planning_frame='map',
        progress_callback=None,
        status_callback=None,
        seed_segments=None,
        seed_segments_are_real_time=False,
        artifact_root=None,
        artifact_request_id=None,
        start_orientation=None,
        target_orientation=None,
        gate_transition_targets=None,
    ):
        planner_kwargs = {}
        if self._planner_type == 'casadi_obstacle':
            planner_kwargs = self._casadi_planner_kwargs()
        requested_segments = int(self.get_parameter('n_segments').value)
        start_position = {'x': float(start_pose.x), 'y': float(start_pose.y), 'z': float(start_pose.z)}
        target_position = {'x': float(target_pose.x), 'y': float(target_pose.y), 'z': float(target_pose.z)}
        start_velocity = {'x': float(start_linearv.x), 'y': float(start_linearv.y), 'z': float(start_linearv.z)}
        target_velocity = {'x': float(target_linearv.x), 'y': float(target_linearv.y), 'z': float(target_linearv.z)}
        start_tangent = None
        target_tangent = None
        start_up_axis = None
        target_up_axis = None
        if start_orientation is not None and target_orientation is not None:
            start_tangent, start_up_axis = self._travel_tangent_from_pose_and_velocity(
                start_orientation, start_linearv)
            target_tangent, target_up_axis = self._travel_tangent_from_pose_and_velocity(
                target_orientation, target_linearv)
        numeric_obstacles = [
            obstacles.numeric_obstacle(obstacle)
            for obstacle in obstacle_poses
        ]

        def vector_list_or_none(vector):
            if vector is None:
                return None
            return [float(value) for value in vector]

        if seed_segments is not None:
            initial_segments = seed_segments
            self.get_logger().info(
                f'Using cached trajectory as optimizer seed ({len(initial_segments)} segment(s)).'
            )
        else:
            initial_segments = self._build_initial_seed(
                planner_kwargs,
                start_position,
                target_position,
                start_velocity,
                target_velocity,
                numeric_obstacles,
                bounds,
                requested_segments,
                start_tangent=start_tangent,
                target_tangent=target_tangent,
                z_axis=start_up_axis,
                goal_z_axis=target_up_axis,
            )
        self._publish_optimizing_path(
            initial_segments,
            planning_frame,
            normalized_time=not seed_segments_are_real_time,
        )
        n_segments = len(initial_segments) if initial_segments is not None else requested_segments
        numeric_inputs = {
            'planner': self._planner_type,
            'planner_kwargs': planner_kwargs,
            'limits': dict(self._limits),
            'bounds': dict(bounds),
            'n_segments': n_segments,
            'start_position': start_position,
            'target_position': target_position,
            'start_velocity': start_velocity,
            'target_velocity': target_velocity,
            'start_tangent': vector_list_or_none(start_tangent),
            'target_tangent': vector_list_or_none(target_tangent),
            'gate_transition_targets': list(gate_transition_targets or []),
            'obstacles': numeric_obstacles,
            'initial_segments': (
                optimization.serialize_optimizer_segments(initial_segments)
                if initial_segments is not None else None
            ),
            'initial_segments_are_real_time': bool(seed_segments_are_real_time),
            'artifact_root': (
                str(self.get_parameter('planner_artifact_dir').value).strip()
                if artifact_root is None else str(artifact_root).strip()
            ),
            'artifact_request_id': (
                f'request_{time.time_ns()}'
                if artifact_request_id is None else str(artifact_request_id).strip()
            ),
            'initial_ipopt_options': self._ipopt_stage_options('initial'),
            'constrained_ipopt_options': self._ipopt_stage_options('constrained'),
        }

        if self._planner_type != 'casadi_obstacle':
            segments = optimization.run_optimizer_process(
                self._mp_context,
                numeric_inputs,
                cancel_event,
                deadline,
                self._check_cancel_or_timeout,
                progress_callback=progress_callback,
            )
            if segments:
                self._publish_optimizing_path(
                    segments, planning_frame, normalized_time=False)
            return segments

        def pipeline_message(payload):
            message = payload.get('msg', '')
            stage = payload.get('stage', 'planner')
            step = payload.get('step')
            metric_parts = []
            duration_sec = payload.get('duration_sec')
            cost = payload.get('cost')
            if duration_sec is not None:
                metric_parts.append(f'duration={float(duration_sec):.3f}s')
            if cost is not None:
                metric_parts.append(f'cost={float(cost):.6g}')
            if metric_parts:
                metrics = ' '.join(metric_parts)
                message = f'{message} [{metrics}]'
            status_text = self._compact_pipeline_feedback(payload)
            log_text = self._format_pipeline_log(stage, message, step)
            remaining = payload.get('remaining_sec')
            if remaining is not None:
                log_text = f'{log_text} ({float(remaining):.1f}s remaining)'
            if status_callback:
                status_callback(status_text)
            self.get_logger().info(log_text)
            serialized = payload.get('trajectory')
            if serialized is not None:
                snapshot_segments = optimization.deserialize_optimizer_segments(serialized)
                self._publish_optimizing_path(
                    snapshot_segments, planning_frame, normalized_time=False)
            # A stage that solved against a narrowed obstacle set (e.g. the
            # warm-start stage limited to gate-adjacent obstacles) republishes
            # that subset here; the next stage republishes the full set once it's
            # back in play. Absent -> leave the currently displayed markers alone.
            stage_obstacles = payload.get('obstacles')
            if stage_obstacles is not None:
                self.obstacles.publish_markers_if_subscribed(stage_obstacles, planning_frame)

        segments = optimization.run_planner_pipeline_process(
            self._mp_context,
            numeric_inputs,
            cancel_event,
            deadline,
            self._check_cancel_or_timeout,
            message_callback=pipeline_message,
            progress_callback=progress_callback,
        )
        if segments:
            self._publish_optimizing_path(
                segments, planning_frame, normalized_time=False)
        return segments

    def _build_initial_seed(
        self,
        planner_kwargs,
        start_position,
        target_position,
        start_velocity,
        target_velocity,
        obstacle_poses,
        bounds,
        n_segments,
        start_tangent=None,
        target_tangent=None,
        z_axis=None,
        goal_z_axis=None,
    ):
        planner_cls = PLANNERS[self._planner_type]
        planner = planner_cls(
            limits=dict(self._limits),
            bounds=dict(bounds),
            n_segments=n_segments,
            **planner_kwargs,
        )
        start = {
            'position': start_position,
            'velocity': start_velocity,
        }
        goal = {
            'position': target_position,
            'velocity': target_velocity,
        }
        if start_tangent is not None:
            start['travel_tangent'] = start_tangent
        if target_tangent is not None:
            goal['travel_tangent'] = target_tangent
        if z_axis is not None:
            start['up_axis'] = z_axis
        if goal_z_axis is not None:
            goal['up_axis'] = goal_z_axis
        seed = planner.build_base_seed(
            start,
            goal,
            obstacles=obstacle_poses,
            bounds=bounds,
            limits=dict(self._limits),
            n_segments=n_segments,
        )
        self._log_seed_diagnostics(getattr(planner, 'last_seed_diagnostics', None))
        return seed

    def _log_seed_diagnostics(self, diagnostics):
        if not diagnostics:
            return
        if diagnostics.get('strategy') == 'unconstrained_optimizer':
            self.get_logger().info(
                'Seed strategy: unconstrained optimizer '
                f"({diagnostics.get('base_strategy', 'unknown')} "
                f"seed, {diagnostics.get('returned_segments', 'unknown')} segment(s))."
            )
            return
        if diagnostics.get('strategy') == 'dubins':
            self.get_logger().info(
                'Seed strategy: dubins '
                f"(radius={float(diagnostics.get('turning_radius', 0.0)):.3f} m, "
                f"path={float(diagnostics.get('path_length', 0.0)):.3f} m, "
                f"segments={diagnostics.get('returned_segments', 'unknown')})."
            )
            return
        moves = diagnostics.get('moves', [])
        if diagnostics.get('subdivided'):
            self.get_logger().info(
                'Seed subdivision: '
                f"{diagnostics.get('base_segments', 3)} -> "
                f"{diagnostics.get('returned_segments', 'unknown')} segment(s)."
            )
        if not moves:
            self.get_logger().info('Seed obstacle adjustment: no guide point intersections.')
            return
        for move in moves:
            before = move['point_before']
            after = move['point_after']
            self.get_logger().info(
                'Seed obstacle adjustment: '
                f'{move["label"]} moved for obstacle {move["obstacle_index"]} '
                f'({move["obstacle_type"]}): '
                f'distance {move["distance_before"]:.3f} -> {move["distance_after"]:.3f}, '
                f'point ({before[0]:.3f}, {before[1]:.3f}, {before[2]:.3f}) -> '
                f'({after[0]:.3f}, {after[1]:.3f}, {after[2]:.3f}).'
            )

    def save_to_csv(self, pieces, file_path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = file_path.with_name(f'.{file_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
        try:
            save_trajectory_pieces(tmp_path, pieces)
            os.replace(tmp_path, file_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _write_final_trajectory_artifact(self, pieces, artifact_root, artifact_request_id):
        artifact_root = str(artifact_root or '').strip()
        if not artifact_root:
            return None
        safe_request_id = re.sub(
            r'[^A-Za-z0-9_.-]+',
            '_',
            str(artifact_request_id or 'request').strip(),
        ).strip('_') or 'request'
        artifact_path = Path(artifact_root) / (
            f'flexible_drones_planner_{safe_request_id}_final_with_yaw_'
            f'{int(time.time())}_{uuid.uuid4().hex[:8]}.csv'
        )
        self.save_to_csv(pieces, artifact_path)
        return artifact_path

    def transform_pose_to_frame(self, pose, source_frame, target_frame='map'):
        stamped = PoseStamped()
        stamped.header.frame_id = source_frame
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.pose = pose

        if source_frame == target_frame:
            result = PoseStamped()
            result.header.frame_id = target_frame
            result.header.stamp = stamped.header.stamp
            result.pose = deepcopy(pose)
            return result

        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # Correct transform: transform the Pose inside the PoseStamped
            transformed_pose = tf2_geometry_msgs.do_transform_pose(stamped.pose, tf)

            # Put result back into a PoseStamped (optional)
            result = PoseStamped()
            result.header = tf.header
            result.pose = transformed_pose
            return result

        except TransformException as e:
            self.get_logger().error(f'[TF] Pose transform failed: {str(e)}')
            return None

    def transform_twist_to_frame(self, twist, source_frame, target_frame='map'):
        stamped = TwistStamped()
        stamped.header.frame_id = source_frame
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.twist = twist

        if source_frame == target_frame:
            result = TwistStamped()
            result.header.frame_id = target_frame
            result.header.stamp = stamped.header.stamp
            result.twist = deepcopy(twist)
            return result

        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # Extract rotation from transform
            q = tf.transform.rotation
            rot_matrix = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

            # Rotate linear and angular vectors
            lin = twist.linear
            ang = twist.angular

            lin_target = rot_matrix @ [lin.x, lin.y, lin.z]
            ang_target = rot_matrix @ [ang.x, ang.y, ang.z]

            # Build transformed TwistStamped
            result = TwistStamped()
            result.header = tf.header
            result.twist.linear.x, result.twist.linear.y, result.twist.linear.z = lin_target
            result.twist.angular.x, result.twist.angular.y, result.twist.angular.z = ang_target

            return result

        except TransformException as e:
            self.get_logger().error(f'[TF] Twist transform failed: {str(e)}')
            return None


def main(args=None):
    rclpy.init(args=args)

    trajectory_planner_server = PlanTrajectoryActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(trajectory_planner_server)
    try:
        executor.spin()
    except BENIGN_SHUTDOWN_EXCEPTIONS:
        pass
    finally:
        executor.shutdown()
        trajectory_planner_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':

    try:
        main()
    except BENIGN_SHUTDOWN_EXCEPTIONS:
        pass
    except Exception as exc:
        print(exc, flush=True)
    finally:
        print('plan_trajectory_action_server is done!', flush=True)

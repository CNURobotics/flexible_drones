#!/usr/bin/env python3
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

"""Planner demo: plan A-B-C-D, visualize, upload, fly, land, disarm."""

import argparse
import select
import sys
import time
from copy import deepcopy
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Twist
from rclpy.action import ActionClient
from rclpy.exceptions import ROSInterruptException
from rclpy.executors import ExternalShutdownException, ShutdownException

from flexible_drones_msgs.action import GetTrajectory
from flexible_drones_msgs.msg import TrajectoryBoundary
from flexible_drones_msgs.msg import TrajectoryBounds, TrajectoryObstacle
from flexible_drones_msgs.srv import SetTrajectoryBounds, SetTrajectoryObstacles
from flexible_drones_tools.demos.demo_cli import print_failure_summary
from flexible_drones_tools.demos.trajectory_demo import TrajectoryDemo
from flexible_drones_tools.trajectories.utilities.conversion import (
    coefficients_from_pieces,
    duration_to_seconds,
)
from flexible_drones_tools.trajectories.utilities.io import save_trajectory_pieces
from flexible_drones_tools.trajectories.utilities.model import PolyOrderTrajectory


DEFAULT_PLANNED_TRAJECTORY_TOPIC = '/planned_trajectory'


BOUNDS = {
    'x_min': -3.0, 'x_max': 3.0,
    'y_min': -3.0, 'y_max': 3.0,
    'z_min': 0.2, 'z_max': 2.5,
}
OBSTACLES = (
    # Gate Obstacles
    {'type': 'cylinder', 'x': 0.0, 'y': -2.5, 'radius': 0.6, 'height': 3.0},
    {'type': 'cylinder', 'x': 0.0, 'y': 2.5, 'radius': 0.6, 'height': 3.0},
    # Gate Poles
    {'type': 'cylinder', 'x': -2.0, 'y': 0.0, 'radius': 0.04, 'height': 2.0},
    {'type': 'cylinder', 'x': -3.0, 'y': 0.0, 'radius': 0.04, 'height': 2.0},
    {'type': 'cylinder', 'x': 2.0, 'y': 0.0, 'radius': 0.04, 'height': 2.0},
    {'type': 'cylinder', 'x': 3.0, 'y': 0.0, 'radius': 0.04, 'height': 2.0},
)
WAYPOINTS = (
    ('A', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
    ('B', (2.5, 0.0, 1.25), (0.0, 0.7, 0.0)),
    ('C', (-2.5, 0.0, 0.75), (0.0, -0.7, 0.0)),
    ('D', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
)


def _goal_status_label(status: int) -> str:
    labels = {
        GoalStatus.STATUS_UNKNOWN: 'unknown',
        GoalStatus.STATUS_ACCEPTED: 'accepted',
        GoalStatus.STATUS_EXECUTING: 'executing',
        GoalStatus.STATUS_CANCELING: 'canceling',
        GoalStatus.STATUS_SUCCEEDED: 'succeeded',
        GoalStatus.STATUS_CANCELED: 'canceled',
        GoalStatus.STATUS_ABORTED: 'aborted',
    }
    return labels.get(int(status), 'unrecognized')


def _pose(position: tuple[float, float, float]) -> Pose:
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation.w = 1.0
    return pose


def _twist(velocity: tuple[float, float, float]) -> Twist:
    twist = Twist()
    twist.linear.x = float(velocity[0])
    twist.linear.y = float(velocity[1])
    twist.linear.z = float(velocity[2])
    return twist


def _trajectory_from_pieces(pieces) -> PolyOrderTrajectory:
    durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs = coefficients_from_pieces(
        pieces,
        coefficient_order='numpy',
    )
    return PolyOrderTrajectory(durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs)


def _poly_derivative_value(coefficients, derivative_order: int, t: float) -> float:
    value = 0.0
    for power, coefficient in enumerate(coefficients):
        if power < derivative_order:
            continue
        multiplier = 1.0
        for factor in range(derivative_order):
            multiplier *= power - factor
        value += float(coefficient) * multiplier * (float(t) ** (power - derivative_order))
    return value


def _piece_state_at(piece, t: float) -> dict[str, float]:
    return {
        'x': _poly_derivative_value(piece.poly_x, 0, t),
        'y': _poly_derivative_value(piece.poly_y, 0, t),
        'z': _poly_derivative_value(piece.poly_z, 0, t),
        'yaw': _poly_derivative_value(piece.poly_yaw, 0, t),
        'vx': _poly_derivative_value(piece.poly_x, 1, t),
        'vy': _poly_derivative_value(piece.poly_y, 1, t),
        'vz': _poly_derivative_value(piece.poly_z, 1, t),
        'wz': _poly_derivative_value(piece.poly_yaw, 1, t),
    }


def _expected_state(position, velocity) -> dict[str, float]:
    return {
        'x': float(position[0]),
        'y': float(position[1]),
        'z': float(position[2]),
        'yaw': 0.0,
        'vx': float(velocity[0]),
        'vy': float(velocity[1]),
        'vz': float(velocity[2]),
        'wz': 0.0,
    }


def _format_state(state: dict[str, float]) -> str:
    keys = ('x', 'y', 'z', 'yaw', 'vx', 'vy', 'vz', 'wz')
    return ' '.join(f'{key:>3}={state[key]:7.3f}' for key in keys)


def _format_delta(actual: dict[str, float], expected: dict[str, float]) -> str:
    keys = ('x', 'y', 'z', 'yaw', 'vx', 'vy', 'vz', 'wz')
    return ' '.join(f'{key:>3}={actual[key] - expected[key]:+7.3f}' for key in keys)


def _save_stitched_trajectory_to_tmp(pieces) -> Path:
    path = Path('/tmp') / f'planner_demo_stitched_{time.time_ns()}.csv'
    save_trajectory_pieces(path, pieces)
    return path


def _sample_piece_clearances(pieces, obstacles, samples_per_piece: int = 80):
    clearances = [
        {'index': index, 'obstacle': obstacle, 'clearance': float('inf'), 'point': None}
        for index, obstacle in enumerate(obstacles)
    ]
    for piece in pieces:
        duration = duration_to_seconds(piece.duration)
        sample_count = max(2, int(samples_per_piece))
        for sample_index in range(sample_count):
            t = duration * sample_index / (sample_count - 1)
            state = _piece_state_at(piece, t)
            for clearance in clearances:
                obstacle = clearance['obstacle']
                distance = (
                    (state['x'] - float(obstacle['x']))**2
                    + (state['y'] - float(obstacle['y']))**2
                ) ** 0.5
                margin = distance - float(obstacle['radius'])
                if margin < clearance['clearance']:
                    clearance['clearance'] = margin
                    clearance['point'] = (state['x'], state['y'], state['z'])
    return clearances


class PlannerDemo(TrajectoryDemo):
    def __init__(
        self,
        planner_action: str = '/plan_trajectory',
        bounds_service: str = '/set_trajectory_bounds',
        obstacles_service: str = '/set_trajectory_obstacles',
    ):
        super().__init__('planner_demo')
        self._planner_action = planner_action
        self._bounds_service = bounds_service
        self._obstacles_service = obstacles_service
        self._last_planner_feedback_time = {}
        self._bounds_client = self.create_client(
            SetTrajectoryBounds,
            self._bounds_service,
        )
        self._obstacles_client = self.create_client(
            SetTrajectoryObstacles,
            self._obstacles_service,
        )
        self._planner_client = ActionClient(self, GetTrajectory, self._planner_action)

    def set_planner_bounds(self, timeout_sec: float) -> None:
        self.get_logger().info(
            f'sending SetTrajectoryBounds to {self._bounds_service}: '
            f"x=[{BOUNDS['x_min']:.1f}, {BOUNDS['x_max']:.1f}], "
            f"y=[{BOUNDS['y_min']:.1f}, {BOUNDS['y_max']:.1f}], "
            f"z=({BOUNDS['z_min']:.1f}, {BOUNDS['z_max']:.1f}]"
        )
        if not self._bounds_client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError('SetTrajectoryBounds service unavailable')
        req = SetTrajectoryBounds.Request()
        req.bounds = TrajectoryBounds(**BOUNDS)
        result = self._spin_until(
            self._bounds_client.call_async(req),
            timeout_sec,
            'set_trajectory_bounds response',
        )
        if result is None or not result.success:
            message = getattr(result, 'message', 'no response')
            raise RuntimeError(f'SetTrajectoryBounds failed: {message}')
        self.get_logger().info(f'SetTrajectoryBounds succeeded: {result.message}')

    def set_planner_obstacles(self, timeout_sec: float) -> None:
        self.get_logger().info(
            f'sending SetTrajectoryObstacles to {self._obstacles_service}: '
            '(0.0, -2.5, z=0.0), (0.0, 2.5, z=0.0), '
            'radius=0.6 m, height=2.0 m'
        )
        if not self._obstacles_client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError('SetTrajectoryObstacles service unavailable')
        req = SetTrajectoryObstacles.Request()
        req.obstacles = [TrajectoryObstacle(**obstacle) for obstacle in OBSTACLES]
        result = self._spin_until(
            self._obstacles_client.call_async(req),
            timeout_sec,
            'set_trajectory_obstacles response',
        )
        if result is None or not result.success:
            message = getattr(result, 'message', 'no response')
            raise RuntimeError(f'SetTrajectoryObstacles failed: {message}')
        self.get_logger().info(f'SetTrajectoryObstacles succeeded: {result.message}')

    def plan_segment(
        self,
        index: int,
        start: tuple[str, tuple[float, float, float], tuple[float, float, float]],
        target: tuple[str, tuple[float, float, float], tuple[float, float, float]],
        timeout_sec: float,
    ):
        start_name, start_position, start_velocity = start
        target_name, target_position, target_velocity = target
        label = f'{start_name}-{target_name}'
        self.get_logger().info(
            f'sending GetTrajectory plan {label} to {self._planner_action}: '
            f'p{start_name}={start_position}, v{start_name}={start_velocity}, '
            f'p{target_name}={target_position}, v{target_name}={target_velocity}, '
            'a_boundary=(0.0, 0.0, 0.0)'
        )
        if not self._planner_client.wait_for_server(timeout_sec=timeout_sec):
            raise TimeoutError('GetTrajectory action unavailable')

        goal = GetTrajectory.Goal()
        goal.trajectory_id = int(index)
        goal.name = f'planner_demo_{label}'
        goal.timeout_sec = float(timeout_sec)
        goal.start = TrajectoryBoundary()
        goal.target = TrajectoryBoundary()
        goal.start.frame_id = 'map'
        goal.target.frame_id = 'map'
        goal.start.pose = _pose(start_position)
        goal.target.pose = _pose(target_position)
        goal.start.twist = _twist(start_velocity)
        goal.target.twist = _twist(target_velocity)

        handle = self._spin_until(
            self._planner_client.send_goal_async(
                goal,
                feedback_callback=lambda msg: self._planner_feedback(label, msg),
            ),
            timeout_sec,
            f'plan {label} goal',
        )
        if handle is None or not handle.accepted:
            raise RuntimeError(f'GetTrajectory goal rejected for {label}')

        wrapped_result = self._spin_until(
            handle.get_result_async(),
            timeout_sec,
            f'plan {label} result',
        )
        status = int(wrapped_result.status)
        result = wrapped_result.result
        pieces = list(result.trajectory.pieces)
        summary = (
            f'planner result {label}: status={status} '
            f'({_goal_status_label(status)}), success={result.success}, '
            f'pieces={len(pieces)}, message="{result.message}"'
        )
        if status != GoalStatus.STATUS_SUCCEEDED or not result.success:
            self.get_logger().error(summary)
            raise RuntimeError(f'GetTrajectory failed for {label}: {summary}')
        self.get_logger().info(summary)
        copied_pieces = [deepcopy(piece) for piece in pieces]
        self._print_segment_start_diagnostics(
            label,
            copied_pieces,
            start_position,
            start_velocity,
        )
        return copied_pieces

    def _print_segment_start_diagnostics(
        self,
        label: str,
        pieces,
        start_position,
        start_velocity,
    ) -> None:
        expected = _expected_state(start_position, start_velocity)
        print(
            f'Planner return {label}',
            flush=True,
        )
        print(f'  {"requested start":<24} {_format_state(expected)}', flush=True)
        for piece_index, piece in enumerate(pieces):
            actual = _piece_state_at(piece, 0.0)
            if piece_index == 0:
                comparison = expected
                comparison_label = 'requested start'
            else:
                previous = pieces[piece_index - 1]
                comparison = _piece_state_at(
                    previous,
                    duration_to_seconds(previous.duration),
                )
                comparison_label = f'piece {piece_index - 1} end'

            print(
                f'  {f"piece {piece_index:02d} start":<24} {_format_state(actual)}',
                flush=True,
            )
            print(f'  {f"error vs {comparison_label}":<24} {_format_delta(actual, comparison)}',
                  flush=True)

    def _planner_feedback(self, label: str, feedback_msg) -> None:
        now = time.monotonic()
        last_time = self._last_planner_feedback_time.get(label)
        if last_time is not None and now - last_time < 1.0:
            return
        self._last_planner_feedback_time[label] = now

        feedback = feedback_msg.feedback
        status = getattr(feedback, 'status', '')
        status_text = f", status='{status}'" if status else ''
        self.get_logger().info(
            f'GetTrajectory {label}: remaining={feedback.remaining_sec:.1f}s'
            f'{status_text}'
        )

    def plan_stitched_trajectory(self, timeout_sec: float):
        self.set_planner_bounds(timeout_sec)
        self.set_planner_obstacles(timeout_sec)

        stitched = []
        summaries = []
        for index, (start, target) in enumerate(zip(WAYPOINTS[:-1], WAYPOINTS[1:])):
            pieces = self.plan_segment(index, start, target, timeout_sec)
            stitched.extend(pieces)
            summaries.append(f'{start[0]}-{target[0]}:{len(pieces)} pieces')

        total_duration = sum(duration_to_seconds(piece.duration) for piece in stitched)
        self.get_logger().info(
            'stitched planner trajectory: '
            f'{len(stitched)} pieces, duration={total_duration:.2f}s '
            f'({", ".join(summaries)})'
        )
        saved_path = _save_stitched_trajectory_to_tmp(stitched)
        print(f'Saved stitched planner trajectory to: {saved_path}', flush=True)
        self.get_logger().info(f'saved stitched planner trajectory to {saved_path}')
        for clearance in _sample_piece_clearances(stitched, OBSTACLES):
            obstacle = clearance['obstacle']
            point = clearance['point']
            point_text = (
                f'nearest=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})'
                if point is not None else 'nearest=(n/a)'
            )
            print(
                f"Obstacle {clearance['index']} clearance: "
                f"center=({obstacle['x']:.3f}, {obstacle['y']:.3f}), "
                f"radius={obstacle['radius']:.3f}, "
                f"margin={clearance['clearance']:+.3f} m, {point_text}",
                flush=True,
            )
        return stitched, total_duration

    def prompt_continue(self) -> bool:
        print(
            'Stitched A-B-C-D trajectory is publishing for visualization. '
            "Press 'c' or 'C' then Enter to continue; anything else aborts: ",
            end='',
            flush=True,
        )
        while rclpy.ok():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable:
                return sys.stdin.readline().strip() in ('c', 'C')
            rclpy.spin_once(self, timeout_sec=0.0)
        return False


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Planner demo: plan A-B-C-D, visualize, upload, execute, land, disarm.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set the drone namespace with '-r __ns:=/drone1'.
  Planner services/actions default to absolute names, outside the drone namespace.
  Start the planner action server before this demo, for example:
    ros2 run flexible_drones_tools plan_trajectory_action_server --ros-args \\
      -p planner:=casadi_obstacle \\
      -p use_cache:=false
  The default n_segments:=2 is sufficient for this demo; increase it if you
  want the optimizer to use more polynomial pieces per planned leg.
  The planner defaults yaw_follow_tangent:=true; pass -p yaw_follow_tangent:=false
  to keep yaw at zero.
  The demo configures /set_trajectory_bounds and /set_trajectory_obstacles
  before sending three /plan_trajectory goals.

Example:
  ros2 run flexible_drones_tools planner_demo --ros-args -r __ns:=/drone1
""",
    )
    parser.add_argument('--planner-timeout-sec', type=float, default=180.0,
                        help='Timeout for each planner service/action call (default: 180.0)')
    parser.add_argument('--go-to-duration', type=float, default=5.0,
                        help='Seconds for the GoTo(0,0,1) leg (default: 5.0)')
    parser.add_argument('--timescale', type=float, default=1.0,
                        help='Trajectory playback timescale (default: 1.0)')
    parser.add_argument('--timeout-sec', type=float, default=10.0,
                        help='Timeout for arm, takeoff, land, and disarm (default: 10.0)')
    parser.add_argument('--planner-action', default='/plan_trajectory',
                        help="GetTrajectory action name (default: '/plan_trajectory')")
    parser.add_argument('--bounds-service', default='/set_trajectory_bounds',
                        help="SetTrajectoryBounds service name (default: '/set_trajectory_bounds')")
    parser.add_argument('--obstacles-service', default='/set_trajectory_obstacles',
                        help=(
                            'SetTrajectoryObstacles service name '
                            "(default: '/set_trajectory_obstacles')"
                        ))
    parser.add_argument('--visualization-samples', type=int, default=160,
                        help='Number of poses for planned trajectory visualization (default: 160)')
    parser.add_argument('--visualization-period-sec', type=float, default=2.0,
                        help='Seconds between planned trajectory Path publishes (default: 2.0)')
    parser.add_argument('--visualization-topic', default=DEFAULT_PLANNED_TRAJECTORY_TOPIC,
                        help=(
                            'Topic for planned trajectory visualization '
                            f"(default: '{DEFAULT_PLANNED_TRAJECTORY_TOPIC}')"
                        ))
    parsed, ros_args = parser.parse_known_args(args=args)
    if parsed.planner_timeout_sec < 2.0:
        parser.error('--planner-timeout-sec must be at least 2.0')
    if parsed.go_to_duration < 1.0:
        parser.error('--go-to-duration must be at least 1.0')
    if parsed.timescale <= 0.0:
        parser.error('--timescale must be positive')
    if parsed.timeout_sec < 2.0:
        parser.error('--timeout-sec must be at least 2.0')
    if parsed.visualization_samples < 2:
        parser.error('--visualization-samples must be at least 2')
    if parsed.visualization_period_sec <= 0.0:
        parser.error('--visualization-period-sec must be positive')

    rclpy.init(args=ros_args)
    demo = PlannerDemo(
        planner_action=parsed.planner_action,
        bounds_service=parsed.bounds_service,
        obstacles_service=parsed.obstacles_service,
    )

    try:
        failure = None
        cleanup_errors = []
        should_cleanup = False
        flight_started = False
        try:
            pieces, total_duration = demo.plan_stitched_trajectory(parsed.planner_timeout_sec)
            poly_trajectory = _trajectory_from_pieces(pieces)
            demo.publish_planned_trajectory(
                poly_trajectory,
                samples=parsed.visualization_samples,
                topic=parsed.visualization_topic,
                period_sec=parsed.visualization_period_sec,
                auto_ground_offset=False,
            )
            if not demo.prompt_continue():
                demo.get_logger().warning('operator aborted before upload')
                return 1

            print('Waiting for initial pose...', flush=True)
            demo.wait_for_pose()
            demo.upload_trajectory(pieces)
            demo.arm(timeout_sec=parsed.timeout_sec)
            should_cleanup = True
            flight_started = True
            demo.takeoff(timeout_sec=parsed.timeout_sec)
            demo._spin_sleep(1.0)
            demo.go_to(0.0, 0.0, 1.0, 0.0, parsed.go_to_duration, 'planner start')
            demo._spin_sleep(0.5)
            demo.execute_trajectory(
                relative=False,
                timescale=parsed.timescale,
                timeout_sec=total_duration / parsed.timescale + 30.0,
            )

        except (KeyboardInterrupt, ExternalShutdownException, ShutdownException,
                ROSInterruptException):
            demo.get_logger().warning(f'[{demo._ns}] interrupted; aborting planner_demo')
            should_cleanup = flight_started
        except Exception as e:
            demo.get_logger().error(f'planner_demo aborting: {e}')
            failure = e
            should_cleanup = True
        finally:
            if should_cleanup:
                try:
                    demo.land(timeout_sec=parsed.timeout_sec)
                except Exception as e:
                    demo.get_logger().error(f'[{demo._ns}] cleanup land failed: {e}')
                    cleanup_errors.append(f'land failed: {e}')
                finally:
                    try:
                        demo.disarm(timeout_sec=parsed.timeout_sec)
                    except Exception as e:
                        demo.get_logger().error(f'[{demo._ns}] cleanup disarm failed: {e}')
                        cleanup_errors.append(f'disarm failed: {e}')
                        if demo.appears_airborne():
                            cleanup_errors.append(demo.warn_manual_recovery_needed())

            if cleanup_errors and failure is None:
                failure = RuntimeError('; '.join(cleanup_errors))

        if failure is not None:
            print_failure_summary('planner_demo', demo._ns, failure, cleanup_errors)
            return 1
        return 0
    finally:
        demo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

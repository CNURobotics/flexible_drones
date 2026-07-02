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

"""ROS planner test client with selectable obstacle scenarios."""

import argparse
from dataclasses import dataclass
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Twist
from rclpy.action import ActionClient
from rclpy.node import Node

from flexible_drones_msgs.action import GetTrajectory
from flexible_drones_msgs.msg import TrajectoryBoundary
from flexible_drones_msgs.msg import TrajectoryBounds, TrajectoryObstacle
from flexible_drones_msgs.srv import SetTrajectoryBounds, SetTrajectoryObstacles
from flexible_drones_tools.trajectories.utilities.conversion import coefficients_from_pieces
from flexible_drones_tools.trajectories.utilities.model import PolyOrderTrajectory
from flexible_drones_tools.trajectories.visualize_trajectory import trajectory_to_path_msg


DEFAULT_PLANNED_TRAJECTORY_TOPIC = '/planned_trajectory'
DEFAULT_BOUNDS = {
    'x_min': -3.0, 'x_max': 3.0,
    'y_min': -3.0, 'y_max': 3.0,
    'z_min': 0.48, 'z_max': 3.0,
}


@dataclass(frozen=True)
class Boundary:
    """A planning boundary expressed in its own frame."""

    name: str
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]


@dataclass(frozen=True)
class Scenario:
    """One service/action test case for plan_trajectory_action_server."""

    name: str
    description: str
    start: Boundary
    target: Boundary
    obstacles: tuple[dict[str, float | str], ...]
    bounds: dict[str, float]
    expect_success: bool = True


def _cylinder(x, y, radius=0.9, height=3.0):
    return {
        'type': 'cylinder',
        'x': float(x),
        'y': float(y),
        'radius': float(radius),
        'height': float(height),
    }


SCENARIOS = {
    'map_to_gate_a': Scenario(
        name='map_to_gate_a',
        description='Center of course to gate A, with gates B/C/D as service obstacles.',
        start=Boundary('map', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        target=Boundary('gate_A_target_1', (0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        obstacles=(
            _cylinder(0.0, 2.5),
            _cylinder(-2.5, 0.0),
            _cylinder(0.0, -2.5),
        ),
        bounds=dict(DEFAULT_BOUNDS),
    ),
    'gate_a_to_gate_b': Scenario(
        name='gate_a_to_gate_b',
        description='Gate A target 1 to gate B target 2; timed out in one logged run.',
        start=Boundary('gate_A_target_1', (0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        target=Boundary('gate_B_target_2', (0.0, 0.0, 0.0), (-0.5, 0.0, 0.0)),
        obstacles=(
            _cylinder(-2.5, 0.0),
            _cylinder(0.0, -2.5),
        ),
        bounds=dict(DEFAULT_BOUNDS),
    ),
    'gate_c_to_map': Scenario(
        name='gate_c_to_map',
        description='Gate C target 2 to center map point; useful infeasible/slow case.',
        start=Boundary('gate_C_target_2', (0.0, 0.0, 0.0), (-0.5, 0.0, 0.0)),
        target=Boundary('map', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        obstacles=(
            _cylinder(2.5, 0.0),
            _cylinder(0.0, 2.5),
            _cylinder(0.0, -2.5),
        ),
        bounds=dict(DEFAULT_BOUNDS),
    ),
    'gate_b_to_gate_d': Scenario(
        name='gate_b_to_gate_d',
        description='Gate B target 2 to gate D target 1 with A/C gate obstacles.',
        start=Boundary('gate_B_target_2', (0.0, 0.0, 0.0), (-0.5, 0.0, 0.0)),
        target=Boundary('gate_D_target_1', (0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        obstacles=(
            _cylinder(2.5, 0.0),
            _cylinder(-2.5, 0.0),
        ),
        bounds=dict(DEFAULT_BOUNDS),
    ),
    'gate_d_to_gate_c': Scenario(
        name='gate_d_to_gate_c',
        description='Gate D target 1 to gate C target 2 with A/B gate obstacles.',
        start=Boundary('gate_D_target_1', (0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        target=Boundary('gate_C_target_2', (0.0, 0.0, 0.0), (-0.5, 0.0, 0.0)),
        obstacles=(
            _cylinder(2.5, 0.0),
            _cylinder(0.0, 2.5),
        ),
        bounds=dict(DEFAULT_BOUNDS),
    ),
}

SCENARIO_RUN_ORDER = (
    'map_to_gate_a',
    'gate_a_to_gate_b',
    'gate_b_to_gate_d',
    'gate_d_to_gate_c',
    'gate_c_to_map',
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


def _pose(position, velocity=(0.0, 0.0, 0.0)):
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    if float(velocity[0]) < 0.0:
        pose.orientation.z = 1.0
        pose.orientation.w = 0.0
    else:
        pose.orientation.w = 1.0
    return pose


def _twist(velocity):
    twist = Twist()
    twist.linear.x = float(velocity[0])
    twist.linear.y = float(velocity[1])
    twist.linear.z = float(velocity[2])
    return twist


class PlannerTest(Node):
    """ROS client for setting planner state and sending GetTrajectory goals."""

    def __init__(
        self,
        planner_action='/plan_trajectory',
        bounds_service='/set_trajectory_bounds',
        obstacles_service='/set_trajectory_obstacles',
        visualization_topic=DEFAULT_PLANNED_TRAJECTORY_TOPIC,
        visualization_samples=160,
        visualization_period_sec=2.0,
        publish_duration_sec=2.0,
    ):
        super().__init__('planner_test')
        self._planner_action = planner_action
        self._bounds_service = bounds_service
        self._obstacles_service = obstacles_service
        self._visualization_topic = visualization_topic
        self._visualization_samples = int(visualization_samples)
        self._visualization_period_sec = float(visualization_period_sec)
        self._publish_duration_sec = float(publish_duration_sec)
        self._last_feedback_time = {}
        self._planned_path_pub = None
        self._planned_path_timer = None
        self._planned_path_msg = None
        self._bounds_client = self.create_client(SetTrajectoryBounds, bounds_service)
        self._obstacles_client = self.create_client(SetTrajectoryObstacles, obstacles_service)
        self._planner_client = ActionClient(self, GetTrajectory, planner_action)

    def _spin_until(self, future, timeout_sec, label):
        deadline = time.monotonic() + float(timeout_sec)
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f'Timed out waiting for {label}')
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done():
            raise TimeoutError(f'Timed out waiting for {label}')
        return future.result()

    def set_bounds(self, bounds, timeout_sec):
        self.get_logger().info(
            f"SetTrajectoryBounds: x=[{bounds['x_min']:.2f}, {bounds['x_max']:.2f}], "
            f"y=[{bounds['y_min']:.2f}, {bounds['y_max']:.2f}], "
            f"z=[{bounds['z_min']:.2f}, {bounds['z_max']:.2f}]"
        )
        self.get_logger().info(
            f"Waiting for SetTrajectoryBounds service '{self._bounds_service}'..."
        )
        if not self._bounds_client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError(f'{self._bounds_service} unavailable')
        self.get_logger().info(
            f"Sending SetTrajectoryBounds request to '{self._bounds_service}'."
        )
        request = SetTrajectoryBounds.Request()
        request.bounds = TrajectoryBounds(**bounds)
        result = self._spin_until(
            self._bounds_client.call_async(request),
            timeout_sec,
            'SetTrajectoryBounds response',
        )
        if result is None or not result.success:
            message = getattr(result, 'message', 'no response')
            raise RuntimeError(f'SetTrajectoryBounds failed: {message}')
        self.get_logger().info(f'SetTrajectoryBounds succeeded: {result.message}')

    def set_obstacles(self, obstacles, timeout_sec):
        summary = ', '.join(
            f"({obstacle['x']:.2f}, {obstacle['y']:.2f}, r={obstacle['radius']:.2f})"
            for obstacle in obstacles
        )
        self.get_logger().info(f'SetTrajectoryObstacles: {summary or "empty"}')
        self.get_logger().info(
            f"Waiting for SetTrajectoryObstacles service '{self._obstacles_service}'..."
        )
        if not self._obstacles_client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError(f'{self._obstacles_service} unavailable')
        self.get_logger().info(
            f"Sending SetTrajectoryObstacles request to '{self._obstacles_service}'."
        )
        request = SetTrajectoryObstacles.Request()
        request.obstacles = [TrajectoryObstacle(**obstacle) for obstacle in obstacles]
        result = self._spin_until(
            self._obstacles_client.call_async(request),
            timeout_sec,
            'SetTrajectoryObstacles response',
        )
        if result is None or not result.success:
            message = getattr(result, 'message', 'no response')
            raise RuntimeError(f'SetTrajectoryObstacles failed: {message}')
        self.get_logger().info(f'SetTrajectoryObstacles succeeded: {result.message}')

    def run_scenario(self, scenario, timeout_sec, trajectory_id, expect_success=None):
        expect_success = scenario.expect_success if expect_success is None else bool(expect_success)
        self.set_bounds(scenario.bounds, timeout_sec)
        self.set_obstacles(scenario.obstacles, timeout_sec)

        self.get_logger().info(f"Waiting for GetTrajectory action '{self._planner_action}'...")
        if not self._planner_client.wait_for_server(timeout_sec=timeout_sec):
            raise TimeoutError(f'{self._planner_action} unavailable')
        goal = self._goal_from_scenario(scenario, trajectory_id, timeout_sec)
        label = scenario.name
        self.get_logger().info(
            f'GetTrajectory {label}: '
            f'start={scenario.start.position}, v={scenario.start.velocity}, '
            f'target={scenario.target.position}, v={scenario.target.velocity}'
        )
        handle = self._spin_until(
            self._planner_client.send_goal_async(
                goal,
                feedback_callback=lambda msg: self._feedback(label, msg),
            ),
            timeout_sec,
            f'{label} goal response',
        )
        if handle is None or not handle.accepted:
            raise RuntimeError(f'GetTrajectory goal rejected for {label}')
        wrapped_result = self._spin_until(
            handle.get_result_async(),
            timeout_sec + 5.0,
            f'{label} result',
        )
        status = int(wrapped_result.status)
        result = wrapped_result.result
        pieces = list(result.trajectory.pieces)
        success = status == GoalStatus.STATUS_SUCCEEDED and bool(result.success)
        self.get_logger().info(
            f'GetTrajectory {label}: status={status} ({_goal_status_label(status)}), '
            f'success={result.success}, pieces={len(pieces)}, message="{result.message}"'
        )
        if success != expect_success:
            expectation = 'success' if expect_success else 'failure'
            raise RuntimeError(f'{label}: expected {expectation}, got message="{result.message}"')
        if success and self._visualization_topic:
            self.publish_planned_trajectory(label, pieces)
        return success

    def _goal_from_scenario(self, scenario, trajectory_id, timeout_sec):
        goal = GetTrajectory.Goal()
        goal.trajectory_id = int(trajectory_id)
        goal.name = f'planner_test_{scenario.name}'
        goal.timeout_sec = float(timeout_sec)
        goal.start = TrajectoryBoundary()
        goal.target = TrajectoryBoundary()
        goal.start.frame_id = scenario.start.name
        goal.target.frame_id = scenario.target.name
        goal.start.pose = _pose(scenario.start.position, scenario.start.velocity)
        goal.target.pose = _pose(scenario.target.position, scenario.target.velocity)
        goal.start.twist = _twist(scenario.start.velocity)
        goal.target.twist = _twist(scenario.target.velocity)
        return goal

    def _feedback(self, label, feedback_msg):
        now = time.monotonic()
        last = self._last_feedback_time.get(label)
        if last is not None and now - last < 1.0:
            return
        self._last_feedback_time[label] = now
        feedback = feedback_msg.feedback
        status = getattr(feedback, 'status', '')
        status_text = f", status='{status}'" if status else ''
        self.get_logger().info(
            f'GetTrajectory {label}: remaining={feedback.remaining_sec:.1f}s'
            f'{status_text}'
        )

    def publish_planned_trajectory(self, label, pieces):
        """Publish a returned GetTrajectory result as an RViz Path."""
        durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs = coefficients_from_pieces(
            pieces,
            coefficient_order='numpy',
        )
        trajectory = PolyOrderTrajectory(durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs)
        self._planned_path_msg = trajectory_to_path_msg(
            trajectory,
            frame_id='map',
            samples=self._visualization_samples,
            stamp=self.get_clock().now().to_msg(),
        )
        if self._planned_path_pub is None:
            self._planned_path_pub = self.create_publisher(
                type(self._planned_path_msg),
                self._visualization_topic,
                1,
            )
        if self._planned_path_timer is not None:
            self._planned_path_timer.cancel()
            self.destroy_timer(self._planned_path_timer)
            self._planned_path_timer = None

        def publish_path():
            self._planned_path_msg.header.stamp = self.get_clock().now().to_msg()
            self._planned_path_pub.publish(self._planned_path_msg)

        publish_path()
        self._planned_path_timer = self.create_timer(
            max(0.1, self._visualization_period_sec),
            publish_path,
        )
        self.get_logger().info(
            f'Publishing {label} planned trajectory on {self._visualization_topic} '
            f'with {len(self._planned_path_msg.poses)} poses'
        )
        self._spin_sleep(max(0.0, self._publish_duration_sec))

    def _spin_sleep(self, duration_sec):
        deadline = time.monotonic() + float(duration_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)


def _scenario_names(case):
    if case == 'all':
        return SCENARIO_RUN_ORDER
    return (case,)


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Send planner service/action calls for named obstacle scenarios.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Required ROS nodes:
  1. Start RViz with the per-drone displays and gates config:
       ros2 launch flexible_drones_description bringup_rviz.launch.py rviz_config:=gates.rviz

  2. If planning with gate obstacles, publish the gate descriptions:
       ros2 launch flexible_drones_description publish_drone_course.launch.py

  3. Start the planner action/service server with the gate-obstacle launch file:
       ros2 launch flexible_drones_tools plan_trajectory_with_gates.launch.py

  4. Run this test client in another terminal:
       ros2 run flexible_drones_tools planner_test --case map_to_gate_a

  5. The gates RViz config includes Planned Path, Optimizing Path,
     Flight Path, and Planner Obstacles displays.

Useful variants:
  List built-in scenarios:
       ros2 run flexible_drones_tools planner_test --case list

  Run every built-in scenario:
       ros2 run flexible_drones_tools planner_test --case all --timeout-sec 300

  Run against remapped planner interfaces:
       ros2 run flexible_drones_tools planner_test \\
         --planner-action /plan_trajectory \\
         --bounds-service /set_trajectory_bounds \\
         --obstacles-service /set_trajectory_obstacles

  Disable RViz path publishing:
       ros2 run flexible_drones_tools planner_test --case map_to_gate_a --no-publish
""",
    )
    parser.add_argument(
        '--case',
        choices=tuple(SCENARIOS) + ('all', 'list'),
        default='map_to_gate_a',
        help='Scenario to run, "all" to run all scenarios, or "list" to print names.',
    )
    parser.add_argument('--timeout-sec', type=float, default=180.0,
                        help='Per-scenario planner timeout in seconds (default: 180.0).')
    parser.add_argument('--planner-action', default='/plan_trajectory',
                        help="GetTrajectory action name (default: '/plan_trajectory').")
    parser.add_argument('--bounds-service', default='/set_trajectory_bounds',
                        help="SetTrajectoryBounds service name (default: '/set_trajectory_bounds').")
    parser.add_argument('--obstacles-service', default='/set_trajectory_obstacles',
                        help="SetTrajectoryObstacles service name (default: '/set_trajectory_obstacles').")
    parser.add_argument('--expect-failure', action='store_true',
                        help='Treat action failure as the expected result for selected scenario(s).')
    parser.add_argument('--visualization-topic', default=DEFAULT_PLANNED_TRAJECTORY_TOPIC,
                        help=(
                            'Topic for returned trajectory Path publishing '
                            f"(default: '{DEFAULT_PLANNED_TRAJECTORY_TOPIC}')."
                        ))
    parser.add_argument('--visualization-samples', type=int, default=160,
                        help='Number of poses in the published Path (default: 160).')
    parser.add_argument('--visualization-period-sec', type=float, default=2.0,
                        help='Seconds between repeated Path publishes (default: 2.0).')
    parser.add_argument('--publish-duration-sec', type=float, default=10.0,
                        help='Seconds to keep publishing each successful Path (default: 10.0).')
    parser.add_argument('--no-publish', action='store_true',
                        help='Do not publish returned trajectories as nav_msgs/Path.')
    parsed, ros_args = parser.parse_known_args(args=args)

    if parsed.case == 'list':
        for name in SCENARIO_RUN_ORDER:
            scenario = SCENARIOS[name]
            print(f'{name}: {scenario.description}', flush=True)
        return 0
    if parsed.timeout_sec < 2.0:
        parser.error('--timeout-sec must be at least 2.0')
    if parsed.visualization_samples < 2:
        parser.error('--visualization-samples must be at least 2')
    if parsed.visualization_period_sec <= 0.0:
        parser.error('--visualization-period-sec must be positive')
    if parsed.publish_duration_sec < 0.0:
        parser.error('--publish-duration-sec must be non-negative')

    rclpy.init(args=ros_args)
    node = PlannerTest(
        planner_action=parsed.planner_action,
        bounds_service=parsed.bounds_service,
        obstacles_service=parsed.obstacles_service,
        visualization_topic='' if parsed.no_publish else parsed.visualization_topic,
        visualization_samples=parsed.visualization_samples,
        visualization_period_sec=parsed.visualization_period_sec,
        publish_duration_sec=parsed.publish_duration_sec,
    )
    try:
        for index, name in enumerate(_scenario_names(parsed.case), start=1):
            scenario = SCENARIOS[name]
            node.get_logger().info(f'Running planner_test case {name}: {scenario.description}')
            node.run_scenario(
                scenario,
                parsed.timeout_sec,
                trajectory_id=index,
                expect_success=not parsed.expect_failure,
            )
        return 0
    except Exception as exc:  # noqa: B902
        node.get_logger().error(f'planner_test failed: {type(exc).__name__}: {exc}')
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())

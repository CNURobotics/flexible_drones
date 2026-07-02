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

"""CLI client for the ``GetSampledTrajectory`` action."""

import ast
import math

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from flexible_drones_msgs.action import GetSampledTrajectory
from flexible_drones_msgs.msg import TrajectoryBoundary
from flexible_drones_tools.clients.client_utils import (
    declare_wait_timeout_parameter,
    print_usage_if_requested,
    wait_for_action_server,
)
from flexible_drones_tools.ros_shutdown import spin_node

USAGE = """\
Usage:
  ros2 run flexible_drones_tools get_sampled_trajectory_client \
    --ros-args -p trajectory_id:=0 -p start_frame_id:=map -p target_frame_id:=map

Common examples:
  ros2 run flexible_drones_tools get_sampled_trajectory_client --ros-args -p trajectory_id:=1
  ros2 run flexible_drones_tools get_sampled_trajectory_client --ros-args -p action_name:=load_sampled_trajectory -p name:=figure8
  ros2 run flexible_drones_tools get_sampled_trajectory_client --ros-args -r __node:=sampled_trajectory_client
  ros2 run flexible_drones_tools get_sampled_trajectory_client --ros-args \
    -p start_pose:="'{x: 0.0, y: 0.0, z: 1.0, yaw: 0.0}'" \
    -p target_pose:="'{x: 1.0, y: 0.0, z: 1.0, yaw: 1.57}'"
  ros2 run flexible_drones_tools get_sampled_trajectory_client --ros-args -p plot_trajectory:=true

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Wrap YAML string values in quotes inside the parameter value, e.g. -p start_pose:="'{x: 1.0}'".
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Parameters declared by this node:
  action_name (string, default: plan_sampled_trajectory)
    Action name for the GetSampledTrajectory server.
  trajectory_id (integer, default: 0)
    ID assigned to the planned trajectory.
  name (string, default: empty)
    Name assigned to the planned trajectory. Defaults to "Trajectory <trajectory_id>" when empty.
  start_frame_id, target_frame_id (string, default: map)
    Frames used by the planner for start and target poses.
  start_pose (YAML string, default: {x: 0.0, y: 0.0, z: 0.75, yaw: 0.0})
    Optional mapping with x, y, z, yaw values. Unspecified values default to 0.0.
  target_pose (YAML string, default: {x: 1.0, y: 1.0, z: 1.25, yaw: 1.5708})
    Optional mapping with x, y, z, yaw values. Unspecified values default to 0.0.
  duration_sec (double, default: 2.0)
    Duration used to derive start and target velocity estimates.
  period_sec (double, default: 0.05)
    Sample period sent as GetSampledTrajectory.sample_dt.
  plot_trajectory (bool, default: false)
    Plot the returned sampled trajectory.
  planning_timeout_sec (double, default: 0.0)
    Requested planner timeout. Values <= 0 use the server maximum.
  wait_timeout_sec (double, default: 5.0)
    Maximum time to wait for the action server.

Examples with parameter overrides:
  ros2 run flexible_drones_tools get_sampled_trajectory_client --ros-args -p trajectory_id:=2
  ros2 run flexible_drones_tools get_sampled_trajectory_client --ros-args \
    -p start_frame_id:=map -p target_frame_id:=map -p duration_sec:=3.0 -p period_sec:=0.05
  ros2 run flexible_drones_tools get_sampled_trajectory_client --ros-args \
    -p start_frame_id:=gate_C_target_2 -p target_frame_id:=gate_A_target_1 \
    -p start_pose:="'{}'" -p target_pose:="'{}'"
"""


POSE_KEYS = ('x', 'y', 'z', 'yaw')


def _parse_pose_mapping(value):
    text = value.strip()
    if not text:
        return {}

    try:
        import yaml
    except ImportError:
        parsed = _parse_simple_flow_mapping(text)
    else:
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(str(e)) from e

    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError('pose value must be a mapping with x, y, z, yaw keys')

    unknown_keys = set(parsed) - set(POSE_KEYS)
    if unknown_keys:
        keys = ', '.join(sorted(str(key) for key in unknown_keys))
        raise ValueError(f'unsupported pose key(s): {keys}')

    return {key: float(parsed[key]) for key in POSE_KEYS if key in parsed}


def _parse_simple_flow_mapping(text):
    stripped = text.strip()
    if not (stripped.startswith('{') and stripped.endswith('}')):
        raise ValueError('pose value must use YAML flow mapping syntax, e.g. {x: 1.0}')

    parsed = {}
    body = stripped[1:-1].strip()
    if not body:
        return parsed

    for item in body.split(','):
        if ':' not in item:
            raise ValueError(f'invalid pose item: {item.strip()}')
        key, raw_value = item.split(':', 1)
        key = key.strip().strip("\"'")
        parsed[key] = ast.literal_eval(raw_value.strip())

    return parsed


def _pose_from_values(values):
    pose = Pose()
    x = values.get('x', 0.0)
    y = values.get('y', 0.0)
    z = values.get('z', 0.0)
    yaw = values.get('yaw', 0.0)

    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.z = math.sin(yaw / 2.0)
    pose.orientation.w = math.cos(yaw / 2.0)
    return pose


class GetSampledTrajectoryClient(Node):
    def __init__(self):
        super().__init__('get_sampled_trajectory_client')
        self.declare_parameter('action_name', 'plan_sampled_trajectory')
        self.declare_parameter('trajectory_id', 0)
        self.declare_parameter('name', '')
        self.declare_parameter('start_frame_id', 'map')
        self.declare_parameter('target_frame_id', 'map')
        self.declare_parameter('start_pose', '{x: 0.0, y: 0.0, z: 0.75, yaw: 0.0}')
        self.declare_parameter('target_pose', '{x: 1.0, y: 1.0, z: 1.25, yaw: 1.5708}')
        self.declare_parameter('duration_sec', 2.0)
        self.declare_parameter('period_sec', 0.05)
        self.declare_parameter('plot_trajectory', False)
        self.declare_parameter('planning_timeout_sec', 0.0)
        declare_wait_timeout_parameter(self)
        action_name = self.get_parameter('action_name').get_parameter_value().string_value
        self._action_client = ActionClient(self, GetSampledTrajectory, action_name)

    def send_goal(self):
        duration_sec = self.get_parameter('duration_sec').get_parameter_value().double_value
        period_sec = self.get_parameter('period_sec').get_parameter_value().double_value
        if duration_sec <= 0.0:
            self.get_logger().error('duration_sec must be greater than 0.0.')
            rclpy.shutdown()
            return False
        if period_sec <= 0.0:
            self.get_logger().error('period_sec must be greater than 0.0.')
            rclpy.shutdown()
            return False

        try:
            start_values = _parse_pose_mapping(self.get_parameter('start_pose').get_parameter_value().string_value)
            target_values = _parse_pose_mapping(self.get_parameter('target_pose').get_parameter_value().string_value)
        except ValueError as e:
            self.get_logger().error(f'Invalid pose YAML: {e}')
            rclpy.shutdown()
            return False

        goal_msg = GetSampledTrajectory.Goal()
        goal_msg.trajectory_id = self.get_parameter('trajectory_id').get_parameter_value().integer_value
        requested_name = self.get_parameter('name').get_parameter_value().string_value
        goal_msg.name = requested_name if requested_name else f'Trajectory {goal_msg.trajectory_id}'
        goal_msg.sample_dt = period_sec
        goal_msg.timeout_sec = self.get_parameter('planning_timeout_sec').get_parameter_value().double_value
        goal_msg.start = TrajectoryBoundary()
        goal_msg.target = TrajectoryBoundary()
        goal_msg.start.frame_id = self.get_parameter('start_frame_id').get_parameter_value().string_value
        goal_msg.target.frame_id = self.get_parameter('target_frame_id').get_parameter_value().string_value
        goal_msg.start.pose = _pose_from_values(start_values)
        goal_msg.target.pose = _pose_from_values(target_values)

        vx = (goal_msg.target.pose.position.x - goal_msg.start.pose.position.x) / duration_sec
        vy = (goal_msg.target.pose.position.y - goal_msg.start.pose.position.y) / duration_sec
        vz = (goal_msg.target.pose.position.z - goal_msg.start.pose.position.z) / duration_sec
        wz = (target_values.get('yaw', 0.0) - start_values.get('yaw', 0.0)) / duration_sec

        goal_msg.start.twist = Twist()
        goal_msg.start.twist.linear.x = vx
        goal_msg.start.twist.linear.y = vy
        goal_msg.start.twist.linear.z = vz
        goal_msg.start.twist.angular.z = wz

        goal_msg.target.twist = Twist()
        goal_msg.target.twist.linear.x = vx
        goal_msg.target.twist.linear.y = vy
        goal_msg.target.twist.linear.z = vz
        goal_msg.target.twist.angular.z = wz

        self.get_logger().info(
            'Sending GetSampledTrajectory goal... '
            f"start_frame_id='{goal_msg.start.frame_id}', "
            f"target_frame_id='{goal_msg.target.frame_id}', "
            f"action_name='{self.get_parameter('action_name').get_parameter_value().string_value}', "
            f'duration_sec={duration_sec}, period_sec={period_sec}, '
            f"name='{goal_msg.name}', timeout_sec={goal_msg.timeout_sec:.1f}"
        )

        if not wait_for_action_server(self, self._action_client, 'GetSampledTrajectory action server'):
            return False

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

        return True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            rclpy.shutdown()
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        status = getattr(feedback, 'status', '')
        status_text = f" status='{status}'" if status else ''
        self.get_logger().info(
            'GetSampledTrajectory: '
            f'remaining={feedback.remaining_sec:.1f}s '
            f'{status_text}'
        )

    def get_result_callback(self, future):
        result = future.result().result
        point_count = len(result.trajectory.points) if result.success else 0
        if result.success:
            self.get_logger().info(f"Result: success={result.success}, message='{result.message}', " f'points={point_count}')
        else:
            self.get_logger().info(f"Result: success={result.success}, message='{result.message}'")

        if self.get_parameter('plot_trajectory').get_parameter_value().bool_value:
            self.plot_trajectory(result)
        else:
            self.get_logger().info('Skipping trajectory plot; set plot_trajectory:=true to enable plotting.')

        rclpy.shutdown()

    def plot_trajectory(self, result):
        import matplotlib.pyplot as plt

        if not result.success or not result.trajectory.points:
            self.get_logger().warning('No trajectory points to plot.')
            return

        traj_x = [p.x for p in result.trajectory.points]
        traj_y = [p.y for p in result.trajectory.points]
        traj_z = [p.z for p in result.trajectory.points]
        time = [p.time_from_start for p in result.trajectory.points]

        traj_x0 = [traj_x[0]]
        traj_y0 = [traj_y[0]]
        traj_z0 = [traj_z[0]]
        times = [time[0]]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj_x, traj_y, traj_z, label='Planned trajectory')
        ax.plot(traj_x0, traj_y0, traj_z0, 'g.', linewidth=2, label='Segment Start')
        ax.plot(traj_x[-1], traj_y[-1], traj_z[-1], 'r.', linewidth=2, label='End')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('Trajectory from Action Server')

        plt.figure()
        plt.plot(time, traj_x, 'r', label='x')
        plt.plot(time, traj_y, 'g', label='y')
        plt.plot(time, traj_z, 'b', label='z')
        plt.plot(times, traj_x0, 'ro', linewidth=2)
        plt.plot(times, traj_y0, 'go', linewidth=2)
        plt.plot(times, traj_z0, 'bo', linewidth=2)
        plt.legend()
        plt.title('Trajectory from Action Server')
        plt.xlabel('Time (s)')
        plt.show()


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)

    action_client = GetSampledTrajectoryClient()

    if action_client.send_goal():
        print('Ready to spin ...', flush=True)
        spin_node(action_client)


if __name__ == '__main__':
    main()

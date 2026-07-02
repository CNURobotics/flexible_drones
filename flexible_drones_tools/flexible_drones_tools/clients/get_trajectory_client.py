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

"""CLI client for the ``GetTrajectory`` action."""

import rclpy
from geometry_msgs.msg import Pose, Twist
from rclpy.action import ActionClient
from rclpy.node import Node

from flexible_drones_msgs.action import GetTrajectory
from flexible_drones_msgs.msg import TrajectoryBoundary
from flexible_drones_tools.clients.client_utils import (
    declare_wait_timeout_parameter,
    print_usage_if_requested,
    wait_for_action_server,
)
from flexible_drones_tools.ros_shutdown import spin_node

USAGE = """\
Usage:
  ros2 run flexible_drones_tools get_trajectory_client --ros-args -p start_frame_id:=map -p target_frame_id:=map

Common examples:
  ros2 run flexible_drones_tools get_trajectory_client --ros-args -p start_frame_id:=map -p target_frame_id:=map
  ros2 run flexible_drones_tools get_trajectory_client --ros-args \
    -p start_frame_id:=gate_C_target_2 -p target_frame_id:=gate_A_target_1
  ros2 run flexible_drones_tools get_trajectory_client --ros-args -p action_name:=load_trajectory -p name:=figure8
  ros2 run flexible_drones_tools get_trajectory_client --ros-args -p trajectory_id:=1 -p name:="'Trajectory 1'"
  ros2 run flexible_drones_tools get_trajectory_client --ros-args -p start_vx:=-0.5 -p target_vx:=0.5

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Parameters declared by this node:
  action_name (string, default: plan_trajectory)
    Action name for the GetTrajectory server.
  trajectory_id (integer, default: 0)
    ID assigned to the returned trajectory.
  name (string, default: empty)
    Name sent with the request. Defaults to "Trajectory <trajectory_id>" when empty.
  start_frame_id, target_frame_id (string, default: map)
    Frames used by the planner for start and target.
  start_x, start_y, start_z, target_x, target_y, target_z (double, default: 0.0)
    Start and target pose positions.
  start_qx, start_qy, start_qz, start_qw, target_qx, target_qy, target_qz, target_qw
    Start and target pose quaternions. Defaults are identity orientation.
  start_vx, start_vy, start_vz, target_vx, target_vy, target_vz (double, default: 0.0)
    Start and target linear velocities.
  start_wx, start_wy, start_wz, target_wx, target_wy, target_wz (double, default: 0.0)
    Start and target angular velocities.
  planning_timeout_sec (double, default: 0.0)
    Requested planner timeout. Values <= 0 use the server maximum.
  wait_timeout_sec (double, default: 5.0)
    Maximum time to wait for the action server.
"""


def _get_double(node, name):
    return node.get_parameter(name).get_parameter_value().double_value


class GetTrajectoryClient(Node):

    def __init__(self):
        super().__init__('get_trajectory_client')

        self.declare_parameter('action_name', 'plan_trajectory')
        self.declare_parameter('trajectory_id', 0)
        self.declare_parameter('name', '')
        self.declare_parameter('start_frame_id', 'map')
        self.declare_parameter('target_frame_id', 'map')
        self.declare_parameter('planning_timeout_sec', 0.0)
        declare_wait_timeout_parameter(self)

        for prefix in ('start', 'target'):
            self.declare_parameter(f'{prefix}_x', 0.0)
            self.declare_parameter(f'{prefix}_y', 0.0)
            self.declare_parameter(f'{prefix}_z', 0.0)
            self.declare_parameter(f'{prefix}_qx', 0.0)
            self.declare_parameter(f'{prefix}_qy', 0.0)
            self.declare_parameter(f'{prefix}_qz', 0.0)
            self.declare_parameter(f'{prefix}_qw', 1.0)
            self.declare_parameter(f'{prefix}_vx', 0.0)
            self.declare_parameter(f'{prefix}_vy', 0.0)
            self.declare_parameter(f'{prefix}_vz', 0.0)
            self.declare_parameter(f'{prefix}_wx', 0.0)
            self.declare_parameter(f'{prefix}_wy', 0.0)
            self.declare_parameter(f'{prefix}_wz', 0.0)

        action_name = self.get_parameter('action_name').get_parameter_value().string_value
        self._action_client = ActionClient(self, GetTrajectory, action_name)

    def _pose_from_parameters(self, prefix):
        pose = Pose()
        pose.position.x = _get_double(self, f'{prefix}_x')
        pose.position.y = _get_double(self, f'{prefix}_y')
        pose.position.z = _get_double(self, f'{prefix}_z')
        pose.orientation.x = _get_double(self, f'{prefix}_qx')
        pose.orientation.y = _get_double(self, f'{prefix}_qy')
        pose.orientation.z = _get_double(self, f'{prefix}_qz')
        pose.orientation.w = _get_double(self, f'{prefix}_qw')
        return pose

    def _twist_from_parameters(self, prefix):
        twist = Twist()
        twist.linear.x = _get_double(self, f'{prefix}_vx')
        twist.linear.y = _get_double(self, f'{prefix}_vy')
        twist.linear.z = _get_double(self, f'{prefix}_vz')
        twist.angular.x = _get_double(self, f'{prefix}_wx')
        twist.angular.y = _get_double(self, f'{prefix}_wy')
        twist.angular.z = _get_double(self, f'{prefix}_wz')
        return twist

    def _boundary_from_parameters(self, prefix, frame_param):
        boundary = TrajectoryBoundary()
        boundary.frame_id = self.get_parameter(frame_param).get_parameter_value().string_value
        boundary.pose = self._pose_from_parameters(prefix)
        boundary.twist = self._twist_from_parameters(prefix)
        return boundary

    def send_goal(self):
        goal_msg = GetTrajectory.Goal()
        goal_msg.trajectory_id = self.get_parameter('trajectory_id').get_parameter_value().integer_value
        requested_name = self.get_parameter('name').get_parameter_value().string_value
        goal_msg.name = requested_name if requested_name else f'Trajectory {goal_msg.trajectory_id}'
        goal_msg.timeout_sec = self.get_parameter('planning_timeout_sec').get_parameter_value().double_value
        goal_msg.start = self._boundary_from_parameters('start', 'start_frame_id')
        goal_msg.target = self._boundary_from_parameters('target', 'target_frame_id')

        if not wait_for_action_server(self, self._action_client, 'GetTrajectory action server'):
            return False

        self.get_logger().info(
            'Sending GetTrajectory goal... '
            f"start_frame_id='{goal_msg.start.frame_id}', "
            f"target_frame_id='{goal_msg.target.frame_id}', "
            f"trajectory_id={goal_msg.trajectory_id}, name='{goal_msg.name}', "
            f'timeout_sec={goal_msg.timeout_sec:.1f}'
        )
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('GetTrajectory goal rejected.')
            rclpy.shutdown()
            return

        self.get_logger().info('GetTrajectory goal accepted.')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        status = getattr(feedback, 'status', '')
        status_text = f" status='{status}'" if status else ''
        self.get_logger().info(
            'GetTrajectory: '
            f'remaining={feedback.remaining_sec:.1f}s '
            f'{status_text}'
        )

    def get_result_callback(self, future):
        result = future.result().result
        trajectory = result.trajectory
        if not result.success:
            self.get_logger().warning(f"GetTrajectory failed: message='{result.message}'")
            rclpy.shutdown()
            return

        self.get_logger().info(
            'GetTrajectory finished successfully: '
            f"message='{result.message}', "
            f'trajectory_id={trajectory.trajectory_id}, '
            f'pieces={len(trajectory.pieces)}'
        )

        rclpy.shutdown()


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)
    action_client = GetTrajectoryClient()
    if action_client.send_goal():
        spin_node(action_client)


if __name__ == '__main__':
    main()

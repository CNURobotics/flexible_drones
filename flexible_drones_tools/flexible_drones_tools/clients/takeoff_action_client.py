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

"""CLI client for the ``takeoff`` action."""

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node

from flexible_drones_msgs.action import (
    Takeoff
)
from flexible_drones_tools.clients.client_utils import (
    declare_wait_timeout_parameter,
    print_usage_if_requested,
    wait_for_action_server,
)
from flexible_drones_tools.ros_shutdown import spin_node

USAGE = """\
Usage:
  ros2 run flexible_drones_tools takeoff --ros-args -r __ns:=/drone1 -p height:=1.0 -p duration_sec:=2.0

Common examples:
  ros2 run flexible_drones_tools takeoff --ros-args -p height:=1.0
  ros2 run flexible_drones_tools takeoff --ros-args -r __ns:=/drone1 -p height:=1.2 -p duration_sec:=3.0
  ros2 run flexible_drones_tools takeoff --ros-args -r __ns:=/drone1 -r __node:=takeoff_client -p group_mask:=0

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Parameters declared by this node:
  group_mask (integer, default: 0)
    Drone group mask sent in the Takeoff goal.
  height (double, default: 1.0)
    Target takeoff height in metres.
  duration_sec (double, default: 2.0)
    Takeoff duration in seconds.
  wait_timeout_sec (double, default: 5.0)
    Maximum time to wait for the action server.

Examples with parameter overrides:
  ros2 run flexible_drones_tools takeoff --ros-args -p height:=1.5 -p duration_sec:=4.0
  ros2 run flexible_drones_tools takeoff --ros-args -r __ns:=/drone2 -p group_mask:=1 -p height:=1.0
"""


class TakeoffActionClient(Node):

    def __init__(self):
        super().__init__('takeoff_action_client')

        self.declare_parameter('group_mask', 0)
        self.declare_parameter('height', 1.0)
        self.declare_parameter('duration_sec', 2.0)
        declare_wait_timeout_parameter(self)

        self._action_client = ActionClient(
            self,
            Takeoff,
            'takeoff')  # Figure out how to add the name of each Drone here? (Anwser: Remapping the namespace via CLA!)

    def send_goal(self):
        group_mask = self.get_parameter('group_mask').get_parameter_value().integer_value
        height = self.get_parameter('height').get_parameter_value().double_value
        duration_sec = self.get_parameter('duration_sec').get_parameter_value().double_value

        goal_msg = Takeoff.Goal()  # Assigns the variables recieved from the Action Server to Takeoff.action's 'goal' field
        self.get_logger().info('TakeoffActionClient accessed!')
        goal_msg.group_mask = group_mask
        goal_msg.height = height
        goal_msg.duration = Duration(seconds=duration_sec).to_msg()

        if not wait_for_action_server(self, self._action_client, 'Takeoff action server'):
            return False

        self.get_logger().info('Sending takeoff goal...')
        # Returns a 'future' to the client's goal handle. Returns feedback as well.
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        # Will execute the callback once the future is marked as 'done'.
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('Takeoff goal rejected.')
            rclpy.shutdown()
            return  # If takeoff goal is rejected, return early

        self.get_logger().info('Takeoff goal accepted.')
        # Recieves a 'future' that will complete when the result is ready.
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)  # Registers a callback when the 'future' is complete

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Current height: {feedback.current_height:.2f} m')

    def get_result_callback(self, future):
        result = future.result().result
        if result.return_code == 0:
            self.get_logger().info('Takeoff finished successfully!')
        else:
            self.get_logger().warning(f'Takeoff finished, but failed. return_code={result.return_code}')

        rclpy.shutdown()  # Shuts down ROS 2 cleanly


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)
    action_client = TakeoffActionClient()
    if action_client.send_goal():
        spin_node(action_client)


if __name__ == '__main__':
    main()

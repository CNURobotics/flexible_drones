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

"""CLI client for the ``execute_trajectory`` action."""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from flexible_drones_msgs.action import ExecuteTrajectory
from flexible_drones_tools.clients.client_utils import (
    declare_wait_timeout_parameter,
    print_usage_if_requested,
    wait_for_action_server,
)
from flexible_drones_tools.ros_shutdown import spin_node

USAGE = """\
Usage:
  ros2 run flexible_drones_tools start_execute_trajectory --ros-args -r __ns:=/drone1 -p trajectory_id:=0

Common examples:
  ros2 run flexible_drones_tools start_execute_trajectory --ros-args -p trajectory_id:=0
  ros2 run flexible_drones_tools start_execute_trajectory --ros-args -r __ns:=/drone1 -p trajectory_id:=1 -p timescale:=1.0
  ros2 run flexible_drones_tools start_execute_trajectory --ros-args -r __ns:=/drone1 -r __node:=execute_client -p relative:=true

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Set boolean parameters with '-p reversed:=true' or '-p relative:=true'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Parameters declared by this node:
  group_mask (integer, default: 0)
    Drone group mask sent in the ExecuteTrajectory goal.
  trajectory_id (integer, default: 0)
    Uploaded trajectory ID to execute.
  timescale (double, default: 1.0)
    Playback timescale.
  reversed (boolean, default: false)
    Execute the trajectory in reverse.
  relative (boolean, default: false)
    Execute the trajectory relative to the current pose.
  wait_timeout_sec (double, default: 5.0)
    Maximum time to wait for the action server.

Examples with parameter overrides:
  ros2 run flexible_drones_tools start_execute_trajectory --ros-args -p trajectory_id:=1 -p timescale:=0.75
  ros2 run flexible_drones_tools start_execute_trajectory --ros-args \
    -r __ns:=/drone2 -p group_mask:=1 -p reversed:=true -p relative:=false
"""


class ExecuteTrajActionClient(Node):

    def __init__(self):
        super().__init__('execute_trajectory_action_client')

        # Declare parameters (you can override via CLI or launch files)
        self.declare_parameter('group_mask', 0)
        self.declare_parameter('trajectory_id', 0)
        self.declare_parameter('timescale', 1.0)
        self.declare_parameter('reversed', False)  # Completes trajectory in reverse
        self.declare_parameter('relative', False)  # Starts trajectory at relative xyz position
        declare_wait_timeout_parameter(self)

        # Create the action client
        self._action_client = ActionClient(
            self,
            ExecuteTrajectory,
            'execute_trajectory',  # Figure out how to add the name of each Drone here? (Anwser: Remapping the namespace via CLA!)
        )

    def send_goal(self):
        group_mask = self.get_parameter('group_mask').get_parameter_value().integer_value
        trajectory_id = self.get_parameter('trajectory_id').get_parameter_value().integer_value
        timescale = self.get_parameter('timescale').get_parameter_value().double_value
        reversed_ = self.get_parameter('reversed').get_parameter_value().bool_value
        relative = self.get_parameter('relative').get_parameter_value().bool_value

        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.group_mask = group_mask
        goal_msg.trajectory_id = trajectory_id
        goal_msg.timescale = timescale
        goal_msg.reversed = reversed_  # Underscore helps to avoid name collision with Python's built-in function of the same name
        goal_msg.relative = relative

        if not wait_for_action_server(self, self._action_client, 'ExecuteTrajectory action server'):
            return False

        self.get_logger().info(
            f'Sending ExecuteTrajectory goal: trajectory_id={trajectory_id}, '
            f'timescale={timescale}, reversed={reversed_}, relative={relative}, '
            f'group_mask={group_mask}'
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
            self.get_logger().warning('ExecuteTrajectory goal rejected.')
            rclpy.shutdown()
            return  # If execute trajectory goal is rejected, return early

        self.get_logger().info('ExecuteTrajectory goal accepted.')
        self._get_result_future = (
            goal_handle.get_result_async()
        )  # Recieves a 'future' that will complete when the result is ready.
        self._get_result_future.add_done_callback(self.get_result_callback)  # Registers a callback when the 'future' is complete

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            'ExecuteTrajectory progress: '
            f'{100.0 * feedback.progress:.1f}% '
            f'elapsed={feedback.elapsed_sec:.1f}s '
            f'remaining={feedback.remaining_sec:.1f}s '
            f'duration={feedback.duration_sec:.1f}s'
        )

    def get_result_callback(self, future):
        result = future.result().result
        return_code = result.return_code  # Directs the return code for the logger function below
        if return_code == 0:  # Log the resulting sequence
            self.get_logger().info(f'ExecuteTrajectory finished successfully! (Return code: {return_code})')
        else:
            self.get_logger().warning(f'ExecuteTrajectory finished, but failed. (Return code: {return_code})')

        rclpy.shutdown()


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)
    action_client = ExecuteTrajActionClient()
    if action_client.send_goal():
        spin_node(action_client)


if __name__ == '__main__':
    main()

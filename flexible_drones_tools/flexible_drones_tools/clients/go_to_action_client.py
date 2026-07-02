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

"""CLI client for the ``go_to`` action."""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Point

from flexible_drones_msgs.action import (
    GoTo
)
from flexible_drones_tools.clients.client_utils import (
    declare_wait_timeout_parameter,
    print_usage_if_requested,
    wait_for_action_server,
)
from flexible_drones_tools.ros_shutdown import spin_node

USAGE = """\
Usage:
  ros2 run flexible_drones_tools go_to --ros-args -r __ns:=/drone1 -p goal_x:=1.0 -p goal_y:=0.0 -p goal_z:=1.0

Common examples:
  ros2 run flexible_drones_tools go_to --ros-args -p goal_x:=1.0 -p goal_y:=0.0 -p goal_z:=1.0
  ros2 run flexible_drones_tools go_to --ros-args -r __ns:=/drone1 \
    -p goal_x:=0.5 -p goal_y:=0.5 -p goal_z:=1.2 -p duration_sec:=4.0
  ros2 run flexible_drones_tools go_to --ros-args -r __ns:=/drone1 -r __node:=go_to_client -p frame:=1

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Parameters declared by this node:
  group_mask (integer, default: 0)
    Drone group mask sent in the GoTo goal.
  frame (integer, default: 0)
    Goal frame: 0=absolute ENU, 1=relative map/ENU, 2=relative body.
  goal_x (double, default: 1.0)
    Target x position in metres.
  goal_y (double, default: 0.0)
    Target y position in metres.
  goal_z (double, default: 1.0)
    Target z position in metres.
  yaw (double, default: 0.0)
    Target yaw in radians.
  duration_sec (double, default: 3.0)
    Go-to duration in seconds.
  wait_timeout_sec (double, default: 5.0)
    Maximum time to wait for the action server.

Examples with parameter overrides:
  ros2 run flexible_drones_tools go_to --ros-args -p goal_x:=0.0 -p goal_y:=1.0 -p goal_z:=1.0 -p yaw:=1.57
  ros2 run flexible_drones_tools go_to --ros-args -r __ns:=/drone2 -p frame:=1 -p goal_x:=0.25 -p duration_sec:=2.0
  ros2 run flexible_drones_tools go_to --ros-args -r __ns:=/drone2 -p frame:=2 -p goal_x:=0.25 -p duration_sec:=2.0
"""

FRAME_NAMES = {
    GoTo.Goal.FRAME_ABSOLUTE: 'absolute',
    GoTo.Goal.FRAME_RELATIVE_MAP: 'relative-map',
    GoTo.Goal.FRAME_RELATIVE_BODY: 'body-relative',
}


class GoToActionClient(Node):

    def __init__(self):
        super().__init__('go_to_action_client')

        self.declare_parameter('group_mask', 0)
        self.declare_parameter('frame', GoTo.Goal.FRAME_ABSOLUTE)
        self.declare_parameter('goal_x', 1.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_z', 1.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('duration_sec', 3.0)
        declare_wait_timeout_parameter(self)

        self._action_client = ActionClient(
            self,
            GoTo,
            'go_to')  # Figure out how to add the name of each Drone here? (Anwser: Remapping the namespace via CLA!)

    def send_goal(self):
        group_mask = self.get_parameter('group_mask').get_parameter_value().integer_value
        frame = self.get_parameter('frame').get_parameter_value().integer_value
        goal_x = self.get_parameter('goal_x').get_parameter_value().double_value
        goal_y = self.get_parameter('goal_y').get_parameter_value().double_value
        goal_z = self.get_parameter('goal_z').get_parameter_value().double_value
        yaw = self.get_parameter('yaw').get_parameter_value().double_value
        duration_sec = self.get_parameter('duration_sec').get_parameter_value().double_value

        if frame not in FRAME_NAMES:
            valid_frames = ', '.join(
                f'{value}={name}' for value, name in FRAME_NAMES.items()
            )
            self.get_logger().error(
                f'Invalid frame={frame}; expected one of: {valid_frames}'
            )
            return False

        goal_msg = GoTo.Goal()
        goal_msg.group_mask = group_mask
        goal_msg.frame = frame
        goal_msg.goal = Point(x=goal_x, y=goal_y, z=goal_z)
        goal_msg.yaw = yaw
        goal_msg.duration = Duration(seconds=duration_sec).to_msg()

        if not wait_for_action_server(self, self._action_client, 'GoTo action server'):
            return False

        self.get_logger().info(
            f'Sending go-to goal... ({goal_x}, {goal_y}, {goal_z}), '
            f'yaw={yaw} rad, frame={frame} ({FRAME_NAMES[frame]})'
        )
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('GoTo goal rejected.')
            rclpy.shutdown()
            return  # If go-to goal is rejected, return early

        self.get_logger().info('GoTo goal accepted.')
        # Recieves a 'future' that will complete when the result is ready.
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)  # Registers a callback when the 'future' is complete

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        position = feedback.current_position
        self.get_logger().info(
            f'Current position: ({position.x}, {position.y}, {position.z}), yaw={feedback.current_yaw} rad'
        )

    def get_result_callback(self, future):
        result = future.result().result
        return_code = result.return_code  # Directs the return code for the logger function below
        if return_code == 0:  # Log the resulting sequence
            self.get_logger().info(f'GoTo finished successfully! (Return code: {return_code})')
        else:
            self.get_logger().warning(f'GoTo finished, but failed. (Return code: {return_code})')

        rclpy.shutdown()


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)
    action_client = GoToActionClient()
    if action_client.send_goal():
        spin_node(action_client)


if __name__ == '__main__':
    main()

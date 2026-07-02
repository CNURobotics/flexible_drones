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

"""CLI client for the ``realign_local_position`` service."""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from flexible_drones_msgs.srv import RealignLocalPosition
from flexible_drones_tools.clients.client_utils import (
    declare_wait_timeout_parameter,
    print_usage_if_requested,
    wait_for_service,
)
from flexible_drones_tools.ros_shutdown import spin_node

USAGE = """\
Usage:
  ros2 run flexible_drones_tools realign_local_position --ros-args -r __ns:=/drone1

Common examples:
  ros2 run flexible_drones_tools realign_local_position --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools realign_local_position --ros-args -r __ns:=/drone1 -p z:=1.0
  ros2 run flexible_drones_tools realign_local_position --ros-args -r __ns:=/drone1 -p frame_id:=map -p x:=0.0 -p y:=0.0 -p z:=0.0

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Parameters declared by this node:
  frame_id (string, default: map)
    Frame ID for the reset pose.
  x, y, z (double, default: 0.0)
    Position for the reset pose.
  qx, qy, qz, qw (double, default: 0.0, 0.0, 0.0, 1.0)
    Orientation quaternion for the reset pose.
  wait_timeout_sec (double, default: 5.0)
    Maximum time to wait for the service.
"""


class RealignLocalPositionClient(Node):

    def __init__(self):
        super().__init__('realign_local_position_client')

        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 0.0)
        self.declare_parameter('qx', 0.0)
        self.declare_parameter('qy', 0.0)
        self.declare_parameter('qz', 0.0)
        self.declare_parameter('qw', 1.0)
        declare_wait_timeout_parameter(self)

        self._client = self.create_client(RealignLocalPosition, 'realign_local_position')

    def send_request(self):
        pose = PoseStamped()
        pose.header.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.get_parameter('x').get_parameter_value().double_value
        pose.pose.position.y = self.get_parameter('y').get_parameter_value().double_value
        pose.pose.position.z = self.get_parameter('z').get_parameter_value().double_value
        pose.pose.orientation.x = self.get_parameter('qx').get_parameter_value().double_value
        pose.pose.orientation.y = self.get_parameter('qy').get_parameter_value().double_value
        pose.pose.orientation.z = self.get_parameter('qz').get_parameter_value().double_value
        pose.pose.orientation.w = self.get_parameter('qw').get_parameter_value().double_value

        request = RealignLocalPosition.Request()
        request.pose = pose

        if not wait_for_service(self, self._client, 'RealignLocalPosition service'):
            return False

        self.get_logger().info(
            'Sending RealignLocalPosition request... '
            f"frame_id='{pose.header.frame_id}', "
            f'position=({pose.pose.position.x}, {pose.pose.position.y}, {pose.pose.position.z})'
        )
        self._send_request_future = self._client.call_async(request)
        self._send_request_future.add_done_callback(self.response_callback)
        return True

    def response_callback(self, future):
        response = future.result()
        if response is None:
            self.get_logger().warning(
                'RealignLocalPosition request failed: no response received.'
            )
        elif response.success:
            self.get_logger().info('RealignLocalPosition finished successfully!')
        else:
            self.get_logger().warning('RealignLocalPosition finished, but failed.')

        rclpy.shutdown()


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)
    client = RealignLocalPositionClient()
    if client.send_request():
        spin_node(client)


if __name__ == '__main__':
    main()

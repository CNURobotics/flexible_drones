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

"""CLI client for the ``set_led_color`` service."""

import rclpy
from rclpy.node import Node

from flexible_drones_msgs.srv import SetLEDColor
from flexible_drones_tools.clients.client_utils import (
    declare_wait_timeout_parameter,
    print_usage_if_requested,
    wait_for_service,
)
from flexible_drones_tools.ros_shutdown import spin_node

USAGE = """\
Usage:
  ros2 run flexible_drones_tools set_led_color --ros-args -p color:=green

Common examples:
  ros2 run flexible_drones_tools set_led_color --ros-args -p color:=blue
  ros2 run flexible_drones_tools set_led_color --ros-args -r __ns:=/drone1 -p color:=green
  ros2 run flexible_drones_tools set_led_color --ros-args -r __ns:=/drone1 -r __node:=led_client -p color:=red

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Parameters declared by this node:
  color (string, default: green)
    LED color name sent in the SetLEDColor service request.
  wait_timeout_sec (double, default: 5.0)
    Maximum time to wait for the service.

Examples with parameter overrides:
  ros2 run flexible_drones_tools set_led_color --ros-args -p color:=white
  ros2 run flexible_drones_tools set_led_color --ros-args -r __ns:=/drone2 -p color:=purple
"""


class SetLEDColorClient(Node):

    def __init__(self):
        super().__init__('set_led_color_client')

        self.declare_parameter('color', 'green')
        declare_wait_timeout_parameter(self)

        self._client = self.create_client(SetLEDColor, 'set_led_color')

    def send_request(self):
        color = self.get_parameter('color').get_parameter_value().string_value

        request = SetLEDColor.Request()
        request.color = color

        if not wait_for_service(self, self._client, 'SetLEDColor service'):
            return False

        self.get_logger().info(f"Sending SetLEDColor request... color='{color}'")
        self._send_request_future = self._client.call_async(request)
        self._send_request_future.add_done_callback(self.response_callback)
        return True

    def response_callback(self, future):
        response = future.result()
        if response is None:
            self.get_logger().warning('SetLEDColor request failed: no response received.')
        elif response.success:
            self.get_logger().info('SetLEDColor finished successfully!')
        else:
            self.get_logger().warning('SetLEDColor finished, but failed.')

        rclpy.shutdown()


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)
    client = SetLEDColorClient()
    if client.send_request():
        spin_node(client)


if __name__ == '__main__':
    main()

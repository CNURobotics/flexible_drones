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

"""CLI client for the ``arm`` service."""

import rclpy
from rclpy.node import Node

from flexible_drones_msgs.srv import Arm
from flexible_drones_tools.clients.client_utils import (
    declare_wait_timeout_parameter,
    print_usage_if_requested,
    wait_for_service,
)
from flexible_drones_tools.ros_shutdown import spin_node

USAGE = """\
Usage:
  ros2 run flexible_drones_tools arm --ros-args -r __ns:=/drone1

Common examples:
  ros2 run flexible_drones_tools arm --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools arm --ros-args -r __ns:=/drone1 -p arm:=false
  ros2 run flexible_drones_tools arm --ros-args -r __ns:=/drone1 -p timeout_sec:=5.0

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Set boolean parameters with '-p arm:=true' or '-p arm:=false'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Parameters declared by this node:
  arm (boolean, default: true)
    Arm when true; disarm when false.
  timeout_sec (double, default: 2.0)
    Timeout sent in the Arm service request.
  wait_timeout_sec (double, default: 5.0)
    Maximum time to wait for the service.
"""


class ArmClient(Node):

    def __init__(self):
        super().__init__('arm_client')

        self.declare_parameter('arm', True)
        self.declare_parameter('timeout_sec', 2.0)
        declare_wait_timeout_parameter(self)

        self._client = self.create_client(Arm, 'arm')

    def send_request(self):
        arm = self.get_parameter('arm').get_parameter_value().bool_value
        timeout_sec = self.get_parameter('timeout_sec').get_parameter_value().double_value

        request = Arm.Request()
        request.arm = arm
        request.timeout_sec = timeout_sec

        command = 'arm' if arm else 'disarm'
        if not wait_for_service(self, self._client, 'Arm service'):
            return False

        self.get_logger().info(f"Sending Arm request... command='{command}', timeout_sec={timeout_sec}")
        self._send_request_future = self._client.call_async(request)
        self._send_request_future.add_done_callback(self.response_callback)
        return True

    def response_callback(self, future):
        response = future.result()
        if response is None:
            self.get_logger().warning('Arm request failed: no response received.')
        elif response.success:
            self.get_logger().info('Arm request finished successfully!')
        else:
            self.get_logger().warning('Arm request finished, but failed.')

        rclpy.shutdown()


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)
    client = ArmClient()
    if client.send_request():
        spin_node(client)


if __name__ == '__main__':
    main()

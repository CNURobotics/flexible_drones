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

"""CLI client for the ``emergency_stop`` service."""

import rclpy
from rclpy.node import Node

from flexible_drones_msgs.srv import EmergencyStop
from flexible_drones_tools.clients.client_utils import (
    declare_wait_timeout_parameter,
    print_usage_if_requested,
    wait_for_service,
)
from flexible_drones_tools.ros_shutdown import spin_node

USAGE = """\
Usage:
  ros2 run flexible_drones_tools emergency_stop --ros-args -r __ns:=/drone1

Common examples:
  ros2 run flexible_drones_tools emergency_stop --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools emergency_stop --ros-args -r __ns:=/drone1 -r __node:=estop_client

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Parameters declared by this node:
  wait_timeout_sec (double, default: 5.0)
    Maximum time to wait for the service.
"""


class EmergencyStopClient(Node):

    def __init__(self):
        super().__init__('emergency_stop_client')

        declare_wait_timeout_parameter(self)
        self._client = self.create_client(EmergencyStop, 'emergency_stop')

    def send_request(self):
        request = EmergencyStop.Request()

        if not wait_for_service(self, self._client, 'EmergencyStop service'):
            return False

        self.get_logger().warning('Sending EmergencyStop request...')
        self._send_request_future = self._client.call_async(request)
        self._send_request_future.add_done_callback(self.response_callback)
        return True

    def response_callback(self, future):
        response = future.result()
        if response is None:
            self.get_logger().warning('EmergencyStop request failed: no response received.')
        elif response.success:
            self.get_logger().info(f'EmergencyStop finished successfully: {response.message}')
        else:
            self.get_logger().warning(f'EmergencyStop finished, but failed: {response.message}')

        rclpy.shutdown()


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)
    client = EmergencyStopClient()
    if client.send_request():
        spin_node(client)


if __name__ == '__main__':
    main()

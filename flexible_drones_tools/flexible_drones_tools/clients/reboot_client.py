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

"""CLI client for the ``reboot`` service."""

import sys

import rclpy
from rclpy.node import Node

from flexible_drones_msgs.srv import Reboot
from flexible_drones_tools.clients.client_utils import (
    declare_wait_timeout_parameter,
    print_usage_if_requested,
    wait_for_service,
)
from flexible_drones_tools.ros_shutdown import spin_node

USAGE = """\
Usage:
  ros2 run flexible_drones_tools reboot --ros-args -r __ns:=/drone1

Common examples:
  ros2 run flexible_drones_tools reboot --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools reboot --ros-args -r __ns:=/drone1 -p timeout_sec:=5.0

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.
  Set parameters with '-p name:=value'.

Parameters declared by this node:
  timeout_sec (double, default: 5.0)
    Reboot/drop confirmation timeout requested from the service.
  wait_timeout_sec (double, default: 5.0)
    Maximum time to wait for the service.

Safety:
  This client always asks for typed confirmation before sending the request.
"""


class RebootClient(Node):

    def __init__(self):
        super().__init__('reboot_client')

        self.declare_parameter('timeout_sec', 5.0)
        declare_wait_timeout_parameter(self)

        self._client = self.create_client(Reboot, 'reboot')

    def _confirmation_phrase(self):
        namespace = self.get_namespace().strip('/')
        target = namespace or self.get_name()
        return f'reboot {target}'

    def _confirm(self):
        phrase = self._confirmation_phrase()
        namespace = self.get_namespace().rstrip('/')
        service_name = f'{namespace}/reboot' if namespace else '/reboot'
        print('', flush=True)
        print('WARNING: This will request a vehicle flight-controller/firmware reboot.', flush=True)
        print(f'Target service: {service_name}', flush=True)
        print(f"Type exactly '{phrase}' to continue.", flush=True)
        try:
            entered = input('Confirmation: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nReboot request canceled.', flush=True)
            return False

        if entered != phrase:
            print('Reboot request canceled: confirmation did not match.', flush=True)
            return False

        return True

    def send_request(self):
        if not self._confirm():
            rclpy.shutdown()
            return False

        timeout_sec = self.get_parameter('timeout_sec').get_parameter_value().double_value

        request = Reboot.Request()
        request.timeout_sec = timeout_sec

        if not wait_for_service(self, self._client, 'Reboot service'):
            return False

        self.get_logger().warning(
            f'Sending Reboot request... confirmation_timeout_sec={timeout_sec}'
        )
        self._send_request_future = self._client.call_async(request)
        self._send_request_future.add_done_callback(self.response_callback)
        return True

    def response_callback(self, future):
        response = future.result()
        if response is None:
            self.get_logger().warning('Reboot request failed: no response received.')
        elif response.success:
            self.get_logger().warning(f'Reboot request finished successfully: {response.message}')
        else:
            self.get_logger().warning(f'Reboot request rejected or failed: {response.message}')

        rclpy.shutdown()


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)
    client = RebootClient()
    if client.send_request():
        spin_node(client)


if __name__ == '__main__':
    main(sys.argv[1:])

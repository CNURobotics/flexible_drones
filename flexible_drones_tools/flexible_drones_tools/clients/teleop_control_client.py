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

"""CLI client for the ``teleop_control`` controller action."""

from threading import Event, Thread

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from flexible_drones_msgs.action import EnableController
from flexible_drones_tools.clients.client_utils import (
    declare_wait_timeout_parameter,
    print_usage_if_requested,
    wait_for_action_server,
)
from flexible_drones_tools.ros_shutdown import BENIGN_SHUTDOWN_EXCEPTIONS


USAGE = """\
Usage:
  ros2 run flexible_drones_tools teleop_control --ros-args -r __ns:=/drone1

Common examples:
  ros2 run flexible_drones_tools teleop_control --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools teleop_control --ros-args -r __ns:=/drone1 -p wait_timeout_sec:=10.0

Notes:
  The drone must already be armed, flying, and receiving a fresh cmd_vel stream.
  Press Enter to cancel teleop control and wait for the action result.
"""


class TeleOpControlClient(Node):

    def __init__(self):
        super().__init__('teleop_control_client')
        declare_wait_timeout_parameter(self)
        self._action_client = ActionClient(
            self,
            EnableController,
            'teleop_control',
        )
        self._goal_handle = None
        self._done = Event()
        self._cancel_requested = False
        self._cancel_after_accept = False

    def send_goal(self):
        if not wait_for_action_server(
            self,
            self._action_client,
            'teleop_control action server',
        ):
            return False

        self.get_logger().info('Sending teleop_control goal...')
        future = self._action_client.send_goal_async(
            EnableController.Goal(controller_name='teleop_control'),
            feedback_callback=self.feedback_callback,
        )
        future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().warning('teleop_control goal rejected.')
            self._cancel_after_accept = False
            self._done.set()
            return

        self.get_logger().info('Teleop control is active. Press Enter to cancel.')
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)
        if self._cancel_after_accept:
            self._cancel_after_accept = False
            self.cancel()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'teleop_active={feedback.controller_active}, '
            f'cmd_vel age={feedback.input_age_sec:.2f} s'
        )

    def cancel(self):
        if self._cancel_requested:
            return
        if self._goal_handle is None:
            if self._cancel_after_accept:
                return
            self._cancel_after_accept = True
            self.get_logger().info('Teleop goal is pending; will cancel after acceptance.')
            return
        self._cancel_requested = True
        self._cancel_after_accept = False
        self.get_logger().info('Canceling teleop control...')
        self._goal_handle.cancel_goal_async()

    def result_callback(self, future):
        result = future.result().result
        if result.return_code == 0:
            self.get_logger().info(f'Teleop control finished: {result.message}')
        else:
            self.get_logger().warning(
                f'Teleop control failed: return_code={result.return_code}, '
                f'message={result.message}'
            )
        self._done.set()

    def spin_until_done(self):
        Thread(target=self._wait_for_enter, daemon=True).start()
        while rclpy.ok() and not self._done.is_set():
            rclpy.spin_once(self, timeout_sec=0.1)

    def _wait_for_enter(self):
        try:
            input()
        except EOFError:
            pass
        self.cancel()


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)
    client = TeleOpControlClient()
    try:
        if client.send_goal():
            client.spin_until_done()
    except BENIGN_SHUTDOWN_EXCEPTIONS:
        client.cancel()
        client.spin_until_done()
    finally:
        client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

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

"""Shared helpers for ROS 2 command-line clients."""

import sys

import rclpy


DEFAULT_WAIT_TIMEOUT_SEC = 5.0
WAIT_TIMEOUT_PARAMETER = 'wait_timeout_sec'


def declare_wait_timeout_parameter(node):
    """Declare the common server/service wait timeout parameter."""
    node.declare_parameter(WAIT_TIMEOUT_PARAMETER, DEFAULT_WAIT_TIMEOUT_SEC)


def get_wait_timeout_sec(node):
    """Read the common server/service wait timeout parameter."""
    return node.get_parameter(WAIT_TIMEOUT_PARAMETER).get_parameter_value().double_value


def print_usage_if_requested(usage, args=None):
    """Print static usage text when ``-h`` or ``--help`` is present."""
    cli_args = list(sys.argv[1:] if args is None else args)
    if any(arg in ('-h', '--help') for arg in cli_args):
        print(usage)
        return True

    return False


def wait_for_action_server(node, action_client, description):
    """Wait for an action server and shut down ROS on timeout."""
    timeout_sec = get_wait_timeout_sec(node)
    node.get_logger().info(f'Waiting for {description}...')
    if action_client.wait_for_server(timeout_sec=timeout_sec):
        return True

    node.get_logger().error(f'{description} not available after {timeout_sec:.1f} s.')
    rclpy.shutdown()
    return False


def wait_for_service(node, client, description):
    """Wait for a service and shut down ROS on timeout."""
    timeout_sec = get_wait_timeout_sec(node)
    node.get_logger().info(f'Waiting for {description}...')
    if client.wait_for_service(timeout_sec=timeout_sec):
        return True

    node.get_logger().error(f'{description} not available after {timeout_sec:.1f} s.')
    rclpy.shutdown()
    return False

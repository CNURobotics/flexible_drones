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

"""Helpers for quiet ROS 2 command-line shutdown."""

import rclpy
from rclpy.executors import ExternalShutdownException, ShutdownException


BENIGN_SHUTDOWN_EXCEPTIONS = (
    KeyboardInterrupt,
    ExternalShutdownException,
    ShutdownException,
)


def spin_node(node) -> None:
    """Spin a node and suppress normal Ctrl+C / external shutdown exceptions."""
    try:
        rclpy.spin(node)
    except BENIGN_SHUTDOWN_EXCEPTIONS:
        pass

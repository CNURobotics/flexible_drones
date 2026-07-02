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

"""Unit tests for quiet ROS shutdown helpers."""

from flexible_drones_tools import ros_shutdown


def test_spin_node_suppresses_keyboard_interrupt(monkeypatch):
    """Ctrl+C during a console script spin should not escape as a traceback."""

    def _raise_keyboard_interrupt(_node):
        raise KeyboardInterrupt

    monkeypatch.setattr(ros_shutdown.rclpy, 'spin', _raise_keyboard_interrupt)

    ros_shutdown.spin_node(object())

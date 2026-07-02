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

"""Tests for DroneManager URDF origin parsing helpers."""

import xml.etree.ElementTree as ET

import pytest

from flexible_drones_core.manager.drone_manager import DroneManager


class _Parser(DroneManager):
    """Minimal concrete subclass used only to exercise base-class helpers."""

    def start_up(self):
        pass

    def shut_down(self):
        pass

    def create_instance(self):
        pass

    def _cmd_vel_changed(self, msg):
        pass

    def _cmd_full_state_changed(self, msg):
        pass

    def _emergency_callback(self, req, resp):
        pass

    def _set_led_color_callback(self, req, resp):
        pass

    def _arm_callback(self, req, resp):
        pass

    def _takeoff_callback(self, gh):
        pass

    def _land_callback(self, gh):
        pass

    def _go_to_callback(self, gh):
        pass

    def _upload_trajectory_callback(self, gh):
        pass

    def _execute_trajectory_callback(self, gh):
        pass

    def _cancel_trajectory_callback(self, gh):
        pass


class _FakeTfBroadcaster:
    """Minimal broadcaster with the same publisher attribute as tf2_ros."""

    def __init__(self, node):
        self.node = node
        self.pub_tf = object()


def _parser():
    return object.__new__(_Parser)


def test_manager_requires_non_empty_robot_description():
    """Base manager validation should reject missing and blank robot descriptions."""
    with pytest.raises(ValueError, match='non-empty robot_description'):
        _Parser('cf1', 'crazyflie', object(), ros_params={})

    with pytest.raises(ValueError, match='non-empty robot_description'):
        _Parser('cf1', 'crazyflie', object(), ros_params={'robot_description': '   '})


def test_parse_xyz_rejects_truncated_origin_attribute():
    """Malformed xyz attributes should fail with a clear parse error."""
    origin = ET.fromstring('<origin xyz="0 0"/>')

    with pytest.raises(ValueError, match='xyz.*exactly 3 values'):
        _parser()._parse_xyz(origin)


def test_parse_rpy_rejects_non_float_origin_attribute():
    """Malformed rpy attributes should name the bad URDF attribute."""
    origin = ET.fromstring('<origin rpy="0 nope 0"/>')

    with pytest.raises(ValueError, match='rpy.*non-float'):
        _parser()._parse_rpy(origin)


def test_parse_origin_defaults_to_zero_vector():
    """Missing xyz/rpy attributes retain the historical zero default."""
    origin = ET.fromstring('<origin/>')
    parser = _parser()

    assert parser._parse_xyz(origin) == [0.0, 0.0, 0.0]
    assert parser._parse_rpy(origin) == [0.0, 0.0, 0.0]


def test_tf_broadcaster_publishers_are_registered_for_cleanup():
    """TF broadcaster publishers should be destroyed with manager publishers."""
    parser = _parser()
    parser.node = object()
    parser._ros_publishers = []

    broadcaster = parser._create_tf_broadcaster(_FakeTfBroadcaster)

    assert broadcaster.node is parser.node
    assert parser._ros_publishers == [broadcaster.pub_tf]

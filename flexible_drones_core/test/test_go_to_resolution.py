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

"""Tests for shared GoTo target-resolution helpers in DroneManager."""

from math import cos, pi, sin
from types import SimpleNamespace

import pytest
from rclpy.action import GoalResponse

from flexible_drones_core.manager.drone_manager import DroneManager

_F_ABSOLUTE = DroneManager._GOTO_FRAME_ABSOLUTE  # 0
_F_MAP = DroneManager._GOTO_FRAME_RELATIVE_MAP  # 1
_F_BODY = DroneManager._GOTO_FRAME_RELATIVE_BODY  # 2


class _Resolver(DroneManager):
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


def _manager_at(x=0.0, y=0.0, z=0.0, yaw=0.0):
    """Return a resolver with minimal odometry state at the given pose."""
    manager = object.__new__(_Resolver)
    manager.drone_state = SimpleNamespace(
        odom=SimpleNamespace(
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=x, y=y, z=z),
                    orientation=SimpleNamespace(
                        x=0.0,
                        y=0.0,
                        z=sin(yaw / 2.0),
                        w=cos(yaw / 2.0),
                    ),
                )
            )
        )
    )
    return manager


def _manager_without_odom():
    """Return a resolver whose backend has not provided odometry."""
    manager = object.__new__(_Resolver)
    manager.drone_state = SimpleNamespace(odom=None)
    manager.drone_name = 'test_drone'
    manager._logger = SimpleNamespace(warning=lambda msg: None)
    return manager


def _go_to_request(frame, x, y, z, yaw=0.0):
    """Build a GoTo.Goal-like object using the uint8 frame field."""
    return SimpleNamespace(
        frame=frame,
        goal=SimpleNamespace(x=x, y=y, z=z),
        yaw=yaw,
    )


def test_frame_absolute_returns_goal_directly():
    """FRAME_ABSOLUTE: goal coordinates returned as-is; odometry is ignored."""
    manager = _manager_at(x=4.0, y=5.0, z=6.0, yaw=pi / 2.0)
    request = _go_to_request(_F_ABSOLUTE, 1.0, 2.0, 3.0, yaw=0.25)
    assert manager._resolve_expected_go_to_position(request) == (1.0, 2.0, 3.0)
    assert manager._resolve_expected_go_to_yaw(request) == 0.25


def test_frame_absolute_ignores_missing_odometry():
    """FRAME_ABSOLUTE does not require current pose state."""
    manager = _manager_without_odom()
    request = _go_to_request(_F_ABSOLUTE, 1.0, 2.0, 3.0, yaw=0.25)

    assert manager._resolve_expected_go_to_position(request) == (1.0, 2.0, 3.0)
    assert manager._resolve_expected_go_to_yaw(request) == 0.25
    assert manager._validate_go_to_goal(request) == GoalResponse.ACCEPT


def test_frame_relative_map_adds_enu_offset_without_rotation():
    """FRAME_RELATIVE_MAP: offset added in ENU regardless of heading."""
    manager = _manager_at(x=1.0, y=2.0, z=3.0, yaw=pi / 2.0)
    request = _go_to_request(_F_MAP, 1.0, 0.5, -0.25, yaw=0.75)
    # Offset applied directly: (1+1, 2+0.5, 3-0.25)
    assert manager._resolve_expected_go_to_position(request) == pytest.approx((2.0, 2.5, 2.75))
    assert manager._resolve_expected_go_to_yaw(request) == pytest.approx(pi / 2.0 + 0.75)


def test_frame_relative_body_rotates_offset_by_heading():
    """FRAME_RELATIVE_BODY: x/y offset rotated by current heading."""
    manager = _manager_at(x=1.0, y=2.0, z=3.0, yaw=pi / 2.0)
    request = _go_to_request(_F_BODY, 1.0, 0.5, -0.25, yaw=0.75)
    # At yaw=pi/2: cos=0, sin=1
    # target_x = 1.0 + 1.0*0 - 0.5*1 = 0.5
    # target_y = 2.0 + 1.0*1 + 0.5*0 = 3.0
    assert manager._resolve_expected_go_to_position(request) == pytest.approx((0.5, 3.0, 2.75))
    assert manager._resolve_expected_go_to_yaw(request) == pytest.approx(pi / 2.0 + 0.75)


def test_frame_relative_map_rejects_missing_odometry():
    """Relative frames need odometry and should fail clearly when it is absent."""
    manager = _manager_without_odom()
    request = _go_to_request(_F_MAP, 1.0, 0.5, -0.25, yaw=0.75)

    with pytest.raises(ValueError, match='requires odometry'):
        manager._resolve_expected_go_to_position(request)
    with pytest.raises(ValueError, match='requires odometry'):
        manager._resolve_expected_go_to_yaw(request)
    assert manager._validate_go_to_goal(request) == GoalResponse.REJECT


def test_go_to_request_missing_frame_raises():
    """Raise AttributeError for old relative= requests without frame."""
    manager = _manager_at()
    stale_request = SimpleNamespace(
        relative=False,
        goal=SimpleNamespace(x=1.0, y=0.0, z=0.0),
        yaw=0.0,
    )
    with pytest.raises(AttributeError):
        manager._resolve_expected_go_to_position(stale_request)

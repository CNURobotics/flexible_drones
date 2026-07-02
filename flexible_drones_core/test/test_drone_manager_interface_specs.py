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

"""Tests for DroneManager ROS interface creation specs."""

import asyncio
import math
import time
from types import SimpleNamespace

import pytest
from rclpy.action import GoalResponse

from flexible_drones_core.manager.drone_manager import DroneManager
from flexible_drones_msgs.msg import DroneStatus


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class _SpecManager(DroneManager):
    """Minimal concrete subclass used to exercise base interface policy."""

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

    def _command_readiness_error(self):
        return None


def _manager():
    mgr = object.__new__(_SpecManager)
    mgr.drone_name = 'px1'
    mgr.node = object()
    mgr.ros_params = {'robot_description': '<robot name="px1"/>'}
    mgr.service_group = object()
    mgr.sub_group = object()
    mgr._action_group = object()
    mgr._logger = _FakeLogger()
    mgr._tfbr = None
    mgr._odom_pub = None
    mgr._status_pub = None
    mgr._target_state_pub = None
    mgr._last_target_state_signature = None
    mgr._robot_description_pub = None
    mgr._allow_teleop = False
    mgr._motion_action_active = False
    mgr._teleop_last_seen = None
    mgr._teleop_stale_after = 0.5
    return mgr


def _stamp(sec=10, nanosec=0):
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _fresh_odom(z=1.0):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp()),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(z=z),
            )
        ),
    )


def _teleop_ready_manager():
    mgr = _manager()
    mgr.connected = True
    mgr.drone_state = SimpleNamespace(
        status=SimpleNamespace(
            status_flags=(
                DroneStatus.STATUS_ARMED
                | DroneStatus.STATUS_AIRBORNE
            )
        ),
        odom=_fresh_odom(),
    )
    mgr.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: _stamp()))
    )
    mgr._teleop_last_seen = time.monotonic()
    return mgr


def _names(specs, attr):
    return [getattr(spec, attr) for spec in specs]


def test_default_interface_specs_match_existing_ros_names():
    """Default specs preserve the existing manager ROS interface surface."""
    mgr = _manager()

    assert _names(mgr._publisher_specs(), 'topic_name') == [
        'px1/odom',
        'px1/status',
        'px1/target_state',
        'px1/robot_description',
    ]
    assert _names(mgr._service_specs(), 'service_name') == [
        'px1/emergency_stop',
        'px1/set_led_color',
        'px1/arm',
        'px1/get_status',
        'px1/realign_local_position',
        'px1/reboot',
    ]
    assert _names(mgr._subscription_specs(), 'topic_name') == [
        'px1/cmd_vel',
        'px1/cmd_full_state',
    ]
    assert _names(mgr._action_server_specs(), 'action_name') == [
        'px1/upload_trajectory',
        'px1/takeoff',
        'px1/go_to',
        'px1/execute_trajectory',
        'px1/land',
        'px1/teleop_control',
    ]


def test_topic_service_initialization_iterates_default_specs(monkeypatch):
    """Topic/service init should create exactly the interfaces described by specs."""
    mgr = _manager()
    publishers = []
    services = []
    subscriptions = []

    def _create_tf_broadcaster(_broadcaster_type):
        return object()

    def _create_publisher(msg_type, topic_name, qos_profile):
        publisher = _FakePublisher()
        publishers.append((msg_type, topic_name, qos_profile, publisher))
        return publisher

    def _create_service(srv_type, service_name, callback, callback_group=None):
        services.append((srv_type, service_name, callback, callback_group))
        return object()

    def _create_subscription(msg_type, topic_name, callback, qos_profile, callback_group=None):
        subscriptions.append((msg_type, topic_name, callback, qos_profile, callback_group))
        return object()

    monkeypatch.setattr(mgr, '_create_tf_broadcaster', _create_tf_broadcaster)
    monkeypatch.setattr(mgr, '_create_publisher', _create_publisher)
    monkeypatch.setattr(mgr, '_create_service', _create_service)
    monkeypatch.setattr(mgr, '_create_subscription', _create_subscription)

    mgr._init_topics_and_services()

    assert [entry[1] for entry in publishers] == [
        'px1/odom',
        'px1/status',
        'px1/target_state',
        'px1/robot_description',
    ]
    assert mgr._odom_pub is publishers[0][3]
    assert mgr._status_pub is publishers[1][3]
    assert mgr._target_state_pub is publishers[2][3]
    assert mgr._robot_description_pub is publishers[3][3]
    assert mgr._robot_description_pub.published[0].data == '<robot name="px1"/>'
    assert [entry[1] for entry in services] == _names(mgr._service_specs(), 'service_name')
    assert [entry[1] for entry in subscriptions] == _names(
        mgr._subscription_specs(),
        'topic_name',
    )


def test_topic_service_initialization_allows_omitting_robot_description(monkeypatch):
    """Topic/service init should tolerate platform policies without robot description."""
    mgr = _manager()
    publishers = []
    services = []
    subscriptions = []
    default_publishers = mgr._publisher_specs()

    def _create_tf_broadcaster(_broadcaster_type):
        return object()

    def _create_publisher(msg_type, topic_name, qos_profile):
        publisher = _FakePublisher()
        publishers.append((msg_type, topic_name, qos_profile, publisher))
        return publisher

    def _create_service(srv_type, service_name, callback, callback_group=None):
        services.append((srv_type, service_name, callback, callback_group))
        return object()

    def _create_subscription(msg_type, topic_name, callback, qos_profile, callback_group=None):
        subscriptions.append((msg_type, topic_name, callback, qos_profile, callback_group))
        return object()

    monkeypatch.setattr(
        mgr,
        '_publisher_specs',
        lambda: tuple(
            spec for spec in default_publishers
            if spec.attribute_name != '_robot_description_pub'
        ),
    )
    monkeypatch.setattr(mgr, '_create_tf_broadcaster', _create_tf_broadcaster)
    monkeypatch.setattr(mgr, '_create_publisher', _create_publisher)
    monkeypatch.setattr(mgr, '_create_service', _create_service)
    monkeypatch.setattr(mgr, '_create_subscription', _create_subscription)

    mgr._init_topics_and_services()

    assert [entry[1] for entry in publishers] == [
        'px1/odom',
        'px1/status',
        'px1/target_state',
    ]
    assert mgr._odom_pub is publishers[0][3]
    assert mgr._status_pub is publishers[1][3]
    assert mgr._robot_description_pub is None
    assert [entry[1] for entry in services] == _names(mgr._service_specs(), 'service_name')
    assert [entry[1] for entry in subscriptions] == _names(
        mgr._subscription_specs(),
        'topic_name',
    )


def test_target_state_publishes_go_to_goal_on_change():
    """Manager target_state publishes resolved action targets once per change."""
    mgr = _manager()
    mgr.odom_tf_name = 'px1/odom'
    mgr._target_state_pub = _FakePublisher()
    mgr.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: _stamp()))
    )
    yaw = math.pi / 2.0
    mgr.drone_state = SimpleNamespace(
        odom=SimpleNamespace(
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                    orientation=SimpleNamespace(
                        x=0.0,
                        y=0.0,
                        z=math.sin(yaw / 2.0),
                        w=math.cos(yaw / 2.0),
                    ),
                )
            )
        )
    )
    request = SimpleNamespace(
        frame=DroneManager._GOTO_FRAME_RELATIVE_BODY,
        goal=SimpleNamespace(x=1.0, y=0.5, z=-0.25),
        yaw=0.75,
    )

    assert mgr._publish_action_target_state('go_to', request) is True
    assert mgr._publish_action_target_state('go_to', request) is False

    msg = mgr._target_state_pub.published[-1]
    assert msg.header.frame_id == 'px1/odom'
    assert msg.child_frame_id == 'px1'
    assert msg.pose.position.x == pytest.approx(0.5)
    assert msg.pose.position.y == pytest.approx(3.0)
    assert msg.pose.position.z == pytest.approx(2.75)


def test_action_initialization_iterates_default_specs(monkeypatch):
    """Action init should assign the same server attributes as before."""
    mgr = _manager()
    action_servers = []

    def _create_action_server(
        node,
        action_type,
        action_name,
        execute_callback=None,
        goal_callback=None,
        cancel_callback=None,
        callback_group=None,
    ):
        server = SimpleNamespace(action_name=action_name)
        action_servers.append((
            node,
            action_type,
            action_name,
            execute_callback,
            goal_callback,
            cancel_callback,
            callback_group,
            server,
        ))
        return server

    monkeypatch.setattr(mgr, '_create_action_server', _create_action_server)

    mgr._init_action_servers()

    assert [entry[2] for entry in action_servers] == _names(
        mgr._action_server_specs(),
        'action_name',
    )
    assert mgr.upload_action_server is action_servers[0][7]
    assert mgr.takeoff_action_server is action_servers[1][7]
    assert mgr.go_to_action_server is action_servers[2][7]
    assert mgr.execute_action_server is action_servers[3][7]
    assert mgr.land_action_server is action_servers[4][7]
    assert mgr.teleop_control_action_server is action_servers[5][7]


def test_motion_goals_reject_while_teleop_is_active():
    mgr = _manager()
    mgr._allow_teleop = True

    assert mgr._go_to_goal_callback(SimpleNamespace(group_mask=0)) == GoalResponse.REJECT


def test_teleop_control_goal_rejects_while_motion_action_is_active():
    mgr = _teleop_ready_manager()
    mgr._motion_action_active = True

    assert mgr._teleop_control_goal_callback(SimpleNamespace()) == GoalResponse.REJECT


def test_teleop_control_goal_requires_live_cmd_vel_stream():
    mgr = _teleop_ready_manager()
    mgr._teleop_last_seen = None

    assert mgr._teleop_control_goal_callback(SimpleNamespace()) == GoalResponse.REJECT


def test_motion_action_wrapper_returns_sync_callback_result():
    mgr = _manager()
    result = object()

    assert asyncio.run(mgr._run_motion_action('takeoff', lambda _gh: result, object())) is result
    assert mgr._motion_action_active is False


def test_motion_action_wrapper_awaits_async_callback_result():
    mgr = _manager()
    result = object()

    async def _callback(_goal_handle):
        assert mgr._motion_action_active is True
        await asyncio.sleep(0)
        return result

    assert asyncio.run(mgr._run_motion_action('takeoff', _callback, object())) is result
    assert mgr._motion_action_active is False


def test_apply_teleop_height_limits_hard_and_soft_clamps():
    mgr = _manager()
    min_height = 0.3
    max_height = 2.5
    slowdown_fraction = 0.1
    slowdown_zone = slowdown_fraction * (max_height - min_height)
    mgr._teleop_min_height = min_height
    mgr._teleop_max_height = max_height
    mgr._teleop_height_slowdown_fraction = slowdown_fraction

    mgr.drone_state = SimpleNamespace(odom=_fresh_odom(z=max_height - slowdown_zone / 2.0))
    assert mgr._apply_teleop_height_limits(1.0) < 1.0

    mgr.drone_state.odom = _fresh_odom(z=max_height)
    assert mgr._apply_teleop_height_limits(1.0) == 0.0

    mgr.drone_state.odom = _fresh_odom(z=min_height)
    assert mgr._apply_teleop_height_limits(-1.0) == 0.0


def test_get_status_serves_fresh_cached_status():
    """get_status returns success for a recently received status."""
    import time as _time

    mgr = _manager()
    mgr.drone_state = SimpleNamespace(status=SimpleNamespace(drone_name='px1'))
    mgr._status_received_mono = _time.monotonic()
    response = SimpleNamespace(success=None, status=None)

    result = mgr._get_status_callback(SimpleNamespace(drone_id=''), response)

    assert result.success is True
    assert result.status is mgr.drone_state.status


def test_get_status_rejects_stale_cached_status():
    """A latched status from a dead publisher must not be served as live."""
    import time as _time

    mgr = _manager()
    mgr.drone_state = SimpleNamespace(status=SimpleNamespace(drone_name='px1'))
    mgr._status_received_mono = (
        _time.monotonic() - mgr._STATUS_STALENESS_S - 0.5
    )
    response = SimpleNamespace(success=None, status=None)

    result = mgr._get_status_callback(SimpleNamespace(drone_id=''), response)

    assert result.success is False
    assert result.status is mgr.drone_state.status


def test_get_status_rejects_when_no_receive_time_recorded():
    """A status cached without a receive timestamp counts as stale."""
    mgr = _manager()
    mgr.drone_state = SimpleNamespace(status=SimpleNamespace(drone_name='px1'))
    mgr._status_received_mono = None
    response = SimpleNamespace(success=None, status=None)

    result = mgr._get_status_callback(SimpleNamespace(drone_id=''), response)

    assert result.success is False

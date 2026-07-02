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

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
import math
import threading
import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.action import ActionServer
from rclpy.action import CancelResponse, GoalResponse
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry

from flexible_drones_msgs.msg import DroneStatus, FullState
from flexible_drones_msgs.srv import (
    Arm,
    EmergencyStop,
    GetStatus,
    RealignLocalPosition,
    Reboot,
    SetLEDColor,
)
from flexible_drones_msgs.action import (
    Takeoff,
    Land,
    GoTo,
    UploadTrajectory,
    ExecuteTrajectory,
    EnableController,
)
from flexible_drones_core.geometry import euler_from_quaternion_xyzw, quaternion_from_euler_xyzw

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


@dataclass(frozen=True)
class PublisherSpec:
    attribute_name: str
    msg_type: object
    topic_name: str
    qos_profile: object


@dataclass(frozen=True)
class SubscriptionSpec:
    msg_type: object
    topic_name: str
    callback: object
    qos_profile: object
    callback_group: object = None


@dataclass(frozen=True)
class ServiceSpec:
    srv_type: object
    service_name: str
    callback: object
    callback_group: object = None


@dataclass(frozen=True)
class ActionServerSpec:
    attribute_name: str
    action_type: object
    action_name: str
    execute_callback: object
    goal_callback: object
    cancel_callback: object
    callback_group: object = None


class _DroneState:
    def __init__(self, odom=None, status=None):
        self.odom = odom
        self.status = status


class DroneManager(ABC):

    # get_status fails when the cached DroneStatus is older than this, so a
    # latched status from a dead publisher is not served as live state.
    # Backends that publish status slower than this should override it.
    _STATUS_STALENESS_S = 3.0

    # Action return codes shared across all DroneManager implementations.
    _RC_SUCCESS = 0
    _RC_HLC_FAILURE = 1
    _RC_ODOM_FAILURE = 2
    _RC_STATUS_FAILURE = 3
    _RC_CANCELLED = 4
    _RC_CRASHED = 5
    _RC_TIMEOUT = 6
    _RC_UPLOAD_FAILURE = 7
    _RC_START_FAILURE = 8
    _RC_INVALID_GOAL = 9

    _TELEOP_FEEDBACK_HZ = 20.0

    _RC_MESSAGES = {
        0: 'success',
        1: 'high-level commander failure',
        2: 'odometry unavailable or out of range',
        3: 'drone status check failed',
        4: 'action cancelled',
        5: 'drone crashed',
        6: 'action timed out',
        7: 'trajectory upload failed',
        8: 'trajectory start failed',
        9: 'invalid goal parameters',
    }

    @classmethod
    def _rc_message(cls, return_code: int) -> str:
        return cls._RC_MESSAGES.get(return_code, f'unknown return code {return_code}')

    def __del__(self):
        """Best-effort cleanup when a manager is garbage-collected."""
        try:
            self.cleanup()
        except Exception:  # noqa: B902
            pass

    def __init__(
        self,
        drone_name: str,
        drone_type: str,
        node: Node,
        ros_params=None,
        drone_state=None,
        odom_tf_name='odom',
        on_connected=None,
        on_disconnected=None,
    ):
        self.drone_name = drone_name
        self.drone_type = drone_type
        self.drone_state = drone_state if drone_state is not None else _DroneState()

        self.node = node
        self.odom_tf_name = f'{drone_name}/{odom_tf_name}'

        robot_description = str((ros_params or {}).get('robot_description', '')).strip()
        if not robot_description:
            raise ValueError(f"Drone '{drone_name}' must define a non-empty robot_description in ros_params!")
        ros_params['robot_description'] = robot_description

        self.ros_params = ros_params
        self.group_mask = self._resolve_group_mask(self.ros_params)

        self._logger = node.get_logger()

        self._on_connected_cb = on_connected
        self._on_disconnected_cb = on_disconnected

        self.connected = False  # boolean
        self._cleanup_lock = threading.RLock()
        self._cleanup_complete = False
        self._ros_action_servers = []
        self._ros_publishers = []
        self._ros_services = []
        self._ros_subscriptions = []
        self._ros_timers = []

        self._allow_teleop = False
        self._teleop_last_seen = None
        self._teleop_min_height = float(self.ros_params.get('teleop_min_height', 0.3))
        self._teleop_max_height = float(self.ros_params.get('teleop_max_height', 2.5))
        self._teleop_height_slowdown_fraction = max(0.0, min(0.5, float(
            self.ros_params.get('teleop_height_slowdown_fraction', 0.1)
        )))
        self._teleop_stale_after = float(self.ros_params.get('teleop_stale_after', 0.5))
        self._validate_teleop_params()
        self._motion_action_active = False

        if not hasattr(self, '_action_group'):
            self._action_group = MutuallyExclusiveCallbackGroup()
        self.service_group = MutuallyExclusiveCallbackGroup()
        self.sub_group = MutuallyExclusiveCallbackGroup()
        self._watchdog_group = MutuallyExclusiveCallbackGroup()

        self._tfbr = None
        self._static_tfbr = self._create_tf_broadcaster(StaticTransformBroadcaster)
        self._odom_pub = None
        self._status_pub = None
        self._status_received_mono = None
        self._target_state_pub = None
        self._last_target_state_signature = None
        self._robot_description_pub = None

        self._continuous_joints = self._parse_robot_description_joints()
        self._prop_angle = 0.0
        self._prop_last_stamp = None

        # Watchdog: fires every 5 s while the drone is not fully operational so
        # operators can see why goals are being rejected.  Cancels itself once
        # _watchdog_not_ready_reason() returns None; restarted on disconnect via
        # _restart_watchdog().
        self._watchdog_timer = self._create_timer(
            5.0,
            self._watchdog_tick,
            callback_group=self._watchdog_group,
        )

        # Let the backend-specific manager allocate its driver/client state.
        self.create_instance()

    def _create_publisher(self, *args, **kwargs):
        publisher = self.node.create_publisher(*args, **kwargs)
        self._ros_publishers.append(publisher)
        return publisher

    def _create_service(self, *args, **kwargs):
        service = self.node.create_service(*args, **kwargs)
        self._ros_services.append(service)
        return service

    def _create_subscription(self, *args, **kwargs):
        subscription = self.node.create_subscription(*args, **kwargs)
        self._ros_subscriptions.append(subscription)
        return subscription

    def _create_timer(self, *args, **kwargs):
        timer = self.node.create_timer(*args, **kwargs)
        self._ros_timers.append(timer)
        return timer

    def _create_action_server(self, *args, **kwargs):
        action_server = ActionServer(*args, **kwargs)
        self._ros_action_servers.append(action_server)
        return action_server

    def _create_tf_broadcaster(self, broadcaster_type):
        broadcaster = broadcaster_type(self.node)
        publisher = getattr(broadcaster, 'pub_tf', None)
        if publisher is not None:
            self._ros_publishers.append(publisher)
        return broadcaster

    def _publisher_specs(self):
        return (
            PublisherSpec(
                '_odom_pub',
                Odometry,
                f'{self.drone_name}/odom',
                10,
            ),
            PublisherSpec(
                '_status_pub',
                DroneStatus,
                f'{self.drone_name}/status',
                rclpy.qos.QoSProfile(
                    depth=1,
                    durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
                ),
            ),
            PublisherSpec(
                '_target_state_pub',
                FullState,
                f'{self.drone_name}/target_state',
                # Latched + reliable to match the PiHawk onboard_node publisher
                # so late-joining GUIs see the current target on every platform.
                rclpy.qos.QoSProfile(
                    depth=1,
                    durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
                ),
            ),
            PublisherSpec(
                '_robot_description_pub',
                String,
                f'{self.drone_name}/robot_description',
                rclpy.qos.QoSProfile(
                    depth=1,
                    durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
                ),
            ),
        )

    def _service_specs(self):
        return (
            ServiceSpec(
                EmergencyStop,
                f'{self.drone_name}/emergency_stop',
                self._emergency_callback,
                self.service_group,
            ),
            ServiceSpec(
                SetLEDColor,
                f'{self.drone_name}/set_led_color',
                self._set_led_color_callback,
                self.service_group,
            ),
            ServiceSpec(
                Arm,
                f'{self.drone_name}/arm',
                self._arm_callback,
                self.service_group,
            ),
            ServiceSpec(
                GetStatus,
                f'{self.drone_name}/get_status',
                self._get_status_callback,
                self.service_group,
            ),
            ServiceSpec(
                RealignLocalPosition,
                f'{self.drone_name}/realign_local_position',
                self._realign_local_position_callback,
                self.service_group,
            ),
            ServiceSpec(
                Reboot,
                f'{self.drone_name}/reboot',
                self._reboot_callback,
                self.service_group,
            ),
        )

    def _subscription_specs(self):
        return (
            SubscriptionSpec(
                TwistStamped,
                f'{self.drone_name}/cmd_vel',
                self._cmd_vel_changed,
                10,
                self.sub_group,
            ),
            SubscriptionSpec(
                FullState,
                f'{self.drone_name}/cmd_full_state',
                self._cmd_full_state_changed,
                10,
                self.sub_group,
            ),
        )

    def _action_server_specs(self):
        return (
            ActionServerSpec(
                'upload_action_server',
                UploadTrajectory,
                f'{self.drone_name}/upload_trajectory',
                self._upload_trajectory_callback,
                self._upload_trajectory_goal_callback,
                self._upload_trajectory_cancel_callback,
                self._action_group,
            ),
            ActionServerSpec(
                'takeoff_action_server',
                Takeoff,
                f'{self.drone_name}/takeoff',
                self._takeoff_action_callback,
                self._takeoff_goal_callback,
                self._takeoff_cancel_callback,
                self._action_group,
            ),
            ActionServerSpec(
                'go_to_action_server',
                GoTo,
                f'{self.drone_name}/go_to',
                self._go_to_action_callback,
                self._go_to_goal_callback,
                self._go_to_cancel_callback,
                self._action_group,
            ),
            ActionServerSpec(
                'execute_action_server',
                ExecuteTrajectory,
                f'{self.drone_name}/execute_trajectory',
                self._execute_trajectory_action_callback,
                self._execute_trajectory_goal_callback,
                self._execute_trajectory_cancel_callback,
                self._action_group,
            ),
            ActionServerSpec(
                'land_action_server',
                Land,
                f'{self.drone_name}/land',
                self._land_action_callback,
                self._land_goal_callback,
                self._land_cancel_callback,
                self._action_group,
            ),
            ActionServerSpec(
                'teleop_control_action_server',
                EnableController,
                f'{self.drone_name}/teleop_control',
                self._teleop_control_callback,
                self._teleop_control_goal_callback,
                self._teleop_control_cancel_callback,
                self._action_group,
            ),
        )

    def _destroy_registered_entities(self, label, entities, destroy_method):
        for entity in reversed(list(entities)):
            try:
                destroy_method(entity)
            except Exception as exc:  # noqa: B902
                self._logger.warning(f'[{self.drone_name}] Failed to destroy {label}: {exc}')
        entities.clear()

    def cleanup(self):
        """Stop backend activity and release ROS entities owned by this manager."""
        if not hasattr(self, '_cleanup_lock'):
            return
        with self._cleanup_lock:
            if self._cleanup_complete:
                return
            self._cleanup_complete = True

            try:
                self.shut_down()
            except Exception as exc:  # noqa: B902
                if hasattr(self, '_logger'):
                    self._logger.warning(f'[{self.drone_name}] Manager shutdown during cleanup failed: {exc}')

            watchdog_timer = getattr(self, '_watchdog_timer', None)
            if watchdog_timer is not None:
                try:
                    watchdog_timer.cancel()
                except Exception:  # noqa: B902
                    pass

            self._destroy_registered_entities(
                'action server',
                self._ros_action_servers,
                lambda action_server: action_server.destroy(),
            )
            self._destroy_registered_entities(
                'timer',
                self._ros_timers,
                lambda timer: self.node.destroy_timer(timer),
            )
            self._destroy_registered_entities(
                'subscription',
                self._ros_subscriptions,
                lambda subscription: self.node.destroy_subscription(subscription),
            )
            self._destroy_registered_entities(
                'service',
                self._ros_services,
                lambda service: self.node.destroy_service(service),
            )

            # Tombstone the broadcaster references before destroying their
            # underlying publishers.  Any concurrent _on_pose call that already
            # passed the `if self._tfbr:` guard holds a reference to a
            # broadcaster whose publisher is still valid at this point.  After
            # this assignment, new _on_pose calls see None and skip the send.
            self._tfbr = None
            self._static_tfbr = None

            self._destroy_registered_entities(
                'publisher',
                self._ros_publishers,
                lambda publisher: self.node.destroy_publisher(publisher),
            )

    def _resolve_group_mask(self, ros_params):
        value = ros_params.get('group_mask') if isinstance(ros_params, dict) else None
        if value is None:
            try:
                value = self.node.get_parameter('group_mask').value
            except Exception:  # noqa: B902
                value = None
        try:
            if value is None:
                return 0xFF
            if isinstance(value, str):
                return int(value, 0)
            return int(value)
        except (TypeError, ValueError):
            self._logger.warning('Invalid group_mask %r; defaulting to 0xFF' % (value,))
            return 0xFF

    # ===== Watchdog =====

    def _watchdog_not_ready_reason(self):
        """Return a warning string if the drone is not yet fully operational, else None.

        Override in subclasses to add backend-specific readiness checks (e.g. HLC
        gate).  The base implementation only checks the connection flag.
        """
        if not self.connected:
            return (
                f'[{self.drone_name}] Not connected —'
                ' services and actions are unavailable until the link comes up.'
            )
        return None

    def _watchdog_tick(self):
        reason = self._watchdog_not_ready_reason()
        if reason is None:
            self._watchdog_timer.cancel()
        else:
            self._logger.warning(reason)

    def _restart_watchdog(self):
        """Restart the watchdog after a disconnect so it begins logging again."""
        self._watchdog_timer.reset()

    @abstractmethod
    def start_up(self):
        pass

    @abstractmethod
    def shut_down(self):
        pass

    @abstractmethod
    def create_instance(self):
        pass

    @abstractmethod
    def _cmd_vel_changed(self, msg):
        pass

    @abstractmethod
    def _cmd_full_state_changed(self, msg):
        pass

    @abstractmethod
    def _emergency_callback(self, request, response):
        pass

    @abstractmethod
    def _set_led_color_callback(self, request, response):
        pass

    @abstractmethod
    def _arm_callback(self, request, response):
        pass

    def _get_status_callback(self, request, response):
        requested_drone_id = request.drone_id.strip()
        if requested_drone_id and requested_drone_id != self.drone_name:
            response.success = False
            return response

        status = self.drone_state.status
        if status is None:
            response.success = False
            return response

        # A latched (TRANSIENT_LOCAL) status from a dead publisher must not
        # masquerade as live state: fail when the cached status is stale.
        received_mono = self._status_received_mono
        if (
            received_mono is None
            or time.monotonic() - received_mono > self._STATUS_STALENESS_S
        ):
            self._logger.warning(
                f'[{self.drone_name}] get_status rejected: cached status is '
                f'older than {self._STATUS_STALENESS_S:.1f}s.'
            )
            response.status = status
            response.success = False
            return response

        response.status = status
        response.success = True
        return response

    def _realign_local_position_callback(self, request, response):
        p = request.pose.pose.position
        self._logger.info(
            f'[{self.drone_name}] realign_local_position at '
            f'({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) — '
            'no estimator reset needed for this backend.'
        )
        response.success = True
        return response

    def _reboot_callback(self, request, response):
        response.success = True
        response.message = f'reboot ignored for backend {self.drone_type}'
        self._logger.info(f'[{self.drone_name}] {response.message}')
        return response

    @abstractmethod
    def _takeoff_callback(self, goal_handle):
        pass

    @abstractmethod
    def _land_callback(self, goal_handle):
        pass

    @abstractmethod
    def _go_to_callback(
        self,
        goal_handle,
    ):
        pass

    @abstractmethod
    def _upload_trajectory_callback(self, goal_handle):
        pass

    @abstractmethod
    def _execute_trajectory_callback(self, goal_handle):
        pass

    @abstractmethod
    def _cancel_trajectory_callback(self, goal_handle):
        pass

    def _upload_trajectory_goal_callback(self, goal_request):
        return self._validate_upload_trajectory_goal(goal_request)

    def _takeoff_goal_callback(self, goal_request):
        self._logger.info(f'[{self.drone_name}] Takeoff goal callback ...')
        reject = self._reject_goal_for_group_mask(goal_request, 'takeoff')
        if reject is not None:
            return reject
        reject = self._reject_goal_for_active_teleop('takeoff')
        if reject is not None:
            return reject
        self._logger.info(f'[{self.drone_name}] Takeoff goal validation check ...')
        return self._validate_takeoff_goal(goal_request)

    def _land_goal_callback(self, goal_request):
        reject = self._reject_goal_for_group_mask(goal_request, 'land')
        if reject is not None:
            return reject
        reject = self._reject_goal_for_active_teleop('land')
        if reject is not None:
            return reject
        return self._validate_land_goal(goal_request)

    def _go_to_goal_callback(self, goal_request):
        reject = self._reject_goal_for_group_mask(goal_request, 'go_to')
        if reject is not None:
            return reject
        reject = self._reject_goal_for_active_teleop('go_to')
        if reject is not None:
            return reject
        return self._validate_go_to_goal(goal_request)

    def _execute_trajectory_goal_callback(self, goal_request):
        reject = self._reject_goal_for_group_mask(goal_request, 'execute_trajectory')
        if reject is not None:
            return reject
        reject = self._reject_goal_for_active_teleop('execute_trajectory')
        if reject is not None:
            return reject
        return self._validate_execute_trajectory_goal(goal_request)

    def _upload_trajectory_cancel_callback(self, _cancel_request):
        return CancelResponse.REJECT

    def _takeoff_cancel_callback(self, _cancel_request):
        return CancelResponse.REJECT

    def _land_cancel_callback(self, _cancel_request):
        return CancelResponse.REJECT

    def _go_to_cancel_callback(self, _cancel_request):
        return CancelResponse.REJECT

    def _execute_trajectory_cancel_callback(self, cancel_request):
        return self._cancel_trajectory_callback(cancel_request)

    def _teleop_control_cancel_callback(self, _cancel_request):
        self._logger.info(f'[{self.drone_name}] teleop_control cancel accepted.')
        return CancelResponse.ACCEPT

    def _teleop_control_goal_callback(self, goal_request):
        controller_name = str(getattr(goal_request, 'controller_name', '') or '').strip()
        if controller_name and controller_name != 'teleop_control':
            self._logger.warning(
                f'[{self.drone_name}] Rejecting teleop_control: controller_name '
                f"must be empty or 'teleop_control', got {controller_name!r}."
            )
            return GoalResponse.REJECT
        params_yaml = str(getattr(goal_request, 'params_yaml', '') or '').strip()
        if params_yaml:
            self._logger.warning(
                f'[{self.drone_name}] Rejecting teleop_control: params_yaml is '
                'not supported by this manager.'
            )
            return GoalResponse.REJECT
        readiness = self._teleop_control_readiness_error()
        if readiness is not None:
            _code, message = self._teleop_control_result_code_message(readiness)
            self._logger.warning(f'[{self.drone_name}] Rejecting teleop_control: {message}.')
            return GoalResponse.REJECT
        if self._teleop_command_age() > self._teleop_stale_after:
            self._logger.warning(
                f'[{self.drone_name}] Rejecting teleop_control: cmd_vel stream is stale.'
            )
            return GoalResponse.REJECT
        if self._motion_action_active:
            self._logger.warning(
                f'[{self.drone_name}] Rejecting teleop_control: another motion '
                'action is active.'
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _teleop_control_readiness_error(self):
        return self._command_readiness_error()

    def _teleop_control_result_code_message(self, readiness_error):
        if isinstance(readiness_error, tuple):
            return int(readiness_error[0]), str(readiness_error[1])
        message = str(readiness_error)
        result_code = (
            self._RC_CRASHED
            if 'crashed' in message.lower()
            else self._RC_STATUS_FAILURE
        )
        return result_code, message

    def _teleop_control_result(self, return_code, message):
        result = EnableController.Result()
        result.return_code = int(return_code)
        result.message = str(message or '')
        result.active_controller = 'teleop_control' if self._allow_teleop else ''
        return result

    def _teleop_control_feedback(self, message=''):
        feedback = EnableController.Feedback()
        age = self._teleop_command_age()
        feedback.controller_name = 'teleop_control'
        feedback.input_age_sec = float(age)
        feedback.controller_active = bool(self._allow_teleop and age < self._teleop_stale_after)
        feedback.status_text = str(message or '')
        feedback.status_yaml = ''
        return feedback

    def _start_teleop_control(self):
        return None

    def _send_teleop_exit_command(self):
        return None

    def _teleop_control_callback(self, goal_handle):
        result = self._teleop_control_result(self._RC_SUCCESS, 'teleop released')
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            return result

        self._start_teleop_control()
        self._allow_teleop = True
        rate_s = 1.0 / self._TELEOP_FEEDBACK_HZ
        try:
            while rclpy.ok():
                age = self._teleop_command_age()
                goal_handle.publish_feedback(self._teleop_control_feedback('teleop active'))

                if goal_handle.is_cancel_requested:
                    self._allow_teleop = False
                    self._send_teleop_exit_command()
                    goal_handle.canceled()
                    return result

                if age > self._teleop_stale_after:
                    self._allow_teleop = False
                    self._send_teleop_exit_command()
                    result = self._teleop_control_result(
                        self._RC_TIMEOUT,
                        'cmd_vel stream timed out',
                    )
                    goal_handle.abort()
                    return result

                readiness = self._teleop_control_readiness_error()
                if readiness is not None:
                    self._allow_teleop = False
                    self._send_teleop_exit_command()
                    return_code, message = self._teleop_control_result_code_message(readiness)
                    result = self._teleop_control_result(return_code, message)
                    goal_handle.abort()
                    return result

                time.sleep(rate_s)

            self._allow_teleop = False
            self._send_teleop_exit_command()
            result = self._teleop_control_result(self._RC_STATUS_FAILURE, 'ROS shutdown')
            goal_handle.abort()
            return result
        finally:
            self._allow_teleop = False

    def _validate_upload_trajectory_goal(self, _goal_request):
        return GoalResponse.ACCEPT

    def _validate_takeoff_goal(self, _goal_request):
        return GoalResponse.ACCEPT

    def _validate_land_goal(self, _goal_request):
        return GoalResponse.ACCEPT

    def _validate_go_to_goal(self, goal_request):
        try:
            self._go_to_pose_for_relative_frame(goal_request)
        except ValueError as exc:
            self._logger.warning(f'[{self.drone_name}] Rejecting go_to: {exc}')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _validate_execute_trajectory_goal(self, _goal_request):
        return GoalResponse.ACCEPT

    def _reject_goal_for_group_mask(self, goal_request, action_name):
        mask = goal_request.group_mask
        if mask != 0 and not (mask & self.group_mask):
            self._logger.warning(
                f'[{self.drone_name}] Rejecting {action_name} goal due to group '
                'mask mismatch:'
                f' goal_mask=0x{mask:02X} drone_mask=0x{self.group_mask:02X}'
            )
            return GoalResponse.REJECT
        return None

    def _reject_goal_for_active_teleop(self, action_name):
        if self._allow_teleop:
            self._logger.warning(f'[{self.drone_name}] Rejecting {action_name}: teleop is active.')
            return GoalResponse.REJECT
        return None

    def _current_odom_position_yaw(self):
        odom = self.drone_state.odom
        if odom is None:
            return None
        pose = odom.pose.pose
        quat = pose.orientation
        _, _, yaw = euler_from_quaternion_xyzw([quat.x, quat.y, quat.z, quat.w])
        return (
            (
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ),
            float(yaw),
        )

    def _target_state_signature(self, msg):
        pose = msg.pose
        twist = msg.twist
        values = (
            msg.header.frame_id,
            msg.child_frame_id,
            int(getattr(msg, 'valid_mask', 0)),
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        )
        signature = []
        for value in values:
            if isinstance(value, str):
                signature.append(value)
            else:
                signature.append(round(float(value), 4))
        return tuple(signature)

    def _publish_target_state_msg(self, msg):
        if self._target_state_pub is None:
            return False
        signature = self._target_state_signature(msg)
        if signature == self._last_target_state_signature:
            return False
        self._last_target_state_signature = signature
        self._target_state_pub.publish(msg)
        return True

    def _publish_target_state_pose(self, position, yaw):
        msg = FullState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_tf_name
        msg.child_frame_id = self.drone_name
        msg.valid_mask = (
            FullState.VALID_POSITION
            | FullState.VALID_ORIENTATION
        )
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        qx, qy, qz, qw = quaternion_from_euler_xyzw(0.0, 0.0, float(yaw))
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return self._publish_target_state_msg(msg)

    def _publish_action_target_state(self, action_name, request):
        try:
            if action_name == 'go_to':
                return self._publish_target_state_pose(
                    self._resolve_expected_go_to_position(request),
                    self._resolve_expected_go_to_yaw(request),
                )

            current = self._current_odom_position_yaw()
            if current is None:
                return False
            position, yaw = current

            if action_name == 'takeoff':
                return self._publish_target_state_pose(
                    (position[0], position[1], float(request.height)),
                    yaw,
                )
            if action_name == 'land':
                return self._publish_target_state_pose(
                    (position[0], position[1], float(request.height)),
                    yaw,
                )
            if (
                action_name == 'execute_trajectory'
                and hasattr(self, '_resolve_expected_trajectory_endpoint')
            ):
                endpoint = self._resolve_expected_trajectory_endpoint(
                    int(request.trajectory_id),
                    bool(request.relative),
                    bool(request.reversed),
                )
                if endpoint is not None:
                    return self._publish_target_state_pose(endpoint, yaw)
        except (AttributeError, TypeError, ValueError) as exc:
            self._logger.warning(f'[{self.drone_name}] Could not publish {action_name} target_state: {exc}')
        return False

    async def _run_motion_action(self, action_name, callback, goal_handle):
        self._logger.info(f'[{self.drone_name}] Starting {action_name} action.')
        self._motion_action_active = True
        try:
            result = callback(goal_handle)
            if inspect.isawaitable(result):
                return await result
            return result
        finally:
            self._motion_action_active = False

    async def _takeoff_action_callback(self, goal_handle):
        return await self._run_motion_action('takeoff', self._takeoff_callback, goal_handle)

    async def _land_action_callback(self, goal_handle):
        return await self._run_motion_action('land', self._land_callback, goal_handle)

    async def _go_to_action_callback(self, goal_handle):
        return await self._run_motion_action('go_to', self._go_to_callback, goal_handle)

    async def _execute_trajectory_action_callback(self, goal_handle):
        return await self._run_motion_action(
            'execute_trajectory',
            self._execute_trajectory_callback,
            goal_handle,
        )

    def _validate_teleop_params(self):
        if self._teleop_stale_after <= 0.0:
            self._logger.warning(
                f'[{self.drone_name}] teleop_stale_after must be > 0 '
                f'(got {self._teleop_stale_after}); defaulting to 0.5.'
            )
            self._teleop_stale_after = 0.5
        if self._teleop_min_height < 0.0:
            self._logger.warning(
                f'[{self.drone_name}] teleop_min_height must be >= 0 '
                f'(got {self._teleop_min_height}); clamping to 0.0.'
            )
            self._teleop_min_height = 0.0
        if self._teleop_max_height <= 0.0:
            self._logger.warning(
                f'[{self.drone_name}] teleop_max_height must be > 0 '
                f'(got {self._teleop_max_height}); defaulting to 2.5.'
            )
            self._teleop_max_height = 2.5
        if self._teleop_max_height <= self._teleop_min_height:
            self._logger.warning(
                f'[{self.drone_name}] teleop_max_height ({self._teleop_max_height}) must be '
                f'> teleop_min_height ({self._teleop_min_height}); '
                'vertical teleop velocity will be clamped to zero.'
            )

    def _teleop_command_age(self):
        if self._teleop_last_seen is None:
            return float('inf')
        return max(0.0, time.monotonic() - self._teleop_last_seen)

    def _status_flag(self, flag):
        status = self.drone_state.status
        if status is None:
            return False
        return bool(int(status.status_flags) & flag)

    def _odom_is_fresh(self, max_age_sec=None):
        odom = self.drone_state.odom
        if odom is None:
            return False
        max_age = self._teleop_stale_after if max_age_sec is None else float(max_age_sec)
        try:
            stamp = odom.header.stamp
            now = self.node.get_clock().now().to_msg()
            age = (now.sec - stamp.sec) + (now.nanosec - stamp.nanosec) * 1e-9
        except Exception:  # noqa: B902
            return True
        return age <= max_age

    def _apply_teleop_height_limits(self, vz: float) -> float:
        height_range = self._teleop_max_height - self._teleop_min_height
        if height_range <= 0.0:
            return 0.0

        odom = self.drone_state.odom
        if odom is None:
            return 0.0

        z = odom.pose.pose.position.z
        if vz > 0.0 and z >= self._teleop_max_height:
            return 0.0
        if vz < 0.0 and z <= self._teleop_min_height:
            return 0.0

        slowdown = self._teleop_height_slowdown_fraction * height_range
        if slowdown <= 0.0:
            return vz

        if vz > 0.0:
            distance_to_ceiling = self._teleop_max_height - z
            if distance_to_ceiling < slowdown:
                return vz * max(0.0, distance_to_ceiling / slowdown)
        elif vz < 0.0:
            distance_to_floor = z - self._teleop_min_height
            if distance_to_floor < slowdown:
                return vz * max(0.0, distance_to_floor / slowdown)

        return vz

    # Frame constants mirror GoTo.Goal for use without importing the message type.
    _GOTO_FRAME_ABSOLUTE = 0
    _GOTO_FRAME_RELATIVE_MAP = 1
    _GOTO_FRAME_RELATIVE_BODY = 2

    def _go_to_pose_for_relative_frame(self, request):
        if request.frame == self._GOTO_FRAME_ABSOLUTE:
            return None

        odom = self.drone_state.odom
        if odom is None:
            raise ValueError('relative go_to requires odometry')
        return odom.pose.pose

    def _resolve_expected_go_to_position(self, request):
        """Resolve a GoTo goal into an absolute ENU target position.

        FRAME_ABSOLUTE:      goal fields are an absolute ENU target; returned as-is.
        FRAME_RELATIVE_MAP:  offset added directly in ENU/map frame.
        FRAME_RELATIVE_BODY: offset rotated by current heading (x=forward, y=left).
        """
        dx = float(request.goal.x)
        dy = float(request.goal.y)
        dz = float(request.goal.z)

        if request.frame == self._GOTO_FRAME_ABSOLUTE:
            return dx, dy, dz

        pose = self._go_to_pose_for_relative_frame(request)
        origin = pose.position

        if request.frame == self._GOTO_FRAME_RELATIVE_BODY:
            quat = pose.orientation
            _, _, current_yaw = euler_from_quaternion_xyzw([quat.x, quat.y, quat.z, quat.w])
            return (
                float(origin.x) + dx * math.cos(current_yaw) - dy * math.sin(current_yaw),
                float(origin.y) + dx * math.sin(current_yaw) + dy * math.cos(current_yaw),
                float(origin.z) + dz,
            )

        # FRAME_RELATIVE_MAP: add offset in ENU without rotation.
        return (
            float(origin.x) + dx,
            float(origin.y) + dy,
            float(origin.z) + dz,
        )

    def _resolve_expected_go_to_yaw(self, request):
        """Resolve a GoTo yaw into an absolute ENU yaw."""
        yaw = float(request.yaw)
        if request.frame == self._GOTO_FRAME_ABSOLUTE:
            return yaw

        pose = self._go_to_pose_for_relative_frame(request)
        quat = pose.orientation
        _, _, current_yaw = euler_from_quaternion_xyzw([quat.x, quat.y, quat.z, quat.w])
        return current_yaw + yaw

    def _parse_xyz(self, origin):
        if origin is None:
            return [0.0, 0.0, 0.0]
        return self._parse_origin_vector(origin, 'xyz')

    def _parse_rpy(self, origin):
        if origin is None:
            return [0.0, 0.0, 0.0]
        return self._parse_origin_vector(origin, 'rpy')

    def _parse_origin_vector(self, origin, attribute: str):
        raw = origin.get(attribute, '0 0 0')
        values = raw.split()
        if len(values) != 3:
            raise ValueError(
                f"URDF origin attribute '{attribute}' must contain exactly 3 values; "
                f'got {len(values)} from {raw!r}'
            )
        try:
            return [float(v) for v in values]
        except ValueError as exc:
            raise ValueError(f"URDF origin attribute '{attribute}' contains a non-float value: {raw!r}") from exc

    def _parse_robot_description_joints(self):
        """Parse robot_description and return continuous joints."""
        try:
            root = ET.fromstring(self.ros_params['robot_description'])
        except ET.ParseError as e:
            self._logger.warning(f'[{self.drone_name}] Failed to parse robot_description URDF: {e}')
            return []

        continuous_joints = []
        fixed_tfs = []
        for j in root.findall('joint'):
            parent = j.find('parent')
            child = j.find('child')
            if parent is None or child is None:
                continue

            origin = j.find('origin')
            joint_type = j.get('type')

            if joint_type == 'fixed':
                xyz = self._parse_xyz(origin)
                rpy = self._parse_rpy(origin)
                qx, qy, qz, qw = quaternion_from_euler_xyzw(*rpy)

                tf = TransformStamped()
                tf.header.stamp = Time().to_msg()
                tf.header.frame_id = parent.get('link')
                tf.child_frame_id = child.get('link')
                tf.transform.translation.x = xyz[0]
                tf.transform.translation.y = xyz[1]
                tf.transform.translation.z = xyz[2]
                tf.transform.rotation.x = qx
                tf.transform.rotation.y = qy
                tf.transform.rotation.z = qz
                tf.transform.rotation.w = qw
                fixed_tfs.append(tf)
                continue

            if joint_type != 'continuous':
                continue

            axis = j.find('axis')
            xyz = self._parse_xyz(origin)
            ax = [float(v) for v in axis.get('xyz', '0 0 1').split()] if axis is not None else [0.0, 0.0, 1.0]
            continuous_joints.append(
                {
                    'parent': parent.get('link'),
                    'child': child.get('link'),
                    'xyz': xyz,
                    'axis': ax,
                }
            )

        if fixed_tfs:
            self._static_tfbr.sendTransform(fixed_tfs)

        return continuous_joints

    def _on_status(self, msg: DroneStatus):
        """Store and publish a pre-built DroneStatus message. Called by subclasses."""
        if self.drone_state.status is None:
            self._logger.info(f'[{self.drone_name}] Initial status: {msg}')

        self.drone_state.status = msg
        self._status_received_mono = time.monotonic()
        if self._status_pub:
            self._status_pub.publish(msg)

    def _on_pose(self, pose_stamped: PoseStamped, armed: bool, cmd_twist=None):
        """Publish odometry, broadcast base-link TF, and spin prop joints."""
        now = pose_stamped.header.stamp
        pose = pose_stamped.pose

        # Odometry
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.odom_tf_name
        odom.child_frame_id = self.drone_name
        odom.pose.pose = pose
        if cmd_twist is not None:
            odom.twist.twist = cmd_twist
        self.drone_state.odom = odom

        if self._odom_pub:
            self._odom_pub.publish(odom)

        if self._tfbr:
            # Base link: <drone>/odom → <drone>
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self.odom_tf_name
            t.child_frame_id = self.drone_name
            t.transform.translation.x = pose.position.x
            t.transform.translation.y = pose.position.y
            t.transform.translation.z = pose.position.z
            t.transform.rotation = pose.orientation

            if not self._continuous_joints:
                self._tfbr.sendTransform(t)
                return

            # Advance prop angle at ~20 rad/s when armed; freeze when disarmed
            if self._prop_last_stamp is not None:
                dt = (now.sec - self._prop_last_stamp.sec) + (now.nanosec - self._prop_last_stamp.nanosec) * 1e-9
                if armed and dt > 0:
                    self._prop_angle = (self._prop_angle + 20.0 * dt) % (2 * math.pi)
            self._prop_last_stamp = now

            # Prop joint TFs
            prop_tfs = []
            half = self._prop_angle / 2.0
            sin_half, cos_half = math.sin(half), math.cos(half)
            for jnt in self._continuous_joints:
                tc = TransformStamped()
                tc.header.stamp = now
                tc.header.frame_id = jnt['parent']
                tc.child_frame_id = jnt['child']
                tc.transform.translation.x = jnt['xyz'][0]
                tc.transform.translation.y = jnt['xyz'][1]
                tc.transform.translation.z = jnt['xyz'][2]
                ax = jnt['axis']
                tc.transform.rotation.x = ax[0] * sin_half
                tc.transform.rotation.y = ax[1] * sin_half
                tc.transform.rotation.z = ax[2] * sin_half
                tc.transform.rotation.w = cos_half
                prop_tfs.append(tc)

            self._tfbr.sendTransform([t] + prop_tfs)

    def _init_topics_and_services(self):
        self._logger.info(f'[{self.drone_name}] Initializing topics and services ...')

        # Transform, odometry, and status publishers
        # (used by _on_pose and _on_status)
        self._tfbr = self._create_tf_broadcaster(TransformBroadcaster)
        for spec in self._publisher_specs():
            publisher = self._create_publisher(
                spec.msg_type,
                spec.topic_name,
                spec.qos_profile,
            )
            setattr(self, spec.attribute_name, publisher)

        msg = String()
        msg.data = self.ros_params['robot_description']
        if self._robot_description_pub:
            self._robot_description_pub.publish(msg)

        for spec in self._service_specs():
            self._create_service(
                spec.srv_type,
                spec.service_name,
                spec.callback,
                callback_group=spec.callback_group,
            )

        # Command inputs are topic subscriptions; each callback receives the
        # incoming message.
        for spec in self._subscription_specs():
            self._create_subscription(
                spec.msg_type,
                spec.topic_name,
                spec.callback,
                spec.qos_profile,
                callback_group=spec.callback_group,
            )

        self._logger.info(f'[{self.drone_name}] Topics and services initialized!')

    def _init_action_servers(self):
        self._logger.info(f'[{self.drone_name}] Initializing action servers ...')

        # Keep upload on the same mutually-exclusive action group as motion
        # actions so a manager never accepts trajectory mutations mid-flight.
        for spec in self._action_server_specs():
            action_server = self._create_action_server(
                self.node,
                spec.action_type,
                spec.action_name,
                execute_callback=spec.execute_callback,
                goal_callback=spec.goal_callback,
                cancel_callback=spec.cancel_callback,
                callback_group=spec.callback_group,
            )
            setattr(self, spec.attribute_name, action_server)

        self._logger.info(f'[{self.drone_name}] Action servers initialized!')

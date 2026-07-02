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


# With assist from ChatGPT

import math
import signal
import sys
import time

try:
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QSlider, QCheckBox,
        QSpacerItem, QSizePolicy
    )
    from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
    from PySide6.QtCore import Qt, QPoint, QRect, QThread, QTimer
except ImportError as exc:
    raise SystemExit(
        'teleop_gui requires PySide6: python3 -m pip install PySide6\n'
        'See flexible_drones_tools/README.md for optional GUI dependencies.'
    ) from exc

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as RosDuration
from rclpy.node import Node

from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult

from flexible_drones_msgs.srv import Arm, EmergencyStop
from flexible_drones_msgs.action import EnableController, Takeoff, Land
from flexible_drones_msgs.msg import DroneStatus as Status

from geometry_msgs.msg import Quaternion, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from flexible_drones_tools.monitoring.cli_help import print_usage_if_requested

STATUS_PUBLISH_PERIOD_S = 1.0
STATUS_STALE_TIMEOUT_S = 5.0 * STATUS_PUBLISH_PERIOD_S
DEFAULT_WINDOW_X = 100
DEFAULT_WINDOW_Y = 100
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300


CLI_USAGE = """\
Usage:
  ros2 run flexible_drones_tools teleop_gui --ros-args \\
    -r __ns:=/drone1 -p window_x:=100 -p window_y:=100

Common examples:
  ros2 run flexible_drones_tools teleop_gui --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools teleop_gui --ros-args -r __ns:=/drone1 -p use_sim_time:=true
  ros2 run flexible_drones_tools teleop_gui --ros-args -r __ns:=/drone1 -p takeoff_height:=0.5

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.
  Set the initial window position with '-p window_x:=100 -p window_y:=100'.

Keyboard controls:
  Ctrl+A arm, Ctrl+D disarm, Ctrl+T takeoff, Ctrl+E enable/release teleop,
  Ctrl+L land, Ctrl+X controlled stop.
  Arrow keys adjust yaw rate and elevation; spacebar zeros yaw rate.
"""


def usage():
    txt = ('\nTeleop Control GUI:\n'
           '    Ctrl-a (or A) - Arm drone\n'
           '    Ctrl-d (or D) - Disarm drone\n'
           '    Ctrl-t (or T) - take off\n'
           '    Ctrl-e (or E) - enable/release teleop control\n'
           '    Ctrl-l (or L) - land\n'
           '    Ctrl-x (or X) - controlled stop: land, disarm; emergency_stop on error\n'
           '      left/right  - adjust yaw rate\n'
           '       spacebar   - zero yaw rate\n'
           '       up/down    - adjust elevation\n'
           '        h (or H)  - print usage\n\n'
           '     Also allows connection to physical joystick via ROS2 joy message\n'
           '                      Y - zero yaw rate\n'
           '          Left joystick - body velocity control\n'
           '         Right joystick - vertical velocity control\n'
           '       Left trigger - decrease yaw rate\n'
           '       Right trigger - increase yaw rate\n'
           '       Left button + A - take off\n'
           '       Left button + B - land\n'
           '       Right button + A - arm\n'
           '       Right button + Y - disarm\n'
           '       Right button + Left button + X - controlled stop\n\n')
    return txt


class TeleopPublisher(Node):
    def __init__(self):
        super().__init__('cmd_teleop_gui')

        print(f"Creating Teleop Control GUI for '{self.get_namespace()}'")
        print(usage())

        self.publisher_ = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self.cli_enable_teleop = ActionClient(self, EnableController, 'teleop_control')
        self.cli_takeoff = ActionClient(self, Takeoff, 'takeoff')
        self.cli_land = ActionClient(self, Land, 'land')
        self.cli_arm = self.create_client(Arm, 'arm')
        self.cli_estop = self.create_client(EmergencyStop, 'emergency_stop')

        self.status_msg = None
        self.status_time = None
        self.status_is_armed = False
        self.status_is_airborne = False
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.pose = None
        self.odom_linear_z = 0.0
        self.create_subscription(Status, 'status', self.status_callback, 10)
        self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.joy_msg = None

        self.declare_parameter('vx_scale', 1.0,
                               descriptor=ParameterDescriptor(
                                   description='Maximum forward velocity vx (m/s)'
                               ))
        self.declare_parameter('vy_scale', 1.0,
                               descriptor=ParameterDescriptor(
                                   description='Maximum sideways velocity vy (m/s)'
                               ))
        self.declare_parameter('yaw_scale', 0.5,
                               descriptor=ParameterDescriptor(
                                   description='Maximum yaw rate (rad/s)'
                               ))
        self.declare_parameter('vz_scale', 0.5,
                               descriptor=ParameterDescriptor(
                                   description='Maximum vertical velocity vz (m/s)'
                               ))
        self.declare_parameter('max_height', 2.5,
                               descriptor=ParameterDescriptor(
                                   description='Maximum height (m)'
                               ))
        self.declare_parameter('horizon_height', 0.50,
                               descriptor=ParameterDescriptor(
                                   description='Height for neutral horizon in attitude indicator (m)'
                               ))  # height for 0 attitude indicator
        self.declare_parameter('landing_height', 0.02,
                               descriptor=ParameterDescriptor(
                                   description='Target height for landing (m)'
                               ))
        self.declare_parameter('landed_height', 0.1,
                               descriptor=ParameterDescriptor(
                                   description='Odometry height below which the GUI treats the drone as landed (m)'
                               ))
        self.declare_parameter('landing_duration', 2.0,
                               descriptor=ParameterDescriptor(
                                   description='Duration for landing (seconds)'
                               ))
        self.declare_parameter('takeoff_height', 0.5,
                               descriptor=ParameterDescriptor(
                                   description='Target height for takeoff (m)'
                               ))
        self.declare_parameter('takeoff_duration', 2.0,
                               descriptor=ParameterDescriptor(
                                   description='Duration for takeoff (seconds)'
                               ))
        self.declare_parameter('arm_timeout_sec', 2.0,
                               descriptor=ParameterDescriptor(
                                   description='Timeout for arm/disarm service requests (seconds)'
                               ))
        self.declare_parameter('teleop_heartbeat_hz', 20.0,
                               descriptor=ParameterDescriptor(
                                   description='Minimum cmd_vel heartbeat rate while flying (Hz)'
                               ))
        self.declare_parameter('group_mask', 0,
                               descriptor=ParameterDescriptor(
                                   description='Drone group mask for takeoff/land/stop actions'
                               ))
        self.declare_parameter('pitch_scaling', 60.0,
                               descriptor=ParameterDescriptor(
                                   description='Pitch range for attitude indicator (degrees)'
                               ))  # Scaled for 60 degree pitch over window
        self.declare_parameter(
            'window_x',
            DEFAULT_WINDOW_X,
            descriptor=ParameterDescriptor(
                description='Initial window x position in screen pixels.'
            ))
        self.declare_parameter(
            'window_y',
            DEFAULT_WINDOW_Y,
            descriptor=ParameterDescriptor(
                description='Initial window y position in screen pixels.'
            ))

        # Cached values
        self.vx_scale = self.get_parameter('vx_scale').get_parameter_value().double_value
        self.vy_scale = self.get_parameter('vy_scale').get_parameter_value().double_value
        self.yaw_scale = self.get_parameter('yaw_scale').get_parameter_value().double_value
        self.vz_scale = self.get_parameter('vz_scale').get_parameter_value().double_value
        self.max_height = self.get_parameter('max_height').get_parameter_value().double_value
        self.horizon_height = self.get_parameter('horizon_height').get_parameter_value().double_value
        self.landing_duration = self.get_parameter('landing_duration').get_parameter_value().double_value
        self.landing_height = self.get_parameter('landing_height').get_parameter_value().double_value
        self.landed_height = self.get_parameter('landed_height').get_parameter_value().double_value
        self.takeoff_duration = self.get_parameter('takeoff_duration').get_parameter_value().double_value
        self.takeoff_height = self.get_parameter('takeoff_height').get_parameter_value().double_value
        self.arm_timeout_sec = self.get_parameter('arm_timeout_sec').get_parameter_value().double_value
        self.teleop_heartbeat_hz = self.get_parameter('teleop_heartbeat_hz').get_parameter_value().double_value
        self.group_mask = self.get_parameter('group_mask').get_parameter_value().integer_value
        self.pitch_scaling = self.get_parameter('pitch_scaling').get_parameter_value().double_value
        self.window_x = self.get_parameter('window_x').get_parameter_value().integer_value
        self.window_y = self.get_parameter('window_y').get_parameter_value().integer_value

        # Default to halfway between landing and takeoff as trigger for flying status and control floor
        self.flying_height = self.landing_height + 0.5 * (self.takeoff_height - self.landing_height)
        self.declare_parameter('flying_height', self.flying_height)
        self.flying_height = self.get_parameter('flying_height').get_parameter_value().double_value
        self.is_flying = False
        self.is_armed = False
        self.mode = None
        self.teleop_active = False
        self.teleop_pending = False
        self.teleop_goal_handle = None
        self.teleop_cancel_after_accept = False
        self.teleop_release_callbacks = []
        self.teleop_status_text = 'idle'
        self.teleop_frame_id = 'base_link'

        # Listen for parameter changes
        self.add_on_set_parameters_callback(self._on_param_change)

    def odom_callback(self, msg):
        self.pose = msg.pose.pose
        self.odom_linear_z = msg.twist.twist.linear.z

    def status_callback(self, msg):
        self.status_msg = msg
        self.status_time = time.monotonic()
        status_flags = int(msg.status_flags)
        armed = (status_flags & Status.STATUS_ARMED) != 0
        airborne = (status_flags & Status.STATUS_AIRBORNE) != 0
        self.status_is_armed = armed
        self.status_is_airborne = airborne
        if not armed and not airborne:
            self.is_flying = False
            self.is_armed = False

    def status_is_fresh(self):
        if self.status_time is None:
            return False
        return time.monotonic() - self.status_time <= STATUS_STALE_TIMEOUT_S

    def joy_callback(self, msg):
        self.joy_msg = msg

    def _on_param_change(self, params):
        for param in params:
            if param.name == 'vx_scale':
                self.vx_scale = param.value
            elif param.name == 'vy_scale':
                self.vy_scale = param.value
            elif param.name == 'yaw_scale':
                self.yaw_scale = param.value
            elif param.name == 'vz_scale':
                self.vz_scale = param.value
            elif param.name == 'max_height':
                self.max_height = param.value
            elif param.name == 'horizon_height':
                self.horizon_height = param.value
            elif param.name == 'landing_height':
                self.landing_height = param.value
            elif param.name == 'landed_height':
                self.landed_height = param.value
            elif param.name == 'landing_duration':
                self.landing_duration = param.value
            elif param.name == 'takeoff_height':
                self.takeoff_height = param.value
            elif param.name == 'takeoff_duration':
                self.takeoff_duration = param.value
            elif param.name == 'flying_height':
                self.flying_height = param.value
            elif param.name == 'pitch_scaling':
                self.pitch_scaling = param.value
            elif param.name == 'arm_timeout_sec':
                self.arm_timeout_sec = param.value
            elif param.name == 'teleop_heartbeat_hz':
                self.teleop_heartbeat_hz = param.value
            elif param.name == 'group_mask':
                self.group_mask = param.value

        return SetParametersResult(successful=True)

    def send_teleop_command(self, vx, vy, yawrate, vz):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.teleop_frame_id
        msg.twist.linear.x = vx * self.vx_scale
        msg.twist.linear.y = vy * self.vy_scale
        msg.twist.linear.z = vz * self.vz_scale
        msg.twist.angular.z = yawrate * self.yaw_scale
        self.publisher_.publish(msg)

    def set_teleop_world_frame(self, enabled):
        self.teleop_frame_id = 'map' if enabled else 'base_link'
        frame_name = 'world' if enabled else 'body'
        self.get_logger().info(f'Teleop velocity frame set to {frame_name} ({self.teleop_frame_id})')
        # self.get_logger().info(
        #     f'Published TwistStamped: vx={msg.twist.linear.x:.3f}, '
        #     f'vy={msg.twist.linear.y:.3f}, vz={msg.twist.linear.z:.3f}, '
        #     f'yawrate={msg.twist.angular.z:.2f}')

    def send_arm(self, armed=True, done_callback=None, allow_while_flying=False):
        if not self.cli_arm.service_is_ready():
            self.get_logger().warning('Arm service is not available!')
            return False
        else:
            if self.is_flying and not allow_while_flying:
                self.get_logger().warning('Arm/disarm is requested but already is_flying=True!')
                return False
            else:
                command = 'arm' if armed else 'disarm'
                self.get_logger().info(
                    f'Requesting {command} from service with timeout {self.arm_timeout_sec} s...'
                )
                request = Arm.Request()
                request.arm = armed
                request.timeout_sec = self.arm_timeout_sec

                future = self.cli_arm.call_async(request)
                future.add_done_callback(
                    lambda future: self.arm_callback(future, armed, done_callback)
                )
                return True

    def send_enable_teleop(self, done_callback=None):
        if self.teleop_active or self.teleop_pending:
            self.get_logger().warning('Teleop control is already active or pending.')
            return False
        if not self.is_flying:
            self.get_logger().warning('Enable teleop requested but not flying.')
            return False
        if not self.cli_enable_teleop.server_is_ready():
            self.get_logger().warning('teleop_control action is not available.')
            return False

        self.teleop_pending = True
        self.teleop_cancel_after_accept = False
        self.teleop_status_text = 'enabling'
        self.get_logger().info('Requesting teleop control ...')
        future = self.cli_enable_teleop.send_goal_async(
            EnableController.Goal(controller_name='teleop_control'),
            feedback_callback=self.teleop_feedback_callback,
        )
        future.add_done_callback(
            lambda future: self.teleop_goal_response_callback(future, done_callback)
        )
        return True

    def teleop_goal_response_callback(self, future, done_callback=None):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warning('teleop_control goal was rejected.')
                self.teleop_pending = False
                self.teleop_active = False
                self.teleop_goal_handle = None
                self.teleop_cancel_after_accept = False
                self.teleop_status_text = 'rejected'
                self._drain_teleop_release_callbacks()
                if done_callback is not None:
                    done_callback()
                return

            self.get_logger().info('Teleop control is active.')
            self.teleop_goal_handle = goal_handle
            self.teleop_pending = False
            self.teleop_active = True
            self.teleop_status_text = 'active'
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.teleop_result_callback)
            if self.teleop_cancel_after_accept:
                self.teleop_cancel_after_accept = False
                self.cancel_teleop()
                return
        except Exception as e:
            self.get_logger().error(f'teleop_control goal failed: {e}')
            self.teleop_pending = False
            self.teleop_active = False
            self.teleop_goal_handle = None
            self.teleop_cancel_after_accept = False
            self.teleop_status_text = 'failed'
            self._drain_teleop_release_callbacks()

        if done_callback is not None:
            done_callback()

    def teleop_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        active = 'active' if feedback.controller_active else 'stale'
        self.teleop_status_text = f'{active}, cmd_vel age={feedback.input_age_sec:.2f}s'

    def cancel_teleop(self, done_callback=None):
        if self.teleop_goal_handle is None:
            if self.teleop_pending:
                self.get_logger().info('Teleop is still pending; will cancel after acceptance.')
                self.teleop_cancel_after_accept = True
                self.teleop_status_text = 'releasing'
                if done_callback is not None:
                    self.teleop_release_callbacks.append(done_callback)
                return True
            self.teleop_cancel_after_accept = False
            self.teleop_active = False
            self.teleop_status_text = 'idle'
            if done_callback is not None:
                done_callback()
            return False

        self.get_logger().info('Canceling teleop control ...')
        self.teleop_cancel_after_accept = False
        self.teleop_status_text = 'releasing'
        if done_callback is not None:
            self.teleop_release_callbacks.append(done_callback)
        future = self.teleop_goal_handle.cancel_goal_async()
        future.add_done_callback(self.teleop_cancel_callback)
        return True

    def teleop_cancel_callback(self, future):
        try:
            response = future.result()
            if len(response.goals_canceling) > 0:
                self.get_logger().info('Teleop cancel accepted.')
            else:
                self.get_logger().warning('Teleop cancel returned no canceling goals.')
        except Exception as e:
            self.get_logger().error(f'Teleop cancel failed: {e}')

    def _drain_teleop_release_callbacks(self):
        callbacks = self.teleop_release_callbacks
        self.teleop_release_callbacks = []
        for callback in callbacks:
            callback()

    def teleop_result_callback(self, future):
        try:
            result = future.result().result
            if result.return_code == 0:
                self.get_logger().info(f'Teleop control finished: {result.message}')
            else:
                self.get_logger().warning(
                    f'Teleop control finished with return_code={result.return_code}: '
                    f'{result.message}'
                )
        except Exception as e:
            self.get_logger().error(f'Teleop result failed: {e}')

        self.teleop_goal_handle = None
        self.teleop_pending = False
        self.teleop_active = False
        self.teleop_cancel_after_accept = False
        self.teleop_status_text = 'idle'
        # Release callbacks are sequenced work that must run only after teleop has fully ended.
        self._drain_teleop_release_callbacks()

    def arm_callback(self, future, armed, done_callback=None):
        success = False
        command = 'arm' if armed else 'disarm'
        try:
            response = future.result()
            if response is None:
                self.get_logger().warning(f'{command.capitalize()} service returned no response.')
            elif response.success:
                self.get_logger().info(f'{command.capitalize()} service succeeded.')
                self.is_armed = armed
                success = True
            else:
                self.get_logger().warning(f'{command.capitalize()} service returned failure.')
        except Exception as e:
            self.get_logger().error(f'{command.capitalize()} service call failed: {e}')

        if done_callback is not None:
            done_callback(success)

    def send_takeoff(self, ui_callback):
        if self.cli_takeoff is None or not self.cli_takeoff.server_is_ready():
            self.get_logger().warning('Takeoff action is not available.')
            self.mode = 'takeoff'
            return True
        else:
            if self.is_flying:
                self.get_logger().warning('Takeoff requested but is_flying=True!')
                return False
            else:
                self.get_logger().info(f'Requesting takeoff action for {self.takeoff_duration} s...')
                goal = Takeoff.Goal()
                goal.group_mask = self.group_mask
                goal.height = self.takeoff_height
                goal.duration = RosDuration(seconds=self.takeoff_duration).to_msg()

                future = self.cli_takeoff.send_goal_async(goal)
                future.add_done_callback(lambda future: self.takeoff_goal_response_callback(future, ui_callback))
                return False

    def takeoff_goal_response_callback(self, future, ui_callback):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warning('Takeoff goal was rejected.')
                self.is_flying = False
                return
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(lambda future: self.takeoff_callback(future, ui_callback))
        except Exception as e:
            self.get_logger().error(f'Takeoff goal failed: {e}')

    def takeoff_callback(self, future, ui_callback):
        try:
            result = future.result().result  # raises if error occurred
            if result.return_code == 0:
                self.get_logger().info('Takeoff action succeeded.')
                self.mode = 'takeoff'
                ui_callback()  # Set UI to match
            else:
                self.get_logger().warning(f'Takeoff action failed with return_code={result.return_code}.')
                self.is_flying = False
        except Exception as e:
            self.get_logger().error(f'Takeoff action failed: {e}')

    def send_land(self, ui_callback):
        if self.cli_land is None or not self.cli_land.server_is_ready():
            self.get_logger().warning('Land action is not available.')
            self.mode = 'land'
            return True
        else:
            if not self.is_flying:
                self.get_logger().warning('Land requested but not flying!')
                return False
            else:
                self.get_logger().info('Requesting land action ...')
                goal = Land.Goal()
                goal.group_mask = self.group_mask
                goal.height = self.landing_height
                goal.duration = RosDuration(seconds=self.landing_duration).to_msg()

                future = self.cli_land.send_goal_async(goal)
                future.add_done_callback(lambda future: self.land_goal_response_callback(future, ui_callback))
                return False

    def land_goal_response_callback(self, future, ui_callback):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warning('Land goal was rejected.')
                return
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(lambda future: self.landing_callback(future, ui_callback))
        except Exception as e:
            self.get_logger().error(f'Land goal failed: {e}')

    def landing_callback(self, future, ui_callback):
        try:
            result = future.result().result  # raises if error occurred
            if result.return_code == 0:
                self.get_logger().info('Landing action succeeded')
                self.mode = 'land'
                ui_callback()  # Set UI to match
            else:
                self.get_logger().warning(f'Landing action failed with return_code={result.return_code}.')
        except Exception as e:
            self.get_logger().error(f'Landing action failed: {e}')

    def send_emergency_stop(self, ui_callback):
        self.get_logger().warning(
            'Ctrl-X requested controlled stop: land, then disarm. '
            'Using emergency_stop only if that fails.'
        )

        if not self.is_flying:
            self.get_logger().info('Drone is not flying; requesting disarm.')
            if not self.send_arm(
                False,
                done_callback=lambda success: self.controlled_stop_disarm_callback(
                    success, ui_callback
                ),
            ):
                self.call_emergency_stop_service('Disarm request could not be sent.', ui_callback)
            return False

        if self.cli_land is None or not self.cli_land.server_is_ready():
            self.call_emergency_stop_service('Land action is not available.', ui_callback)
            return False

        self.get_logger().info('Requesting controlled-stop land action ...')
        goal = Land.Goal()
        goal.group_mask = self.group_mask
        goal.height = self.landing_height
        goal.duration = RosDuration(seconds=self.landing_duration).to_msg()

        future = self.cli_land.send_goal_async(goal)
        future.add_done_callback(
            lambda future: self.controlled_stop_land_goal_response_callback(future, ui_callback)
        )
        return False

    def controlled_stop_land_goal_response_callback(self, future, ui_callback):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.call_emergency_stop_service('Controlled-stop land goal was rejected.', ui_callback)
                return
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(
                lambda future: self.controlled_stop_landing_callback(future, ui_callback)
            )
        except Exception as e:
            self.call_emergency_stop_service(f'Controlled-stop land goal failed: {e}', ui_callback)

    def controlled_stop_landing_callback(self, future, ui_callback):
        try:
            result = future.result().result
            if result.return_code != 0:
                self.call_emergency_stop_service(
                    f'Controlled-stop landing failed with return_code={result.return_code}.',
                    ui_callback,
                )
                return

            self.get_logger().info('Controlled-stop landing succeeded; requesting disarm.')
            self.mode = 'land'
            if not self.send_arm(
                False,
                done_callback=lambda success: self.controlled_stop_disarm_callback(
                    success, ui_callback
                ),
                allow_while_flying=True,
            ):
                self.call_emergency_stop_service('Disarm request could not be sent.', ui_callback)
        except Exception as e:
            self.call_emergency_stop_service(f'Controlled-stop landing failed: {e}', ui_callback)

    def controlled_stop_disarm_callback(self, success, ui_callback):
        if success:
            self.get_logger().info('Controlled stop completed.')
            ui_callback()
            return

        if self.is_flying or self.status_is_airborne:
            self.get_logger().error(
                'Controlled-stop disarm failed while the drone still appears airborne. '
                'Not requesting emergency_stop automatically. Try landing again manually, '
                'or use emergency_stop if the situation requires it.'
            )
            return

        self.call_emergency_stop_service('Controlled-stop disarm failed.', ui_callback)

    def call_emergency_stop_service(self, reason, ui_callback):
        self.get_logger().error(f'{reason} Requesting emergency_stop service.')

        if not self.cli_estop.service_is_ready():
            self.get_logger().error('EmergencyStop service is not available!')
            return False

        request = EmergencyStop.Request()
        future = self.cli_estop.call_async(request)
        future.add_done_callback(lambda future: self.estop_callback(future, ui_callback))
        return True

    def estop_callback(self, future, ui_callback):
        try:
            response = future.result()  # raises if error occurred
            if response.success:
                self.get_logger().info(f'EmergencyStop succeeded: {response.message}')
                self.is_flying = False
                ui_callback()  # Set UI to match
            else:
                self.get_logger().warning(f'EmergencyStop service returned failure: {response.message}')
        except Exception as e:
            self.get_logger().error(f'EmergencyStop service call failed: {e}')


class ROS2Thread(QThread):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.running = True

    def quit(self):  # noqa: A003
        self.running = False
        super().quit()

    def run(self):
        try:
            while rclpy.ok() and self.running:
                # Block and wait for work
                while rclpy.spin_once(self.node, timeout_sec=0.001):
                    # Spins until no work processed before timeout
                    pass
                time.sleep(0.005)  # Cooperative yield to reduce CPU
        except Exception as e:
            print(f'[ROS2Thread] Exception: {e}')


class AttitudeJoystick(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.widget_size = 200
        self.center_position = QPoint(self.widget_size // 2, self.widget_size // 2)
        self.stick_position = QPoint(self.center_position)
        self.setFixedSize(self.widget_size, self.widget_size)
        self._on_change = lambda: None
        self.enabled = False
        self.roll_degrees = 0.0
        self.pitch_offset = 0.0
        self.yaw_degrees = 0.0
        self.physical_joystick_control = False
        self.update_pending = False

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self.enabled = enabled
        if not self.update_pending:
            self.update_pending = True
            QTimer.singleShot(0, self.repaint_widget)

    def paintEvent(self, event):
        painter = QPainter(self)

        self.update_attitude(painter)
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawEllipse(1, 1, self.widget_size - 2, self.widget_size - 2)

        if not self.enabled:
            painter.setBrush(QColor(200, 90, 90))
        else:
            painter.setBrush(QColor(90, 200, 90))

        painter.drawEllipse(self.stick_position.x() - 10, self.stick_position.y() - 10, 20, 20)

    def update_attitude(self, painter: QPainter):
        clip_rect = QRect(0, 0, self.widget_size, self.widget_size)
        clip_path = QPainterPath()
        clip_path.addEllipse(clip_rect)  # Circular clip region
        painter.setClipPath(clip_path)
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.setBrush(QBrush(QColor(173, 216, 230)))  # Light blue
        painter.drawEllipse(clip_rect)
        painter.setBrush(Qt.NoBrush)
        painter.save()  # Save the painter state

        # Move origin to the center of the container (for rotation)
        painter.translate(self.center_position.x(), self.center_position.y())

        painter.rotate(-self.roll_degrees)  # rotation on screen is opposite body frame
        # Ground
        painter.setBrush(QBrush(QColor(139, 69, 19)))  # Brown
        painter.setPen(QPen(Qt.black))
        painter.drawRect(-self.widget_size / 2, self.pitch_offset * (self.widget_size / 2), self.widget_size, self.widget_size)

        # Yaw indicator
        painter.rotate(self.roll_degrees)  # undo rotation to draw yaw indicator after ground is drawn
        screen_angle = self.yaw_degrees * math.pi / 180.0  # Relative to x-world frame (ENU)
        dx = -math.sin(screen_angle) * self.widget_size / 2  # Relative to screen frame
        dy = -math.cos(screen_angle) * self.widget_size / 2
        painter.setBrush(QBrush(QColor(192, 192, 192)))  # Brown
        painter.setPen(QPen(QColor(0, 0, 192)))
        painter.drawEllipse(dx - 8, dy - 8, 16, 16)

        painter.restore()  # Restore the painter state (back to original position)
        self.update_reference(painter)

    def update_reference(self, painter: QPainter):
        painter.setPen(QPen(QColor(128, 128, 128), 2))
        x0 = self.center_position.x() - 15
        x1 = self.center_position.x() + 15
        for tic in range(-5, 6):
            position = self.center_position.y() + tic * self.widget_size / 12
            painter.drawLine(x0, position, x1, position)
        painter.setPen(QPen(QColor(128, 128, 0), 3))
        painter.drawLine(4, self.center_position.y() - 1, self.widget_size - 4, self.center_position.y() - 1)

    def mousePressEvent(self, event):
        self.move_stick(event.pos())

    def mouseMoveEvent(self, event):
        self.move_stick(event.pos())

    def mouseReleaseEvent(self, event):
        self.stick_position = self.center_position
        if not self.update_pending:
            self.update_pending = True
            QTimer.singleShot(0, self.repaint_widget)
        self._on_change()

    def move_stick(self, point):
        dx = point.x() - self.center_position.x()
        dy = point.y() - self.center_position.y()
        distance = (dx**2 + dy**2)**0.5

        if distance > self.center_position.x():
            scale = self.center_position.x() / distance
            dx *= scale
            dy *= scale

        new_stick_position = QPoint(self.center_position.x() + dx, self.center_position.y() + dy)
        if new_stick_position.x() != self.stick_position.x() and new_stick_position.y() != self.stick_position.y():
            self.stick_position = new_stick_position
            if not self.update_pending:
                self.update_pending = True
                QTimer.singleShot(0, self.repaint_widget)
            self._on_change()

    def process_joy_msg(self, msg):
        if len(msg.axes) < 2:
            return  # Not enough axes

        # Normalize axes (assumed -1.0 to 1.0)
        joy_x = msg.axes[0]  # Left (+1.0) / right (-1.0)
        joy_y = msg.axes[1]  # Forward/backward (invert if needed)
        joy_x *= abs(joy_x)  # square magnitude to reduce off axis control
        joy_y *= abs(joy_y)  # but keep the same sign

        mag_sq = joy_x * joy_x + joy_y * joy_y
        if mag_sq > 0.001:
            self.physical_joystick_control = True
            if not self.enabled:
                print('Joystick velocity control is not enabled until flying!')
                return

            if mag_sq > 1.0:
                # Normalize to keep within radius
                mag = math.sqrt(mag_sq)
            else:
                mag = 1.0  # allow fractional values within bubble
            dx = -joy_x / mag  # Joystick has +1 to left, so invert for screen
            dy = -joy_y / mag  # Invert Y for screen coords (optional)

            # Set the stick position in screen coordinates
            # Assumes square area of radius = center_position
            self.stick_position = QPoint(
                int(self.center_position.x() * (1.0 + dx)),
                int(self.center_position.y() * (1.0 + dy))
            )
            # print(f" left joystick {joy_x} {joy_y} {mag_sq} dx,dy={dx},{dy} "
            #       f"sp({self.stick_position.x()}, {self.stick_position.y()}) "
            #       f"cp({self.center_position.x()}, {self.center_position.y()})")

            if not self.update_pending:
                self.update_pending = True
                QTimer.singleShot(0, self.repaint_widget)
            self._on_change()
        else:
            if self.physical_joystick_control:
                # Recenter if previously under physical control
                self.physical_joystick_control = False
                self.stick_position = QPoint(
                    int(self.center_position.x()),
                    int(self.center_position.y())
                )
                if not self.update_pending:
                    self.update_pending = True
                    QTimer.singleShot(0, self.repaint_widget)
                self._on_change()

    def get_vx_vy(self):
        """Get scaled +/- 1.0 velocity."""
        dx = self.stick_position.x() - self.center_position.x()
        dy = self.stick_position.y() - self.center_position.y()
        vx = -dy / self.center_position.y()  # Up = +vx
        vy = -dx / self.center_position.x()  # Left = +vy
        return vx, vy

    def on_change(self, callback):
        """Update the on_change callback for JoyStick class."""
        self._on_change = callback

    def repaint_widget(self):
        self.update_pending = False
        self.update()


class CenteredSlider(QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.enabled = False

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self.enabled = enabled
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        value = self.value()
        min_value = self.minimum()
        max_value = self.maximum()
        slider_range = max_value - min_value
        if slider_range <= 0:
            return

        zero_value = (max_value + min_value) / 2.0
        half_range = slider_range / 2.0
        normalized = max(-1.0, min(1.0, (value - zero_value) / half_range))
        bar_height = 6  # Thickness of the progress bar
        handle_width = bar_height * 2  # Size of the handle

        # Draw symmetric fill from center
        painter.setPen(Qt.NoPen)

        background_color = self.palette().color(self.backgroundRole())
        tick_color = QColor('#444')  # Color for the tick marks
        if self.enabled:
            progress_color = QColor('#c70')  # Color for the progress bar
            handle_color = QColor('#484')  # Color for the slider handle
        else:
            progress_color = QColor('#888')  # Color for the progress bar
            handle_color = QColor('#ccc')  # Color for the slider handle

        center_x = width // 2
        center_y = height // 2
        travel_margin = handle_width
        painter.setBrush(background_color)

        if self.orientation() == Qt.Horizontal:
            bar_y = int(0.25 * height)
            travel = max(1.0, width / 2.0 - travel_margin)
            handle_x = center_x + normalized * travel
            fill_width = abs(normalized) * travel

            painter.drawRect(0, bar_y, width, bar_height)
            painter.setBrush(progress_color)
            if normalized > 0.0:
                painter.drawRect(center_x, bar_y, int(fill_width), bar_height)
            elif normalized < 0.0:
                painter.drawRect(
                    int(center_x - fill_width), bar_y, int(fill_width), bar_height
                )

            painter.setBrush(handle_color)
            painter.drawEllipse(
                int(handle_x - bar_height),
                bar_y - bar_height // 2,
                handle_width,
                handle_width,
            )

            tick_length = 6
            tick_step = max(1, int(slider_range // 10))
            painter.setPen(QPen(tick_color))
            for tick in range(int(min_value), int(max_value) + 1, tick_step):
                tick_normalized = (tick - zero_value) / half_range
                tick_position = int(center_x + tick_normalized * travel)
                painter.drawLine(
                    tick_position,
                    bar_y + bar_height + 1,
                    tick_position,
                    bar_y + bar_height + 1 + tick_length,
                )
        else:
            bar_width = 6
            travel = max(1.0, height / 2.0 - travel_margin)
            handle_y = center_y - normalized * travel
            fill_height = abs(normalized) * travel
            bar_x = center_x - bar_width // 2

            painter.drawRect(bar_x, 0, bar_width, height)
            painter.setBrush(progress_color)
            if normalized > 0.0:
                painter.drawRect(
                    bar_x, int(center_y - fill_height), bar_width, int(fill_height)
                )
            elif normalized < 0.0:
                painter.drawRect(bar_x, center_y, bar_width, int(fill_height))

            painter.setBrush(handle_color)
            painter.drawEllipse(
                center_x - handle_width // 2,
                int(handle_y - handle_width / 2.0),
                handle_width,
                handle_width,
            )

            tick_length = 6
            tick_step = max(1, int(slider_range // 10))
            painter.setPen(QPen(tick_color))
            for tick in range(int(min_value), int(max_value) + 1, tick_step):
                tick_normalized = (tick - zero_value) / half_range
                tick_position = int(center_y - tick_normalized * travel)
                painter.drawLine(
                    bar_x + bar_width + 1,
                    tick_position,
                    bar_x + bar_width + 1 + tick_length,
                    tick_position,
                )


class CenteredValueBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._minimum = -100.0
        self._maximum = 100.0
        self.setFixedWidth(22)

    def setRange(self, minimum, maximum):
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self.setValue(self._value)

    def setValue(self, value):
        value = max(self._minimum, min(self._maximum, float(value)))
        if abs(value - self._value) <= 0.5:
            return
        self._value = value
        self.update()

    def value(self):
        return self._value

    def maximum(self):
        return self._maximum

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        bar_width = 8
        bar_x = center_x - bar_width // 2
        travel = max(1.0, height / 2.0 - 4)
        half_range = (self._maximum - self._minimum) / 2.0
        zero_value = (self._maximum + self._minimum) / 2.0
        normalized = 0.0 if half_range <= 0.0 else (
            (self._value - zero_value) / half_range
        )
        normalized = max(-1.0, min(1.0, normalized))
        fill_height = abs(normalized) * travel

        painter.setPen(QPen(QColor('#5c5c5c'), 1))
        painter.setBrush(QColor('#f0f0f0'))
        painter.drawRect(bar_x, 0, bar_width, height - 1)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#2a6fdb'))
        if normalized > 0.0:
            painter.drawRect(
                bar_x + 1,
                int(center_y - fill_height),
                bar_width - 1,
                int(fill_height),
            )
        elif normalized < 0.0:
            painter.drawRect(bar_x + 1, center_y, bar_width - 1, int(fill_height))

        painter.setPen(QPen(QColor('#444'), 1))
        painter.drawLine(0, center_y, width, center_y)


class IndicatorLight(QWidget):
    def __init__(self):
        super().__init__()
        self.active = False
        self.known = False
        self.setFixedSize(18, 18)

    def set_state(self, active, known=True):
        active = bool(active)
        known = bool(known)
        if self.active == active and self.known == known:
            return
        self.active = active
        self.known = known
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(35, 35, 35), 1))

        if not self.known:
            color = QColor(120, 120, 120)
        elif self.active:
            color = QColor(60, 190, 90)
        else:
            color = QColor(200, 70, 70)

        painter.setBrush(color)
        painter.drawEllipse(1, 1, self.width() - 2, self.height() - 2)


class TeleopGUI(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle(f'{node.get_namespace()} Teleop Control')
        self.setGeometry(
            int(self.node.window_x),
            int(self.node.window_y),
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        )
        self.setFocusPolicy(Qt.StrongFocus)  # enable keyPressEvent

        main_layout = QVBoxLayout()

        # Top layout: joystick + height slider
        top_layout = QHBoxLayout()

        # Center joystick in layout
        center_layout = QVBoxLayout()
        center_layout.addSpacerItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.joystick = AttitudeJoystick()
        center_layout.addWidget(self.joystick, alignment=Qt.AlignCenter)

        center_layout.addSpacerItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Yawrate slider
        yaw_layout = QVBoxLayout()
        yaw_label = QLabel('Yawrate (rad/s)')
        yaw_label.setAlignment(Qt.AlignHCenter)
        yaw_layout.addWidget(yaw_label)

        self.yaw_slider = CenteredSlider(Qt.Horizontal)
        self.yaw_slider.setMinimum(-100)
        self.yaw_slider.setMaximum(100)
        self.yaw_slider.setValue(0)
        self.yaw_slider.setTickInterval(10)
        self.yaw_slider.setTickPosition(QSlider.TicksBelow)
        self.yaw_slider.setInvertedAppearance(False)  # positive on left, negative on right

        yaw_layout.addWidget(self.yaw_slider)
        center_layout.addLayout(yaw_layout)
        top_layout.addLayout(center_layout)

        # Vertical velocity slider
        zv_layout = QVBoxLayout()
        zh_layout = QHBoxLayout()

        zv_layout.addWidget(QLabel('Vz (m/s)'), alignment=Qt.AlignHCenter)

        self.z_slider = CenteredSlider(Qt.Vertical)
        self.z_slider.setMinimum(-100)
        self.z_slider.setMaximum(100)
        self.z_slider.setValue(0)
        self.z_slider.setTickInterval(10)
        self.z_slider.setTickPosition(QSlider.TicksRight)
        zh_layout.addWidget(self.z_slider)

        self.z_progress = CenteredValueBar()
        self.z_progress.setRange(-100, 100)
        self.z_progress.setValue(0)
        zh_layout.addWidget(self.z_progress)

        zv_layout.addLayout(zh_layout)
        top_layout.addLayout(zv_layout)
        main_layout.addLayout(top_layout)

        self.state_label = QLabel('DISCONNECTED')
        self.state_label.setWordWrap(True)
        main_layout.addWidget(self.state_label)

        status_indicator_layout = QHBoxLayout()
        status_indicator_layout.addWidget(QLabel('Armed'))
        self.armed_indicator = IndicatorLight()
        status_indicator_layout.addWidget(self.armed_indicator)
        status_indicator_layout.addSpacing(12)
        status_indicator_layout.addWidget(QLabel('Flying'))
        self.flying_indicator = IndicatorLight()
        status_indicator_layout.addWidget(self.flying_indicator)
        status_indicator_layout.addStretch()
        main_layout.addLayout(status_indicator_layout)

        self.teleop_status_label = QLabel('TELEOP: idle')
        self.teleop_status_label.setWordWrap(True)
        main_layout.addWidget(self.teleop_status_label)
        self.world_frame_checkbox = QCheckBox('World frame')
        self.world_frame_checkbox.setChecked(self.node.teleop_frame_id == 'map')
        self.world_frame_checkbox.toggled.connect(self.set_world_frame)
        main_layout.addWidget(self.world_frame_checkbox)
        self.teleop_button = QPushButton('Enable Teleop')
        self.teleop_button.clicked.connect(self.teleop_command)
        main_layout.addWidget(self.teleop_button)

        self.setLayout(main_layout)

        # Connect value changes to update command
        self.joystick.on_change(self.send_teleop)
        self.yaw_slider.valueChanged.connect(self.send_teleop)
        self.z_slider.valueChanged.connect(self.send_teleop)

        self.last_pose = None
        self.last_joy_msg = None
        self.last_joy_buttons = None
        self.close_waiting_for_teleop = False
        self.force_close_requested = False
        self.set_controls_enabled(False)
        self.prior_quat = Quaternion()
        self.prior_rpy = (0., 0., 0.)

        self.current_yaw_rate = 0.0
        self.last_yaw_update_time = time.time()
        self.yaw_change_speed = 80.0
        self.trigger_deadband = 0.05

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_pose_display)
        self.timer.start(100)  # Update display at 10 hz

        self.teleop_heartbeat_timer = QTimer(self)
        self.teleop_heartbeat_timer.timeout.connect(self.teleop_heartbeat)
        heartbeat_hz = max(1.0, float(self.node.teleop_heartbeat_hz))
        self.teleop_heartbeat_timer.start(max(10, int(1000.0 / heartbeat_hz)))

    def update_yaw_from_triggers(self, lt, rt):
        """
        Update yaw rate from trigger inputs.

        RT increases yaw rate over time.
        LT decreases yaw rate over time.
        Harder trigger squeeze = faster change.
        Releasing triggers = keep current yaw rate.
        """
        now = time.time()
        dt = now - self.last_yaw_update_time
        self.last_yaw_update_time = now
        dt = min(dt, 0.1)

        # Ignore noise
        if rt < self.trigger_deadband:
            rt = 0.0
        if lt < self.trigger_deadband:
            lt = 0.0

        yaw_delta = 0.0

        # Right trigger increases yaw rate
        yaw_delta = rt * self.yaw_change_speed * dt

        # Left trigger decreases yaw rate
        yaw_delta -= lt * self.yaw_change_speed * dt

        if abs(yaw_delta) > 0.001:
            self.current_yaw_rate += yaw_delta
            self.set_yaw_rate(self.current_yaw_rate)

    def process_joy_msg(self, msg):
        self.joystick.process_joy_msg(msg)
        if len(self.node.joy_msg.axes) > 5:
            # Handle triggers for yaw rate control
            lt_raw = msg.axes[2]
            rt_raw = msg.axes[5]

            lt = (1.0 - lt_raw) / 2.0
            rt = (1.0 - rt_raw) / 2.0
            lt = max(0.0, min(1.0, lt))
            rt = max(0.0, min(1.0, rt))

            if self.yaw_slider.isEnabled():
                self.update_yaw_from_triggers(lt, rt)

            # Vertical axis: map joystick axis to slider around center and
            # zero the slider when the joystick is released.
            if abs(self.node.joy_msg.axes[4]) > 0.001:
                if self.z_slider.isEnabled():
                    # Map axis (-1..1) directly to centered slider range.
                    z_position = int(
                        self.node.joy_msg.axes[4] * self.z_slider.maximum()
                    )
                    self.set_vertical_velocity(z_position)
                else:
                    print('Joystick control of vertical velocity disabled until flying!')
            else:
                # Joystick released: zero vertical rate (center slider)
                if self.z_slider.isEnabled():
                    self.set_vertical_velocity(0)

        buttons = msg.buttons
        if len(buttons) < 6:
            print('Not enough buttons in joy message - ignore!')
            return

        if self.last_joy_buttons == buttons:
            return  # only process first occurrence of button

        self.last_joy_buttons = buttons
        a = buttons[0]
        b = buttons[1]
        x = buttons[2]
        y = buttons[3]
        lb = buttons[4]  # required for takeoff/land
        rb = buttons[5]  # required for e-stop

        if rb and a:
            self.arm_command()
        elif rb and y:
            self.disarm_command()
        elif lb and a:
            self.takeoff_command()
        elif lb and b:
            self.land_command()
        elif rb and lb and x:
            self.emergency_stop_command()
        elif y:
            # Zero yaw rate value
            self.current_yaw_rate = 0.0
            self.yaw_slider.setValue(0)

    def arm_command(self):
        if self.node.send_arm(True):
            self.node.get_logger().info('Arm drone requested ...')

    def disarm_command(self):
        self.run_after_teleop_release(self._send_disarm)

    def _send_disarm(self):
        if self.node.send_arm(False):
            self.node.get_logger().info('Disarm drone requested ...')

    def takeoff_command(self):
        if self.node.send_takeoff(self.drone_takeoff):
            self.node.get_logger().info('Takeoff initiated by cmd_vel ...')
            self.z_slider.setValue(0)
            self.current_yaw_rate = 0.0
            self.yaw_slider.setValue(0)

    def drone_takeoff(self):
        self.current_yaw_rate = 0.0
        self.yaw_slider.setValue(0)
        self.z_slider.setValue(0)

    def drone_off(self):
        self.node.get_logger().info('Drone off ...')
        self.node.cancel_teleop(self.refresh_teleop_status)
        self.set_controls_enabled(False)
        self.current_yaw_rate = 0.0
        self.yaw_slider.setValue(0)
        self.z_slider.setValue(0)
        self.node.is_flying = False
        self.node.is_armed = False

    def land_command(self):
        self.run_after_teleop_release(self._send_land)

    def _send_land(self):
        if self.node.send_land(self.drone_off):
            self.node.get_logger().info('Landing initiated by cmd_vel ...')
            self.z_slider.setValue(0)
            self.current_yaw_rate = 0.0
            self.yaw_slider.setValue(0)

    def set_yaw_rate(self, value=0):
        # Set yaw rate when flying
        if self.node.is_flying:
            if value < self.yaw_slider.minimum():
                value = self.yaw_slider.minimum()
            elif value > self.yaw_slider.maximum():
                value = self.yaw_slider.maximum()

            self.current_yaw_rate = float(value)
            if int(value) != self.yaw_slider.value():
                self.yaw_slider.setValue(int(value))

    def set_vertical_velocity(self, value=0):
        # Set vertical velocity when flying
        if self.node.is_flying:
            value = int(max(self.z_slider.minimum(), min(self.z_slider.maximum(), value)))

            if value != self.z_slider.value():
                self.z_slider.setValue(value)

    def emergency_stop_command(self):
        self.run_after_teleop_release(self._send_emergency_stop)

    def _send_emergency_stop(self):
        self.node.send_emergency_stop(self.drone_off)

    def run_after_teleop_release(self, command):
        if self.node.teleop_active or self.node.teleop_pending:
            self.node.cancel_teleop(lambda: (self.refresh_teleop_status(), command()))
            self.refresh_teleop_status()
            return
        command()

    def teleop_command(self):
        if self.node.teleop_active or self.node.teleop_pending:
            self.node.cancel_teleop(self.refresh_teleop_status)
            self.refresh_teleop_status()
            return

        self.send_teleop()
        self.node.send_enable_teleop(self.refresh_teleop_status)
        self.refresh_teleop_status()

    def teleop_heartbeat(self):
        if self.node.is_flying or self.node.teleop_active or self.node.teleop_pending:
            self.send_teleop(force=self.node.teleop_active or self.node.teleop_pending)
        self.refresh_teleop_status()

    def refresh_teleop_status(self):
        status = self.node.teleop_status_text
        self.teleop_status_label.setText(f'TELEOP:\n{status}')
        teleop_on = self.node.teleop_active or self.node.teleop_pending
        enabled = self.node.is_flying or teleop_on
        if teleop_on:
            self.teleop_button.setText('Release Teleop')
        else:
            self.teleop_button.setText('Enable Teleop')
        self.teleop_button.setEnabled(enabled)
        self.update_teleop_button_style(enabled, teleop_on)

    def update_teleop_button_style(self, enabled, teleop_on=False):
        if not enabled:
            self.teleop_button.setStyleSheet('')
            return
        color = '#b3261e' if teleop_on else '#1f7a3a'
        hover = '#8c1d18' if teleop_on else '#17612d'
        self.teleop_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                font-weight: 600;
                border: 1px solid {hover};
                border-radius: 4px;
                padding: 6px 10px;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """)

    def set_world_frame(self, enabled):
        self.node.set_teleop_world_frame(enabled)
        self.send_teleop(force=self.node.teleop_active or self.node.teleop_pending)

    def update_pose_display(self):
        status_known = self.node.status_is_fresh()
        self.armed_indicator.set_state(self.node.status_is_armed, status_known)
        self.flying_indicator.set_state(self.node.status_is_airborne, status_known)

        if self.node.joy_msg and self.node.joy_msg != self.last_joy_msg:
            # Process any incoming physical joystick messages
            self.last_joy_msg = self.node.joy_msg
            self.process_joy_msg(self.last_joy_msg)

        label_text = 'UNKNOWN\n'
        if self.node.pose:
            pos = self.node.pose.position
            quat = self.node.pose.orientation
            if pos.z <= self.node.landed_height:
                self.node.is_flying = False
                if not self.node.is_armed:
                    label_text = 'DISARMED\n'
                else:
                    label_text = 'ARMED\n'
            elif pos.z > self.node.flying_height:
                label_text = 'FLYING\n'
                if not self.node.is_flying:
                    print('Flying! - enable GUI controls and let match current height')
                    self.set_controls_enabled(True)
                    self.node.is_flying = True
                    if not self.last_pose:
                        # If this is first update after connection, set slider height to match
                        print('Already in flight - match current height')
                        self.node.is_armed = True  # Must be armed if flying!
                        self.z_slider.setValue(0)

            self.last_pose = self.node.pose
            z_scale = max(0.001, abs(self.node.vz_scale))
            z_progress = 100.0 * self.node.odom_linear_z / z_scale

            if (abs(quat.x - self.prior_quat.x) > 0.001 or
                abs(quat.y - self.prior_quat.y) > 0.001 or
                    abs(quat.z - self.prior_quat.z) > 0.001):
                import tf_transformations

                self.prior_quat = quat
                r, p, y = tf_transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
                # Convert to degrees for display
                r *= 180. / math.pi
                p *= 180. / math.pi
                y *= 180. / math.pi
                self.prior_rpy = (r, p, y)

                self.joystick.roll_degrees = r
                self.joystick.yaw_degrees = y
                self.joystick.pitch_offset = (
                    min((pos.z / self.node.horizon_height - 0.5), pos.z / self.node.max_height)
                    - p / self.node.pitch_scaling)
                if not self.joystick.update_pending:
                    self.joystick.update_pending = True
                    QTimer.singleShot(0, self.joystick.repaint_widget)
            else:
                r, p, y = self.prior_rpy
                if abs(pos.z - self.last_pose.position.z) > 0.001:
                    self.joystick.roll_degrees = r
                    self.joystick.yaw_degrees = y
                    self.joystick.pitch_offset = (
                        min((pos.z / self.node.horizon_height - 0.5), pos.z / self.node.max_height)
                        - p / self.node.pitch_scaling)
                    if not self.joystick.update_pending:
                        self.joystick.update_pending = True
                        QTimer.singleShot(0, self.joystick.repaint_widget)

            label_text += (
                f'flying={self.node.is_flying} | xyz=({pos.x:6.3f}, {pos.y:6.3f}, {pos.z:6.3f}) m\n'
                f'rpy=({r:6.1f}, {p:6.1f}, {y:6.1f}) deg')
        else:
            label_text += 'DISCONNECTED'
            self.joystick.roll_degrees = -30.0 * 0
            self.joystick.yaw_degrees = -49.0
            p = -10.0 * 0
            z_progress = 0.0
            self.joystick.pitch_offset = -p / self.node.pitch_scaling
            self.joystick.update()

        if abs(z_progress - self.z_progress.value()) > 0.5:
            self.z_progress.setValue(z_progress)

        if label_text != self.state_label.text():
            self.state_label.setText(label_text)

    def set_controls_enabled(self, enabled):
        print('Enable GUI control inputs')
        self.z_slider.setEnabled(enabled)
        self.yaw_slider.setEnabled(enabled)
        self.joystick.setEnabled(enabled)
        self.teleop_button.setEnabled(enabled)
        self.update_teleop_button_style(
            enabled,
            self.node.teleop_active or self.node.teleop_pending,
        )

    def keyPressEvent(self, event):
        key = event.key()
        ctrl_key = event.modifiers() & Qt.ControlModifier

        if key == Qt.Key_Left:
            self.set_yaw_rate(self.yaw_slider.value() + 5)
        elif key == Qt.Key_Right:
            self.set_yaw_rate(self.yaw_slider.value() - 5)
        elif key == Qt.Key_Space:
            self.yaw_slider.setValue(0)
        elif key == Qt.Key_Up:
            self.set_vertical_velocity(self.z_slider.value() + 5)
        elif key == Qt.Key_Down:
            self.set_vertical_velocity(self.z_slider.value() - 5)
        elif key == Qt.Key_H:
            print(usage())
        elif key == Qt.Key_T:
            if ctrl_key:
                self.takeoff_command()
            else:
                self.node.get_logger().info('Takeoff requires CTRL modifier!')
        elif key == Qt.Key_E:
            if ctrl_key:
                self.teleop_command()
            else:
                self.node.get_logger().info('Enable/release teleop requires CTRL modifier!')
        elif key == Qt.Key_L:
            if ctrl_key:
                self.land_command()
            else:
                self.node.get_logger().info('Landing requires CTRL modifier!')
        elif key == Qt.Key_X:
            if ctrl_key:
                self.emergency_stop_command()
            else:
                self.node.get_logger().info('Controlled stop requires CTRL modifier!')
        elif key == Qt.Key_A:
            if ctrl_key:
                self.arm_command()
            else:
                self.node.get_logger().info('Arm requires CTRL modifier!')
        elif key == Qt.Key_D:
            if ctrl_key:
                self.disarm_command()
            else:
                self.node.get_logger().info('Disarm requires CTRL modifier!')

    def closeEvent(self, event):
        if (
            (self.node.teleop_active or self.node.teleop_pending)
            and not self.force_close_requested
        ):
            event.ignore()
            if not self.close_waiting_for_teleop:
                self.close_waiting_for_teleop = True
                self.node.get_logger().info('GUI closing; releasing teleop control first.')
                if not self.node.cancel_teleop(self._close_after_teleop_release):
                    self._close_after_teleop_release()
                    return
                QTimer.singleShot(2000, self._force_close_after_teleop_timeout)
            return

        self.node.cancel_teleop()
        super().closeEvent(event)

    def _close_after_teleop_release(self):
        self.close_waiting_for_teleop = False
        self.force_close_requested = True
        self.close()

    def _force_close_after_teleop_timeout(self):
        if not self.close_waiting_for_teleop:
            return
        self.node.get_logger().warning('Closing before teleop release confirmation.')
        self.close_waiting_for_teleop = False
        self.force_close_requested = True
        self.close()

    def send_teleop(self, *_args, force=False):
        """Send scaled values +/- 1.0 range."""
        if self.node.is_flying or force:
            vx, vy = self.joystick.get_vx_vy()

            # Slider is inverted to have positive on left, which matches ROS convention
            yawrate = self.yaw_slider.value() / self.yaw_slider.maximum()
        else:
            vx, vy, yawrate = 0.0, 0.0, 0.0

        vz = self.z_slider.value() / self.z_slider.maximum()
        self.node.send_teleop_command(vx, vy, yawrate, vz)


def main(args=None):
    if print_usage_if_requested(CLI_USAGE, args):
        return

    rclpy.init(args=args)
    teleop_node = TeleopPublisher()
    # Start the ROS spinner in a separate thread
    ros_thread = ROS2Thread(teleop_node)
    ros_thread.start()

    app = QApplication(sys.argv)
    gui = TeleopGUI(teleop_node)
    gui.show()
    gui.setFixedSize(gui.size())  # Lock the size after show

    # Gracefully shutdown on Ctrl+C
    def sigint_handler(sig, frame):
        print('Shutting down gracefully...', flush=True)
        gui.close()

    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        app.exec()
    finally:
        ros_thread.quit()
        ros_thread.wait()  # Ensure the thread has fully stopped
        teleop_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        app.quit()
    print("We're outta here!")


if __name__ == '__main__':
    main()

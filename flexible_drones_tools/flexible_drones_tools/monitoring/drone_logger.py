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

import numpy as np
import os
from pathlib import Path
import signal
from threading import Event, Thread
import time

from rcl_interfaces.msg import ParameterDescriptor

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TwistStamped

from flexible_drones_msgs.msg import FullState
from flexible_drones_msgs.msg import DroneStatus as Status
from flexible_drones_tools.monitoring.cli_help import print_usage_if_requested
from flexible_drones_tools.ros_shutdown import BENIGN_SHUTDOWN_EXCEPTIONS


FULLSTATE_COLUMNS = 15


def _frame_is_world(frame_id):
    """
    Return True for a world/ENU control frame_id (vs the body frame).

    World velocities are published either in the legacy 'map' frame (teleop
    cmd_vel input) or in a drone's '<drone>/odom' frame (generated control).
    The body frame is the bare drone name, so anything ending in 'odom' or
    equal to 'map' is treated as world for the logged frame_code (1 = world,
    0 = body).
    """
    normalized = str(frame_id or '').strip().lower()
    return normalized == 'map' or normalized.endswith('odom')


USAGE = """\
Usage:
  ros2 run flexible_drones_tools drone_logger --ros-args -r __ns:=/drone1

Common examples:
  ros2 run flexible_drones_tools drone_logger --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools drone_logger --ros-args -r __ns:=/drone1 -p use_sim_time:=true
  ros2 run flexible_drones_tools drone_logger --ros-args -r __ns:=/drone1 -p logging_folder:=log/drone1

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Parameters declared by this node:
  logging_folder (string, default: log/<namespace>)
    Output folder below $WORKSPACE_ROOT.
  logging_buffer_length (integer, default: 6000)
    Number of samples stored before writing a buffer.

Logging starts automatically when the drone status reports armed and flushes/stops
when the status reports disarmed. Leave the node running between flights.
"""


class DroneLoggerNode(Node):
    LOG_DATASETS = (
        'odom',
        'cmd_vel',
        'cmd',
        'pose',
        'cmd_full',
        'cmd_state',
        'target_state',
        'full_state',
        'status',
    )

    def __init__(self):
        super().__init__('drone_data_logger')

        self.declare_parameter('logging_folder', f'log{self.get_namespace()}',
                               descriptor=ParameterDescriptor(
            description='Output folder for logs'
        ))
        self.declare_parameter('logging_buffer_length', 100 * 60,  # 1 minutes at 100Hz
                               descriptor=ParameterDescriptor(
                                   description='Number of samples to store before saving (default: 6000)'
                               ))

        self.logging_buffer_length = self.get_parameter('logging_buffer_length').value

        log_folder = self.get_parameter('logging_folder').value
        workspace_root = os.environ.get('WORKSPACE_ROOT', '.')
        self.log_path = Path(workspace_root) / log_folder

        # Define a buffer pool to hold data while writing
        self.logging_buffers = [self._new_logging_buffer(), self._new_logging_buffer()]
        self.num_buffers = len(self.logging_buffers)
        self.index_buffer = 0
        self.log_index = 0
        self.log_stamp = 0
        self.logging_start = None
        self._logging_active = False
        self._last_armed = False

        self._write_threads = []

        # Subscriptions
        sensor_qos = qos_profile_sensor_data
        self.create_subscription(PoseStamped, 'pose', self.pose_cb, sensor_qos)
        self.create_subscription(FullState, 'cmd_full_state', self.fullstate_cb, sensor_qos)
        self.create_subscription(FullState, 'cmd_state', self.cmdstate_cb, sensor_qos)
        self.create_subscription(FullState, 'target_state', self.targetstate_cb, sensor_qos)
        self.create_subscription(FullState, 'full_state', self.optitrack_fullstate_cb, sensor_qos)
        self.create_subscription(Odometry, 'odom', self.odom_cb, sensor_qos)
        self.create_subscription(TwistStamped, 'cmd_vel', self.cmd_vel_cb, sensor_qos)
        self.create_subscription(Twist, 'command/twist', self.cmd_cb, sensor_qos)
        self.create_subscription(Status, 'status', self.status_cb, sensor_qos)

        self.get_logger().info('Logger ready; waiting for armed status to start recording.')

    def _new_logging_buffer(self):
        return {
            'odom': np.zeros((self.logging_buffer_length, 14)),
            'cmd_vel': np.zeros((self.logging_buffer_length, 6)),
            'cmd': np.zeros((self.logging_buffer_length, 7)),
            'pose': np.zeros((self.logging_buffer_length, 8)),       # stamp, pos x,y,z, ori x,y,z,w
            # stamp, pos x,y,z, ori x,y,z,w, linear vel x,y,z,
            # angular vel x,y,z, valid_mask
            'cmd_full': np.zeros((self.logging_buffer_length, FULLSTATE_COLUMNS)),
            'cmd_state': np.zeros((self.logging_buffer_length, FULLSTATE_COLUMNS)),
            'target_state': np.zeros((self.logging_buffer_length, FULLSTATE_COLUMNS)),
            'full_state': np.zeros((self.logging_buffer_length, FULLSTATE_COLUMNS)),
            # stamp_us, state, status_flags, battery_mV, battery_pct,
            # link_latency_ms, battery_current_a
            'status': np.zeros((self.logging_buffer_length, 7)),
            'odom_index': 0,
            'cmd_vel_index': 0,
            'cmd_index': 0,
            'pose_index': 0,
            'cmd_full_index': 0,
            'cmd_state_index': 0,
            'target_state_index': 0,
            'full_state_index': 0,
            'status_index': 0,
        }

    def _reset_logging_buffers(self):
        for buffer in self.logging_buffers:
            for data in self.LOG_DATASETS:
                buffer[f'{data}_index'] = 0

    def _message_stamp_nsec(self, msg):
        header = getattr(msg, 'header', None)
        stamp = getattr(header, 'stamp', None)
        if stamp is not None:
            nanoseconds = stamp.sec * 1_000_000_000 + stamp.nanosec
            if nanoseconds > 0:
                return nanoseconds
        return self.get_clock().now().nanoseconds

    def _begin_logging_session(self, start_nsec):
        self.log_path.mkdir(parents=True, exist_ok=True)
        self._reset_logging_buffers()
        self.index_buffer = 0
        self.log_index = 0
        self.log_stamp = 0
        self.logging_start = start_nsec
        self._logging_active = True
        if self.logging_start == 0:
            self.get_logger().warning(
                'Clock is zero at arm; log filenames and timestamps will start at zero.'
            )
        self.get_logger().info(f'Armed status received; recording to {self.log_path}.')

    def _end_logging_session(self, end_nsec):
        if not self._logging_active:
            return
        relative_end_nsec = max(0, end_nsec - (self.logging_start or 0))
        self.write_log(self.logging_buffers[self.index_buffer], relative_end_nsec)
        self._logging_active = False
        self.logging_start = None
        self.log_stamp = 0
        self.get_logger().info('Disarmed status received; recording stopped.')

    def _relative_nanoseconds(self, msg):
        if not self._logging_active or self.logging_start is None:
            return None
        nanoseconds = self._message_stamp_nsec(msg) - self.logging_start
        if nanoseconds < 0:
            return None
        return nanoseconds

    def _active_buffer_for_write(self, data, nanoseconds):
        row_index = self.logging_buffers[self.index_buffer][f'{data}_index']
        if row_index >= self.logging_buffer_length or nanoseconds - self.log_stamp > 60_000_000_000:
            write_index = self.index_buffer
            self.index_buffer = (self.index_buffer + 1) % self.num_buffers
            self.write_log(self.logging_buffers[write_index], nanoseconds)
            row_index = self.logging_buffers[self.index_buffer][f'{data}_index']

        return self.logging_buffers[self.index_buffer], row_index

    def _flush_if_needed(self, row_index, nanoseconds):
        if row_index >= self.logging_buffer_length:
            write_index = self.index_buffer
            self.index_buffer = (self.index_buffer + 1) % self.num_buffers
            self.write_log(self.logging_buffers[write_index], nanoseconds)

    def pose_cb(self, msg):
        nanoseconds = self._relative_nanoseconds(msg)
        if nanoseconds is None:
            return
        stamp = nanoseconds * 1.e-9
        buffer, row_index = self._active_buffer_for_write('pose', nanoseconds)
        pose_buffer = buffer['pose']
        pose_buffer[row_index] = [
            stamp,
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w,
        ]
        row_index += 1
        buffer['pose_index'] = row_index
        self._flush_if_needed(row_index, nanoseconds)

    def _fullstate_row(self, msg, stamp):
        return np.array([
            stamp,
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w,
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z,
            float(getattr(msg, 'valid_mask', 0)),
        ], dtype=float)

    def _apply_valid_mask_nan(self, row, valid_mask):
        if not (valid_mask & FullState.VALID_POSITION):
            row[1:4] = np.nan
        if not (valid_mask & FullState.VALID_ORIENTATION):
            row[4:8] = np.nan
        if not (valid_mask & FullState.VALID_LINEAR_VELOCITY):
            row[8:11] = np.nan
        if not (valid_mask & FullState.VALID_ANGULAR_RATE):
            row[11:14] = np.nan
        return row

    def _write_fullstate(self, data, msg, *, mask_invalid=False):
        nanoseconds = self._relative_nanoseconds(msg)
        if nanoseconds is None:
            return
        stamp = nanoseconds * 1.e-9
        buffer, row_index = self._active_buffer_for_write(data, nanoseconds)
        full_buffer = buffer[data]
        row = self._fullstate_row(msg, stamp)
        if mask_invalid:
            row = self._apply_valid_mask_nan(row, int(getattr(msg, 'valid_mask', 0)))
        full_buffer[row_index] = row
        row_index += 1
        buffer[f'{data}_index'] = row_index
        self._flush_if_needed(row_index, nanoseconds)

    def fullstate_cb(self, msg):
        self._write_fullstate('cmd_full', msg)

    def cmdstate_cb(self, msg):
        self._write_fullstate('cmd_state', msg, mask_invalid=True)

    def targetstate_cb(self, msg):
        self._write_fullstate('target_state', msg, mask_invalid=True)

    def optitrack_fullstate_cb(self, msg):
        self._write_fullstate('full_state', msg)

    def odom_cb(self, msg):
        nanoseconds = self._relative_nanoseconds(msg)
        if nanoseconds is None:
            return
        stamp = nanoseconds * 1.e-9
        buffer, row_index = self._active_buffer_for_write('odom', nanoseconds)
        odom_buffer = buffer['odom']
        odom_buffer[row_index][0] = stamp
        odom_buffer[row_index][1] = msg.pose.pose.position.x
        odom_buffer[row_index][2] = msg.pose.pose.position.y
        odom_buffer[row_index][3] = msg.pose.pose.position.z
        odom_buffer[row_index][4] = msg.pose.pose.orientation.x
        odom_buffer[row_index][5] = msg.pose.pose.orientation.y
        odom_buffer[row_index][6] = msg.pose.pose.orientation.z
        odom_buffer[row_index][7] = msg.pose.pose.orientation.w
        odom_buffer[row_index][8] = msg.twist.twist.linear.x
        odom_buffer[row_index][9] = msg.twist.twist.linear.y
        odom_buffer[row_index][10] = msg.twist.twist.linear.z
        odom_buffer[row_index][11] = msg.twist.twist.angular.x
        odom_buffer[row_index][12] = msg.twist.twist.angular.y
        odom_buffer[row_index][13] = msg.twist.twist.angular.z
        row_index += 1
        buffer['odom_index'] = row_index
        self._flush_if_needed(row_index, nanoseconds)

    def cmd_vel_cb(self, msg):
        nanoseconds = self._relative_nanoseconds(msg)
        if nanoseconds is None:
            return
        stamp = nanoseconds * 1.e-9
        buffer, row_index = self._active_buffer_for_write('cmd_vel', nanoseconds)
        cmd_vel_buffer = buffer['cmd_vel']
        cmd_vel_buffer[row_index][0] = stamp
        cmd_vel_buffer[row_index][1] = msg.twist.linear.x
        cmd_vel_buffer[row_index][2] = msg.twist.linear.y
        cmd_vel_buffer[row_index][3] = msg.twist.angular.z
        cmd_vel_buffer[row_index][4] = msg.twist.linear.z
        frame_id = getattr(msg.header, 'frame_id', '')
        cmd_vel_buffer[row_index][5] = 1.0 if _frame_is_world(frame_id) else 0.0
        row_index += 1
        buffer['cmd_vel_index'] = row_index
        self._flush_if_needed(row_index, nanoseconds)

    def cmd_cb(self, msg):
        nanoseconds = self._relative_nanoseconds(msg)
        if nanoseconds is None:
            return
        stamp = nanoseconds * 1e-9
        buffer, row_index = self._active_buffer_for_write('cmd', nanoseconds)
        cmd_buffer = buffer['cmd']
        cmd_buffer[row_index][0] = stamp
        cmd_buffer[row_index][1] = msg.linear.x
        cmd_buffer[row_index][2] = msg.linear.y
        cmd_buffer[row_index][3] = msg.linear.z
        cmd_buffer[row_index][4] = msg.angular.x
        cmd_buffer[row_index][5] = msg.angular.y
        cmd_buffer[row_index][6] = msg.angular.z
        row_index += 1
        buffer['cmd_index'] = row_index
        self._flush_if_needed(row_index, nanoseconds)

    def status_cb(self, msg):
        stamp_nsec = self._message_stamp_nsec(msg)
        armed = (int(msg.status_flags) & Status.STATUS_ARMED) != 0
        if armed and not self._logging_active:
            self._begin_logging_session(stamp_nsec)
        if not armed and self._logging_active and self._last_armed:
            nanoseconds = max(0, stamp_nsec - (self.logging_start or 0))
            self._write_status(msg, nanoseconds)
            self._end_logging_session(stamp_nsec)
            self._last_armed = armed
            return

        self._last_armed = armed
        if not self._logging_active:
            return

        nanoseconds = max(0, stamp_nsec - (self.logging_start or 0))
        self._write_status(msg, nanoseconds)

    def _write_status(self, msg, nanoseconds):
        buffer, row_index = self._active_buffer_for_write('status', nanoseconds)
        status_buffer = buffer['status']
        status_buffer[row_index][0] = nanoseconds // 1000
        status_buffer[row_index][1] = int(msg.state)
        status_buffer[row_index][2] = int(msg.status_flags)
        status_buffer[row_index][3] = int(msg.battery_voltage_v * 1000)
        status_buffer[row_index][4] = int(msg.battery_remaining_pct)
        status_buffer[row_index][5] = float(msg.link_latency_ms)
        status_buffer[row_index][6] = float(getattr(msg, 'battery_current_a', -1.0))
        row_index += 1
        buffer['status_index'] = row_index
        self._flush_if_needed(row_index, nanoseconds)

    def write_log(self, buffer, log_stamp):
        if self.logging_start is None:
            return
        self.log_stamp = log_stamp
        counts = ', '.join(f"{data}={buffer[f'{data}_index']}" for data in self.LOG_DATASETS)
        print(
            f'\nWriting log index={self.log_index} with ({counts}) points '
            f'at {log_stamp * 1.e-9:.6f} seconds ...'
        )

        wrote_any = False
        for data in self.LOG_DATASETS:
            index_key = f'{data}_index'
            row_count = buffer[index_key]
            if row_count > 0:
                file_base = self.log_path / f'{data}_{self.logging_start}_{self.log_index}.npy'
                data_snapshot = buffer[data][:row_count].copy()
                thread = Thread(target=np.save, args=(file_base, data_snapshot), daemon=True)
                self._write_threads.append(thread)
                thread.start()
                wrote_any = True
            buffer[index_key] = 0  # reset the index for next logging

        if wrote_any:
            self.log_index += 1
            print('Writing logs .', end='')


# --- ROS spin in background thread ---
shutdown_event = Event()


def ros_spin(node):

    try:
        while not shutdown_event.is_set() and rclpy.ok():
            cnt = 0
            while rclpy.spin_once(node, timeout_sec=0.001) and cnt < 100:
                # Spins until no work processed before timeout
                cnt += 1
            if cnt < 80:
                time.sleep(0.005)  # Cooperatively release CPU
        if shutdown_event.is_set():
            print('External shutdown triggered ...')
    except BENIGN_SHUTDOWN_EXCEPTIONS:
        print('ROS shutdown in ros_spin triggered ...')

    print('Dump remaining log data ...', flush=True)
    # Dump any final information to logs
    node.write_log(node.logging_buffers[node.index_buffer], -1)
    for thread in node._write_threads:
        print('.', end='')
        thread.join()  # Wait for each log writing thread to finish
    print('Logging is done!', flush=True)


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)
    node = DroneLoggerNode()

    def signal_handler(sig, frame):
        print('Caught Ctrl-C, shutting down...')
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        ros_spin(node)
    except BENIGN_SHUTDOWN_EXCEPTIONS:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

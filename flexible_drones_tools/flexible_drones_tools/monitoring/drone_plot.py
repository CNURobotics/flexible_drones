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

# Usage:
# bokeh serve --show drone_plot.py --args --ros-args -r __ns:=/blue1
#
# Or for simulation, be sure to use sim time:
# bokeh serve --show drone_plot.py --args --ros-args -r __ns:=/blue1 -p use_sim_time:=true

# Assumes bokeh installed in the workspace virtual environment, and venv/bin added to path

import argparse
from functools import partial
import numpy as np
import os
from pathlib import Path
from queue import Queue, Empty
import signal
from types import SimpleNamespace
import sys
from threading import Event, Thread
import time

try:
    from bokeh.layouts import column, row
    from bokeh.models import ColumnDataSource, CheckboxGroup, CustomJS, Div, Range1d
    from bokeh.plotting import figure, curdoc
except ImportError as exc:
    raise SystemExit(
        'drone_plot requires bokeh: python3 -m pip install bokeh\n'
        'See flexible_drones_tools/README.md for optional GUI dependencies.'
    ) from exc

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TwistStamped
from flexible_drones_msgs.msg import FullState
from flexible_drones_msgs.msg import DroneStatus as Status
from flexible_drones_tools.ros_shutdown import BENIGN_SHUTDOWN_EXCEPTIONS


# Constants
FULLSTATE_COLUMNS = 15
QUEUE_WARN_LIMIT = 100
UI_UPDATE = 500  # ms (2 Hz)
SAMPLE_RATE = 20
MIN_UPDATE_PERIOD = (10**9) / SAMPLE_RATE  # 20 Hz
TIME_WINDOW_SECONDS = 10  # Seconds
ROLL_WINDOW = int(TIME_WINDOW_SECONDS * SAMPLE_RATE)  # Samples x seconds
ROLLOVER_WINDOW = {
    'cmd': ROLL_WINDOW,
    'ref': ROLL_WINDOW,
    'odom': 10 * ROLL_WINDOW,
    'cmd_vel': ROLL_WINDOW,
    'cmd_state': ROLL_WINDOW,
    'target_state': ROLL_WINDOW,
}

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--duration', type=float, default=10.0, help='History to plot (seconds)')
args, unknown_args = parser.parse_known_args(sys.argv[1:])

history_duration = args.duration

# Dummy live data sources (replace with ROS-linked sources)


def make_twist_source():
    return ColumnDataSource(data={'time': [], 'vx': [], 'vy': [], 'vz': [], 'wx': [], 'wy': [], 'wz': []})


cmd_twist_source = make_twist_source()
ref_twist_source = make_twist_source()
odom_twist_source = make_twist_source()
cmd_state_twist_source = make_twist_source()
target_state_twist_source = make_twist_source()
cmd_vel_source = ColumnDataSource(data={'time': [], 'vx': [], 'vy': [], 'wz': [], 'vz': []})
position_source = ColumnDataSource(data={'x': [0.0], 'y': [0.0], 'z': [0.0]})
trajectory_source = ColumnDataSource(data={'time': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': []})
cmd_state_source = ColumnDataSource(data={'time': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': []})
target_state_source = ColumnDataSource(data={'time': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': []})

delay_div = Div(text='Delay text ...', width=100, height=40, name='delay_div')
time_source_data = {'now': [0]}
time_source = ColumnDataSource(data=time_source_data, name='time_source')
time_source.js_on_change('data', CustomJS(args={'source': time_source, 'delay_div': delay_div}, code="""
    const now = Date.now();  // ms of wall clock
    const py_time = source.data['now'][0];
    const delay = now - py_time;

    let color = "darkgreen";
    if (delay > 500) {
        color = "red";
    } else if (delay > 200) {
        color = "orangered";
    }

    delay_div.text = `<b>Delay:</b> <span style="color:${color};">${delay} ms</span>`;
    console.log("JS callback triggered: ", delay);
    source.change.emit();
"""))
status_div = Div(text='Status: waiting for data...', width=420, height=40)
dummy_plot = figure(height=1, width=1, toolbar_location=None)
dummy_plot.xaxis.visible = False
dummy_plot.yaxis.visible = False
dummy_plot.outline_line_color = None
dummy_plot.grid.visible = False
dummy_plot.border_fill_color = None
dummy_plot.min_border = 0
dummy_plot.scatter(x='now', y='now', source=time_source, size=0)  # invisible glyph


# Color maps
linear_colors = {'x': '#1f77b4', 'y': '#2ca02c', 'z': '#d62728'}
angular_colors = {'x': '#9467bd', 'y': '#ff7f0e', 'z': '#17becf'}
angular_colors.update({'roll': '#9467bd', 'pitch': '#ff7f0e', 'yaw': '#17becf'})
line_width = 2

# LINEAR VELOCITY PLOT
lin_plot = figure(title='Linear Velocity (meters/sec)', width=600, height=250, y_range=Range1d(-0.05, 0.05))
linear_lines = []
for axis in 'xyz':
    color = linear_colors[axis]
    linear_lines.append(
        lin_plot.line('time', f'v{axis}', source=cmd_twist_source, legend_label=f'v{axis} cmd',
                      name=f'vx_set_{axis}', line_color=color, line_dash='dashed', line_width=line_width)
    )
    linear_lines.append(
        lin_plot.line('time', f'v{axis}', source=cmd_state_twist_source, legend_label=f'v{axis} cmd_state',
                      name=f'vx_cmd_state_{axis}', line_color=color, line_dash='dashed',
                      line_width=line_width)
    )
    linear_lines.append(
        lin_plot.line('time', f'v{axis}', source=target_state_twist_source, legend_label=f'v{axis} target_state',
                      name=f'vx_target_state_{axis}', line_color=color, line_dash='dotdash',
                      line_width=line_width)
    )
    if axis != 'z':
        linear_lines.append(
            lin_plot.line('time', f'v{axis}', source=cmd_vel_source, legend_label=f'v{axis} cmd_vel',
                          name=f'vx_cmd_vel_{axis}', line_color=color, line_dash='dotted', line_width=line_width)
        )
    linear_lines.append(
        lin_plot.line('time', f'v{axis}', source=odom_twist_source, legend_label=f'v{axis} act',
                      name=f'vx_act_{axis}', line_color=color, line_width=line_width)
    )
lin_plot.legend.click_policy = 'hide'
lin_plot.legend.label_text_font_size = '10pt'
lin_plot.legend.location = 'top_left'
lin_plot.legend.orientation = 'vertical'

# ANGULAR VELOCITY PLOT
ang_plot = figure(title='Angular Velocity (radians/sec)', width=600, height=250, y_range=Range1d(-0.05, 0.05))
angular_lines = []
for axis in 'xyz':
    color = angular_colors[axis]
    angular_lines.append(
        ang_plot.line('time', f'w{axis}', source=cmd_twist_source, legend_label=f'w{axis} cmd',
                      name=f'wx_set_{axis}', line_color=color, line_dash='dashed', line_width=line_width)
    )
    angular_lines.append(
        ang_plot.line('time', f'w{axis}', source=cmd_state_twist_source, legend_label=f'w{axis} cmd_state',
                      name=f'wx_cmd_state_{axis}', line_color=color, line_dash='dashed',
                      line_width=line_width)
    )
    angular_lines.append(
        ang_plot.line('time', f'w{axis}', source=target_state_twist_source, legend_label=f'w{axis} target_state',
                      name=f'wx_target_state_{axis}', line_color=color, line_dash='dotdash',
                      line_width=line_width)
    )
    if axis == 'z':
        angular_lines.append(
            ang_plot.line('time', f'w{axis}', source=cmd_vel_source, legend_label=f'w{axis} cmd_vel',
                          name=f'wx_cmd_vel_{axis}', line_color=color, line_dash='dotted', line_width=line_width)
        )
    angular_lines.append(
        ang_plot.line('time', f'w{axis}', source=odom_twist_source, legend_label=f'w{axis} act',
                      name=f'wx_act_{axis}', line_color=color, line_width=line_width)
    )
ang_plot.legend.click_policy = 'hide'
ang_plot.legend.label_text_font_size = '10pt'
ang_plot.legend.location = 'top_left'
ang_plot.legend.orientation = 'vertical'

# Position Time Plot
xyz_plot = figure(title='Position (meters)', width=600, height=250, y_range=Range1d(-0.05, 0.05))
xyz_lines = []
for axis in 'xyz':
    color = linear_colors[axis]
    xyz_lines.append(
        xyz_plot.line('time', f'{axis}', source=trajectory_source, legend_label=f'{axis} (m)',
                      name=f'act_{axis}', line_color=color, line_width=line_width)
    )
    xyz_lines.append(
        xyz_plot.line('time', f'{axis}', source=cmd_state_source, legend_label=f'{axis} cmd_state',
                      name=f'cmd_state_{axis}', line_color=color, line_dash='dashed',
                      line_width=line_width)
    )
    xyz_lines.append(
        xyz_plot.line('time', f'{axis}', source=target_state_source, legend_label=f'{axis} target_state',
                      name=f'target_state_{axis}', line_color=color, line_dash='dotdash',
                      line_width=line_width)
    )
xyz_plot.legend.click_policy = 'hide'
xyz_plot.legend.label_text_font_size = '10pt'
xyz_plot.legend.location = 'top_left'
xyz_plot.legend.orientation = 'vertical'

# RPY Time Plot
rpy_plot = figure(title='Euler (degrees)', width=600, height=250, y_range=Range1d(-0.05, 0.05))
rpy_lines = []
for axis in ['roll', 'pitch', 'yaw']:
    color = angular_colors[axis]
    rpy_lines.append(
        rpy_plot.line('time', f'{axis}', source=trajectory_source, legend_label=f'{axis}',
                      name=f'euler_act_{axis}', line_color=color, line_width=line_width)
    )
    rpy_lines.append(
        rpy_plot.line('time', f'{axis}', source=cmd_state_source, legend_label=f'{axis} cmd_state',
                      name=f'euler_cmd_state_{axis}', line_color=color, line_dash='dashed',
                      line_width=line_width)
    )
    rpy_lines.append(
        rpy_plot.line('time', f'{axis}', source=target_state_source, legend_label=f'{axis} target_state',
                      name=f'euler_target_state_{axis}', line_color=color, line_dash='dotdash',
                      line_width=line_width)
    )
rpy_plot.legend.click_policy = 'hide'
rpy_plot.legend.label_text_font_size = '10pt'
rpy_plot.legend.location = 'top_left'
rpy_plot.legend.orientation = 'vertical'


# XY and Z Position Plots
xy_plot = figure(title='XY Position', width=500, height=500, x_range=(-3, 3), y_range=(-3, 3))
xy_plot.line('x', 'y', source=trajectory_source, line_color=color, line_width=line_width)
xy_plot.line('x', 'y', source=cmd_state_source, line_color='black', line_dash='dashed', line_width=line_width)
xy_plot.line('x', 'y', source=target_state_source, line_color='black', line_dash='dotdash', line_width=line_width)
xy_plot.scatter('x', 'y', source=position_source, size=10, color='navy')

z_bar = figure(title='Height (Z)', width=80, height=500, toolbar_location=None, x_range=(0, 1), y_range=(0, 2.5))
z_bar.vbar(x=0.5, top='z', width=0.8, source=position_source, color='blue')
z_bar.xaxis.visible = False
z_bar.yaxis.axis_label = 'Height'

xz_plot = figure(title='XZ Position', width=500, height=250, x_range=(-3, 3), y_range=(0, 3))
xz_plot.line('x', 'z', source=trajectory_source, line_color=color, line_width=line_width)
xz_plot.line('x', 'z', source=cmd_state_source, line_color='black', line_dash='dashed', line_width=line_width)
xz_plot.line('x', 'z', source=target_state_source, line_color='black', line_dash='dotdash', line_width=line_width)
xz_plot.scatter('x', 'z', source=position_source, size=10, color='navy')

yz_plot = figure(title='YZ Position', width=500, height=250, x_range=(-3, 3), y_range=(0, 3))
yz_plot.line('y', 'z', source=trajectory_source, line_color=color, line_width=line_width)
yz_plot.line('y', 'z', source=cmd_state_source, line_color='black', line_dash='dashed', line_width=line_width)
yz_plot.line('y', 'z', source=target_state_source, line_color='black', line_dash='dotdash', line_width=line_width)
yz_plot.scatter('y', 'z', source=position_source, size=10, color='navy')

# Axis selectors
lin_axis_selector = CheckboxGroup(labels=['vx', 'vy', 'vz'], active=[0, 1, 2])
ang_axis_selector = CheckboxGroup(labels=['wx', 'wy', 'wz'], active=[0, 1, 2])
xyz_axis_selector = CheckboxGroup(labels=['x', 'y', 'z'], active=[0, 1, 2])
rpy_axis_selector = CheckboxGroup(labels=['roll', 'pitch', 'yaw'], active=[0, 1, 2])

lin_axis_selector.js_on_change('active', CustomJS(args={'lines': linear_lines}, code="""
    const axes = ['x', 'y', 'z'];
    const visible = new Set(cb_obj.active.map(i => axes[i]));
    for (const r of lines) {
        const axis = r.name.slice(-1);
        r.visible = visible.has(axis);
    }
"""))

ang_axis_selector.js_on_change('active', CustomJS(args={'lines': angular_lines}, code="""
    const axes = ['x', 'y', 'z'];
    const visible = new Set(cb_obj.active.map(i => axes[i]));
    for (const r of lines) {
        const axis = r.name.slice(-1);
        r.visible = visible.has(axis);
    }
"""))

# Live status text
status_div = Div(text='', width=800)

# Simulated data queues
twist_queue = Queue()
pos_queue = Queue()
cmd_vel_queue = Queue()
traj_queue = Queue()
cmd_state_queue = Queue()
target_state_queue = Queue()

# --- ROS 2 Node Definition ---


class DronePlotNode(Node):
    LOG_DATASETS = ('odom', 'cmd_vel', 'cmd', 'ref', 'cmd_state', 'target_state', 'status')

    def __init__(self):
        super().__init__('drone_plot_node')
        self.start_time = self.get_clock().now().nanoseconds

        # --- ROS topic names ---
        odom_topic = 'odom'
        cmd_vel_topic = 'cmd_vel'
        twist_topic = 'reference/twist'
        cmd_topic = 'command/twist'
        cmd_state_topic = 'cmd_state'
        target_state_topic = 'target_state'
        status_topic = 'status'

        cnt = 0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)
            if self.get_clock().now().nanoseconds > 0:
                print('Received valid clock!')
                break

            if cnt > 2500:
                print('Never received valid clock!')
                break
            cnt += 1

        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.create_subscription(TwistStamped, cmd_vel_topic, self.cmd_vel_cb, 10)
        self.create_subscription(Twist, twist_topic, self.ref_cb, 10)
        self.create_subscription(Twist, cmd_topic, self.cmd_cb, 10)
        self.create_subscription(FullState, cmd_state_topic, self.cmdstate_cb, 10)
        self.create_subscription(FullState, target_state_topic, self.targetstate_cb, 10)
        self.create_subscription(Status, status_topic, self.status_cb, 10)

        self.declare_parameter('logging', True,
                               descriptor=ParameterDescriptor(
                                   description='Do simple data logging using numpy'
                               ))
        self.declare_parameter('logging_folder', f'log{self.get_namespace()}',
                               descriptor=ParameterDescriptor(
            description='Folder for data logging'
        ))
        self.declare_parameter('logging_buffer_length', 100 * 60,  # 1 minutes at 100Hz
                               descriptor=ParameterDescriptor(
                                   description='Number of samples to store before saving (default: 6000)'
                               ))

        self.logging = self.get_parameter('logging').value

        logging_folder = self.get_parameter('logging_folder').value

        self.logging_start = self.get_clock().now().nanoseconds
        self.log_stamp = 0

        if self.logging:
            self.logging_buffer_length = self.get_parameter('logging_buffer_length').value
            workspace_root = os.environ.get('WORKSPACE_ROOT')
            if workspace_root is None:
                raise EnvironmentError('WORKSPACE_ROOT environment variable is not set.')
            log_dir = Path(workspace_root) / logging_folder
            self.get_logger().info(f"Logging to '{log_dir}'")
            log_dir.mkdir(parents=True, exist_ok=True)
            self.logging_path = log_dir

            # Define a buffer pool to hold data while writing
            self.logging_buffers = [self._new_logging_buffer(), self._new_logging_buffer()]
            self.num_buffers = len(self.logging_buffers)
            self.index_buffer = 0
            self.log_index = 0
            self._write_threads = []
        else:
            self.get_logger().warning('Logs are NOT being saved!')

    def elapsed_time(self):
        return (self.get_clock().now().nanoseconds - self.start_time) * 1e-9

    def _new_logging_buffer(self):
        return {
            'odom': np.zeros((self.logging_buffer_length, 14)),
            'cmd_vel': np.zeros((self.logging_buffer_length, 5)),
            'cmd': np.zeros((self.logging_buffer_length, 7)),
            'ref': np.zeros((self.logging_buffer_length, 7)),
            # stamp, pos x,y,z, ori x,y,z,w, linear vel x,y,z,
            # angular vel x,y,z, valid_mask
            'cmd_state': np.zeros((self.logging_buffer_length, FULLSTATE_COLUMNS)),
            'target_state': np.zeros((self.logging_buffer_length, FULLSTATE_COLUMNS)),
            # stamp_us, state, status_flags, battery_mV, battery_pct,
            # link_latency_ms, battery_current_a
            'status': np.zeros((self.logging_buffer_length, 7)),
            'odom_index': 0,
            'cmd_vel_index': 0,
            'cmd_index': 0,
            'ref_index': 0,
            'cmd_state_index': 0,
            'target_state_index': 0,
            'status_index': 0,
        }

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

    def odom_cb(self, msg):
        import tf_transformations

        now = self.elapsed_time()
        nanoseconds = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec - self.logging_start
        stamp = nanoseconds * 1.e-9
        twist_queue.put(('odom', now, stamp, msg.twist.twist.linear, msg.twist.twist.angular))
        pos_queue.put(msg.pose.pose.position)
        r, p, y = tf_transformations.euler_from_quaternion([msg.pose.pose.orientation.x,
                                                            msg.pose.pose.orientation.y,
                                                            msg.pose.pose.orientation.z,
                                                            msg.pose.pose.orientation.w])
        # Convert to degrees for display
        r *= 180. / np.pi
        y *= 180. / np.pi
        p *= 180. / np.pi

        traj_queue.put((now, stamp, msg.pose.pose.position, r, p, y))
        if self.logging:
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
        now = self.elapsed_time()
        nanoseconds = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec - self.logging_start
        stamp = nanoseconds * 1.e-9
        cmd_vel_queue.put((
            'cmd_vel',
            now,
            stamp,
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.angular.z,
            msg.twist.linear.z,
        ))

        if self.logging:
            buffer, row_index = self._active_buffer_for_write('cmd_vel', nanoseconds)
            cmd_vel_buffer = buffer['cmd_vel']
            cmd_vel_buffer[row_index][0] = stamp
            cmd_vel_buffer[row_index][1] = msg.twist.linear.x
            cmd_vel_buffer[row_index][2] = msg.twist.linear.y
            cmd_vel_buffer[row_index][3] = msg.twist.angular.z
            cmd_vel_buffer[row_index][4] = msg.twist.linear.z
            row_index += 1
            buffer['cmd_vel_index'] = row_index
            self._flush_if_needed(row_index, nanoseconds)

    def cmd_cb(self, msg):
        now = self.elapsed_time()
        nanoseconds = self.get_clock().now().nanoseconds - self.logging_start
        stamp = nanoseconds * 1e-9
        twist_queue.put(('cmd', now, stamp, msg.linear, msg.angular))
        if self.logging:
            buffer, row_index = self._active_buffer_for_write('cmd', nanoseconds)
            twist_buffer = buffer['cmd']
            twist_buffer[row_index][0] = stamp
            twist_buffer[row_index][1] = msg.linear.x
            twist_buffer[row_index][2] = msg.linear.y
            twist_buffer[row_index][3] = msg.linear.z
            twist_buffer[row_index][4] = msg.angular.x
            twist_buffer[row_index][5] = msg.angular.y
            twist_buffer[row_index][6] = msg.angular.z
            row_index += 1
            buffer['cmd_index'] = row_index
            self._flush_if_needed(row_index, nanoseconds)

    def ref_cb(self, msg):
        now = self.elapsed_time()
        nanoseconds = self.get_clock().now().nanoseconds - self.logging_start
        stamp = nanoseconds * 1e-9
        twist_queue.put(('ref', now, stamp, msg.linear, msg.angular))
        if self.logging:
            buffer, row_index = self._active_buffer_for_write('ref', nanoseconds)
            twist_buffer = buffer['ref']
            twist_buffer[row_index][0] = stamp
            twist_buffer[row_index][1] = msg.linear.x
            twist_buffer[row_index][2] = msg.linear.y
            twist_buffer[row_index][3] = msg.linear.z
            twist_buffer[row_index][4] = msg.angular.x
            twist_buffer[row_index][5] = msg.angular.y
            twist_buffer[row_index][6] = msg.angular.z
            row_index += 1
            buffer['ref_index'] = row_index
            self._flush_if_needed(row_index, nanoseconds)

    def _fullstate_display_cb(self, msg, data_name, pose_queue):
        import tf_transformations

        nanoseconds = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec - self.logging_start
        stamp = nanoseconds * 1.e-9
        valid_mask = int(getattr(msg, 'valid_mask', 0))
        position_valid = bool(valid_mask & FullState.VALID_POSITION)
        orientation_valid = bool(valid_mask & FullState.VALID_ORIENTATION)
        linear_velocity_valid = bool(valid_mask & FullState.VALID_LINEAR_VELOCITY)
        angular_rate_valid = bool(valid_mask & FullState.VALID_ANGULAR_RATE)

        target_position = msg.pose.position
        if not position_valid:
            target_position = SimpleNamespace(x=np.nan, y=np.nan, z=np.nan)

        roll = pitch = yaw = np.nan
        if orientation_valid:
            try:
                roll, pitch, yaw = tf_transformations.euler_from_quaternion([
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ])
                roll = np.degrees(roll)
                pitch = np.degrees(pitch)
                yaw = np.degrees(yaw)
            except Exception:
                roll = pitch = yaw = np.nan

        target_linear = msg.twist.linear
        if not linear_velocity_valid:
            target_linear = SimpleNamespace(x=np.nan, y=np.nan, z=np.nan)
        target_angular = SimpleNamespace(
            x=msg.twist.angular.x if angular_rate_valid else np.nan,
            y=msg.twist.angular.y if angular_rate_valid else np.nan,
            z=msg.twist.angular.z if angular_rate_valid else np.nan,
        )
        now = self.elapsed_time()
        twist_queue.put((data_name, now, stamp, target_linear, target_angular))
        pose_queue.put((now, stamp, target_position, roll, pitch, yaw))

        if not self.logging:
            return

        buffer, row_index = self._active_buffer_for_write(data_name, nanoseconds)
        fullstate_buffer = buffer[data_name]
        fullstate_buffer[row_index] = [
            stamp,
            target_position.x, target_position.y, target_position.z,
            msg.pose.orientation.x if orientation_valid else np.nan,
            msg.pose.orientation.y if orientation_valid else np.nan,
            msg.pose.orientation.z if orientation_valid else np.nan,
            msg.pose.orientation.w if orientation_valid else np.nan,
            target_linear.x, target_linear.y, target_linear.z,
            target_angular.x, target_angular.y, target_angular.z,
            float(valid_mask),
        ]
        row_index += 1
        buffer[f'{data_name}_index'] = row_index
        self._flush_if_needed(row_index, nanoseconds)

    def cmdstate_cb(self, msg):
        self._fullstate_display_cb(msg, 'cmd_state', cmd_state_queue)

    def targetstate_cb(self, msg):
        self._fullstate_display_cb(msg, 'target_state', target_state_queue)

    def status_cb(self, msg):
        nanoseconds = self.get_clock().now().nanoseconds - self.logging_start
        if not self.logging:
            return

        buffer, row_index = self._active_buffer_for_write('status', nanoseconds)
        status_buffer = buffer['status']
        status_buffer[row_index][0] = nanoseconds // 1000  # int microseconds
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
        counts = ', '.join(f"{data}={buffer[f'{data}_index']}" for data in self.LOG_DATASETS)
        print(
            f'Writing log index={self.log_index} with ({counts}) points '
            f'at {log_stamp * 1.e-9:.6f} seconds ...'
        )
        for data in self.LOG_DATASETS:
            index_key = f'{data}_index'
            row_count = buffer[index_key]
            if row_count > 0:
                file_base = self.logging_path / f'{data}_{self.logging_start}_{self.log_index}.npy'
                data_snapshot = buffer[data][:row_count].copy()
                thread = Thread(target=np.save, args=(file_base, data_snapshot), daemon=True)
                self._write_threads.append(thread)
                thread.start()
            buffer[index_key] = 0  # reset the index for next logging
        self.log_index += 1
        self.log_stamp = log_stamp
        print('Wrote logs. ')


def truncate_data_by_time(new_data, cutoff_time):
    time_array = np.array(new_data['time'])
    mask = time_array >= cutoff_time
    for k in new_data:
        new_data[k] = np.array(new_data[k])[mask].tolist()
    return new_data


def adjust_plot_range(plot, sources, keys, min_abs=0.01, max_abs=25.0):
    min_val, max_val = float('inf'), -float('inf')
    for source in sources:
        for key in keys:
            if key in source.data and len(source.data[key]) > 0:
                arr = np.array(source.data[key])
                min_val = min(min_val, arr.min())
                max_val = max(max_val, arr.max())

    min_val = abs(min(min_val, -min_abs))
    max_val = abs(max(max_val, min_abs))
    abs_val = min(max_abs, max(min_val, max_val))  # Use absolute and keep symmetric about 0

    plot.y_range.start = -abs_val
    plot.y_range.end = abs_val


def adjust_xy_plot_range(plot, sources, keys,
                         min_x_val=-10.0, max_x_val=10.0, min_x_abs=1.0,
                         min_y_val=-10.0, max_y_val=10.0, min_y_abs=1.0,
                         range_scale=1.0, keep_symmetric=False):
    limits = {}
    for key in keys:
        limits[key] = {'min': float('inf'), 'max': -float('inf')}
    for source in sources:
        for key in keys:
            if key in source.data and len(source.data[key]) > 0:
                arr = np.array(source.data[key])
                limits[key]['min'] = min(limits[key]['min'], arr.min())
                limits[key]['max'] = max(limits[key]['max'], arr.max())

    if keep_symmetric:
        x_min_val = max(min_x_val, (min(limits[keys[0]]['min'], -min_x_abs)))
        x_max_val = min(max_x_val, (max(limits[keys[0]]['max'], min_x_abs)))
        y_min_val = max(min_y_val, (min(limits[keys[1]]['min'], -min_y_abs)))
        y_max_val = min(max_y_val, (max(limits[keys[1]]['max'], min_y_abs)))

        mx = max(abs(x_min_val), abs(x_max_val))
        my = max(abs(y_min_val), abs(y_max_val), mx)
        x_min_val = -my
        x_max_val = my
        y_min_val = -my
        y_max_val = my
    else:
        x_min_val = max(min_x_val, (min(limits[keys[0]]['min'], -min_x_abs)))
        x_max_val = min(max_x_val, (max(limits[keys[0]]['max'], min_x_abs)))
        y_min_val = max(min_y_val, (min(limits[keys[1]]['min'], -min_y_abs)))
        y_max_val = min(max_y_val, (max(limits[keys[1]]['max'], min_y_abs)))

        x_range = x_max_val - x_min_val
        y_range = y_max_val - y_min_val
        if abs(y_range - range_scale * x_range) < 1.e-6:
            # Enforce proportional axes in view
            y_max_val = y_min_val + x_range * range_scale

    plot.x_range.start = x_min_val
    plot.x_range.end = x_max_val
    plot.y_range.start = y_min_val
    plot.y_range.end = y_max_val
    return x_min_val, x_max_val, y_min_val, y_max_val

# Update function with .stream and status reporting


def update_sources():
    # pr = cProfile.Profile()
    # pr.enable()

    now = time.time()
    updates = 0
    dropped = []
    queue_size_text = (
        f' cmd_vel {cmd_vel_queue.qsize()} &nbsp; | Cmd {twist_queue.qsize()} &nbsp; |'
        f' Act {pos_queue.qsize()} &nbsp; |  Traj {traj_queue.qsize()} &nbsp; |'
        f' CmdState {cmd_state_queue.qsize()} &nbsp; |'
        f' Target {target_state_queue.qsize()} &nbsp; |'
    )

    if twist_queue.qsize() > 0:
        try:
            source_data = {'cmd': {'time': [], 'vx': [], 'vy': [], 'vz': [], 'wx': [], 'wy': [], 'wz': []},
                           'ref': {'time': [], 'vx': [], 'vy': [], 'vz': [], 'wx': [], 'wy': [], 'wz': []},
                           'odom': {'time': [], 'vx': [], 'vy': [], 'vz': [], 'wx': [], 'wy': [], 'wz': []},
                           'cmd_state': {'time': [], 'vx': [], 'vy': [], 'vz': [], 'wx': [], 'wy': [], 'wz': []},
                           'target_state': {'time': [], 'vx': [], 'vy': [], 'vz': [], 'wx': [], 'wy': [], 'wz': []}}
            while True:
                kind, now_lin, t, v, w = twist_queue.get_nowait()
                source_data[kind]['time'].append(t)
                source_data[kind]['vx'].append(v.x)
                source_data[kind]['vy'].append(v.y)
                source_data[kind]['vz'].append(v.z)
                source_data[kind]['wx'].append(w.x)
                source_data[kind]['wy'].append(w.y)
                source_data[kind]['wz'].append(w.z)
                updates += 1
        except Empty:
            # Stream once per update
            for key, source in {
                'cmd': cmd_twist_source,
                'odom': odom_twist_source,
                'ref': ref_twist_source,
                'cmd_state': cmd_state_twist_source,
                'target_state': target_state_twist_source,
            }.items():
                source.stream(source_data[key], rollover=ROLLOVER_WINDOW[key])

    if traj_queue.qsize() > 0:
        try:
            source_data = {'time': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': []}
            while True:
                now_traj, t, pos, r, p, y = traj_queue.get_nowait()
                source_data['time'].append(t)
                source_data['x'].append(pos.x)
                source_data['y'].append(pos.y)
                source_data['z'].append(pos.z)
                source_data['roll'].append(r)
                source_data['pitch'].append(p)
                source_data['yaw'].append(y)
                updates += 1
        except Empty:
            # Stream once per update
            trajectory_source.stream(source_data, rollover=ROLLOVER_WINDOW['odom'])

    if cmd_state_queue.qsize() > 0:
        try:
            source_data = {'time': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': []}
            while True:
                now_traj, t, pos, r, p, y = cmd_state_queue.get_nowait()
                source_data['time'].append(t)
                source_data['x'].append(pos.x)
                source_data['y'].append(pos.y)
                source_data['z'].append(pos.z)
                source_data['roll'].append(r)
                source_data['pitch'].append(p)
                source_data['yaw'].append(y)
                updates += 1
        except Empty:
            cmd_state_source.stream(source_data, rollover=ROLLOVER_WINDOW['cmd_state'])

    if target_state_queue.qsize() > 0:
        try:
            source_data = {'time': [], 'x': [], 'y': [], 'z': [], 'roll': [], 'pitch': [], 'yaw': []}
            while True:
                now_traj, t, pos, r, p, y = target_state_queue.get_nowait()
                source_data['time'].append(t)
                source_data['x'].append(pos.x)
                source_data['y'].append(pos.y)
                source_data['z'].append(pos.z)
                source_data['roll'].append(r)
                source_data['pitch'].append(p)
                source_data['yaw'].append(y)
                updates += 1
        except Empty:
            target_state_source.stream(source_data, rollover=ROLLOVER_WINDOW['target_state'])

    if pos_queue.qsize() > 0:
        pos = None
        try:
            while True:
                pos = pos_queue.get_nowait()
        except Empty:
            pass

        if pos:
            position_source.data = {'x': [pos.x], 'y': [pos.y], 'z': [pos.z]}
        else:
            print('invalid pos_queue - no data!')

    if cmd_vel_queue.qsize() > 0:
        try:
            source_data = {'time': [], 'vx': [], 'vy': [], 'wz': [], 'vz': []}
            while True:
                kind, now_ang, t, vx, vy, wz, vz = cmd_vel_queue.get_nowait()
                source_data['time'].append(t)
                source_data['vx'].append(vx)
                source_data['vy'].append(vy)
                source_data['wz'].append(wz)
                source_data['vz'].append(vz)
                updates += 1
        except Empty:
            cmd_vel_source.stream(source_data, rollover=ROLLOVER_WINDOW['cmd_vel'])
            pass

    # Queue monitoring
    for name, q in [('lin', twist_queue), ('pos', pos_queue), ('cmd_vel', cmd_vel_queue),
                    ('cmd_state', cmd_state_queue), ('target_state', target_state_queue)]:
        if q.qsize() > QUEUE_WARN_LIMIT:
            dropped.append(f'⚠️ {name}_queue={q.qsize()}')

    if len(odom_twist_source.data['time']) > 0:
        latest_t = odom_twist_source.data['time'][-1]
    else:
        latest_t = 0

    now_str = time.strftime('%H:%M:%S', time.localtime(now)) + f'.{int((now % 1) * 1000):03d}'
    status_text = (f'<b>Last update:</b> {now_str} ({latest_t:.6f} s) &nbsp; | '
                   f'<b>queue sizes: </b> &nbsp; '
                   f'{queue_size_text}')
    status_div.text = status_text
    # print(status_text.replace("<b>","").replace("</b>", "").replace("&nbsp;", "")
    #       .replace("<span style='color:","").replace(";'>", "").replace("</span>",""))

    time_source.data['now'][0] = int(now * 1000.0)
    time_source.trigger('data', time_source.data, time_source.data)  # tell Bokeh to refresh JS side

    # Adjust y-axis limits with min threshold to avoid low level noise
    adjust_plot_range(lin_plot,
                      sources=[cmd_twist_source, odom_twist_source, cmd_vel_source,
                               cmd_state_twist_source, target_state_twist_source],
                      keys=['vx', 'vy', 'vz'], min_abs=0.01, max_abs=2.0)

    adjust_plot_range(ang_plot,
                      sources=[cmd_twist_source, odom_twist_source, cmd_vel_source,
                               cmd_state_twist_source, target_state_twist_source],
                      keys=['wx', 'wy', 'wz'], min_abs=0.01, max_abs=6.28)

    x_min_val, x_max_val, y_min_val, y_max_val = adjust_xy_plot_range(
        xy_plot,
        sources=[trajectory_source, cmd_state_source, target_state_source],
        keys=['x', 'y'],
        min_x_val=-10.0, max_x_val=10.0, min_x_abs=1.0,
        min_y_val=-10.0, max_y_val=10.0, min_y_abs=1.0,
        range_scale=1.0, keep_symmetric=True)

    adjust_xy_plot_range(xz_plot,
                         sources=[trajectory_source, cmd_state_source, target_state_source],
                         keys=['x', 'z'],
                         min_x_val=x_min_val, max_x_val=x_max_val, min_x_abs=x_max_val,
                         min_y_val=-0.5, max_y_val=3.0, min_y_abs=1.0,
                         range_scale=0.5)

    adjust_xy_plot_range(yz_plot,
                         sources=[trajectory_source, cmd_state_source, target_state_source],
                         keys=['y', 'z'],
                         min_x_val=y_min_val, max_x_val=y_max_val, min_x_abs=y_max_val,
                         min_y_val=-0.5, max_y_val=3.0, min_y_abs=1.0,
                         range_scale=0.5)

    adjust_plot_range(xyz_plot,
                      sources=[trajectory_source, cmd_state_source, target_state_source],
                      keys=['x', 'y', 'z'], min_abs=0.01, max_abs=10.0)

    adjust_plot_range(rpy_plot,
                      sources=[trajectory_source, cmd_state_source, target_state_source],
                      keys=['roll', 'pitch', 'yaw'], min_abs=1.0, max_abs=360.0)

    # # Print top 10 cumulative time functions
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats(10)
    # print(s.getvalue())


def start_app():
    rclpy.init(args=unknown_args)
    node = DronePlotNode()
    shutdown_event = Event()
    ros_thread = None

    def ros_spin():
        try:
            while not shutdown_event.is_set() and rclpy.ok():
                while rclpy.spin_once(node, timeout_sec=0.001):
                    # Spins until no work processed before timeout
                    pass
                time.sleep(0.005)  # Cooperatively release CPU

        except BENIGN_SHUTDOWN_EXCEPTIONS:
            pass

        if shutdown_event.is_set():
            print('External shutdown triggered ...')

        if rclpy.ok():
            node.destroy_node()  # Stop the ROS stuff

        if node.logging:
            # Dump any final information to logs
            node.write_log(node.logging_buffers[node.index_buffer], -1)
            for thread in node._write_threads:
                thread.join()

        rclpy.shutdown()
        print('ROS node is done!')

    def stop_ros_thread():
        shutdown_event.set()
        if ros_thread is not None and ros_thread.is_alive():
            ros_thread.join(timeout=4)

    print('Starting ROS spin loop')
    ros_thread = Thread(target=ros_spin, daemon=True)
    ros_thread.start()

    def signal_handler(sig, frame):
        print('Caught Ctrl-C, shutting down...')
        print('Waiting on ROS shutdown ...')
        stop_ros_thread()
        print('Done signal handler!')

    signal.signal(signal.SIGINT, signal_handler)

    # Layout
    lin_section = row(lin_plot, lin_axis_selector)
    ang_section = row(ang_plot, ang_axis_selector)
    xyz_section = row(xyz_plot, xyz_axis_selector)
    rpy_section = row(rpy_plot, rpy_axis_selector)

    time_layout = column(lin_section, ang_section, xyz_section, rpy_section)
    xy_layout = row(xy_plot, z_bar)
    xyz_layout = column(xy_layout, xz_plot, yz_plot)

    layout = column(row(status_div, delay_div, dummy_plot),
                    row(xyz_layout, time_layout),
                    sizing_mode='stretch_both')
    doc = curdoc()
    doc.add_root(layout)
    doc.title = f'{node.get_namespace()} Drone Telemetry'
    doc.add_periodic_callback(partial(update_sources), 333)
    doc.on_session_destroyed(lambda session_context: stop_ros_thread())


if __name__ == '__main__' or __name__.startswith('bokeh_app_'):
    start_app()

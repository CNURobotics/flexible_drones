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

import argparse
import os
import re

import numpy as np
from pathlib import Path

from flexible_drones_core import ansi
from flexible_drones_msgs.msg import DroneStatus as Status
from flexible_drones_tools.trajectories.utilities.plotting import set_equal_3d_bounds


LOG_DATASETS = (
    'odom',
    'cmd_vel',
    'cmd',
    'ref',
    'pose',
    'cmd_full',
    'cmd_state',
    'target_state',
    'full_state',
    'status',
    'posctl',
    'velctl',
)

EXPECTED_COLUMNS = {
    'odom': 14,
    'cmd_vel': 6,
    'cmd': 7,
    'ref': 7,
    'pose': 8,
    'cmd_full': 15,
    'cmd_state': 15,
    'target_state': 15,
    'full_state': 15,
    'status': 6,
    'posctl': 4,
    'velctl': 5,
}


BODY_LINEAR_VELOCITY_SOURCES = (
    ('odom', 8, '-'),
    ('full_state', 8, '-'),
    ('cmd_full', 8, '--'),
    ('cmd_state', 8, '--'),
    ('target_state', 8, '--'),
)
# Logged velocity frame_code: 0 = body/FLU, 1 = world/ENU. World-frame samples
# arrive labelled either '<drone>/odom' (generated control) or the legacy 'map'
# frame (teleop cmd_vel); the logger collapses both to FRAME_CODE_WORLD.
FRAME_CODE_BODY = 0
FRAME_CODE_WORLD = 1
ANIMATIONS = []
SOURCE_COLORS = {
    'odom': 'tab:blue',
    'full_state': 'tab:cyan',
    'pose': 'tab:green',
    'cmd_full': 'tab:purple',
    'cmd_state': 'mediumpurple',
    'target_state': 'tab:orange',
    'cmd': 'tab:red',
    'ref': 'tab:pink',
    'cmd_vel': 'tab:brown',
    'posctl': 'tab:olive',
    'velctl': 'tab:gray',
    'desired': 'tab:purple',
    'target': 'tab:orange',
    'drone': 'tab:blue',
}
COMPONENT_LINESTYLES = {
    'x': '-',
    'y': '--',
    'z': ':',
    'roll': '-',
    'pitch': '--',
    'yaw': ':',
    'vx_body': '-',
    'vy_body': '--',
    'vz_body': ':',
    'vx_enu': '-',
    'vy_enu': '--',
    'vz_enu': ':',
    'wx': '-',
    'wy': '--',
    'wz': ':',
}


def source_color(source):
    key = str(source).split()[0]
    return SOURCE_COLORS.get(key, 'tab:gray')


def component_linestyle(component, fallback='-'):
    return COMPONENT_LINESTYLES.get(component, fallback)


def plot_source_line(times, values, source, component, label=None, linestyle=None, **kwargs):
    return plt.plot(
        times,
        values,
        color=source_color(source),
        linestyle=linestyle or component_linestyle(component),
        label=label or f'{component} {source}',
        **kwargs,
    )


def should_plot_angular_component(source, component):
    return True


def yaw_from_row(row):
    quat = row[4:8]
    if not np.all(np.isfinite(quat)):
        return np.nan
    try:
        _roll, _pitch, yaw = tf_transformations.euler_from_quaternion(quat)
    except Exception:
        return np.nan
    return yaw


def yaw_series(values):
    if values.ndim != 2 or values.shape[1] < 8:
        return np.zeros(values.shape[0])
    return np.array([yaw_from_row(row) for row in values], dtype=float)


def euler_degrees_from_values(values):
    rpy = []
    for row in values:
        quat = row[4:8]
        if not np.all(np.isfinite(quat)):
            rpy.append([np.nan, np.nan, np.nan])
            continue
        try:
            r, p, y = tf_transformations.euler_from_quaternion(quat)
        except Exception:
            rpy.append([np.nan, np.nan, np.nan])
            continue
        rpy.append([np.degrees(r), np.degrees(p), np.degrees(y)])
    return np.array(rpy)


def body_flu_to_enu(body_velocity, yaw):
    vx_body = body_velocity[:, 0]
    vy_body = body_velocity[:, 1]
    vz_body = body_velocity[:, 2]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.column_stack((
        vx_body * cos_yaw - vy_body * sin_yaw,
        vx_body * sin_yaw + vy_body * cos_yaw,
        vz_body,
    ))


def frame_coded_velocity_source(data, key, value_columns, frame_column, target_frame, style):
    if not has_data(data, key, frame_column + 1):
        return None
    values = data[key]
    frame_codes = np.rint(values[:, frame_column]).astype(int)
    if target_frame == 'body':
        rows = frame_codes == FRAME_CODE_BODY
    elif target_frame == 'enu':
        rows = frame_codes == FRAME_CODE_WORLD
    else:
        raise ValueError(f'unknown target frame: {target_frame}')
    if not np.any(rows):
        return None
    velocity = np.column_stack([values[rows, column] for column in value_columns])
    return key, values[rows, 0], velocity, style


def cmd_vel_velocity_source(data, target_frame):
    return frame_coded_velocity_source(data, 'cmd_vel', (1, 2, 4), 5, target_frame, ':')


def velctl_velocity_source(data, target_frame):
    return frame_coded_velocity_source(data, 'velctl', (1, 2, 3), 4, target_frame, '--')


def body_velocity_sources(data, include_velctl=True, include_cmd_vel=True):
    for key, start_col, style in BODY_LINEAR_VELOCITY_SOURCES:
        if has_data(data, key, start_col + 3):
            yield key, data[key][:, 0], data[key][:, start_col:start_col + 3], style
    if include_velctl:
        velctl_source = velctl_velocity_source(data, 'body')
        if velctl_source is not None:
            yield velctl_source
    if include_cmd_vel:
        cmd_vel_source = cmd_vel_velocity_source(data, 'body')
        if cmd_vel_source is not None:
            yield cmd_vel_source


def enu_velocity_sources_from_body(data, include_velctl=True, include_cmd_vel=True):
    for source, times, velocity_body, style in body_velocity_sources(
        data,
        include_velctl=False,
        include_cmd_vel=False,
    ):
        yaw = yaw_series(data[source])
        velocity_enu = body_flu_to_enu(velocity_body, yaw)
        yield source, times, velocity_enu, style
    if include_velctl:
        velctl_source = velctl_velocity_source(data, 'enu')
        if velctl_source is not None:
            yield velctl_source
    if include_cmd_vel:
        cmd_vel_source = cmd_vel_velocity_source(data, 'enu')
        if cmd_vel_source is not None:
            yield cmd_vel_source


def plot_velocity_group(title, sources, labels):
    sources = list(sources)
    if not sources:
        return False
    plt.figure(title)
    for source, times, velocity, style in sources:
        for i, label in enumerate(labels):
            plot_source_line(times, velocity[:, i], source, label, linestyle=component_linestyle(label, style))
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    return True


def plot_velocity_components(title_prefix, sources, labels):
    sources = list(sources)
    if not sources:
        return False
    for i, label in enumerate(labels):
        plt.figure(f'{title_prefix} {label}')
        for source, times, velocity, style in sources:
            plot_source_line(times, velocity[:, i], source, label, linestyle=component_linestyle(label, style))
        plt.title(f'{title_prefix} {label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
    return True


def load_log_data(log_dir, start_time, index_range):
    """Load available data files and stack into single numpy array per item."""
    data = {key: [] for key in LOG_DATASETS}
    found_files = {key: [] for key in LOG_DATASETS}
    missing_indexes = {key: [] for key in LOG_DATASETS}
    for idx in range(index_range[0], index_range[1] + 1):
        for key in data.keys():
            fname = log_dir / f'{key}_{start_time}_{idx}.npy'
            if fname.exists():
                print(f'Loading: {fname}')
                try:
                    data[key].append(np.load(fname))
                    found_files[key].append(fname)
                except Exception as exc:
                    print(f"Failed to load data for '{fname}': {exc}")
            else:
                missing_indexes[key].append(idx)
    for key in data:
        data[key] = np.vstack(data[key]) if data[key] else np.zeros((0, 1))
    print_log_inventory(data, found_files, missing_indexes)
    return data


def print_log_inventory(data, found_files, missing_indexes):
    available = [
        f'{key}{data[key].shape} from {len(found_files[key])} file(s)'
        for key in LOG_DATASETS
        if has_data(data, key, 1)
    ]
    missing = [
        key for key in LOG_DATASETS
        if not has_data(data, key, 1)
    ]
    partial = [
        f'{key}: missing index {format_index_list(missing_indexes[key])}'
        for key in LOG_DATASETS
        if has_data(data, key, 1) and missing_indexes[key]
    ]
    short = datasets_with_fewer_columns_than_expected(data)

    if available:
        print('[INFO] Available log datasets:')
        for item in available:
            print(f'  - {item}')
    else:
        print('[WARN] No matching log data found for the requested time/index range.')

    if missing:
        print(f'[INFO] Missing log datasets: {", ".join(missing)}')

    if partial:
        print('[INFO] Partially available log datasets:')
        for item in partial:
            print(f'  - {item}')

    if short:
        print('[WARN] Log datasets with fewer columns than expected:')
        for item in short:
            print(f'  - {item}')


def format_index_list(indexes):
    if not indexes:
        return 'none'
    if len(indexes) <= 6:
        return ', '.join(str(idx) for idx in indexes)
    return f'{indexes[0]}, {indexes[1]}, {indexes[2]}, ... {indexes[-1]}'


def has_data(data, key, min_columns):
    values = data[key]
    return values.ndim == 2 and values.shape[0] > 0 and values.shape[1] >= min_columns


def filter_data_to_armed(data):
    if not has_data(data, 'status', 3):
        print('[WARN] No status data available; cannot filter plots to armed timestamps.')
        return data

    status = data['status']
    status_times = status[:, 0] * 1e-6
    status_flags = status[:, 2].astype(np.int64)
    order = np.argsort(status_times)
    status_times = status_times[order]
    armed = (status_flags[order] & Status.STATUS_ARMED) != 0

    if not np.any(armed):
        print('[WARN] Status data contains no armed samples; filtering will remove all time-series rows.')

    filtered = {}
    count_parts = []
    for key, values in data.items():
        if not has_data(data, key, 1):
            filtered[key] = values
            continue

        if key == 'status':
            keep = armed
        else:
            sample_times = values[:, 0]
            status_indexes = np.searchsorted(status_times, sample_times, side='right') - 1
            keep = (status_indexes >= 0) & armed[np.maximum(status_indexes, 0)]

        filtered[key] = values[keep]
        count_parts.append(f'{key}: {values.shape[0]}->{filtered[key].shape[0]}')

    print(f'[INFO] Filtered logs to armed timestamps ({", ".join(count_parts)}).')
    return filtered


def finite_position_trace(values):
    rows = np.all(np.isfinite(values[:, 0:4]), axis=1)
    if not np.any(rows):
        return None

    trace = values[rows, 0:4]
    order = np.argsort(trace[:, 0])
    trace = trace[order]
    _times, unique_indexes = np.unique(trace[:, 0], return_index=True)
    trace = trace[unique_indexes]
    if trace.shape[0] == 0:
        return None
    return trace


def first_position_trace(data, keys):
    for key in keys:
        if has_data(data, key, 4):
            trace = finite_position_trace(data[key])
            if trace is not None:
                return key, trace
    return None, None


def animation_position_sources(data):
    specs = [
        ('desired', ('cmd_state', 'cmd_full', 'posctl'), 'tab:purple', '--'),
        ('target', ('target_state',), 'tab:orange', '--'),
        ('drone', ('odom', 'full_state', 'pose'), 'tab:blue', '-'),
    ]
    sources = []
    for label, keys, color, style in specs:
        key, trace = first_position_trace(data, keys)
        if trace is None:
            print(f'[INFO] No {label} position source available for 3D animation ({", ".join(keys)}).')
            continue
        sources.append({
            'label': label,
            'key': key,
            'trace': trace,
            'color': color,
            'style': style,
        })
    return sources


def interpolate_position(trace, times):
    return np.column_stack((
        np.interp(times, trace[:, 0], trace[:, 1]),
        np.interp(times, trace[:, 0], trace[:, 2]),
        np.interp(times, trace[:, 0], trace[:, 3]),
    ))


def datasets_with_fewer_columns_than_expected(data):
    short = []
    for key, min_columns in EXPECTED_COLUMNS.items():
        values = data[key]
        if (
            values.ndim == 2
            and values.shape[0] > 0
            and values.shape[1] < min_columns
        ):
            short.append(f'{key}: found {values.shape[1]}, expected {min_columns}')
    return short


def iter_position_sources(data, include_commands=False, include_posctl=False):
    if include_commands:
        keys = ('odom', 'full_state', 'pose', 'cmd_state', 'cmd_full', 'target_state', 'posctl')
    else:
        keys = ('odom', 'full_state', 'pose')
        if include_posctl:
            keys += ('posctl',)
    for key in keys:
        if has_data(data, key, 4):
            yield key, data[key]


def iter_orientation_sources(data, include_commands=False):
    keys = (
        ('odom', 'full_state', 'pose', 'cmd_state', 'cmd_full', 'target_state')
        if include_commands else
        ('odom', 'full_state', 'pose')
    )
    for key in keys:
        if has_data(data, key, 8):
            yield key, data[key]


def iter_velocity_sources(data, include_commands=False):
    keys = (
        ('odom', 'full_state', 'cmd_state', 'cmd_full', 'target_state')
        if include_commands else
        ('odom', 'full_state')
    )
    for key in keys:
        if has_data(data, key, 14):
            yield key, data[key]


def plot_available_logs(data):
    """Plot every useful signal family available in the loaded logs."""
    position_sources = list(iter_position_sources(data, include_commands=True))

    if position_sources:
        plt.figure('Position vs Time')
        for source, values in position_sources:
            for i, label in enumerate(['x', 'y', 'z']):
                plot_source_line(values[:, 0], values[:, 1 + i], source, label)
        plt.title('Position')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.legend()

    orientation_sources = list(iter_orientation_sources(data, include_commands=True))
    if orientation_sources:
        plt.figure('Euler Angles (deg)')
        for source, values in orientation_sources:
            rpy = euler_degrees_from_values(values)
            for i, label in enumerate(['roll', 'pitch', 'yaw']):
                plot_source_line(values[:, 0], rpy[:, i], source, label)
        plt.title('Euler Angles')
        plt.xlabel('Time (s)')
        plt.ylabel('Degrees')
        plt.legend()

    plot_velocity_group(
        'Linear Velocity Body FLU',
        body_velocity_sources(data),
        ('vx_body', 'vy_body', 'vz_body'),
    )
    plot_velocity_group(
        'Linear Velocity ENU',
        enu_velocity_sources_from_body(data),
        ('vx_enu', 'vy_enu', 'vz_enu'),
    )

    angular_velocity_sources = [(key, values, start_col, style) for key, values, start_col, style in (
        ('odom', data['odom'], 11, '-'),
        ('full_state', data['full_state'], 11, '-'),
        ('cmd_full', data['cmd_full'], 11, '--'),
        ('cmd_state', data['cmd_state'], 11, '--'),
        ('target_state', data['target_state'], 11, '--'),
        ('cmd', data['cmd'], 4, '--'),
        ('ref', data['ref'], 4, '--'),
    ) if has_data(data, key, start_col + 3)]
    if angular_velocity_sources or has_data(data, 'cmd_vel', 5):
        plt.figure('Angular Velocity')
        for source, values, start_col, style in angular_velocity_sources:
            for i, label in enumerate(['wx', 'wy', 'wz']):
                if not should_plot_angular_component(source, label):
                    continue
                plot_source_line(values[:, 0], values[:, start_col + i], source, label,
                                 linestyle=component_linestyle(label, style))
        if has_data(data, 'cmd_vel', 5):
            plot_source_line(data['cmd_vel'][:, 0], data['cmd_vel'][:, 3], 'cmd_vel', 'wz')
        plt.title('Angular Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.legend()

    if not any(has_data(data, key, 1) for key in LOG_DATASETS):
        print('No plottable log data available.')


def plot_bundled(data):
    # if data['odom'].shape[1] < 14 or data['cmd'].shape[1] < 7 or data['cmd_vel'].shape[1] < 5:
    #     print("One or more log files have fewer columns than expected. Aborting.")
    #     return

    # === LINEAR VELOCITY ===
    plot_velocity_group(
        'Linear Velocity Body FLU',
        body_velocity_sources(data),
        ('vx_body', 'vy_body', 'vz_body'),
    )
    plot_velocity_group(
        'Linear Velocity ENU',
        enu_velocity_sources_from_body(data),
        ('vx_enu', 'vy_enu', 'vz_enu'),
    )

    # === ANGULAR VELOCITY ===
    plt.figure('Angular Velocity')
    for i, label in enumerate(['wx', 'wy', 'wz']):
        if has_data(data, 'cmd', 7):
            plot_source_line(data['cmd'][:, 0], data['cmd'][:, 4 + i], 'cmd', label)
        if has_data(data, 'ref', 7):
            plot_source_line(data['ref'][:, 0], data['ref'][:, 4 + i], 'ref', label)
        if has_data(data, 'cmd_full', 14):
            plot_source_line(data['cmd_full'][:, 0], data['cmd_full'][:, 11 + i], 'cmd_full', label)
        if has_data(data, 'cmd_state', 14):
            plot_source_line(data['cmd_state'][:, 0], data['cmd_state'][:, 11 + i], 'cmd_state', label)
        if has_data(data, 'target_state', 14) and should_plot_angular_component('target_state', label):
            plot_source_line(data['target_state'][:, 0], data['target_state'][:, 11 + i], 'target_state', label)
        if has_data(data, 'odom', 14):
            plot_source_line(data['odom'][:, 0], data['odom'][:, 11 + i], 'odom', label)
        if has_data(data, 'full_state', 14):
            plot_source_line(data['full_state'][:, 0], data['full_state'][:, 11 + i], 'full_state', label)
        if label == 'wz' and has_data(data, 'cmd_vel', 5):
            plot_source_line(data['cmd_vel'][:, 0], data['cmd_vel'][:, 3], 'cmd_vel', label)
    plt.title('Angular Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend()

    # === POSITION ===
    plt.figure('Position vs Time')
    for i, label in enumerate(['x', 'y', 'z']):
        if has_data(data, 'odom', 14):
            plot_source_line(data['odom'][:, 0], data['odom'][:, 1 + i], 'odom', label)
        if has_data(data, 'full_state', 14):
            plot_source_line(data['full_state'][:, 0], data['full_state'][:, 1 + i], 'full_state', label)
        if has_data(data, 'cmd_full', 14):
            plot_source_line(data['cmd_full'][:, 0], data['cmd_full'][:, 1 + i], 'cmd_full', label)
        if has_data(data, 'cmd_state', 14):
            plot_source_line(data['cmd_state'][:, 0], data['cmd_state'][:, 1 + i], 'cmd_state', label)
        if has_data(data, 'target_state', 14):
            plot_source_line(data['target_state'][:, 0], data['target_state'][:, 1 + i], 'target_state', label)
        if has_data(data, 'pose', 8):
            plot_source_line(data['pose'][:, 0], data['pose'][:, 1 + i], 'pose', label)
    plt.title('Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()

    # === HEIGHT ===
    plt.figure('Z Position (m)')
    if has_data(data, 'odom', 14):
        plot_source_line(data['odom'][:, 0], data['odom'][:, 3], 'odom', 'z')
    if has_data(data, 'full_state', 14):
        plot_source_line(data['full_state'][:, 0], data['full_state'][:, 3], 'full_state', 'z')
    if has_data(data, 'cmd_full', 14):
        plot_source_line(data['cmd_full'][:, 0], data['cmd_full'][:, 3], 'cmd_full', 'z')
    if has_data(data, 'cmd_state', 14):
        plot_source_line(data['cmd_state'][:, 0], data['cmd_state'][:, 3], 'cmd_state', 'z')
    if has_data(data, 'target_state', 14):
        plot_source_line(data['target_state'][:, 0], data['target_state'][:, 3], 'target_state', 'z')
    if has_data(data, 'pose', 8):
        plot_source_line(data['pose'][:, 0], data['pose'][:, 3], 'pose', 'z')
    # plt.plot(data['cmd_vel'][:, 0], data['cmd_vel'][:, 4], label='vz cmd_vel')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.legend()

    # === ROLL, PITCH, YAW ===
    if (has_data(data, 'odom', 14) or has_data(data, 'full_state', 14)
            or has_data(data, 'pose', 8) or has_data(data, 'cmd_full', 14)
            or has_data(data, 'cmd_state', 14) or has_data(data, 'target_state', 14)):
        plt.figure('Euler Angles (deg)')
        for source, values in iter_orientation_sources(data, include_commands=True):
            rpy = euler_degrees_from_values(values)
            for i, label in enumerate(['roll', 'pitch', 'yaw']):
                plot_source_line(values[:, 0], rpy[:, i], source, label)
        plt.title('Euler Angles')
        plt.xlabel('Time (s)')
        plt.ylabel('Degrees')
        plt.legend()


def plot_3d(data, animate=False, animation_dt=0.05, animation_interval_ms=50):
    if animate:
        position_sources = [
            (f"{source['label']} ({source['key']})", source['trace'], source)
            for source in animation_position_sources(data)
        ]
    else:
        position_sources = [
            (
                source,
                finite_position_trace(trajectory),
                {'style': '--' if source in ('cmd_state', 'cmd_full', 'pose', 'posctl') else '-'},
            )
            for source, trajectory in iter_position_sources(data, include_commands=True)
        ]
        position_sources = [
            (source, trace, style_info)
            for source, trace, style_info in position_sources
            if trace is not None
        ]

    if not position_sources:
        print('No pose, full_state, odometry, target, or command position data available for 3D plot.')
        return

    # === 3D TRAJECTORY ===
    fig = plt.figure('3D Trajectory')
    ax = fig.add_subplot(111, projection='3d')
    all_x = []
    all_y = []
    all_z = []
    for source, trace, style_info in position_sources:
        style = style_info.get('style', '-')
        color = style_info.get('color')
        x = trace[:, 1]
        y = trace[:, 2]
        z = trace[:, 3]
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)
        color = color or source_color(source)
        ax.plot(x, y, z, style, color=color, label=f'{source} trajectory')

        ax.scatter(x[0], y[0], z[0], color=color, marker='o', s=35, label=f'{source} start')

        ax.scatter(x[-1], y[-1], z[-1], color=color, marker='x', s=45, label=f'{source} end')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    set_equal_3d_bounds(
        ax,
        np.concatenate(all_x),
        np.concatenate(all_y),
        np.concatenate(all_z),
    )

    if animate:
        add_3d_animation(
            fig,
            ax,
            position_sources,
            animation_dt=animation_dt,
            interval_ms=animation_interval_ms,
        )


def add_3d_animation(fig, ax, position_sources, animation_dt=0.05, interval_ms=50):
    from matplotlib.animation import FuncAnimation

    start_time = min(trace[0, 0] for _source, trace, _style_info in position_sources)
    end_time = max(trace[-1, 0] for _source, trace, _style_info in position_sources)
    if end_time < start_time:
        print('[WARN] 3D animation skipped: selected sources have invalid time ranges.')
        return None

    t_common = np.arange(start_time, end_time + max(animation_dt, 1e-3), max(animation_dt, 1e-3))
    if t_common.size == 0:
        t_common = np.array([start_time])

    animated = []
    for source, trace, style_info in position_sources:
        positions = interpolate_position(trace, t_common)
        color = style_info.get('color')
        marker = ax.scatter(
            [], [], [],
            color=color,
            s=90,
            depthshade=True,
            label=f'{source} current',
        )
        animated.append((marker, positions, source))

    time_label = ax.text2D(0.02, 0.96, '', transform=ax.transAxes)

    def update(frame):
        for marker, positions, _source in animated:
            x, y, z = positions[frame]
            marker._offsets3d = ([x], [y], [z])
        time_label.set_text(f't = {t_common[frame]:.2f} s')
        return [marker for marker, _positions, _source in animated] + [time_label]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(t_common),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    ANIMATIONS.append(animation)
    ax.legend()
    return animation


def plot_separate(data):

    cmd_avail = True
    ref_avail = True
    odom_avail = True
    cmd_vel_avail = True
    pose_avail = True
    cmd_full_avail = True
    cmd_state_avail = True
    target_state_avail = True
    full_state_avail = True
    posctl_avail = True
    if not has_data(data, 'odom', 14):
        print('Odometry data is not available')
        odom_avail = False

    if not has_data(data, 'cmd', 7):
        print('Command twist data is not available (Gazebo only)')
        cmd_avail = False

    if not has_data(data, 'ref', 7):
        print('Reference data is not available (Gazebo only)')
        ref_avail = False

    if not has_data(data, 'cmd_vel', 5):
        print('cmd_vel data is not available')
        cmd_vel_avail = False

    if not has_data(data, 'pose', 8):
        print('Pose data is not available')
        pose_avail = False

    cmd_full_avail = has_data(data, 'cmd_full', 14)
    cmd_state_avail = has_data(data, 'cmd_state', 14)
    target_state_avail = has_data(data, 'target_state', 14)

    if not has_data(data, 'full_state', 14):
        print('Full state data is not available')
        full_state_avail = False

    if not cmd_state_avail:
        print('Command state data is not available')

    if not target_state_avail:
        print('Target state data is not available')

    if not has_data(data, 'posctl', 4):
        print('Posctl data is not available')
        posctl_avail = False

    if not has_data(data, 'velctl', 5):
        print('Velctl data is not available')

    short = datasets_with_fewer_columns_than_expected(data)
    if short:
        print(f'{ansi.YELLOW}WARNING: One or more log files have fewer columns than expected.{ansi.RESET}')

    # --- Linear Velocity ---
    plot_velocity_components(
        'Body FLU Linear Velocity',
        body_velocity_sources(data),
        ('vx_body', 'vy_body', 'vz_body'),
    )
    plot_velocity_components(
        'ENU Linear Velocity',
        enu_velocity_sources_from_body(data),
        ('vx_enu', 'vy_enu', 'vz_enu'),
    )

    # --- Angular Velocity ---
    for i, label in enumerate(['wx', 'wy', 'wz']):
        plt.figure(f'{label} Angular Velocity')
        if cmd_avail:
            plot_source_line(data['cmd'][:, 0], data['cmd'][:, 4 + i], 'cmd', label)
        if ref_avail:
            plot_source_line(data['ref'][:, 0], data['ref'][:, 4 + i], 'ref', label)
        if cmd_full_avail:
            plot_source_line(data['cmd_full'][:, 0], data['cmd_full'][:, 11 + i], 'cmd_full', label)
        if cmd_state_avail:
            plot_source_line(data['cmd_state'][:, 0], data['cmd_state'][:, 11 + i], 'cmd_state', label)
        if target_state_avail and should_plot_angular_component('target_state', label):
            plot_source_line(data['target_state'][:, 0], data['target_state'][:, 11 + i], 'target_state', label)
        if odom_avail:
            plot_source_line(data['odom'][:, 0], data['odom'][:, 11 + i], 'odom', label)
        if full_state_avail:
            plot_source_line(data['full_state'][:, 0], data['full_state'][:, 11 + i], 'full_state', label)
        if label == 'wz' and cmd_vel_avail:
            plot_source_line(data['cmd_vel'][:, 0], data['cmd_vel'][:, 3], 'cmd_vel', label)
        plt.title(f'{label.upper()} Angular Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.legend()

    # --- Position ---
    if odom_avail or pose_avail or full_state_avail or cmd_full_avail or cmd_state_avail or target_state_avail:
        for i, label in enumerate(['x', 'y', 'z']):
            plt.figure(f'{label} Position')
            if odom_avail:
                plot_source_line(data['odom'][:, 0], data['odom'][:, 1 + i], 'odom', label)
            if full_state_avail:
                plot_source_line(data['full_state'][:, 0], data['full_state'][:, 1 + i], 'full_state', label)
            if cmd_full_avail:
                plot_source_line(data['cmd_full'][:, 0], data['cmd_full'][:, 1 + i], 'cmd_full', label)
            if cmd_state_avail:
                plot_source_line(data['cmd_state'][:, 0], data['cmd_state'][:, 1 + i], 'cmd_state', label)
            if target_state_avail:
                plot_source_line(data['target_state'][:, 0], data['target_state'][:, 1 + i], 'target_state', label)
            if pose_avail:
                plot_source_line(data['pose'][:, 0], data['pose'][:, 1 + i], 'pose', label)
            if posctl_avail:
                plot_source_line(data['posctl'][:, 0], data['posctl'][:, 1 + i], 'posctl', label)
            plt.title(f'{label.upper()} Position')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            plt.legend()

    # --- RPY ---
    if odom_avail or pose_avail or full_state_avail or cmd_full_avail or cmd_state_avail or target_state_avail:
        for i, label in enumerate(['roll', 'pitch', 'yaw']):
            plt.figure(f'{label} Euler Angle')
            for source, source_label, available in [
                ('odom', 'odom', odom_avail),
                ('full_state', 'full_state', full_state_avail),
                ('pose', 'pose', pose_avail),
                ('cmd_full', 'cmd_full', cmd_full_avail),
                ('cmd_state', 'cmd_state', cmd_state_avail),
                ('target_state', 'target_state', target_state_avail),
            ]:
                if available:
                    rpy = euler_degrees_from_values(data[source])
                    plot_source_line(data[source][:, 0], rpy[:, i], source, label,
                                     label=f'{label} {source_label}')
            plt.title(f'{label.upper()} (deg)')
            plt.xlabel('Time (s)')
            plt.ylabel('Degrees')
            plt.legend()


def plot_projections(data):
    position_sources = list(iter_position_sources(data, include_commands=True))
    if not position_sources:
        print('No pose, full_state, odometry, command, or target data available for projections.')
        return

    # --- XY ---
    plt.figure('XY Projection')
    for source, trajectory in position_sources:
        style = '--' if source in ('cmd_state', 'cmd_full', 'pose', 'posctl') else '-'
        x = trajectory[:, 1]
        y = trajectory[:, 2]
        color = source_color(source)
        plt.plot(x, y, linestyle=style, color=color, label=f'XY {source} trajectory')
        plt.scatter(x[0], y[0], color=color, marker='o', s=25, label=f'{source} start')
        plt.scatter(x[-1], y[-1], color=color, marker='x', s=35, label=f'{source} end')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('XY Projection')
    plt.axis('equal')
    plt.legend()

    # --- XZ ---
    plt.figure('XZ Projection')
    for source, trajectory in position_sources:
        style = '--' if source in ('cmd_state', 'cmd_full', 'pose', 'posctl') else '-'
        x = trajectory[:, 1]
        z = trajectory[:, 3]
        color = source_color(source)
        plt.plot(x, z, linestyle=style, color=color, label=f'XZ {source} trajectory')
        plt.scatter(x[0], z[0], color=color, marker='o', s=25, label=f'{source} start')
        plt.scatter(x[-1], z[-1], color=color, marker='x', s=35, label=f'{source} end')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title('XZ Projection')
    plt.legend()

    # --- YZ ---
    plt.figure('YZ Projection')
    for source, trajectory in position_sources:
        style = '--' if source in ('cmd_state', 'cmd_full', 'pose', 'posctl') else '-'
        y = trajectory[:, 2]
        z = trajectory[:, 3]
        color = source_color(source)
        plt.plot(y, z, linestyle=style, color=color, label=f'YZ {source} trajectory')
        plt.scatter(y[0], z[0], color=color, marker='o', s=25, label=f'{source} start')
        plt.scatter(y[-1], z[-1], color=color, marker='x', s=35, label=f'{source} end')
    plt.xlabel('Y (m)')
    plt.ylabel('Z (m)')
    plt.title('YZ Projection')
    plt.legend()


def plot_statuses(data):
    if not has_data(data, 'status', 6):
        print('Status data too short')
        return

    time = data['status'][:, 0] * 1e-6  # microseconds to seconds
    state = data['status'][:, 1].astype(np.int64)
    flags = data['status'][:, 2].astype(np.int64)
    volts = data['status'][:, 3] / 1000.  # mV to Volts
    latency = data['status'][:, 5]
    current_a = data['status'][:, 6] if data['status'].shape[1] > 6 else None

    fig, ax1 = plt.subplots()
    fig.suptitle('Battery Voltage and Current')

    ax1.plot(time, volts, label='Voltage', color='red')
    ax1.set_xlabel('Time (secs)')
    ax1.set_ylabel('Volts', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    voltage_max = 12.0 if np.any(volts > 6.0) else 4.0
    ax1.set_ylim(0, voltage_max)

    ax2 = ax1.twinx()
    if current_a is not None:
        known_current = current_a >= 0.0
        ax2.plot(
            time[known_current],
            current_a[known_current],
            label='Current',
            color='blue',
        )
    ax2.set_ylabel('Amps', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    if current_a is not None and np.any(current_a >= 0.0):
        ax2.set_ylim(0, max(1.0, float(np.nanmax(current_a[current_a >= 0.0])) * 1.1))
    else:
        ax2.set_ylim(0, 1.0)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.tight_layout()

    # --- State ---
    plt.figure('State')
    plt.step(time, state, where='post', label='state')
    plt.xlabel('time (secs)')
    plt.ylabel('State')
    plt.title('Drone State')
    plt.tight_layout()
    plt.legend()

    # --- Latency ---
    plt.figure('Latency')
    plt.plot(time, latency, label='Latency (ms)')
    plt.xlabel('time (secs)')
    plt.ylabel('Latency (ms)')
    plt.title('Latency')
    plt.tight_layout()
    plt.legend()

    STATUS_FLAG_TEXT = {
        Status.STATUS_CONNECTED: 'Connected',
        Status.STATUS_ARMED: 'Armed',
        Status.STATUS_AIRBORNE: 'Airborne',
        Status.STATUS_READY_FOR_COMMANDS: 'Ready',
        Status.STATUS_AUTONOMOUS_CONTROL: 'Autonomous',
        Status.STATUS_DEGRADED: 'Degraded',
        Status.STATUS_LOCKED: 'Locked',
    }
    # Extract bits into a (num_bits, num_times) matrix
    bit_values = list(STATUS_FLAG_TEXT)
    num_bits = len(bit_values)
    bit_matrix = np.array([((flags & bit) != 0).astype(np.int8) for bit in bit_values])

    print(' num_bits=', num_bits, flush=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bit_names = reversed([STATUS_FLAG_TEXT[bit] for bit in bit_values])

    for i in range(num_bits):
        y = np.full_like(time, fill_value=num_bits - 1 - i, dtype=np.int8)  # stack bits vertically
        states = bit_matrix[i]

        # Plot true as green filled circles
        ax.plot(time[states == 1], y[states == 1],
                marker='o',
                markerfacecolor='green',     # fill color of marker
                markeredgecolor='green',     # border color of marker
                linestyle=':',
                color='black',
                alpha=0.5,
                label=None)

    ax.set_xlabel('Time [s]')
    ax.set_yticks(range(num_bits))
    ax.set_yticklabels(bit_names)
    ax.set_title('Status Flags Over Time')
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(-1, num_bits)
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()


def find_latest_start_time_and_index(log_dir):
    """
    Scan log_dir and return the start_time and max index.

    Find the most recently created log file based on filesystem time.
    Returns (start_time: str, max_index: int).
    """
    pattern = re.compile(rf"({'|'.join(LOG_DATASETS)})_(\d+)_(\d+)\.npy")
    indexes_by_start_time = {}
    latest_start_time = None
    latest_mtime = -1

    for file in log_dir.glob('*.npy'):
        match = pattern.match(file.name)
        if match:
            start_time = match.group(2)
            index = int(match.group(3))
            indexes_by_start_time.setdefault(start_time, set()).add(index)
            mtime = file.stat().st_mtime  # modification time (or use st_ctime on Windows)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_start_time = start_time

    if latest_start_time is None:
        raise FileNotFoundError('No valid log files found in the directory.')

    return latest_start_time, max(indexes_by_start_time[latest_start_time])


def main():
    parser = argparse.ArgumentParser(
        description='Plot logged drone data from .npy files',
        epilog='Example: ros2 run flexible_drones_tools plot_logs --drone_name drone1 --include-status',
    )
    default_log_dir = Path(os.environ.get('WORKSPACE_ROOT', '.')) / 'log'
    parser.add_argument('--log_dir', type=Path, default=default_log_dir,
                        help='Path to log directory (default: ${WORKSPACE_ROOT}/log)')
    parser.add_argument('--drone_name', type=str, default='red1', help="Name of drone (default: 'red1')")
    parser.add_argument('--start_time', type=str, default=None, help='Start timestamp used in log file names (None = latest)')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of files')
    parser.add_argument('--end_idx', type=int, default=None, help='End index of files (inclusive)')
    parser.add_argument('--include-separate', action='store_true', help='Additionally plot each axis in separate figures')
    parser.add_argument('--include-projections', action='store_true', help='Additionally plot x-y, x-z, y-z projections')
    parser.add_argument('--include-bundled', action='store_true', help='Additionally plot each group in separate figures')
    parser.add_argument('--include-status', action='store_true', help='Additionally plot status data in separate figures')
    parser.add_argument('--animate', action='store_true',
                        help='Animate desired, target, and drone markers on the 3D trajectory plot')
    parser.add_argument('--animation-dt', type=float, default=0.05,
                        help='Animation time step in seconds (default: 0.05)')
    parser.add_argument('--animation-interval-ms', type=int, default=50,
                        help='Animation frame interval in milliseconds (default: 50)')
    parser.add_argument('--include-unarmed', action='store_true',
                        help='Do not filter out samples captured while the drone was not armed')

    args = parser.parse_args()

    global plt, tf_transformations
    import matplotlib.pyplot as plt
    import tf_transformations

    args.log_dir = args.log_dir / args.drone_name
    if args.start_time is None or args.end_idx is None:
        latest_time, latest_idx = find_latest_start_time_and_index(args.log_dir)
        if args.start_time is None:
            args.start_time = latest_time
            print(f'[INFO] No start_time provided. Using latest created file: {args.start_time}')
        if args.end_idx is None:
            args.end_idx = latest_idx
            print(f'[INFO] No end_idx provided. Using latest available index: {args.end_idx}')

    print(f'[INFO] Loading logs from: {args.log_dir} start_time={args.start_time} index={args.start_idx} to {args.end_idx}')
    data = load_log_data(args.log_dir, args.start_time, (args.start_idx, args.end_idx))
    if not args.include_unarmed:
        data = filter_data_to_armed(data)

    plot_3d(
        data,
        animate=args.animate,
        animation_dt=args.animation_dt,
        animation_interval_ms=args.animation_interval_ms,
    )

    default_state_plots = not any([
        args.include_separate,
        args.include_bundled,
        args.include_projections,
        args.include_status,
    ])
    if default_state_plots:
        plot_projections(data)
        plot_available_logs(data)
        if has_data(data, 'status', 6):
            plot_statuses(data)

    if args.include_separate:
        plot_separate(data)

    if args.include_bundled:
        plot_bundled(data)

    if args.include_projections:
        plot_projections(data)

    if args.include_status:
        plot_statuses(data)

    plt.show()


if __name__ == '__main__':
    main()

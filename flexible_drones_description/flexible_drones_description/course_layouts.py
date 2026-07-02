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

"""Shared gate course configuration and launch helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from launch.actions import ExecuteProcess
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from flexible_drones_description.launch_assets import (
    DESCRIPTION_PACKAGE,
    resolve_gate_xacro_path,
    resolve_relative_asset_path,
)


DEFAULT_COURSE_CONFIG = 'serc_bag.yaml'
COURSE_CONFIG_ALIASES = {
    'original': 'small_gates.yaml',
    'serc': 'serc_bag.yaml',
}


@dataclass(frozen=True)
class GatePose:
    """Pose of one course gate."""

    prefix: str
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class GateCourse:
    """Gate course geometry and display metadata."""

    pipe_length: str
    base_height: str
    pipe_radius: str
    gates: tuple[GatePose, ...]
    label_z: float | None = None


def resolve_course_config_path(
    filename: str | None,
    *,
    package_name: str = DESCRIPTION_PACKAGE,
) -> str:
    """Resolve a course config path from an absolute path, relative path, or config name."""
    if filename is None or not filename.strip():
        filename = DEFAULT_COURSE_CONFIG

    filename = COURSE_CONFIG_ALIASES.get(filename.strip(), filename.strip())
    if Path(filename).is_absolute():
        return filename
    if filename.startswith(('.', '/')):
        return str(Path(filename).resolve())
    return resolve_relative_asset_path(
        f'config/{filename}',
        package_name=package_name,
    )


def _required_float(mapping: dict, key: str, *, context: str) -> float:
    if key not in mapping:
        raise RuntimeError(f"Missing '{key}' in {context}.")
    try:
        return float(mapping[key])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"'{key}' in {context} must be a number.") from exc


def _required_number_string(mapping: dict, key: str, *, context: str) -> str:
    if key not in mapping:
        raise RuntimeError(f"Missing '{key}' in {context}.")
    value = mapping[key]
    try:
        float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"'{key}' in {context} must be a number.") from exc
    return str(value)


def _required_str(mapping: dict, key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"'{key}' in {context} must be a non-empty string.")
    return value.strip()


def load_gate_course(
    filename: str | None = None,
    *,
    package_name: str = DESCRIPTION_PACKAGE,
) -> GateCourse:
    """Load and validate a gate course YAML file."""
    path = resolve_course_config_path(filename, package_name=package_name)
    with open(path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise RuntimeError(f"Course config '{path}' must contain a YAML mapping.")

    pipe_length = _required_number_string(data, 'pipe_length', context=path)
    base_height = data.get('base_height', pipe_length)
    try:
        float(base_height)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"'base_height' in {path} must be a number.") from exc
    base_height = str(base_height)
    pipe_radius = _required_number_string(data, 'pipe_radius', context=path)
    label_z = data.get('label_z')
    if label_z is not None:
        label_z = float(label_z)

    gate_data = data.get('gates')
    if not isinstance(gate_data, list) or not gate_data:
        raise RuntimeError(f"Course config '{path}' must define a non-empty 'gates' list.")

    gates = []
    for index, gate in enumerate(gate_data):
        context = f"gate {index} in '{path}'"
        if not isinstance(gate, dict):
            raise RuntimeError(f'{context} must be a YAML mapping.')
        prefix = _required_str(gate, 'prefix', context=context)
        pose = gate.get('pose')
        if not isinstance(pose, dict):
            raise RuntimeError(f"'pose' in {context} must be a YAML mapping.")
        gates.append(GatePose(
            prefix=prefix,
            x=_required_float(pose, 'x', context=context),
            y=_required_float(pose, 'y', context=context),
            z=_required_float(pose, 'z', context=context),
            yaw=_required_float(pose, 'yaw', context=context),
        ))

    return GateCourse(
        pipe_length=pipe_length,
        base_height=base_height,
        pipe_radius=pipe_radius,
        gates=tuple(gates),
        label_z=label_z,
    )


def gate_tuple_list(course: GateCourse) -> list[tuple[str, float, float, float, float]]:
    """Return the legacy tuple shape used by Gazebo launch helpers."""
    return [(gate.prefix, gate.x, gate.y, gate.z, gate.yaw) for gate in course.gates]


def gate_marker_yaml(course: GateCourse, *, label_z: float | None = None) -> str:
    """Build a MarkerArray YAML string for gate labels."""
    if label_z is None:
        label_z = course.label_z
    if label_z is None:
        label_z = float(course.base_height) + float(course.pipe_length) * 2.5

    markers = []
    for index, gate in enumerate(course.gates):
        label = gate.prefix.replace('gate_', '')
        markers.append({
            'header': {
                'frame_id': f'{gate.prefix}_base_link',
                'stamp': {'sec': 0, 'nanosec': 0},
            },
            'ns': 'gate_labels',
            'id': index,
            'type': 9,
            'action': 0,
            'pose': {
                'position': {'x': 0.0, 'y': 0.0, 'z': label_z},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
            },
            'scale': {'x': 0.0, 'y': 0.0, 'z': 0.4},
            'color': {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0},
            'text': label,
        })
    return yaml.dump({'markers': markers}, sort_keys=False)


def make_gate_course_description_actions(
    course: GateCourse,
    *,
    frame_id: str = 'map',
    label_rate='0.2',
    label_z: float | None = None,
    xacro_path: str | None = None,
) -> list:
    """Create launch actions for gate TF, robot descriptions, and labels."""
    xacro_path = xacro_path or resolve_gate_xacro_path()
    frame_id = frame_id.strip() or 'map'
    actions = []

    for gate in course.gates:
        robot_description = ParameterValue(
            Command([
                f'xacro {xacro_path} prefix:={gate.prefix} '
                f'pipe_length:={course.pipe_length} '
                f'base_height:={course.base_height} pipe_radius:={course.pipe_radius}'
            ]),
            value_type=str,
        )
        actions.extend([
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name=f'{gate.prefix}_tf',
                arguments=[
                    '--x', str(gate.x),
                    '--y', str(gate.y),
                    '--z', str(gate.z),
                    '--roll', '0',
                    '--pitch', '0',
                    '--yaw', str(gate.yaw),
                    '--frame-id', frame_id,
                    '--child-frame-id', f'{gate.prefix}_base_link',
                ],
                output='screen',
            ),
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name=f'{gate.prefix}_state_publisher',
                parameters=[{'robot_description': robot_description}],
                output='screen',
                remappings=[('robot_description', f'{gate.prefix}_description')],
            ),
        ])

    actions.append(ExecuteProcess(
        cmd=[
            'ros2',
            'topic',
            'pub',
            '/gate_labels',
            'visualization_msgs/msg/MarkerArray',
            gate_marker_yaml(course, label_z=label_z),
            '-r',
            label_rate,
        ],
        output='log',
    ))
    return actions

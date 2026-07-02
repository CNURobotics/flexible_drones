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

"""Shared asset lookup and launch helpers for Flexible Drones descriptions."""

from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch.actions import LogInfo, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


DESCRIPTION_PACKAGE = 'flexible_drones_description'
MODEL_ROOT = 'models'
GATE_MODEL_NAME = 'gates'


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _package_share(package_name: str = DESCRIPTION_PACKAGE) -> Path | None:
    try:
        return Path(get_package_share_directory(package_name))
    except (PackageNotFoundError, ValueError):
        return None


def _package_root_path(
    package_name: str = DESCRIPTION_PACKAGE,
    package_root: os.PathLike[str] | str | None = None,
) -> Path | None:
    if package_root is not None:
        return Path(package_root)
    if package_name == DESCRIPTION_PACKAGE:
        return _repo_root()
    return None


def _candidate_package_roots(
    *,
    package_name: str = DESCRIPTION_PACKAGE,
    package_root: os.PathLike[str] | str | None = None,
) -> list[Path]:
    roots: list[Path] = []
    source_root = _package_root_path(package_name=package_name, package_root=package_root)
    if source_root is not None:
        roots.append(source_root)

    share_root = _package_share(package_name)
    if share_root is not None and share_root not in roots:
        roots.append(share_root)

    return roots


def _drone_model_xacro_path(model_name: str, *, model_root: str = MODEL_ROOT) -> str:
    return os.path.join(model_root, model_name, 'urdf', f'{model_name}_model.urdf.xacro')


def default_drone_model_xacro(
    *,
    model_root: str = MODEL_ROOT,
    package_name: str = DESCRIPTION_PACKAGE,
    package_root: os.PathLike[str] | str | None = None,
    excluded_model_names: set[str] | None = None,
) -> dict[str, str]:
    """Discover drone model xacros from models/<type>/urdf/<type>_model.urdf.xacro."""
    excluded = set(excluded_model_names or {GATE_MODEL_NAME})
    for root in _candidate_package_roots(package_name=package_name, package_root=package_root):
        models_path = root / model_root
        if not models_path.is_dir():
            continue

        discovered: dict[str, str] = {}
        for model_dir in sorted(models_path.iterdir()):
            if not model_dir.is_dir() or model_dir.name in excluded:
                continue

            rel_path = _drone_model_xacro_path(model_dir.name, model_root=model_root)
            if (root / rel_path).is_file():
                discovered[model_dir.name] = rel_path
            else:
                warnings.warn(
                    f"Model directory '{model_dir.name}' found in '{models_path}' "
                    f"but expected xacro '{rel_path}' is missing — "
                    f'it will not be registered as a drone type. '
                    f"Create '{rel_path}' or add '{model_dir.name}' to "
                    f'excluded_model_names to suppress this warning.',
                    UserWarning,
                    stacklevel=2,
                )

        if discovered:
            return discovered

    return {}


DRONE_MODEL_XACRO = default_drone_model_xacro()
GATE_XACRO = os.path.join(MODEL_ROOT, GATE_MODEL_NAME, 'urdf', 'drone_gate.urdf.xacro')


def resolve_relative_asset_path(
    rel_path: str,
    *,
    package_name: str = DESCRIPTION_PACKAGE,
    package_root: os.PathLike[str] | str | None = None,
) -> str:
    source_root = _package_root_path(package_name=package_name, package_root=package_root)
    if source_root is not None:
        source_path = source_root / rel_path
        if source_path.exists():
            return str(source_path)

    share_path = _package_share(package_name)
    if share_path is not None:
        candidate = share_path / rel_path
        if candidate.exists():
            return str(candidate)

    raise RuntimeError(
        f"Could not locate description asset '{rel_path}' in package '{package_name}'."
    )


def resolve_drone_xacro_path(
    drone_type: str,
    *,
    package_name: str = DESCRIPTION_PACKAGE,
    package_root: os.PathLike[str] | str | None = None,
    model_root: str = MODEL_ROOT,
    drone_model_xacro: Mapping[str, str] | None = None,
) -> str:
    model_map = drone_model_xacro or default_drone_model_xacro(
        model_root=model_root,
        package_name=package_name,
        package_root=package_root,
    )
    rel_path = model_map.get(drone_type)
    if rel_path is None:
        supported = ', '.join(sorted(model_map)) or '(none discovered)'
        raise RuntimeError(
            f"Unknown drone type '{drone_type}'. Supported types: {supported}."
        )
    return resolve_relative_asset_path(
        rel_path,
        package_name=package_name,
        package_root=package_root,
    )


def render_urdf(drone_name: str, drone_type: str) -> str:
    """Render a drone's xacro to a URDF string for the given drone name."""
    xacro_path = resolve_drone_xacro_path(drone_type)
    result = subprocess.run(
        ['xacro', xacro_path, f'drone_name:={drone_name}'],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def resolve_gate_xacro_path(
    *,
    package_name: str = DESCRIPTION_PACKAGE,
    package_root: os.PathLike[str] | str | None = None,
    model_root: str = MODEL_ROOT,
    gate_model_name: str = GATE_MODEL_NAME,
) -> str:
    return resolve_relative_asset_path(
        os.path.join(model_root, gate_model_name, 'urdf', 'drone_gate.urdf.xacro'),
        package_name=package_name,
        package_root=package_root,
    )


def resolve_rviz_config_path(
    filename: str | None,
    *,
    package_name: str = DESCRIPTION_PACKAGE,
    package_root: os.PathLike[str] | str | None = None,
) -> str | None:
    if filename is None:
        return None

    filename = filename.strip()
    if not filename:
        return None

    if os.path.isabs(filename):
        return filename

    if filename.startswith(('.', '/')):
        return os.path.abspath(filename)

    for subdir in ('rviz', 'config'):
        rel_path = os.path.join(subdir, filename)
        try:
            return resolve_relative_asset_path(
                rel_path,
                package_name=package_name,
                package_root=package_root,
            )
        except RuntimeError:
            continue

    raise RuntimeError(
        f"RViz config '{filename}' was not found in package '{package_name}'."
    )


def _default_robot_model_display() -> dict:
    return {
        'Alpha': 1,
        'Class': 'rviz_default_plugins/RobotModel',
        'Collision Enabled': False,
        'Description File': '',
        'Description Source': 'Topic',
        'Description Topic': {
            'Depth': 5,
            'Durability Policy': 'Transient Local',
            'Filter size': 10,
            'History Policy': 'Keep Last',
            'Reliability Policy': 'Reliable',
            'Value': '',
        },
        'Enabled': True,
        'Links': {
            'All Links Enabled': True,
            'Expand Joint Details': False,
            'Expand Link Details': False,
            'Expand Tree': False,
            'Link Tree Style': 'Links in Alphabetic Order',
        },
        'Mass Properties': {
            'Inertia': False,
            'Mass': False,
        },
        'Name': 'RobotModel',
        'TF Prefix': '',
        'Update Interval': 0,
        'Value': True,
        'Visual Enabled': True,
    }


def _make_robot_model_display(template: dict | None, drone_name: str) -> dict:
    display = deepcopy(template) if template is not None else _default_robot_model_display()
    display['Class'] = 'rviz_default_plugins/RobotModel'
    display['Enabled'] = True
    display['Name'] = drone_name
    display['Description Source'] = 'Topic'
    display['Description File'] = ''
    display['TF Prefix'] = ''
    display['Update Interval'] = 0
    display['Value'] = True
    display['Visual Enabled'] = True

    topic_cfg = display.get('Description Topic', {})
    topic_cfg['Depth'] = 5
    topic_cfg['Durability Policy'] = 'Transient Local'
    topic_cfg['Filter size'] = 10
    topic_cfg['History Policy'] = 'Keep Last'
    topic_cfg['Reliability Policy'] = 'Reliable'
    topic_cfg['Value'] = f'/{drone_name}/robot_description'
    display['Description Topic'] = topic_cfg

    links = display.get('Links', {})
    display['Links'] = {
        'All Links Enabled': links.get('All Links Enabled', True),
        'Expand Joint Details': links.get('Expand Joint Details', False),
        'Expand Link Details': links.get('Expand Link Details', False),
        'Expand Tree': links.get('Expand Tree', False),
        'Link Tree Style': links.get('Link Tree Style', 'Links in Alphabetic Order'),
    }
    return display


def generate_rviz_config(
    base_config_path: str,
    drone_names: list[str],
    *,
    tempfile_prefix: str = 'flexible_drones_description_',
    window_x: int | None = None,
    window_y: int | None = None,
    window_width: int | None = None,
    window_height: int | None = None,
) -> str:
    with open(base_config_path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}

    vis_manager = config.setdefault('Visualization Manager', {})
    displays = vis_manager.setdefault('Displays', [])
    global_options = vis_manager.setdefault('Global Options', {})
    global_options['Fixed Frame'] = 'map'

    template = next(
        (
            deepcopy(display)
            for display in displays
            if display.get('Class') == 'rviz_default_plugins/RobotModel'
        ),
        None,
    )

    existing_topics = {
        display.get('Description Topic', {}).get('Value')
        for display in displays
        if display.get('Class') == 'rviz_default_plugins/RobotModel'
    }

    for drone_name in dict.fromkeys(drone_names):
        topic = f'/{drone_name}/robot_description'
        if topic in existing_topics:
            continue
        displays.append(_make_robot_model_display(template, drone_name))
        existing_topics.add(topic)

    if (
        window_x is not None
        or window_y is not None
        or window_width is not None
        or window_height is not None
    ):
        window_geometry = config.setdefault('Window Geometry', {})
        if window_x is not None:
            window_geometry['X'] = window_x
        if window_y is not None:
            window_geometry['Y'] = window_y
        if window_width is not None:
            window_geometry['Width'] = window_width
        if window_height is not None:
            window_geometry['Height'] = window_height

    with tempfile.NamedTemporaryFile(
        mode='w',
        prefix=tempfile_prefix,
        suffix='.rviz',
        delete=False,
        encoding='utf-8',
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        return handle.name


def cleanup_temp_path(path: str) -> list[LogInfo]:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return []
    except OSError as exc:
        return [LogInfo(msg=f"Failed to remove temporary RViz config '{path}': {exc}")]
    return [LogInfo(msg=f"Removed temporary RViz config '{path}'")]


def make_robot_state_publisher(
    spec,
    *,
    package_name: str = DESCRIPTION_PACKAGE,
    package_root: os.PathLike[str] | str | None = None,
    model_root: str = MODEL_ROOT,
    drone_model_xacro: Mapping[str, str] | None = None,
) -> Node:
    """Build a robot_state_publisher node for a normalized launch spec."""
    robot_description = ParameterValue(
        Command(
            [
                'xacro',
                ' ',
                resolve_drone_xacro_path(
                    spec.type,
                    package_name=package_name,
                    package_root=package_root,
                    model_root=model_root,
                    drone_model_xacro=drone_model_xacro,
                ),
                ' ',
                f'drone_name:={spec.name}',
            ]
        ),
        value_type=str,
    )
    return Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name=f'{spec.name}_state_publisher',
        parameters=[{'robot_description': robot_description}],
        remappings=[('robot_description', f'/{spec.name}/robot_description')],
        output='screen',
    )


def make_prop_joint_state_spinner(drone_names: list[str]) -> Node:
    """Build the shared prop joint spinner node for a selected fleet."""
    return Node(
        package='flexible_drones_description',
        executable='prop_joint_state_spinner',
        name='prop_joint_state_spinner',
        parameters=[{'drone_names': drone_names}],
        output='screen',
    )


def make_rviz_launch_actions(
    drone_names: list[str],
    rviz_config: str,
    *,
    package_name: str = DESCRIPTION_PACKAGE,
    package_root: os.PathLike[str] | str | None = None,
    tempfile_prefix: str = 'flexible_drones_description_',
    window_x: int | None = None,
    window_y: int | None = None,
    window_width: int | None = None,
    window_height: int | None = None,
) -> list:
    """Build launch actions for an RViz session visualizing the selected drones."""
    generated_rviz_config = generate_rviz_config(
        resolve_rviz_config_path(
            rviz_config,
            package_name=package_name,
            package_root=package_root,
        ),
        drone_names,
        tempfile_prefix=tempfile_prefix,
        window_x=window_x,
        window_y=window_y,
        window_width=window_width,
        window_height=window_height,
    )
    return [
        LogInfo(msg=f"Starting RViz with generated config '{generated_rviz_config}'"),
        RegisterEventHandler(
            OnShutdown(
                on_shutdown=lambda event, context, path=generated_rviz_config: cleanup_temp_path(path)
            )
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', generated_rviz_config],
            output='screen',
        ),
    ]

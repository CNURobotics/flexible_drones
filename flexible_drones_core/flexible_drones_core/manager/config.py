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

"""Resolved manager construction configuration shared by launch adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from ament_index_python.packages import get_package_share_directory
from flexible_drones_core.registry import DroneSpec


COMMON_MANAGER_PARAMETER_NAMES = frozenset(
    {
        'drone_name',
        'uri',
        'drone_type',
        'manager_type',
    }
)


@dataclass(frozen=True)
class ManagerConfig:
    """Launch-ready manager configuration resolved from a DroneSpec."""

    drone_name: str
    uri: str
    drone_type: str
    manager_type: str
    ros_namespace: str
    node_name: str | None = None
    ros_params: Mapping[str, Any] = field(default_factory=dict)
    extra_args: tuple[str, ...] = ()

    def merge_ros_params(
        self,
        params: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> 'ManagerConfig':
        """Return a copy with additional ROS params layered on top."""
        merged = dict(self.ros_params)
        if params:
            merged.update(dict(params))
        if kwargs:
            merged.update(kwargs)
        return replace(self, ros_params=merged)

    def to_ros_parameters(self) -> dict[str, Any]:
        """Return ROS node parameters, excluding namespace and node-name remaps."""
        params = dict(self.ros_params)
        params.update(
            {
                'drone_name': self.drone_name,
                'uri': self.uri,
                'drone_type': self.drone_type,
                'manager_type': self.manager_type,
            }
        )
        return params


def resolve_package_config_path(
    filename: str,
    *,
    default_filename: str,
    package_name: str,
) -> str:
    """Resolve a config path from absolute, relative, or package-share input."""
    filename = filename.strip() or default_filename
    if os.path.isabs(filename):
        return filename
    if filename.startswith(('.', '/')):
        return os.path.abspath(filename)

    return str(Path(get_package_share_directory(package_name)) / filename)


def load_type_defaults(path: str, *, defaults_name: str = 'Type defaults') -> dict[str, dict]:
    """Load and validate a YAML ``types`` mapping keyed by drone type."""
    with open(path, 'r', encoding='utf-8') as handle:
        document = yaml.safe_load(handle) or {}

    if not isinstance(document, dict):
        raise ValueError(f"{defaults_name} '{path}' must contain a top-level mapping.")

    raw_types = document.get('types', {})
    if not isinstance(raw_types, dict):
        raise ValueError(f"{defaults_name} '{path}' key 'types' must be a mapping.")

    defaults: dict[str, dict] = {}
    for drone_type, entry in raw_types.items():
        if not isinstance(drone_type, str):
            raise ValueError(f'{defaults_name} type key {drone_type!r} must be a string.')
        drone_type = drone_type.strip()
        if not drone_type:
            raise ValueError(f"{defaults_name} '{path}' contains an empty type key.")
        if not isinstance(entry, dict):
            raise ValueError(f"{defaults_name} type '{drone_type}' must map to a dict.")
        defaults[drone_type] = dict(entry)
    return defaults


def merge_manager_config_layers(
    config: ManagerConfig,
    *,
    defaults_by_type: Mapping[str, Mapping[str, Any]] | None = None,
    ros_params: Mapping[str, Any] | None = None,
    forced_params: Mapping[str, Any] | None = None,
) -> ManagerConfig:
    """Merge backend defaults, deployment params, launch params, and forced params."""
    params = dict((defaults_by_type or {}).get(config.drone_type, {}) or {})
    params.update(dict(config.ros_params or {}))
    if ros_params:
        params.update(dict(ros_params))
    if forced_params:
        params.update(dict(forced_params))
    return config.merge_ros_params(params)


def has_node_remap(args: tuple[str, ...] | list[str]) -> bool:
    """Return true when args already contain a ROS __node remap."""
    return any('__node:=' in arg for arg in args)


def base_manager_config_from_spec(
    spec: DroneSpec,
    profile: Mapping[str, Any] | None = None,
) -> ManagerConfig:
    """Resolve backend-neutral manager construction fields from a DroneSpec."""
    profile = profile or {}
    ros_namespace = spec.effective_namespace()
    manager_owns_drone_prefix = bool(profile.get('manager_owns_drone_prefix', False))
    if manager_owns_drone_prefix and not spec.ros_namespace:
        ros_namespace = '/'

    extra_args = tuple(spec.extra_args or ())
    node_name = None
    node_name_template = str(profile.get('node_name_template', '')).strip()
    if node_name_template and not has_node_remap(extra_args):
        executable = str(profile.get('executable', '')).strip()
        rendered = node_name_template.format(
            name=spec.name,
            drone_name=spec.name,
            drone_type=spec.type,
            manager_type=spec.manager_type,
            executable=executable,
        ).strip()
        node_name = rendered or None

    return ManagerConfig(
        drone_name=spec.name,
        uri=spec.uri,
        drone_type=spec.type,
        manager_type=spec.manager_type,
        ros_namespace=ros_namespace,
        node_name=node_name,
        ros_params=dict(spec.ros_params or {}),
        extra_args=extra_args,
    )

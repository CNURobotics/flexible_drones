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

"""Registry for drone ID -> process/namespace/config mappings."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence
import os
import threading
import time

import yaml

from flexible_drones_core.utils import parse_bool_field


_KNOWN_SPEC_KEYS = {
    'name',
    'uri',
    'type',
    'manager_type',
    'manager',
    'enabled',
    'ros_namespace',
    'ros_params',
    'extra_args',
    'extras',
}

_URI_OPTIONAL_MANAGER_TYPES = {
    'gazebo_manager',
    'sim_manager',
    'simulation_manager',
}


@dataclass(frozen=True)
class DroneSpec:
    """
    Desired configuration for a single drone manager process.

    Minimal required fields:
      - name: unique namespace/name for the drone (e.g., "cf01")
      - uri: connection identifier; optional for simulation managers
      - type: platform string (e.g., "crazyflie", "pihawk")
      - manager_type: manager profile key from launch/orchestrator config
    """

    name: str
    uri: str
    type: str  # noqa: A003
    manager_type: str
    enabled: bool = True

    # Optional extras:
    ros_namespace: str | None = None  # if omitted, defaults to "/<name>"
    ros_params: dict[str, Any] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def effective_namespace(self) -> str:
        ns = self.ros_namespace if self.ros_namespace else f'/{self.name}'
        if not ns.startswith('/'):
            ns = '/' + ns
        return ns

    def with_enabled(self, enabled: bool) -> 'DroneSpec':
        return replace(self, enabled=enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'uri': self.uri,
            'type': self.type,
            'manager_type': self.manager_type,
            'enabled': self.enabled,
            'ros_namespace': self.ros_namespace,
            'ros_params': self.ros_params,
            'extra_args': self.extra_args,
            'extras': self.extras,
        }


class Registry:
    """
    Loads and stores desired drone specs. This is "desired state" only.
    Runtime status (connected, etc.) should live elsewhere (orchestrator).
    """

    def __init__(self, specs: list[DroneSpec] | None = None, source_path: str | None = None):
        self._lock = threading.RLock()
        self._specs_by_name: dict[str, DroneSpec] = {}
        self._source_path = source_path
        self._loaded_at_unix = 0.0

        if specs:
            self._specs_by_name = self._build_specs_by_name(specs)

    @property
    def source_path(self) -> str | None:
        with self._lock:
            return self._source_path

    @property
    def loaded_at_unix(self) -> float:
        with self._lock:
            return self._loaded_at_unix

    def list_all(self) -> list[DroneSpec]:
        with self._lock:
            return list(self._specs_by_name.values())

    def list_enabled(self) -> list[DroneSpec]:
        with self._lock:
            return [s for s in self._specs_by_name.values() if s.enabled]

    def get(self, name: str) -> DroneSpec | None:
        with self._lock:
            return self._specs_by_name.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._specs_by_name.keys())

    def reload_from_file(self, path: str | None = None, selected_names: Sequence[str] | None = None) -> None:
        if path is None:
            with self._lock:
                if not self._source_path:
                    raise ValueError('No registry path provided to reload_from_file().')
                path = self._source_path

        specs = Registry.load_specs_from_yaml(path, selected_names=selected_names)
        specs_by_name = self._build_specs_by_name(specs)

        with self._lock:
            self._specs_by_name = specs_by_name
            self._source_path = path
            self._loaded_at_unix = time.time()

    def _insert(self, spec: DroneSpec) -> None:
        if not spec.name:
            raise ValueError('DroneSpec.name cannot be empty')
        with self._lock:
            if spec.name in self._specs_by_name:
                raise ValueError(f'Duplicate drone name in registry: {spec.name}')
            self._specs_by_name[spec.name] = spec

    @staticmethod
    def _build_specs_by_name(specs: Sequence[DroneSpec]) -> dict[str, DroneSpec]:
        specs_by_name: dict[str, DroneSpec] = {}
        for spec in specs:
            if not spec.name:
                raise ValueError('DroneSpec.name cannot be empty')
            if spec.name in specs_by_name:
                raise ValueError(f'Duplicate drone name in registry: {spec.name}')
            specs_by_name[spec.name] = spec
        return specs_by_name

    @staticmethod
    def _parse_spec(item: dict[str, Any], *, idx_label: str, fallback_name: str | None = None) -> DroneSpec:
        name = str(item.get('name', fallback_name or '')).strip()
        uri = str(item.get('uri', '')).strip()
        dtype = str(item.get('type', '')).strip()

        if not name:
            raise ValueError(f'Registry YAML {idx_label} requires name.')
        if not dtype:
            dtype = 'unknown'

        manager_type = str(item.get('manager_type', item.get('manager', ''))).strip()
        if not manager_type:
            raise ValueError(f'Registry YAML {idx_label} requires manager_type.')
        if not uri and manager_type not in _URI_OPTIONAL_MANAGER_TYPES:
            raise ValueError(f'Registry YAML {idx_label} requires uri.')
        enabled = parse_bool_field(item.get('enabled', True), label=f'{idx_label}.enabled')
        ros_namespace = item.get('ros_namespace', None)
        ros_params = item.get('ros_params', {})
        extra_args = item.get('extra_args', [])
        raw_extras = item.get('extras', {})

        if ros_namespace is not None:
            ros_namespace = str(ros_namespace).strip()

        if not isinstance(ros_params, dict):
            raise ValueError(f'{idx_label}.ros_params must be a dict')
        if not isinstance(extra_args, list) or any(not isinstance(x, str) for x in extra_args):
            raise ValueError(f'{idx_label}.extra_args must be a list of strings')
        if not isinstance(raw_extras, dict):
            raise ValueError(f'{idx_label}.extras must be a dict')

        extras = dict(raw_extras)
        extras.update(
            {
                key: value
                for key, value in item.items()
                if key not in _KNOWN_SPEC_KEYS
            }
        )

        return DroneSpec(
            name=name,
            uri=uri,
            type=dtype,
            manager_type=manager_type,
            enabled=enabled,
            ros_namespace=ros_namespace,
            ros_params=ros_params,
            extra_args=extra_args,
            extras=extras,
        )

    @staticmethod
    def load_specs_from_yaml(path: str, selected_names: Sequence[str] | None = None) -> list[DroneSpec]:
        """
        Supported YAML schema:

        drones:
          - name: cf01
            uri: radio://0/80/2M/E7E7E7E701
            type: crazyflie
            manager_type: example_manager
            enabled: true
            ros_namespace: /cf01       # optional
            ros_params: {}             # optional
            extra_args: []             # optional
            extras: {}                 # optional; unknown top-level keys are also preserved here

        selected_names:
          Optional list of names used to filter which entries are loaded.
          None means all entries; an explicit empty list means no entries.
        """
        expanded = os.path.expandvars(os.path.expanduser(path))
        with open(expanded, 'r', encoding='utf-8') as f:
            doc = yaml.safe_load(f) or {}

        selected: list[str] | None = None
        if selected_names is not None:
            selected = [str(name).strip() for name in selected_names if str(name).strip()]

        specs: list[DroneSpec] = []

        if 'drones' in doc:
            drones = doc.get('drones', [])
            if not isinstance(drones, list):
                raise ValueError("Registry YAML must have top-level key 'drones' as a list.")

            selected_set = set(selected) if selected is not None else None
            for idx, item in enumerate(drones):
                if not isinstance(item, dict):
                    raise ValueError(f'Registry YAML drones[{idx}] must be a dict.')

                parsed = Registry._parse_spec(item, idx_label=f'drones[{idx}]')
                if selected_set is not None and parsed.name not in selected_set:
                    continue
                specs.append(parsed)

            return specs

        raise ValueError("Registry YAML must contain a top-level 'drones' list.")

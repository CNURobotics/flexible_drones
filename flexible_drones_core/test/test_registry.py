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

"""Unit tests for registry YAML parsing."""

from pathlib import Path

from flexible_drones_core.registry import Registry
import pytest
import yaml


def _write_yaml(tmp_path: Path, name: str, data) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return path


def test_load_specs_from_drones_schema_filters_selected_names(tmp_path: Path):
    """Load the flat schema and keep only explicitly selected drone names."""
    path = _write_yaml(
        tmp_path,
        'drones.yaml',
        {
            'drones': [
                {
                    'name': 'cf1',
                    'uri': 'radio://cf1',
                    'type': 'crazyflie',
                    'manager_type': 'cf_manager',
                    'enabled': True,
                },
                {
                    'name': 'px01',
                    'uri': 'udp://px01',
                    'type': 'pihawk',
                    'manager_type': 'pihawk_manager',
                    'enabled': True,
                },
            ]
        },
    )

    specs = Registry.load_specs_from_yaml(str(path), selected_names=['px01'])

    assert [spec.name for spec in specs] == ['px01']
    assert specs[0].manager_type == 'pihawk_manager'


def test_load_specs_empty_selected_names_selects_no_drones(tmp_path: Path):
    """An explicit empty selection is distinct from omitting selected_names."""
    path = _write_yaml(
        tmp_path,
        'drones.yaml',
        {
            'drones': [
                {
                    'name': 'cf1',
                    'uri': 'radio://cf1',
                    'type': 'crazyflie',
                    'manager_type': 'cf_manager',
                    'enabled': True,
                },
            ]
        },
    )

    assert Registry.load_specs_from_yaml(str(path), selected_names=[]) == []


def test_load_specs_requires_manager_type(tmp_path: Path):
    """Core registry requires explicit manager profiles instead of guessing by type."""
    path = _write_yaml(
        tmp_path,
        'missing_manager.yaml',
        {
            'drones': [
                {
                    'name': 'cf1',
                    'uri': 'radio://cf1',
                    'type': 'crazyflie',
                }
            ]
        },
    )

    with pytest.raises(ValueError, match=r'drones\[0\] requires manager_type'):
        Registry.load_specs_from_yaml(str(path))


def test_load_specs_allows_empty_uri_for_gazebo_manager(tmp_path: Path):
    """Gazebo specs may omit URI because the simulator provides the transport identity."""
    path = _write_yaml(
        tmp_path,
        'gazebo.yaml',
        {
            'drones': [
                {
                    'name': 'gz1',
                    'type': 'crazyflie',
                    'manager_type': 'gazebo_manager',
                }
            ]
        },
    )

    specs = Registry.load_specs_from_yaml(str(path))

    assert specs[0].name == 'gz1'
    assert specs[0].uri == ''


def test_load_specs_requires_uri_for_hardware_manager(tmp_path: Path):
    """Hardware managers still require an explicit connection URI."""
    path = _write_yaml(
        tmp_path,
        'missing_uri.yaml',
        {
            'drones': [
                {
                    'name': 'cf1',
                    'type': 'crazyflie',
                    'manager_type': 'cf_manager',
                }
            ]
        },
    )

    with pytest.raises(ValueError, match=r'drones\[0\] requires uri'):
        Registry.load_specs_from_yaml(str(path))


def test_load_specs_preserves_extras_and_unknown_keys(tmp_path: Path):
    """Preserve package-specific registry fields without core interpretation."""
    path = _write_yaml(
        tmp_path,
        'drones.yaml',
        {
            'drones': [
                {
                    'name': 'cf1',
                    'uri': 'radio://cf1',
                    'type': 'crazyflie',
                    'manager_type': 'cf_manager',
                    'extras': {'initial_pose': {'position': {'x': 1.0}}},
                    'group_mask': 3,
                }
            ]
        },
    )

    specs = Registry.load_specs_from_yaml(str(path))

    assert specs[0].extras['initial_pose']['position']['x'] == 1.0
    assert specs[0].extras['group_mask'] == 3
    assert specs[0].to_dict()['extras']['group_mask'] == 3


def test_load_specs_parses_enabled_string_false(tmp_path: Path):
    """Quoted false-like values must not be treated as truthy strings."""
    path = _write_yaml(
        tmp_path,
        'drones.yaml',
        {
            'drones': [
                {
                    'name': 'cf1',
                    'uri': 'radio://cf1',
                    'type': 'crazyflie',
                    'manager_type': 'cf_manager',
                    'enabled': 'false',
                }
            ]
        },
    )

    specs = Registry.load_specs_from_yaml(str(path))

    assert specs[0].enabled is False


def test_load_specs_rejects_ambiguous_enabled_string(tmp_path: Path):
    """Ambiguous enabled values should fail closed instead of enabling drones."""
    path = _write_yaml(
        tmp_path,
        'drones.yaml',
        {
            'drones': [
                {
                    'name': 'cf1',
                    'uri': 'radio://cf1',
                    'type': 'crazyflie',
                    'manager_type': 'cf_manager',
                    'enabled': 'maybe',
                }
            ]
        },
    )

    with pytest.raises(ValueError, match=r'drones\[0\]\.enabled must be a boolean'):
        Registry.load_specs_from_yaml(str(path))


def test_load_specs_rejects_legacy_robots_schema(tmp_path: Path):
    """Reject the removed legacy robots schema."""
    path = _write_yaml(
        tmp_path,
        'robots.yaml',
        {
            'robots': {
                'cf1': {
                    'uri': 'radio://cf1',
                    'type': 'crazyflie',
                    'enabled': True,
                },
                'cf2': {
                    'uri': 'radio://cf2',
                    'type': 'crazyflie',
                    'enabled': True,
                },
            }
        },
    )

    with pytest.raises(ValueError, match=r"top-level 'drones' list"):
        Registry.load_specs_from_yaml(str(path), selected_names=['cf2', 'cf1'])


def test_load_specs_rejects_invalid_ros_params_type(tmp_path: Path):
    """Reject a registry entry whose ros_params field is not a mapping."""
    path = _write_yaml(
        tmp_path,
        'invalid.yaml',
        {
            'drones': [
                {
                    'name': 'cf1',
                    'uri': 'radio://cf1',
                    'manager_type': 'cf_manager',
                    'ros_params': ['not', 'a', 'dict'],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match=r'drones\[0\]\.ros_params must be a dict'):
        Registry.load_specs_from_yaml(str(path))


def test_load_specs_rejects_invalid_extra_args_type(tmp_path: Path):
    """Reject a registry entry whose extra_args field is not string-only."""
    path = _write_yaml(
        tmp_path,
        'invalid_args.yaml',
        {
            'drones': [
                {
                    'name': 'cf1',
                    'uri': 'radio://cf1',
                    'manager_type': 'cf_manager',
                    'extra_args': ['ok', 2],
                }
            ]
        },
    )

    with pytest.raises(
        ValueError,
        match=r'drones\[0\]\.extra_args must be a list of strings',
    ):
        Registry.load_specs_from_yaml(str(path))

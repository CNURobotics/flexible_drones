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

"""Unit tests for the generic roster/deployment engine — no ROS runtime required."""

from pathlib import Path

import pytest
import yaml

import flexible_drones_core.deployment as deployment
from flexible_drones_core.deployment import (
    apply_manager_overrides,
    is_filesystem_path,
    load_deployment_overlays,
    load_deployment_specs,
    load_roster_specs,
    parse_selected_names,
    require_package_name,
    resolve_package_relative_path,
)
from flexible_drones_core.registry import DroneSpec


# ── helpers ───────────────────────────────────────────────────────────────────

def _write(tmp_path: Path, name: str, data: dict) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return str(path)


def _roster(tmp_path: Path, drones: list) -> str:
    return _write(tmp_path, 'roster.yaml', {'drones': drones})


def _setup(tmp_path: Path, drones: dict) -> str:
    return _write(tmp_path, 'setup.yaml', {'drones': drones})


# ── parse_selected_names ──────────────────────────────────────────────────────

def test_parse_selected_names_empty_string():
    assert parse_selected_names('') is None
    assert parse_selected_names('  ') is None


def test_parse_selected_names_yaml_list():
    assert parse_selected_names('[cf1, cf2]') == ['cf1', 'cf2']


def test_parse_selected_names_empty_yaml_list():
    assert parse_selected_names('[]') == []


def test_parse_selected_names_null_yaml():
    assert parse_selected_names('null') is None


def test_parse_selected_names_rejects_non_list():
    with pytest.raises(ValueError, match='must be a YAML list'):
        parse_selected_names('{cf1: true}')


def test_parse_selected_names_rejects_duplicates():
    with pytest.raises(ValueError, match='Duplicate fliers entries are not allowed: cf1'):
        parse_selected_names('[cf1, cf1]')


def test_parse_selected_names_rejects_multiple_duplicates_once_each():
    with pytest.raises(ValueError, match='Duplicate fliers entries are not allowed: cf1, cf2'):
        parse_selected_names('[cf1, cf2, cf1, cf2, cf1]')


# ── package-relative path resolution ──────────────────────────────────────────

def test_resolve_package_relative_path_uses_explicit_package_root(tmp_path):
    package_root = tmp_path / 'source_root'
    asset_path = package_root / 'config' / 'roster.yaml'
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text('drones: []\n', encoding='utf-8')

    assert resolve_package_relative_path(
        'config/roster.yaml',
        package_root=package_root,
    ) == str(asset_path)


def test_resolve_package_relative_path_uses_package_share(monkeypatch, tmp_path):
    share_root = tmp_path / 'install_share'
    asset_path = share_root / 'config' / 'roster.yaml'
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text('drones: []\n', encoding='utf-8')

    monkeypatch.setattr(deployment, '_package_share', lambda package_name: share_root)

    assert resolve_package_relative_path(
        'config/roster.yaml',
        package_name='any_package',
    ) == str(asset_path)


def test_resolve_package_relative_path_rejects_missing_dot_relative_path():
    with pytest.raises(RuntimeError, match="Could not locate deployment asset './missing.yaml'"):
        resolve_package_relative_path('./missing.yaml')


def test_resolve_package_relative_path_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv('HOME', str(tmp_path))
    asset_path = tmp_path / 'roster.yaml'
    asset_path.write_text('drones: []\n', encoding='utf-8')

    assert resolve_package_relative_path('~/roster.yaml') == str(asset_path)


def test_is_filesystem_path_classification():
    assert is_filesystem_path('/abs/roster.yaml')
    assert is_filesystem_path('./roster.yaml')
    assert is_filesystem_path('~/roster.yaml')
    assert not is_filesystem_path('config/roster.yaml')
    assert not is_filesystem_path('')


# ── launch validation ────────────────────────────────────────────────────────

def test_require_package_name_strips_non_empty_values():
    assert (
        require_package_name(
            '  chris_serc_deployment  ',
            argument_name='deployment_package',
            purpose='resolve deployment files',
        )
        == 'chris_serc_deployment'
    )


def test_require_package_name_error_explains_launch_arg_and_env_var():
    with pytest.raises(RuntimeError) as excinfo:
        require_package_name(
            '',
            argument_name='deployment_package',
            purpose='resolve roster and deployment setup files',
        )

    msg = str(excinfo.value)
    assert 'deployment_package:=<package_name>' in msg
    assert 'FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE=<package_name>' in msg
    assert 'deployment_package:=chris_serc_deployment' in msg


def test_require_package_name_can_include_extra_hint():
    hint = 'To launch RViz without deployment files, pass `fliers:="[cf1, cf2]"`.'
    with pytest.raises(RuntimeError) as excinfo:
        require_package_name(
            ' ',
            argument_name='deployment_package',
            purpose='resolve roster and deployment setup files when fliers is empty',
            extra_hint=hint,
        )

    assert 'fliers:="[cf1, cf2]"' in str(excinfo.value)


# ── load_roster_specs ─────────────────────────────────────────────────────────

def test_load_roster_specs_list_schema(tmp_path):
    path = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
        {'name': 'cf2', 'uri': 'radio://cf2', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    specs = load_roster_specs(path)
    assert [s.name for s in specs] == ['cf1', 'cf2']
    assert specs[0].uri == 'radio://cf1'
    assert specs[0].enabled is True


def test_load_roster_specs_dict_schema(tmp_path):
    path = _write(tmp_path, 'roster.yaml', {
        'drones': {
            'cf1': {'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
            'cf2': {'uri': 'radio://cf2', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
        }
    })
    specs = load_roster_specs(path)
    assert {s.name for s in specs} == {'cf1', 'cf2'}


def test_load_roster_specs_requires_manager_type(tmp_path):
    path = _roster(tmp_path, [
        {'name': 'px1', 'uri': 'udp://px1', 'type': 'pihawk'},
    ])
    with pytest.raises(ValueError, match='drones\\[0\\] must define manager_type'):
        load_roster_specs(path)


# ── load_deployment_overlays ──────────────────────────────────────────────────

def test_load_deployment_overlays_returns_keyed_dict(tmp_path):
    path = _setup(tmp_path, {
        'cf1': {'enabled': True, 'group_mask': 3},
        'cf2': {'enabled': False},
    })
    overlays = load_deployment_overlays(path)
    assert overlays['cf1']['enabled'] is True
    assert overlays['cf1']['group_mask'] == 3
    assert overlays['cf2']['enabled'] is False


# ── load_deployment_specs ─────────────────────────────────────────────────────

def test_load_deployment_specs_filters_disabled_by_default(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
        {'name': 'cf2', 'uri': 'radio://cf2', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {
        'cf1': {'enabled': True},
        'cf2': {'enabled': False},
    })
    specs = load_deployment_specs(roster, setup)
    assert [s.name for s in specs] == ['cf1']


def test_load_deployment_specs_parses_enabled_string_false(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {'cf1': {'enabled': 'false'}})

    specs = load_deployment_specs(roster, setup)

    assert specs == []


def test_load_deployment_specs_rejects_ambiguous_enabled_string(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {'cf1': {'enabled': 'maybe'}})

    with pytest.raises(ValueError, match=r'merged\.cf1\.enabled must be a boolean'):
        load_deployment_specs(roster, setup)


def test_load_deployment_specs_drone_absent_from_setup_is_not_default_selection(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
        {'name': 'cf2', 'uri': 'radio://cf2', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {'cf1': {'enabled': True}})
    specs = load_deployment_specs(roster, setup)
    assert [s.name for s in specs] == ['cf1']


def test_load_deployment_specs_drone_absent_from_setup_preserves_roster_disabled(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
        {
            'name': 'cf2',
            'uri': 'radio://cf2',
            'type': 'crazyflie',
            'manager_type': 'cf_manager',
            'enabled': False,
        },
    ])
    setup = _setup(tmp_path, {'cf1': {'enabled': True}})
    specs = load_deployment_specs(roster, setup)
    assert [s.name for s in specs] == ['cf1']


def test_load_deployment_specs_overlay_overrides_roster_fields(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://original', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {
        'cf1': {'enabled': True, 'uri': 'radio://override'},
    })
    specs = load_deployment_specs(roster, setup)
    assert specs[0].uri == 'radio://override'


def test_load_deployment_specs_overlay_without_enabled_defaults_to_enabled(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {
        'cf1': {
            'initial_pose': {'position': {'x': 1.0, 'y': 0.0, 'z': 0.5}},
        },
    })
    specs = load_deployment_specs(roster, setup)
    assert [s.name for s in specs] == ['cf1']
    assert specs[0].enabled is True


def test_load_deployment_specs_roster_disabled_persists_through_overlay_omission(tmp_path):
    roster = _roster(tmp_path, [
        {
            'name': 'cf1',
            'uri': 'radio://cf1',
            'type': 'crazyflie',
            'manager_type': 'cf_manager',
            'enabled': False,
        },
    ])
    setup = _setup(tmp_path, {
        'cf1': {
            'initial_pose': {'position': {'x': 1.0, 'y': 0.0, 'z': 0.5}},
        },
    })

    specs = load_deployment_specs(roster, setup)

    assert specs == []


def test_load_deployment_specs_roster_disabled_persists_through_overlay_true(tmp_path):
    roster = _roster(tmp_path, [
        {
            'name': 'cf1',
            'uri': 'radio://cf1',
            'type': 'crazyflie',
            'manager_type': 'cf_manager',
            'enabled': False,
        },
    ])
    setup = _setup(tmp_path, {'cf1': {'enabled': True}})

    specs = load_deployment_specs(roster, setup)

    assert specs == []


def test_load_deployment_specs_selected_names_are_final_subset(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
        {'name': 'cf2', 'uri': 'radio://cf2', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
        {'name': 'cf3', 'uri': 'radio://cf3', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {
        'cf1': {'initial_pose': {'position': {'x': 1.0, 'y': 0.0, 'z': 0.5}}},
    })

    specs = load_deployment_specs(roster, setup, selected_names=['cf2'])

    assert [s.name for s in specs] == ['cf2']


def test_load_deployment_specs_selected_names_can_use_enabled_roster_drone_absent_from_setup(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
        {'name': 'cf2', 'uri': 'radio://cf2', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {'cf1': {'enabled': True}})

    specs = load_deployment_specs(roster, setup, selected_names=['cf2'])

    assert [s.name for s in specs] == ['cf2']


def test_load_deployment_specs_selected_names_rejects_disabled(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
        {'name': 'cf2', 'uri': 'radio://cf2', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {
        'cf1': {'enabled': False},
        'cf2': {'enabled': False},
    })
    with pytest.raises(ValueError, match='disabled by the roster/deployment selection'):
        load_deployment_specs(roster, setup, selected_names=['cf2'])


def test_load_deployment_specs_missing_selected_name_raises(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {'cf1': {'enabled': True}})
    with pytest.raises(ValueError, match='not found'):
        load_deployment_specs(roster, setup, selected_names=['ghost'])


def test_load_deployment_specs_rejects_setup_name_not_in_roster_by_default(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {
        'cfl': {'enabled': True},
    })

    with pytest.raises(ValueError, match='Deployment setup references drones not found in roster: cfl'):
        load_deployment_specs(roster, setup)


def test_load_deployment_specs_rejects_setup_name_not_in_roster_with_selected_names(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {
        'cfl': {'enabled': True},
    })

    with pytest.raises(ValueError, match='Deployment setup references drones not found in roster: cfl'):
        load_deployment_specs(roster, setup, selected_names=['cf1'])


def test_load_deployment_specs_extras_captured_from_overlay(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {
        'cf1': {
            'enabled': True,
            'initial_pose': {'position': {'x': 1.0, 'y': 2.0, 'z': 0.0}},
        }
    })
    specs = load_deployment_specs(roster, setup)
    assert specs[0].extras['initial_pose']['position']['x'] == 1.0


def test_load_deployment_specs_rejects_duplicate_extras_keys(tmp_path):
    roster = _roster(tmp_path, [
        {
            'name': 'cf1',
            'uri': 'radio://cf1',
            'type': 'crazyflie',
            'manager_type': 'cf_manager',
            'extras': {'initial_pose': {'position': {'x': 0.0}}},
            'initial_pose': {'position': {'x': 1.0}},
        },
    ])
    setup = _setup(tmp_path, {'cf1': {}})

    with pytest.raises(ValueError, match='extras duplicates unknown top-level key'):
        load_deployment_specs(roster, setup)


def test_load_deployment_specs_rejects_invalid_extra_args_type(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {
        'cf1': {'enabled': True, 'extra_args': '--bad'},
    })

    with pytest.raises(ValueError, match=r'merged\.cf1\.extra_args must be a list of strings'):
        load_deployment_specs(roster, setup)


@pytest.mark.parametrize('ros_params', [
    ['not', 'a', 'dict'],
    [],
])
def test_load_deployment_specs_rejects_invalid_ros_params_type(tmp_path, ros_params):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie', 'manager_type': 'cf_manager'},
    ])
    setup = _setup(tmp_path, {
        'cf1': {'enabled': True, 'ros_params': ros_params},
    })

    with pytest.raises(ValueError, match=r'merged\.cf1\.ros_params must be a dict'):
        load_deployment_specs(roster, setup)


def test_load_deployment_specs_ros_params_merged(tmp_path):
    roster = _roster(tmp_path, [
        {'name': 'cf1', 'uri': 'radio://cf1', 'type': 'crazyflie',
         'manager_type': 'cf_manager',
         'ros_params': {
             'odom_tf_name': 'odom',
             'nested': {'keep': True, 'replace': 'roster'},
         }},
    ])
    setup = _setup(tmp_path, {
        'cf1': {
            'enabled': True,
            'ros_params': {
                'group_mask': 7,
                'nested': {'replace': 'overlay'},
            },
        },
    })
    specs = load_deployment_specs(roster, setup)
    assert specs[0].ros_params.get('group_mask') == 7
    assert specs[0].ros_params.get('odom_tf_name') == 'odom'
    assert specs[0].ros_params['nested'] == {
        'keep': True,
        'replace': 'overlay',
    }


# ── apply_manager_overrides ───────────────────────────────────────────────────

def test_apply_manager_overrides_changes_selected_specs():
    specs = [
        DroneSpec(name='cf1', uri='radio://cf1', type='crazyflie', manager_type='cf_manager'),
        DroneSpec(name='px1', uri='udp://px1', type='pihawk', manager_type='pihawk_manager'),
    ]

    updated = apply_manager_overrides(specs, {'cf1': 'gazebo_manager'})

    assert [spec.manager_type for spec in updated] == ['gazebo_manager', 'pihawk_manager']
    assert specs[0].manager_type == 'cf_manager'


def test_apply_manager_overrides_rejects_missing_name():
    specs = [
        DroneSpec(name='cf1', uri='radio://cf1', type='crazyflie', manager_type='cf_manager'),
    ]

    with pytest.raises(ValueError, match="Add them to 'fliers' first"):
        apply_manager_overrides(specs, {'ghost': 'gazebo_manager'})

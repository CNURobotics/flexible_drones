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

"""Unit tests for resolved manager construction configuration."""

from flexible_drones_core.manager.config import (
    base_manager_config_from_spec,
    load_type_defaults,
    merge_manager_config_layers,
    resolve_package_config_path,
)
import flexible_drones_core.manager.config as core_config
from flexible_drones_core.registry import DroneSpec


def _make_spec(**kwargs) -> DroneSpec:
    defaults = {
        'name': 'cf1',
        'uri': 'radio://cf1',
        'type': 'crazyflie',
        'manager_type': 'cf_manager',
        'ros_params': {'group_mask': 3},
    }
    defaults.update(kwargs)
    return DroneSpec(**defaults)


def test_base_manager_config_preserves_common_fields_and_params():
    spec = _make_spec()

    config = base_manager_config_from_spec(spec)

    assert config.drone_name == 'cf1'
    assert config.uri == 'radio://cf1'
    assert config.drone_type == 'crazyflie'
    assert config.manager_type == 'cf_manager'
    assert config.ros_namespace == '/cf1'
    assert config.to_ros_parameters() == {
        'drone_name': 'cf1',
        'uri': 'radio://cf1',
        'drone_type': 'crazyflie',
        'manager_type': 'cf_manager',
        'group_mask': 3,
    }


def test_base_manager_config_applies_profile_namespace_and_node_name():
    spec = _make_spec(manager_type='gazebo_manager')
    profile = {
        'executable': 'gazebo_drone_manager',
        'manager_owns_drone_prefix': True,
        'node_name_template': '{name}_gz_manager',
    }

    config = base_manager_config_from_spec(spec, profile)

    assert config.ros_namespace == '/'
    assert config.node_name == 'cf1_gz_manager'


def test_base_manager_config_preserves_explicit_node_remap():
    spec = _make_spec(extra_args=['--ros-args', '-r', '__node:=custom'])
    profile = {'node_name_template': '{name}_manager'}

    config = base_manager_config_from_spec(spec, profile)

    assert config.node_name is None
    assert config.extra_args == ('--ros-args', '-r', '__node:=custom')


def test_merge_ros_params_layers_without_mutating_original():
    config = base_manager_config_from_spec(_make_spec())

    enriched = config.merge_ros_params({'robot_description': '<robot/>'}, group_mask=7)

    assert config.ros_params == {'group_mask': 3}
    assert enriched.ros_params['robot_description'] == '<robot/>'
    assert enriched.ros_params['group_mask'] == 7


def test_resolve_package_config_path_uses_package_share(monkeypatch, tmp_path):
    monkeypatch.setattr(
        core_config,
        'get_package_share_directory',
        lambda package_name: str(tmp_path / package_name),
    )

    path = resolve_package_config_path(
        '',
        default_filename='config/defaults.yaml',
        package_name='demo_package',
    )

    assert path == str(tmp_path / 'demo_package' / 'config' / 'defaults.yaml')


def test_load_type_defaults_validates_and_copies_entries(tmp_path):
    defaults_path = tmp_path / 'defaults.yaml'
    defaults_path.write_text(
        'types:\n'
        '  crazyflie:\n'
        '    odom_tf_name: odom\n',
        encoding='utf-8',
    )

    defaults = load_type_defaults(str(defaults_path), defaults_name='Demo defaults')

    assert defaults == {'crazyflie': {'odom_tf_name': 'odom'}}


def test_merge_manager_config_layers_applies_expected_precedence():
    spec = _make_spec(type='crazyflie', ros_params={'group_mask': 3})
    config = base_manager_config_from_spec(spec)

    enriched = merge_manager_config_layers(
        config,
        defaults_by_type={'crazyflie': {'group_mask': 255, 'odom_tf_name': 'odom'}},
        ros_params={'pose_source': 'optitrack'},
        forced_params={'robot_description': '<robot/>'},
    )

    assert config.ros_params == {'group_mask': 3}
    assert enriched.ros_params == {
        'group_mask': 3,
        'odom_tf_name': 'odom',
        'pose_source': 'optitrack',
        'robot_description': '<robot/>',
    }

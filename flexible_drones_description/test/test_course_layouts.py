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

"""Unit tests for shared gate course configuration."""

from flexible_drones_description.course_layouts import gate_marker_yaml
from flexible_drones_description.course_layouts import gate_tuple_list
from flexible_drones_description.course_layouts import load_gate_course


def test_builtin_course_configs_include_original_and_serc_sizes():
    """Both built-in layouts keep their expected pipe dimensions."""
    small = load_gate_course('small_gates.yaml')
    serc = load_gate_course('serc_bag.yaml')

    assert small.pipe_length == '0.5'
    assert small.base_height == '0.5'
    assert small.pipe_radius == '0.01065'
    assert serc.pipe_length == '1.0'
    assert serc.base_height == '0.5'
    assert serc.pipe_radius == '0.01670'


def test_serc_layout_uses_big_gate_positions():
    """The SERC layout uses the measured big-gate BAG course positions."""
    course = load_gate_course('serc_bag.yaml')

    assert gate_tuple_list(course) == [
        ('gate_A', 3.0, 0.0, 0.0, 1.5708),
        ('gate_B', 0.0, 3.0, 0.0, 3.1416),
        ('gate_C', -3.0, 0.0, 0.0, -1.5708),
        ('gate_D', 0.0, -3.0, 0.0, 0.0),
    ]


def test_marker_yaml_targets_gate_base_frames():
    """Gate label markers are attached to the gate base frames."""
    course = load_gate_course('small_gates.yaml')

    yaml_text = gate_marker_yaml(course)

    assert 'frame_id: gate_A_base_link' in yaml_text
    assert 'text: A' in yaml_text
    assert 'z: 1.75' in yaml_text


def test_marker_yaml_uses_base_height_when_label_is_not_configured():
    """Default label height follows gates with a shorter first segment."""
    course = load_gate_course('serc_bag.yaml')
    course = course.__class__(
        pipe_length=course.pipe_length,
        base_height=course.base_height,
        pipe_radius=course.pipe_radius,
        gates=course.gates,
    )

    yaml_text = gate_marker_yaml(course)

    assert 'z: 3.0' in yaml_text


def test_legacy_course_aliases_still_resolve():
    """Old layout names are kept as aliases for callers moving to YAML configs."""
    assert gate_tuple_list(load_gate_course('serc')) == gate_tuple_list(
        load_gate_course('serc_bag.yaml')
    )
    assert gate_tuple_list(load_gate_course('original')) == gate_tuple_list(
        load_gate_course('small_gates.yaml')
    )

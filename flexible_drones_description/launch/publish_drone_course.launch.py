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

"""Publish a configured gate course for RViz or hardware sessions."""

from flexible_drones_description.course_layouts import DEFAULT_COURSE_CONFIG
from flexible_drones_description.course_layouts import load_gate_course
from flexible_drones_description.course_layouts import make_gate_course_description_actions

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _optional_float_launch_value(context, name: str) -> float | None:
    value = LaunchConfiguration(name).perform(context).strip()
    if value == '':
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f'{name} must be a number, got {value!r}') from exc


def _course_actions(context, *args, **kwargs):
    del args, kwargs

    course = load_gate_course(LaunchConfiguration('course_config').perform(context))
    return make_gate_course_description_actions(
        course,
        frame_id=LaunchConfiguration('frame_id').perform(context),
        label_rate=LaunchConfiguration('label_rate'),
        label_z=_optional_float_launch_value(context, 'label_z'),
    )


def generate_launch_description():
    """Build the gate course publisher launch description."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'course_config',
            default_value=DEFAULT_COURSE_CONFIG,
            description=(
                'Gate course config filename from flexible_drones_description/config, '
                'or an absolute path.'
            ),
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='map',
            description='Parent frame for all gate base_link static transforms.',
        ),
        DeclareLaunchArgument(
            'label_z',
            default_value='',
            description=(
                'Optional label height in each gate frame; empty uses the config '
                'label_z or base_height + 2.5 * pipe_length.'
            ),
        ),
        DeclareLaunchArgument(
            'label_rate',
            default_value='0.2',
            description='Gate label MarkerArray publish rate in Hz.',
        ),
        OpaqueFunction(function=_course_actions),
    ])

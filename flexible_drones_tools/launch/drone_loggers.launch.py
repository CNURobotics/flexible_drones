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

"""Launch one simple drone logger per selected flier."""

import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _parse_fliers(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Could not parse fliers argument as YAML: {raw!r}") from exc

    if parsed is None:
        return []
    if isinstance(parsed, str):
        return [parsed.strip()] if parsed.strip() else []
    if isinstance(parsed, list):
        names = [str(item).strip() for item in parsed]
        return [name for name in names if name]

    raise RuntimeError(
        "fliers must be a YAML list such as '[cf4, cf8]' or a single drone name."
    )


def _build_actions(context, *args, **kwargs):
    del args, kwargs

    fliers = _parse_fliers(LaunchConfiguration('fliers').perform(context))
    if not fliers:
        raise RuntimeError(
            "No fliers specified. Use fliers:='[cf4]' or fliers:='[cf4, cf8]'."
        )

    buffer_length = ParameterValue(LaunchConfiguration('logging_buffer_length'), value_type=int)
    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    return [
        Node(
            package='flexible_drones_tools',
            executable='drone_logger',
            namespace=flier,
            name='drone_data_logger',
            output='screen',
            parameters=[
                {
                    'logging_buffer_length': buffer_length,
                    'use_sim_time': use_sim_time,
                }
            ],
        )
        for flier in fliers
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'fliers',
                default_value='',
                description="YAML list of drone names to log, e.g. [cf4, cf8].",
            ),
            DeclareLaunchArgument(
                'logging_buffer_length',
                default_value='6000',
                description='Number of samples each logger buffers before writing a chunk.',
            ),
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='false',
                description='Use simulation time for all launched loggers.',
            ),
            OpaqueFunction(function=_build_actions),
        ]
    )

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

"""Publish a gate course and optionally launch RViz."""

from flexible_drones_description.course_layouts import DEFAULT_COURSE_CONFIG
from flexible_drones_description.launch_assets import resolve_rviz_config_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _rviz_actions(context, *args, **kwargs):
    del args, kwargs
    return [
        Node(
            condition=IfCondition(LaunchConfiguration('launch_rviz')),
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=[
                '-d',
                resolve_rviz_config_path(
                    LaunchConfiguration('rviz_config').perform(context)
                ),
            ],
            output='screen',
        )
    ]


def generate_launch_description():
    """Build the RViz gate course launch description."""
    publish_course = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('flexible_drones_description'),
                'launch',
                'publish_drone_course.launch.py',
            ])
        ),
        launch_arguments={
            'course_config': LaunchConfiguration('course_config'),
            'frame_id': LaunchConfiguration('frame_id'),
            'label_z': LaunchConfiguration('label_z'),
            'label_rate': LaunchConfiguration('label_rate'),
        }.items(),
    )

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
            'rviz_config',
            default_value='gates.rviz',
            description=(
                'RViz config filename from flexible_drones_description/rviz, '
                'or an absolute path.'
            ),
        ),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
            description='Start RViz in addition to publishing the course.',
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
        publish_course,
        OpaqueFunction(function=_rviz_actions),
    ])

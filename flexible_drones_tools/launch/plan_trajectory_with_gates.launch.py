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

"""Launch the trajectory planner with default gate URDF obstacles."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


DEFAULT_GATE_DESCRIPTION_TOPICS = [
    '/gate_A_description',
    '/gate_B_description',
    '/gate_C_description',
    '/gate_D_description',
]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'preferred_frame',
                default_value='map',
                description='Frame used for trajectory planning and obstacle TF resolution.',
            ),
            DeclareLaunchArgument(
                'obstacle_safety_margin',
                default_value='0.381',
                description='Inflation margin added to each URDF cylinder radius.',
            ),
            DeclareLaunchArgument(
                'seed_strategy',
                default_value='dubins',
                description='Initial trajectory seed strategy.',
            ),
            DeclareLaunchArgument(
                'n_segments',
                default_value='4',
                description='Number of polynomial segments requested from the planner.',
            ),
            DeclareLaunchArgument(
                'min_segment_time',
                default_value='0.5',
                description='Minimum duration, in seconds, for each optimized polynomial segment.',
            ),
            DeclareLaunchArgument(
                'max_duration',
                default_value='60.0',
                description='Maximum duration, in seconds, for each optimized polynomial segment.',
            ),
            DeclareLaunchArgument(
                'gate_transition_distance',
                default_value='0.5',
                description='Distance, in metres, for soft pre/post gate transition targets.',
            ),
            DeclareLaunchArgument(
                'gate_transition_position_weight',
                default_value='10.0',
                description='Soft position weight for pre/post gate transition targets.',
            ),
            DeclareLaunchArgument(
                'gate_transition_velocity_weight',
                default_value='1.0',
                description='Soft velocity weight for moving gate transition targets.',
            ),
            DeclareLaunchArgument(
                'gate_transition_velocity_threshold',
                default_value='0.01',
                description=(
                    'Minimum boundary speed, in m/s, before velocity is used '
                    'as a soft target.'
                ),
            ),
            DeclareLaunchArgument(
                'planner_artifact_dir',
                default_value='/tmp',
                description='Directory for planner stage CSV artifacts; empty disables artifact writing.',
            ),
            DeclareLaunchArgument(
                'kinematic_constraint_scale',
                default_value='0.975',
                description='Scale applied to sampled optimizer kinematic caps; validation uses full limits.',
            ),
            DeclareLaunchArgument(
                'w_yaw_accel_limit',
                default_value='1.0',
                description='Output yaw soft penalty weight for acceleration normalized by yaw_accel_max.',
            ),
            DeclareLaunchArgument(
                'initial_ipopt_max_iter',
                default_value='800',
                description='IPOPT max_iter for the obstacle-free seed solve.',
            ),
            DeclareLaunchArgument(
                'initial_ipopt_tol',
                default_value='1e-4',
                description='IPOPT tolerance for the obstacle-free seed solve.',
            ),
            DeclareLaunchArgument(
                'initial_ipopt_constr_viol_tol',
                default_value='1e-4',
                description='IPOPT constraint violation tolerance for the obstacle-free seed solve.',
            ),
            DeclareLaunchArgument(
                'initial_ipopt_acceptable_tol',
                default_value='1e-3',
                description='IPOPT acceptable tolerance for the obstacle-free seed solve.',
            ),
            DeclareLaunchArgument(
                'initial_ipopt_acceptable_constr_viol_tol',
                default_value='1e-3',
                description='IPOPT acceptable constraint violation tolerance for the obstacle-free seed solve.',
            ),
            DeclareLaunchArgument(
                'initial_ipopt_acceptable_iter',
                default_value='5',
                description='IPOPT acceptable_iter for the obstacle-free seed solve.',
            ),
            DeclareLaunchArgument(
                'constrained_ipopt_max_iter',
                default_value='800',
                description='IPOPT max_iter for the obstacle-constrained solve.',
            ),
            DeclareLaunchArgument(
                'constrained_ipopt_tol',
                default_value='1e-6',
                description='IPOPT tolerance for the obstacle-constrained solve.',
            ),
            DeclareLaunchArgument(
                'constrained_ipopt_constr_viol_tol',
                default_value='1e-6',
                description='IPOPT constraint violation tolerance for the obstacle-constrained solve.',
            ),
            DeclareLaunchArgument(
                'constrained_ipopt_acceptable_tol',
                default_value='1e-5',
                description='IPOPT acceptable tolerance for the obstacle-constrained solve.',
            ),
            DeclareLaunchArgument(
                'constrained_ipopt_acceptable_constr_viol_tol',
                default_value='1e-5',
                description='IPOPT acceptable constraint violation tolerance for the obstacle-constrained solve.',
            ),
            DeclareLaunchArgument(
                'constrained_ipopt_acceptable_iter',
                default_value='5',
                description='IPOPT acceptable_iter for the obstacle-constrained solve.',
            ),
            DeclareLaunchArgument(
                'use_cache',
                default_value='false',
                description='Load dense-valid cached trajectories instead of regenerating.',
            ),
            DeclareLaunchArgument(
                'save_cache',
                default_value='true',
                description='Write generated trajectories to the cache for later reuse.',
            ),
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='false',
                description='Use simulation time for the trajectory planner.',
            ),
            Node(
                package='flexible_drones_tools',
                executable='plan_trajectory_action_server',
                name='plan_trajectory_action_server',
                output='screen',
                parameters=[
                    {
                        'preferred_frame': LaunchConfiguration('preferred_frame'),
                        'obstacle_description_topics': DEFAULT_GATE_DESCRIPTION_TOPICS,
                        'obstacle_safety_margin': ParameterValue(
                            LaunchConfiguration('obstacle_safety_margin'),
                            value_type=float,
                        ),
                        'seed_strategy': LaunchConfiguration('seed_strategy'),
                        'n_segments': ParameterValue(
                            LaunchConfiguration('n_segments'),
                            value_type=int,
                        ),
                        'min_segment_time': ParameterValue(
                            LaunchConfiguration('min_segment_time'),
                            value_type=float,
                        ),
                        'max_duration': ParameterValue(
                            LaunchConfiguration('max_duration'),
                            value_type=float,
                        ),
                        'gate_transition_distance': ParameterValue(
                            LaunchConfiguration('gate_transition_distance'),
                            value_type=float,
                        ),
                        'gate_transition_position_weight': ParameterValue(
                            LaunchConfiguration('gate_transition_position_weight'),
                            value_type=float,
                        ),
                        'gate_transition_velocity_weight': ParameterValue(
                            LaunchConfiguration('gate_transition_velocity_weight'),
                            value_type=float,
                        ),
                        'gate_transition_velocity_threshold': ParameterValue(
                            LaunchConfiguration('gate_transition_velocity_threshold'),
                            value_type=float,
                        ),
                        'planner_artifact_dir': LaunchConfiguration('planner_artifact_dir'),
                        'kinematic_constraint_scale': ParameterValue(
                            LaunchConfiguration('kinematic_constraint_scale'),
                            value_type=float,
                        ),
                        'w_yaw_accel_limit': ParameterValue(
                            LaunchConfiguration('w_yaw_accel_limit'),
                            value_type=float,
                        ),
                        'initial_ipopt_max_iter': ParameterValue(
                            LaunchConfiguration('initial_ipopt_max_iter'),
                            value_type=int,
                        ),
                        'initial_ipopt_tol': ParameterValue(
                            LaunchConfiguration('initial_ipopt_tol'),
                            value_type=float,
                        ),
                        'initial_ipopt_constr_viol_tol': ParameterValue(
                            LaunchConfiguration('initial_ipopt_constr_viol_tol'),
                            value_type=float,
                        ),
                        'initial_ipopt_acceptable_tol': ParameterValue(
                            LaunchConfiguration('initial_ipopt_acceptable_tol'),
                            value_type=float,
                        ),
                        'initial_ipopt_acceptable_constr_viol_tol': ParameterValue(
                            LaunchConfiguration('initial_ipopt_acceptable_constr_viol_tol'),
                            value_type=float,
                        ),
                        'initial_ipopt_acceptable_iter': ParameterValue(
                            LaunchConfiguration('initial_ipopt_acceptable_iter'),
                            value_type=int,
                        ),
                        'constrained_ipopt_max_iter': ParameterValue(
                            LaunchConfiguration('constrained_ipopt_max_iter'),
                            value_type=int,
                        ),
                        'constrained_ipopt_tol': ParameterValue(
                            LaunchConfiguration('constrained_ipopt_tol'),
                            value_type=float,
                        ),
                        'constrained_ipopt_constr_viol_tol': ParameterValue(
                            LaunchConfiguration('constrained_ipopt_constr_viol_tol'),
                            value_type=float,
                        ),
                        'constrained_ipopt_acceptable_tol': ParameterValue(
                            LaunchConfiguration('constrained_ipopt_acceptable_tol'),
                            value_type=float,
                        ),
                        'constrained_ipopt_acceptable_constr_viol_tol': ParameterValue(
                            LaunchConfiguration('constrained_ipopt_acceptable_constr_viol_tol'),
                            value_type=float,
                        ),
                        'constrained_ipopt_acceptable_iter': ParameterValue(
                            LaunchConfiguration('constrained_ipopt_acceptable_iter'),
                            value_type=int,
                        ),
                        'use_cache': ParameterValue(
                            LaunchConfiguration('use_cache'),
                            value_type=bool,
                        ),
                        'save_cache': ParameterValue(
                            LaunchConfiguration('save_cache'),
                            value_type=bool,
                        ),
                        'use_sim_time': ParameterValue(
                            LaunchConfiguration('use_sim_time'),
                            value_type=bool,
                        ),
                    }
                ],
            ),
        ]
    )

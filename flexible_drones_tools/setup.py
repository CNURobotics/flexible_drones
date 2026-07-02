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

"""Install the python scripts."""

from glob import glob
import os
from setuptools import find_packages, setup

PACKAGE_NAME = 'flexible_drones_tools'

setup(
    name=PACKAGE_NAME,
    version='0.0.1',
    packages=find_packages(include=[PACKAGE_NAME, PACKAGE_NAME + '.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + PACKAGE_NAME]),
        ('share/' + PACKAGE_NAME, ['package.xml']),
        (
            os.path.join('share', PACKAGE_NAME, 'launch'),
            glob(os.path.join('launch', '*.launch.py')),
        ),
    ],
    install_requires=[
        'casadi',
        'cvxpy',
        'matplotlib',
        'numpy',
        'PyYAML',
        'scipy',
        'setuptools',
        'urdf_parser_py',
    ],
    zip_safe=True,
    maintainer='David Conner',
    maintainer_email='robotics@cnu.edu',
    description='Command-line clients, action servers, demos, and monitoring tools for Flexible Drones.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                # Clients
                'arm = flexible_drones_tools.clients.arm_client:main',
                'emergency_stop = flexible_drones_tools.clients.emergency_stop_client:main',
                'teleop_control = flexible_drones_tools.clients.teleop_control_client:main',
                'get_trajectory_client = flexible_drones_tools.clients.get_trajectory_client:main',
                'go_to = flexible_drones_tools.clients.go_to_action_client:main',
                'land = flexible_drones_tools.clients.land_action_client:main',
                'reboot = flexible_drones_tools.clients.reboot_client:main',
                'realign_local_position = flexible_drones_tools.clients.realign_local_position_client:main',
                'set_led_color = flexible_drones_tools.clients.set_led_color_client:main',
                'start_execute_trajectory = flexible_drones_tools.clients.execute_trajectory_action_client:main',
                'takeoff = flexible_drones_tools.clients.takeoff_action_client:main',
                'get_sampled_trajectory_client = flexible_drones_tools.clients.get_sampled_trajectory_client:main',
                'upload_trajectory = flexible_drones_tools.clients.upload_trajectory_action_client:main',

                # Servers
                'load_trajectory_action_server = flexible_drones_tools.trajectories.load_trajectory_action_server:main',
                'plan_trajectory_action_server = flexible_drones_tools.trajectories.plan_trajectory_action_server:main',

                # Monitoring
                'drone_logger = flexible_drones_tools.monitoring.drone_logger:main',
                'plot_logs = flexible_drones_tools.monitoring.plot_logs:main',
                'run_drone_plot = flexible_drones_tools.monitoring.run_drone_plot:main',
                'status_gui = flexible_drones_tools.monitoring.status_gui:main',
                'teleop_gui = flexible_drones_tools.monitoring.teleop_gui:main',

                # Trajectories
                'visualize_flight = flexible_drones_tools.monitoring.visualize_flight:main',
                'visualize_trajectory = flexible_drones_tools.trajectories.visualize_trajectory:main',
                'plot_trajectory = flexible_drones_tools.trajectories.plot_trajectory:main',

                # Trajectory generators are NOT exposed as ros2 entry points; they are
                # run from source. See flexible_drones_tools/trajectories/generators/README.md.

                # Demos
                'basic_demo = flexible_drones_tools.demos.basic_demo:main',
                'square_demo = flexible_drones_tools.demos.square_demo:main',
                'planner_demo = flexible_drones_tools.demos.planner_demo:main',
                'planner_test = flexible_drones_tools.demos.planner_test:main',
                'teleop_flight_demo = flexible_drones_tools.demos.teleop_flight_demo:main',
                'trajectory_demo = flexible_drones_tools.demos.trajectory_demo:main',
        ],
    },
)

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

"""CLI client for the ``upload_trajectory`` action."""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from flexible_drones_msgs.action import UploadTrajectory
from flexible_drones_msgs.msg import Trajectory

from flexible_drones_tools.clients.client_utils import (
    declare_wait_timeout_parameter,
    print_usage_if_requested,
    wait_for_action_server,
)
from flexible_drones_tools.ros_shutdown import spin_node

USAGE = """\
Usage:
  ros2 run flexible_drones_tools upload_trajectory --ros-args -r __ns:=/drone1 \
    -p trajectory_package:=<deployment_package> -p trajectory_file:=figure8.csv

Common examples:
  export FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE=<deployment_package>
  ros2 run flexible_drones_tools upload_trajectory --ros-args -p trajectory_file:=figure8.csv
  ros2 run flexible_drones_tools upload_trajectory --ros-args -r __ns:=/drone1 \
    -p trajectory_package:=<deployment_package> -p trajectory_file:=figure8.csv
  ros2 run flexible_drones_tools upload_trajectory --ros-args -r __ns:=/drone1 -r __node:=upload_client -p trajectory_id:=1

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Parameters declared by this node:
  trajectory_id (integer, default: 0)
    ID assigned to the uploaded trajectory.
  trajectory_path (string, default: empty)
    Direct path to a trajectory CSV. When set, it overrides package/folder/file lookup.
  trajectory_package (string, default: empty)
    Package used to resolve a trajectory CSV. Empty falls back to
    FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE.
  trajectory_folder (string, default: trajectories)
    Folder inside the trajectory package.
  trajectory_file (string, default: figure8.csv)
    CSV filename to load.
  wait_timeout_sec (double, default: 5.0)
    Maximum time to wait for the action server.

Examples with parameter overrides:
  ros2 run flexible_drones_tools upload_trajectory --ros-args -p trajectory_path:=/tmp/figure8.csv
  ros2 run flexible_drones_tools upload_trajectory --ros-args -r __ns:=/drone2 \
    -p trajectory_id:=2 -p trajectory_package:=<deployment_package> \
    -p trajectory_folder:=trajectories -p trajectory_file:=figure8.csv
"""


class UploadTrajActionClient(Node):

    def __init__(self):
        super().__init__('upload_trajectory_action_client')

        # Declare parameters with default values
        self.declare_parameter('trajectory_id', 0)
        self.declare_parameter('trajectory_path', '')
        self.declare_parameter('trajectory_package', '')
        self.declare_parameter('trajectory_folder', 'trajectories')
        self.declare_parameter('trajectory_file', 'figure8.csv')
        declare_wait_timeout_parameter(self)

        self._action_client = ActionClient(
            self, UploadTrajectory, 'upload_trajectory'
        )  # Figure out how to add the name of each Drone here? (Anwser: Remapping the namespace via CLA!)

    def send_goal(self):
        from flexible_drones_tools.trajectories.utilities.io import load_trajectory_pieces
        from flexible_drones_tools.trajectories.utilities.trajectory_path import resolve_trajectory_path

        trajectory_id = self.get_parameter('trajectory_id').get_parameter_value().integer_value
        trajectory_path = self.get_parameter('trajectory_path').get_parameter_value().string_value
        trajectory_package = self.get_parameter('trajectory_package').get_parameter_value().string_value
        trajectory_folder = self.get_parameter('trajectory_folder').get_parameter_value().string_value
        trajectory_file = self.get_parameter('trajectory_file').get_parameter_value().string_value

        resolved_path = '<unresolved>'
        try:
            resolved_path = resolve_trajectory_path(
                trajectory_path=trajectory_path,
                trajectory_package=trajectory_package,
                trajectory_folder=trajectory_folder,
                trajectory_file=trajectory_file,
            )
            pieces = load_trajectory_pieces(resolved_path)
        except Exception as e:
            self.get_logger().error(f'Failed to load trajectory from CSV: {e}')
            rclpy.shutdown()
            return False

        goal_msg = UploadTrajectory.Goal()
        goal_msg.trajectory = Trajectory(
            trajectory_id=trajectory_id,
            pieces=pieces,
        )

        if not wait_for_action_server(self, self._action_client, 'UploadTrajectory action server'):
            return False

        self.get_logger().info(f"Sending UploadTrajectory goal with {len(pieces)} pieces from '{resolved_path}'...")
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning('UploadTrajectory goal was rejected.')
            rclpy.shutdown()
            return

        self.get_logger().info('UploadTrajectory goal accepted.')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.return_code == 0:
            self.get_logger().info('UploadTrajectory finished successfully! ' f'Uploaded {result.uploaded_pieces} pieces.')
        else:
            self.get_logger().warning(f'UploadTrajectory finished with return_code={result.return_code}')

        rclpy.shutdown()  # Shuts down ROS 2 cleanly


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    rclpy.init(args=args)
    action_client = UploadTrajActionClient()
    if action_client.send_goal():
        spin_node(action_client)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
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

# usage:
#    ros2 run flexible_drones_tools visualize_flight --ros-args -p drone_name:=drone1

import os
import re
import sys
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node

from flexible_drones_tools.ros_shutdown import spin_node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath


def load_log_data(log_dir, start_time, index_range):
    """Load data files and stack into single numpy array per item."""
    data = {'odom': [], 'pose': []}
    print(f"Loading from '{log_dir}' at {start_time} for indices {index_range}")
    for idx in range(index_range[0], index_range[1] + 1):
        for key in data.keys():
            fname = log_dir / f'{key}_{start_time}_{idx}.npy'
            if fname.exists():
                print(f'Loading: {fname}')
                try:
                    data[key].append(np.load(fname))
                except Exception:
                    print(f"Failed to load data for '{fname}'!")
            else:
                print(f'Missing: {fname}')
    for key in data:
        if len(data[key]) > 0:
            data[key] = np.vstack(data[key])
        else:
            data[key] = np.zeros((1, 1))

    if data['odom'].shape[0] > data['pose'].shape[0]:
        times = data['odom'][:, 0] > 0.0
        return data['odom'][times, :8]  # stamp, pos x,y,z, ori x,y,z,w
    else:
        times = data['pose'][:, 0] > 0.0
        return data['pose'][times, :8]  # stamp, pos x,y,z, ori x,y,z,w


def find_latest_start_time_and_index(log_dir):
    """
    Scan log_dir and return the start_time and max index.

    Find the most recently created log file based on filesystem time.
    Returns (start_time: str, max_index: int).
    """
    pattern = re.compile(r'(odom|pose)_(\d+)_(\d+)\.npy')
    latest_file = None
    latest_mtime = -1

    for file in log_dir.glob('*.npy'):
        match = pattern.match(file.name)
        if match:
            mtime = file.stat().st_mtime  # modification time (or use st_ctime on Windows)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_file = file

    if latest_file is None:
        raise FileNotFoundError('No valid log files found in the directory.')

    # Extract start_time and index from the latest file's name
    match = pattern.match(latest_file.name)
    start_time = match.group(2)
    index = int(match.group(3))

    return start_time, index


class DroneFlightPathClient(Node):
    def __init__(self):
        super().__init__('drone_flight_path_client')

        self.trajectory = None

        default_log_dir = Path(os.environ.get('WORKSPACE_ROOT', '.')) / 'log'

        self.declare_parameter('log_folder', f'{default_log_dir}')
        self.declare_parameter('drone_name', 'drone1')
        self.declare_parameter('period_sec', 2.0)
        self.declare_parameter('samples', 100)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('start_time', '')
        self.declare_parameter('end_idx', -1)

        log_folder = self.get_parameter('log_folder').get_parameter_value().string_value
        drone_name = self.get_parameter('drone_name').get_parameter_value().string_value
        self.period_sec = self.get_parameter('period_sec').get_parameter_value().double_value
        self.num_samples = self.get_parameter('samples').get_parameter_value().integer_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.start_time = self.get_parameter('start_time').get_parameter_value().string_value
        self.end_idx = self.get_parameter('end_idx').get_parameter_value().integer_value
        try:
            self.log_path = Path(log_folder) / drone_name

            latest_time, latest_idx = find_latest_start_time_and_index(self.log_path)

            if self.start_time == '':
                self.start_time = latest_time
                self.get_logger().info(f'[INFO] No start_time provided. Using latest created file: {self.start_time}')
            if self.end_idx < 0:
                self.end_idx = latest_idx
                self.get_logger().info(f'[INFO] No end_idx provided. Using latest available index: {self.end_idx}')

            self.flight_data = load_log_data(self.log_path, self.start_time, (0, self.end_idx))

            if self.flight_data.shape[0] > 1:
                self.get_logger().info(f"Processing flight with {self.flight_data.shape[0]} data points from '{self.log_path}'")
            else:
                self.get_logger().error(f"Invalid trajectory '{self.log_path}'")
                sys.exit(1)

        except Exception as exc:
            self.get_logger().error(f"Invalid trajectory '{self.log_path}'\n    {exc}")
            sys.exit(1)

        if self.flight_data is None:
            self.get_logger().error(f"Invalid trajectory '{self.log_path}'\n")
            sys.exit(1)

        self.get_logger().info(f"Loaded trajectory from '{self.log_path}'.")

        self.path_msg = NavPath()
        self.path_msg.header.frame_id = self.frame_id
        self.path_msg.header.stamp = self.get_clock().now().to_msg()

        for ndx in range(self.flight_data.shape[0]):
            t = self.flight_data[ndx, 0]
            sec = int(t)
            nanosec = int((t - sec) * 1e9)
            pos = self.flight_data[ndx, 1:4].flatten()
            quat = self.flight_data[ndx, 4:8].flatten()
            pose = PoseStamped()
            pose.header.stamp = Time(sec=sec, nanosec=nanosec)
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = pos[0]
            pose.pose.position.y = pos[1]
            pose.pose.position.z = pos[2]
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            self.path_msg.poses.append(pose)

        self.path_pub_ = self.create_publisher(NavPath, 'flight_trajectory', 1)
        self.timer = self.create_timer(self.period_sec, self.publish_path)

    def publish_path(self):

        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub_.publish(self.path_msg)
        self.get_logger().info(f"Publishing flight from '{self.log_path}' as nav_msgs/Path")


def main():
    rclpy.init()
    node = DroneFlightPathClient()
    try:
        spin_node(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

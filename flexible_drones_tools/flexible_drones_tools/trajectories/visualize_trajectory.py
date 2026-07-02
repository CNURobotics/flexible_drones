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

"""Publish a planned trajectory CSV as ``nav_msgs/Path`` for RViz."""

import argparse
import math
import sys

import numpy as np

import rclpy
from rclpy.node import Node

from flexible_drones_tools.ros_shutdown import spin_node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath

from .utilities.model import PolyOrderTrajectory
from .utilities.io import load_trajectory_csv
from .utilities.trajectory_path import resolve_deployment_trajectory_path

DEFAULT_PATH_TOPIC = 'planned_trajectory'


def load_drone_order_trajectory(file_path):
    """Load standard drone-order CSV coefficients for NumPy polynomial evaluation."""
    print(f"Loading trajectory from '{file_path}' ...", flush=True)
    data = load_trajectory_csv(file_path, coefficient_order='numpy')
    return PolyOrderTrajectory(*data)


def visualization_offset_for_trajectory(
    trajectory,
    *,
    offset=None,
    auto_ground_offset=True,
):
    """Return the visualization offset array for a trajectory."""
    offset_array = np.array(offset if offset is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    start_posn = trajectory.evaluate(0.0)[0]
    if auto_ground_offset and start_posn is not None and start_posn[2] < 0.05:
        offset_array[2] = max(0.5, offset_array[2])
    return offset_array


def trajectory_to_path_msg(
    trajectory,
    *,
    frame_id='map',
    samples=100,
    offset=None,
    stamp=None,
):
    """Sample a trajectory and return a ``nav_msgs/Path``."""
    offset_array = np.array(offset if offset is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    path_msg = NavPath()
    path_msg.header.frame_id = frame_id
    if stamp is not None:
        path_msg.header.stamp = stamp

    times = np.linspace(0.0, trajectory.total_duration(), int(samples))
    for t in times:
        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        pos, _, _ = trajectory.evaluate(t)
        if pos is None:
            print(f'no position at t={t} of {times[-1]}', flush=True)
            continue

        pose = PoseStamped()
        pose.header.stamp = Time(sec=sec, nanosec=nanosec)
        pose.header.frame_id = frame_id
        pose.pose.position.x = pos[0] + offset_array[0]
        pose.pose.position.y = pos[1] + offset_array[1]
        pose.pose.position.z = pos[2] + offset_array[2]
        half_yaw = pos[3] / 2.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(half_yaw)
        pose.pose.orientation.w = math.cos(half_yaw)
        path_msg.poses.append(pose)

    return path_msg


class DroneTrajectoryPathClient(Node):
    def __init__(
        self,
        *,
        trajectory,
        trajectory_path,
        period_sec=2.0,
        samples=100,
        frame_id='map',
        offset=None,
        auto_ground_offset=True,
    ):
        super().__init__('drone_trajectory_path_client')

        self.trajectory = trajectory
        self.file_path = trajectory_path
        self.period_sec = float(period_sec)
        self.num_samples = int(samples)
        self.frame_id = frame_id
        self.offset = visualization_offset_for_trajectory(
            self.trajectory,
            offset=offset,
            auto_ground_offset=auto_ground_offset,
        )

        start_posn = self.trajectory.evaluate(0.0)[0]
        self.get_logger().info(f'Trajectory starting position = {start_posn}')
        if auto_ground_offset and start_posn is not None and start_posn[2] < 0.05:
            self.get_logger().info(
                'Trajectory starts near ground; applying z offset '
                f'{self.offset[2]:.3f} for visualization.'
            )

        self.get_logger().info(f"Loaded trajectory from '{self.file_path}'.")

        self.path_msg = trajectory_to_path_msg(
            self.trajectory,
            frame_id=self.frame_id,
            samples=self.num_samples,
            offset=self.offset,
            stamp=self.get_clock().now().to_msg(),
        )
        self.path_pub_ = self.create_publisher(NavPath, DEFAULT_PATH_TOPIC, 1)
        self.timer = self.create_timer(self.period_sec, self.publish_path)

    def publish_path(self):

        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub_.publish(self.path_msg)
        self.get_logger().info(
            f"Publishing planned_trajectory at '{self.file_path}' as nav_msgs/Path"
        )


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Visualize a drone-order trajectory CSV in RViz as nav_msgs/Path.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.

Examples:
  export FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE=<deployment_package>
  ros2 run flexible_drones_tools visualize_trajectory figure8.csv
  ros2 run flexible_drones_tools visualize_trajectory cnu_sail0.csv --samples 300
  ros2 run flexible_drones_tools visualize_trajectory ./my_traj.csv \\
    --ros-args -p use_sim_time:=true
""",
    )
    parser.add_argument(
        'trajectory_file',
        help=(
            'Trajectory CSV. Bare filenames are resolved in '
            '$FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE/trajectories/; relative or '
            'absolute paths are used directly.'
        ),
    )
    parser.add_argument('--period-sec', type=float, default=2.0,
                        help='Seconds between repeated Path publishes (default: 2.0)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of poses to sample along the trajectory (default: 100)')
    parser.add_argument('--frame-id', default='map',
                        help="Header frame for the published Path (default: 'map')")
    parser.add_argument('--offset-x', type=float, default=0.0,
                        help='X offset added to every pose (default: 0.0)')
    parser.add_argument('--offset-y', type=float, default=0.0,
                        help='Y offset added to every pose (default: 0.0)')
    parser.add_argument('--offset-z', type=float, default=0.0,
                        help='Z offset added to every pose (default: 0.0)')
    parser.add_argument('--no-auto-ground-offset', action='store_true',
                        help='Disable automatic z offset for trajectories that start near z=0')
    parsed, ros_args = parser.parse_known_args(args=args)

    if parsed.samples < 2:
        parser.error('--samples must be at least 2')
    if parsed.period_sec <= 0.0:
        parser.error('--period-sec must be positive')

    try:
        trajectory_path = resolve_deployment_trajectory_path(parsed.trajectory_file)
        trajectory = load_drone_order_trajectory(trajectory_path)
    except Exception as exc:
        print(f'Failed to load trajectory: {exc}', file=sys.stderr, flush=True)
        return 1

    rclpy.init(args=ros_args)
    node = DroneTrajectoryPathClient(
        trajectory=trajectory,
        trajectory_path=trajectory_path,
        period_sec=parsed.period_sec,
        samples=parsed.samples,
        frame_id=parsed.frame_id,
        offset=[parsed.offset_x, parsed.offset_y, parsed.offset_z],
        auto_ground_offset=not parsed.no_auto_ground_offset,
    )
    try:
        spin_node(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())

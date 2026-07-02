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

"""Basic demo: arm, takeoff, move +x relative, land, disarm."""

import argparse

import rclpy
from rclpy.exceptions import ROSInterruptException
from rclpy.executors import ExternalShutdownException, ShutdownException

from flexible_drones_tools.demos.demo_cli import print_failure_summary
from flexible_drones_tools.demos.trajectory_demo import TrajectoryDemo


class BasicDemo(TrajectoryDemo):
    def __init__(self, takeoff_height: float, land_height: float):
        super().__init__(
            'basic_demo',
            takeoff_height=takeoff_height,
            land_height=land_height,
        )


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Basic demo: arm/takeoff/relative +x go_to/land/disarm.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set the node namespace with '-r __ns:=/drone1'.
  If no namespace is set, remapping the node name can target '/<node_name>'.

Examples:
  ros2 run flexible_drones_tools basic_demo --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools basic_demo --ros-args -r __node:=drone1
  ros2 run flexible_drones_tools basic_demo --distance 0.5 --takeoff-height 0.75 --ros-args -r __ns:=/drone2
""",
    )
    parser.add_argument('--distance', type=float, default=1.0,
                        help='Relative map +x movement in metres, 0.1 to 5.0 (default: 1.0)')
    parser.add_argument('--takeoff-height', type=float, default=1.0,
                        help='Takeoff altitude in metres, 0.25 to 5.0 (default: 1.0)')
    parser.add_argument('--land-height', type=float, default=0.02,
                        help='Landing target height in metres, 0.0 to 1.0 (default: 0.02)')
    parser.add_argument('--move-duration', type=float, default=5.0,
                        help='Seconds for the relative go_to leg, 5.0 to 60.0 (default: 5.0)')
    parser.add_argument('--timeout-sec', type=float, default=10.0,
                        help='Timeout for arm, takeoff, land, and disarm in seconds '
                             '(default: 10.0)')
    parsed, ros_args = parser.parse_known_args(args=args)
    if not 0.1 <= parsed.distance <= 5.0:
        parser.error('--distance must be between 0.1 and 5.0')
    if not 0.25 <= parsed.takeoff_height <= 5.0:
        parser.error('--takeoff-height must be between 0.25 and 5.0')
    if not 0.0 <= parsed.land_height <= 1.0:
        parser.error('--land-height must be between 0.0 and 1.0')
    if not 5.0 <= parsed.move_duration <= 60.0:
        parser.error('--move-duration must be between 5.0 and 60.0')
    if parsed.timeout_sec < 2.0:
        parser.error('--timeout-sec must be at least 2.0')

    rclpy.init(args=ros_args)
    demo = BasicDemo(parsed.takeoff_height, parsed.land_height)

    try:
        armed = False
        flying = False
        failure = None
        cleanup_errors = []
        try:
            print('Waiting for initial pose...', flush=True)
            demo.wait_for_pose()

            demo.arm(timeout_sec=parsed.timeout_sec)
            armed = True
            demo.takeoff(timeout_sec=parsed.timeout_sec)
            flying = True
            demo._spin_sleep(1.0)
            demo.go_to(
                parsed.distance,
                0.0,
                0.0,
                0.0,
                parsed.move_duration,
                f'relative +x {parsed.distance:.2f} m',
                relative=True,
            )
        except (KeyboardInterrupt, ExternalShutdownException, ShutdownException,
                ROSInterruptException):
            demo.get_logger().info(f'[{demo._ns}] interrupted')
        except Exception as e:
            failure = e
        finally:
            if flying:
                try:
                    demo.land(timeout_sec=parsed.timeout_sec)
                except Exception as e:
                    demo.get_logger().error(f'[{demo._ns}] cleanup land failed: {e}')
                    cleanup_errors.append(f'land failed: {e}')
            if armed:
                try:
                    demo.disarm(timeout_sec=parsed.timeout_sec)
                except Exception as e:
                    demo.get_logger().error(f'[{demo._ns}] cleanup disarm failed: {e}')
                    cleanup_errors.append(f'disarm failed: {e}')
                    if demo.appears_airborne():
                        cleanup_errors.append(demo.warn_manual_recovery_needed())

            if cleanup_errors and failure is None:
                failure = RuntimeError('; '.join(cleanup_errors))

        if failure is not None:
            print_failure_summary('basic_demo', demo._ns, failure, cleanup_errors)
            return 1
        return 0
    finally:
        demo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

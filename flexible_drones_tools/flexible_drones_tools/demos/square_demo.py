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

"""Square demo: arm, takeoff, fly a map-aligned 2x2 square, land, disarm."""

import argparse
import math

import rclpy
from rclpy.exceptions import ROSInterruptException
from rclpy.executors import ExternalShutdownException, ShutdownException

from flexible_drones_tools.demos.demo_cli import print_failure_summary
from flexible_drones_tools.demos.trajectory_demo import TrajectoryDemo


class SquareDemo(TrajectoryDemo):
    def __init__(self, cruise_height: float):
        super().__init__(
            'square_demo',
            takeoff_height=cruise_height,
            land_height=0.0,
        )


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Square demo: arm/takeoff/map-aligned square/land/disarm.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set the node namespace with '-r __ns:=/drone1'.
  If no namespace is set, remapping the node name can target '/<node_name>'.

Examples:
  ros2 run flexible_drones_tools square_demo --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools square_demo --ros-args -r __node:=drone1
  ros2 run flexible_drones_tools square_demo --cruise-height 1.2 --leg-duration 5.0 --ros-args -r __ns:=/drone2
""",
    )
    parser.add_argument('--cruise-height', type=float, default=1.0,
                        help='Cruise altitude in metres, 0.25 to 5.0 (default: 1.0)')
    parser.add_argument('--leg-duration', type=float, default=5.0,
                        help='Seconds per go_to leg, 5.0 to 60.0 (default: 5.0)')
    parser.add_argument('--timeout-sec', type=float, default=10.0,
                        help='Timeout for arm, takeoff, land, and disarm in seconds '
                             '(default: 10.0)')
    parsed, ros_args = parser.parse_known_args(args=args)
    if not 0.25 <= parsed.cruise_height <= 5.0:
        parser.error('--cruise-height must be between 0.25 and 5.0')
    if not 5.0 <= parsed.leg_duration <= 60.0:
        parser.error('--leg-duration must be between 5.0 and 60.0')
    if parsed.timeout_sec < 2.0:
        parser.error('--timeout-sec must be at least 2.0')

    rclpy.init(args=ros_args)
    demo = SquareDemo(parsed.cruise_height)

    try:
        failure = None
        cleanup_errors = []
        try:
            print('Waiting for initial pose...', flush=True)
            demo.wait_for_pose()

            demo.arm(timeout_sec=parsed.timeout_sec)
            demo.takeoff(timeout_sec=parsed.timeout_sec)
            demo._spin_sleep(1.0)

            sx, sy, sz, syaw = demo.current_pose()
            demo.get_logger().info(
                f'[{demo._ns}] start pose: x={sx:.3f}, y={sy:.3f}, '
                f'z={sz:.3f}, yaw={math.degrees(syaw):.1f}°'
            )
            z = parsed.cruise_height

            try:
                # Step 1: advance 1 m along the map +x axis.
                demo.go_to(sx + 1.0, sy, z, syaw, parsed.leg_duration, 'advance +1 m map x')

                # Step 2: rotate 90° CCW in place; the following corners remain map-aligned.
                yaw_rotated = syaw + math.pi / 2.0
                demo.go_to(sx + 1.0, sy, z, yaw_rotated, parsed.leg_duration, 'rotate 90° CCW')

                # Steps 3-7: fly the four corners of the 2x2 square as map-frame
                # offsets from start, ending back at (start+1, start).
                #
                # Corner sequence (x offset, y offset) from start:
                #   (+1, +1) -> (-1, +1) -> (-1, -1) -> (+1, -1) -> (+1, 0)
                corners = [
                    (1.0, 1.0, '+1,+1'),
                    (-1.0, 1.0, '-1,+1'),
                    (-1.0, -1.0, '-1,-1'),
                    (1.0, -1.0, '+1,-1'),
                    (1.0, 0.0, '+1, 0  (1 m map +x from start)'),
                ]
                for dx, dy, label in corners:
                    demo.go_to(sx + dx, sy + dy, z, yaw_rotated, parsed.leg_duration, label)

            except Exception as e:
                demo.get_logger().error(f'[{demo._ns}] error during square: {e}')
                failure = e

        except (KeyboardInterrupt, ExternalShutdownException, ShutdownException,
                ROSInterruptException):
            demo.get_logger().info(f'[{demo._ns}] interrupted')
        except Exception as e:
            failure = e
        finally:
            try:
                demo.land(timeout_sec=parsed.timeout_sec)
            except Exception as e:
                demo.get_logger().error(f'[{demo._ns}] cleanup land failed: {e}')
                cleanup_errors.append(f'land failed: {e}')
            finally:
                try:
                    demo.disarm(timeout_sec=parsed.timeout_sec)
                except Exception as e:
                    demo.get_logger().error(f'[{demo._ns}] cleanup disarm failed: {e}')
                    cleanup_errors.append(f'disarm failed: {e}')
                    if demo.appears_airborne():
                        cleanup_errors.append(demo.warn_manual_recovery_needed())

            try:
                fx, fy, fz, fyaw = demo.current_pose()
                demo.get_logger().info(
                    f'[{demo._ns}] final pose: x={fx:.3f}, y={fy:.3f}, '
                    f'z={fz:.3f}, yaw={math.degrees(fyaw):.1f}°'
                )
            except TimeoutError as e:
                demo.get_logger().warning(f'[{demo._ns}] final pose unavailable: {e}')

            if cleanup_errors and failure is None:
                failure = RuntimeError('; '.join(cleanup_errors))

        if failure is not None:
            print_failure_summary('square_demo', demo._ns, failure, cleanup_errors)
            return 1
        return 0
    finally:
        demo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

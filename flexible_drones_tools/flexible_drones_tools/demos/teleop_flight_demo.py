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

"""Demo: arm, takeoff, enable teleop control, cancel on keypress, land, disarm."""

import argparse
import time
from threading import Event, Thread

import rclpy
from rclpy.action import ActionClient
from rclpy.exceptions import ROSInterruptException
from rclpy.executors import ExternalShutdownException, ShutdownException

from geometry_msgs.msg import TwistStamped
from flexible_drones_msgs.action import EnableController
from flexible_drones_tools.demos.demo_cli import print_failure_summary
from flexible_drones_tools.demos.trajectory_demo import TrajectoryDemo


class TeleopFlightDemo(TrajectoryDemo):
    def __init__(
        self,
        takeoff_height: float,
        land_height: float,
        *,
        frame_id: str = 'base_link',
    ):
        super().__init__(
            'teleop_flight_demo',
            takeoff_height=takeoff_height,
            land_height=land_height,
        )
        self._teleop_client = ActionClient(
            self,
            EnableController,
            self._target_topic('teleop_control'),
        )
        self._cmd_vel_pub = self.create_publisher(TwistStamped, self._target_topic('cmd_vel'), 10)
        self._frame_id = frame_id
        self._teleop_done = Event()
        self._teleop_handle = None
        self._cmd_vel_timer = None

    def start_neutral_cmd_vel(self, rate_hz: float = 20.0) -> None:
        period = 1.0 / rate_hz
        self._cmd_vel_timer = self.create_timer(period, self._publish_neutral_cmd_vel)
        self._publish_neutral_cmd_vel()

    def stop_neutral_cmd_vel(self) -> None:
        if self._cmd_vel_timer is not None:
            self._cmd_vel_timer.cancel()
            self.destroy_timer(self._cmd_vel_timer)
            self._cmd_vel_timer = None
        self._publish_neutral_cmd_vel()

    def _publish_neutral_cmd_vel(self) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        self._cmd_vel_pub.publish(msg)

    def enable_teleop_until_confirmed(self) -> None:
        if not self._teleop_client.wait_for_server(timeout_sec=5.0):
            raise TimeoutError(f'teleop_control action unavailable for /{self._ns}')

        self.get_logger().info(f'[{self._ns}] enabling teleop control...')
        handle = self._spin_until(
            self._teleop_client.send_goal_async(
                EnableController.Goal(controller_name='teleop_control'),
                feedback_callback=self._teleop_feedback_cb,
            ),
            5.0,
            'teleop_control goal',
        )
        if handle is None or not handle.accepted:
            raise RuntimeError(f'teleop_control goal rejected for /{self._ns}')

        self._teleop_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._teleop_result_cb)
        Thread(target=self._wait_for_confirmation, daemon=True).start()

        print('Teleop control is active. Press Enter to cancel and land.', flush=True)
        while rclpy.ok() and not self._teleop_done.is_set():
            rclpy.spin_once(self, timeout_sec=0.1)

    def cancel_teleop(self) -> None:
        if self._teleop_handle is None or self._teleop_done.is_set():
            return
        self.get_logger().info(f'[{self._ns}] canceling teleop control...')
        self._teleop_handle.cancel_goal_async()

    def wait_for_teleop_done(self, timeout_sec: float = 5.0) -> None:
        end_time = time.monotonic() + timeout_sec
        while rclpy.ok() and not self._teleop_done.is_set():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() >= end_time:
                self.get_logger().warning(
                    f'[{self._ns}] timed out waiting for teleop action result'
                )
                return

    def _wait_for_confirmation(self) -> None:
        try:
            input()
        except EOFError:
            pass
        self.cancel_teleop()

    def _teleop_feedback_cb(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'[{self._ns}] teleop_active={feedback.controller_active}, '
            f'cmd_vel age={feedback.input_age_sec:.2f} s'
        )

    def _teleop_result_cb(self, future) -> None:
        result = future.result().result
        if result.return_code == 0:
            self.get_logger().info(f'[{self._ns}] teleop finished: {result.message}')
        else:
            self.get_logger().warning(
                f'[{self._ns}] teleop finished with return_code={result.return_code}: '
                f'{result.message}'
            )
        self._teleop_done.set()


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Demo: arm/takeoff/enable teleop/cancel on Enter/land/disarm.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set the node namespace with '-r __ns:=/drone1'.
  If no namespace is set, remapping the node name can target '/<node_name>'.

Examples:
  ros2 run flexible_drones_tools teleop_flight_demo --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools teleop_flight_demo --ros-args -r __node:=drone1
  ros2 run flexible_drones_tools teleop_flight_demo --takeoff-height 0.75 --ros-args -r __ns:=/drone2
""",
    )
    parser.add_argument('--takeoff-height', type=float, default=1.0,
                        help='Takeoff altitude in metres, 0.25 to 5.0 (default: 1.0)')
    parser.add_argument('--land-height', type=float, default=0.02,
                        help='Landing target height in metres, 0.0 to 1.0 (default: 0.02)')
    parser.add_argument('--frame-id', default='base_link',
                        help='Frame id used for neutral cmd_vel heartbeat (default: base_link)')
    parser.add_argument('--timeout-sec', type=float, default=10.0,
                        help='Timeout for arm, takeoff, land, and disarm in seconds '
                             '(default: 10.0)')
    parsed, ros_args = parser.parse_known_args(args=args)
    if not 0.25 <= parsed.takeoff_height <= 5.0:
        parser.error('--takeoff-height must be between 0.25 and 5.0')
    if not 0.0 <= parsed.land_height <= 1.0:
        parser.error('--land-height must be between 0.0 and 1.0')
    if parsed.timeout_sec < 2.0:
        parser.error('--timeout-sec must be at least 2.0')

    rclpy.init(args=ros_args)
    demo = TeleopFlightDemo(
        parsed.takeoff_height,
        parsed.land_height,
        frame_id=parsed.frame_id,
    )

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
            demo.start_neutral_cmd_vel()
            demo.enable_teleop_until_confirmed()
        except (KeyboardInterrupt, ExternalShutdownException, ShutdownException,
                ROSInterruptException):
            demo.get_logger().info(f'[{demo._ns}] interrupted')
            demo.cancel_teleop()
        except Exception as e:
            failure = e
        finally:
            demo.cancel_teleop()
            demo.wait_for_teleop_done()
            demo.stop_neutral_cmd_vel()
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
            print_failure_summary('teleop_flight_demo', demo._ns, failure, cleanup_errors)
            return 1
        return 0
    finally:
        demo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

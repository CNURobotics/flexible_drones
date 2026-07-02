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

"""Trajectory demo: upload, arm, takeoff, go_to trajectory start, execute, land, disarm."""

import argparse
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.exceptions import ROSInterruptException
from rclpy.executors import ExternalShutdownException, ShutdownException
from rclpy.node import Node
from rclpy.task import Future

from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Odometry, Path as NavPath
from flexible_drones_msgs.action import ExecuteTrajectory, GoTo, Land, Takeoff, UploadTrajectory
from flexible_drones_msgs.msg import Trajectory
from flexible_drones_msgs.srv import Arm
from flexible_drones_tools.demos.demo_cli import print_failure_summary
from flexible_drones_tools.trajectories.utilities.model import PolyOrderTrajectory
from flexible_drones_tools.trajectories.utilities.trajectory_path import resolve_deployment_trajectory_path
from flexible_drones_tools.trajectories.visualize_trajectory import (
    DEFAULT_PATH_TOPIC,
    trajectory_to_path_msg,
    visualization_offset_for_trajectory,
)


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _resolve_target_namespace(node: Node, default_node_name: str) -> str:
    """Return the drone namespace targeted by this demo node."""
    namespace = node.get_namespace().strip('/')
    if namespace:
        return namespace

    node_name = node.get_name().strip('/')
    if node_name and node_name != default_node_name:
        return node_name
    return ''


def _return_code_label(return_code: int) -> str:
    labels = {
        0: 'success',
        1: 'high-level command failure',
        4: 'cancelled',
        6: 'timeout',
        7: 'upload failure',
        8: 'start failure',
        9: 'invalid goal',
    }
    return labels.get(int(return_code), 'unknown')


def _format_return_code(return_code: int) -> str:
    return f'rc={return_code} ({_return_code_label(return_code)})'


class TrajectoryDemo(Node):
    def __init__(
        self,
        node_name: str = 'trajectory_demo',
        *,
        takeoff_height: float = 1.0,
        land_height: float = 0.02,
    ):
        super().__init__(node_name)
        self._ns = _resolve_target_namespace(self, node_name)
        self._takeoff_height = float(takeoff_height)
        self._land_height = float(land_height)
        self._pose: Pose | None = None
        self._target_prefix = f'/{self._ns}' if self._ns else ''
        self._planned_path_pub = None
        self._planned_path_timer = None
        self._planned_path_msg = None
        self._arm_client = self.create_client(Arm, self._target_topic('arm'))
        self._takeoff_client = ActionClient(self, Takeoff, self._target_topic('takeoff'))
        self._land_client = ActionClient(self, Land, self._target_topic('land'))
        self._go_to_client = ActionClient(self, GoTo, self._target_topic('go_to'))
        self._upload_client = ActionClient(
            self, UploadTrajectory, self._target_topic('upload_trajectory')
        )
        self._execute_client = ActionClient(
            self, ExecuteTrajectory, self._target_topic('execute_trajectory')
        )

        self.create_subscription(Odometry, self._target_topic('odom'), self._odom_cb, 10)

    def _target_topic(self, name: str) -> str:
        return f'{self._target_prefix}/{name}' if self._target_prefix else name

    def _target_label(self, name: str | None = None) -> str:
        target = self._target_prefix or '/'
        return f'{target}/{name}' if name and self._target_prefix else (
            f'/{name}' if name else target
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self._pose = msg.pose.pose

    def _spin_until(self, future: Future, timeout_sec: float, what: str):
        end_time = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() >= end_time:
                raise TimeoutError(f'Timed out waiting for {what}')
        return future.result()

    def _spin_sleep(self, duration_sec: float) -> None:
        end_time = time.monotonic() + duration_sec
        while rclpy.ok() and time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=min(0.02, duration_sec))

    def publish_planned_trajectory(
        self,
        trajectory: PolyOrderTrajectory,
        *,
        frame_id: str = 'map',
        samples: int = 100,
        topic: str = DEFAULT_PATH_TOPIC,
        period_sec: float = 2.0,
        offset=None,
        auto_ground_offset: bool = True,
    ) -> None:
        """Publish the planned trajectory as a repeated RViz Path."""
        path_offset = visualization_offset_for_trajectory(
            trajectory,
            offset=offset,
            auto_ground_offset=auto_ground_offset,
        )
        self._planned_path_msg = trajectory_to_path_msg(
            trajectory,
            frame_id=frame_id,
            samples=samples,
            offset=path_offset,
            stamp=self.get_clock().now().to_msg(),
        )
        self._planned_path_pub = self.create_publisher(NavPath, topic, 1)

        def publish_path():
            self._planned_path_msg.header.stamp = self.get_clock().now().to_msg()
            self._planned_path_pub.publish(self._planned_path_msg)

        publish_path()
        self._planned_path_timer = self.create_timer(float(period_sec), publish_path)
        self.get_logger().info(
            f'[{self._ns}] publishing planned trajectory on {topic} '
            f'with {len(self._planned_path_msg.poses)} poses'
        )

    def wait_for_pose(self, timeout_sec: float = 10.0) -> Pose:
        end_time = time.monotonic() + timeout_sec
        while rclpy.ok() and self._pose is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() >= end_time:
                raise TimeoutError(
                    f'Timed out waiting for odometry on {self._target_label("odom")}'
                )
        assert self._pose is not None
        return self._pose

    def current_pose(self) -> tuple[float, float, float, float]:
        """Return (x, y, z, yaw) of latest known pose."""
        pose = self.wait_for_pose()
        yaw = _yaw_from_quat(
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        )
        return pose.position.x, pose.position.y, pose.position.z, yaw

    def appears_airborne(self, landed_height: float = 0.1) -> bool:
        """Return whether the latest pose is above the local landed threshold."""
        try:
            _, _, z, _ = self.current_pose()
        except TimeoutError:
            return False
        return z > max(float(landed_height), self._land_height)

    def warn_manual_recovery_needed(self) -> str:
        message = (
            'Drone still appears airborne and disarm did not complete. '
            'Do not assume cleanup made the drone safe. Try landing again manually, '
            'or use emergency stop if the situation requires it.'
        )
        self.get_logger().error(f'[{self._ns}] {message}')
        print(f'WARNING: {message}', file=sys.stderr, flush=True)
        return message

    def arm(self, timeout_sec: float = 10.0) -> None:
        timeout_sec = float(timeout_sec)
        self.get_logger().info(f'[{self._ns}] sending arm request...')
        if not self._arm_client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError(f'Arm service unavailable for /{self._ns}')
        req = Arm.Request()
        req.arm = True
        req.timeout_sec = timeout_sec
        result = self._spin_until(
            self._arm_client.call_async(req), timeout_sec, 'arm response'
        )
        if result is None or not result.success:
            raise RuntimeError(f'Arm failed for /{self._ns}')
        self.get_logger().info(f'[{self._ns}] armed')

    def disarm(self, timeout_sec: float = 10.0) -> None:
        timeout_sec = float(timeout_sec)
        self.get_logger().info(f'[{self._ns}] sending disarm request...')
        if not self._arm_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(
                f'Arm service unavailable for /{self._ns}; cannot disarm'
            )
            return
        req = Arm.Request()
        req.arm = False
        req.timeout_sec = timeout_sec
        result = self._spin_until(
            self._arm_client.call_async(req), timeout_sec, 'disarm response'
        )
        if result is None or not result.success:
            raise RuntimeError(f'Disarm failed for /{self._ns}')
        self.get_logger().info(f'[{self._ns}] disarmed')

    def takeoff(self, timeout_sec: float = 10.0) -> None:
        timeout_sec = float(timeout_sec)
        self.get_logger().info(
            f'[{self._ns}] sending takeoff goal to {self._takeoff_height:.2f} m...'
        )
        if not self._takeoff_client.wait_for_server(timeout_sec=timeout_sec):
            raise TimeoutError(f'Takeoff action unavailable for /{self._ns}')
        goal = Takeoff.Goal()
        goal.group_mask = 0
        goal.height = self._takeoff_height
        goal.duration = Duration(seconds=timeout_sec).to_msg()
        handle = self._spin_until(
            self._takeoff_client.send_goal_async(goal), timeout_sec, 'takeoff goal'
        )
        if handle is None or not handle.accepted:
            raise RuntimeError(f'Takeoff goal rejected for /{self._ns}')
        result = self._spin_until(
            handle.get_result_async(), timeout_sec, 'takeoff result'
        )
        if result.result.return_code != 0:
            raise RuntimeError(
                f'Takeoff failed for {self._target_label()}: '
                f'{_format_return_code(result.result.return_code)}'
            )
        self.get_logger().info(f'[{self._ns}] takeoff complete at {self._takeoff_height:.2f} m')

    def land(self, timeout_sec: float = 10.0) -> None:
        timeout_sec = float(timeout_sec)
        self.get_logger().info(
            f'[{self._ns}] sending land goal to {self._land_height:.2f} m...'
        )
        if not self._land_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error(f'Land action unavailable for /{self._ns}')
            return
        goal = Land.Goal()
        goal.group_mask = 0
        goal.height = self._land_height
        goal.duration = Duration(seconds=timeout_sec).to_msg()
        handle = self._spin_until(
            self._land_client.send_goal_async(goal), timeout_sec, 'land goal'
        )
        if handle is None or not handle.accepted:
            self.get_logger().error(f'Land goal rejected for /{self._ns}')
            return
        result = self._spin_until(
            handle.get_result_async(), timeout_sec, 'land result'
        )
        if result.result.return_code != 0:
            self.get_logger().error(
                f'Land failed for {self._target_label()}: '
                f'{_format_return_code(result.result.return_code)}'
            )
        self.get_logger().info(f'[{self._ns}] landed')

    def go_to(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        duration_sec: float,
        label: str,
        relative: bool = False,
    ) -> None:
        """Send a GoTo goal and block until it completes."""
        self.get_logger().info(
            f'[{self._ns}] sending go_to {label}: '
            f'x={x:.3f}, y={y:.3f}, z={z:.3f}, '
            f'yaw={math.degrees(yaw):.1f}°...'
        )
        if not self._go_to_client.wait_for_server(timeout_sec=5.0):
            raise TimeoutError(f'GoTo action unavailable for /{self._ns}')

        frame = (
            GoTo.Goal.FRAME_RELATIVE_MAP
            if relative
            else GoTo.Goal.FRAME_ABSOLUTE
        )
        if z < 0.25 and not relative:
            raise RuntimeError(f'GoTo failed for /{self._ns} at {label}: z={z:.2f} < 0.25')

        goal = GoTo.Goal()
        goal.group_mask = 0
        goal.frame = frame
        goal.goal = Point(x=x, y=y, z=z)
        goal.yaw = yaw
        goal.duration = Duration(seconds=duration_sec).to_msg()
        handle = self._spin_until(
            self._go_to_client.send_goal_async(goal), 5.0, f'go_to goal [{label}]'
        )
        if handle is None or not handle.accepted:
            raise RuntimeError(f'GoTo goal rejected for /{self._ns} at {label}')
        result = self._spin_until(
            handle.get_result_async(), duration_sec + 10.0, f'go_to result [{label}]'
        )
        if result.result.return_code != 0:
            raise RuntimeError(
                f'GoTo failed for {self._target_label()} at {label}: '
                f'{_format_return_code(result.result.return_code)}'
            )
        self.get_logger().info(f'[{self._ns}] reached {label}')

    def upload_trajectory(self, pieces, trajectory_id: int = 0) -> None:
        self.get_logger().info(
            f'[{self._ns}] sending upload for {len(pieces)} trajectory pieces...'
        )
        if not self._upload_client.wait_for_server(timeout_sec=5.0):
            raise TimeoutError(f'UploadTrajectory action unavailable for /{self._ns}')
        goal = UploadTrajectory.Goal()
        goal.trajectory = Trajectory(
            trajectory_id=trajectory_id, pieces=pieces
        )
        handle = self._spin_until(
            self._upload_client.send_goal_async(goal), 10.0, 'upload goal'
        )
        if handle is None or not handle.accepted:
            raise RuntimeError(f'UploadTrajectory goal rejected for /{self._ns}')
        result = self._spin_until(handle.get_result_async(), 30.0, 'upload result')
        if result.result.return_code != 0:
            raise RuntimeError(
                f'UploadTrajectory failed for {self._target_label()}: '
                f'{_format_return_code(result.result.return_code)}'
            )
        self.get_logger().info(
            f'[{self._ns}] trajectory uploaded ({_format_return_code(result.result.return_code)})'
        )

    def execute_trajectory(
        self,
        trajectory_id: int = 0,
        timescale: float = 1.0,
        relative: bool = True,
        reversed_: bool = False,
        timeout_sec: float = 120.0,
    ) -> None:
        self.get_logger().info(
            f'[{self._ns}] sending execute trajectory goal '
            f'(relative={relative}, reversed={reversed_}, timescale={timescale:.2f})...'
        )
        if not self._execute_client.wait_for_server(timeout_sec=5.0):
            raise TimeoutError(f'ExecuteTrajectory action unavailable for /{self._ns}')
        goal = ExecuteTrajectory.Goal()
        goal.group_mask = 0
        goal.trajectory_id = trajectory_id
        goal.timescale = timescale
        goal.reversed = reversed_
        goal.relative = relative
        handle = self._spin_until(
            self._execute_client.send_goal_async(goal), 5.0, 'execute goal'
        )
        if handle is None or not handle.accepted:
            raise RuntimeError(f'ExecuteTrajectory goal rejected for /{self._ns}')
        result = self._spin_until(
            handle.get_result_async(), timeout_sec, 'execute result'
        )
        if result.result.return_code != 0:
            raise RuntimeError(
                f'ExecuteTrajectory failed for {self._target_label()}: '
                f'{_format_return_code(result.result.return_code)}'
            )
        self.get_logger().info(f'[{self._ns}] trajectory execution complete')


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Trajectory demo: upload/arm/takeoff/go_to-start/execute/land/disarm.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set the node namespace with '-r __ns:=/drone1'.
  If no namespace is set, remapping the node name can target '/<node_name>'.

Examples:
  export FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE=<deployment_package>
  ros2 run flexible_drones_tools trajectory_demo figure8.csv --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools trajectory_demo figure8.csv --ros-args -r __node:=drone1
  ros2 run flexible_drones_tools trajectory_demo figure8.csv --go-to-duration 5.0 --timescale 0.75 --ros-args -r __ns:=/drone2
  ros2 run flexible_drones_tools trajectory_demo figure8.csv --execute-in-reverse --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools trajectory_demo ./my_traj.csv --ros-args -r __ns:=/drone1
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
    parser.add_argument('--go-to-duration', type=float, default=5.0,
                        help='Seconds for the go_to-start leg, 5.0 to 60.0 (default: 5.0)')
    parser.add_argument('--timescale', type=float, default=1.0,
                        help='Trajectory playback timescale, 0.25 to 10.0; '
                             'below 1.0 is faster, above 1.0 is slower (default: 1.0)')
    parser.add_argument('--relative', dest='relative', action='store_true', default=None,
                        help='Execute trajectory relative to current pose')
    parser.add_argument('--no-relative', dest='relative', action='store_false',
                        help='Execute trajectory in absolute coordinates')
    parser.add_argument('--execute-in-reverse', action='store_true',
                        help='After the forward pass, execute the uploaded trajectory in reverse')
    parser.add_argument('--timeout-sec', type=float, default=10.0,
                        help='Timeout for arm, takeoff, land, and disarm in seconds '
                             '(default: 10.0)')
    parser.add_argument('--no-visualize-trajectory', action='store_true',
                        help='Do not publish the planned trajectory as nav_msgs/Path')
    parser.add_argument('--visualization-samples', type=int, default=100,
                        help='Number of poses for planned trajectory visualization (default: 100)')
    parser.add_argument('--visualization-period-sec', type=float, default=2.0,
                        help='Seconds between planned trajectory Path publishes (default: 2.0)')
    parser.add_argument('--visualization-topic', default=DEFAULT_PATH_TOPIC,
                        help=(
                            'Topic for planned trajectory visualization '
                            f"(default: '{DEFAULT_PATH_TOPIC}')"
                        ))
    parsed, ros_args = parser.parse_known_args(args=args)
    if not 5.0 <= parsed.go_to_duration <= 60.0:
        parser.error('--go-to-duration must be between 5.0 and 60.0')
    if not 0.25 <= parsed.timescale <= 10.0:
        parser.error('--timescale must be between 0.25 and 10.0')
    if parsed.timeout_sec < 2.0:
        parser.error('--timeout-sec must be at least 2.0')
    if parsed.visualization_samples < 2:
        parser.error('--visualization-samples must be at least 2')
    if parsed.visualization_period_sec <= 0.0:
        parser.error('--visualization-period-sec must be positive')

    # Resolve and load trajectory before ROS init so errors surface immediately.
    from flexible_drones_tools.trajectories.utilities.generate_drone_format import GenerateDroneFormat
    from flexible_drones_tools.trajectories.utilities.read_from_csv import load_trajectory_data

    trajectory_path = resolve_deployment_trajectory_path(parsed.trajectory_file)
    print(f'Loading trajectory from: {trajectory_path}', flush=True)

    # This loads drone format and converts to polyval ordering (descending powers)
    durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs = load_trajectory_data(trajectory_path)

    # Converts to pieces for uploading to drone (ascending powers)
    pieces = GenerateDroneFormat(
        durations,
        x_coeffs,
        y_coeffs,
        z_coeffs,
        yaw_coeffs,
    ).to_ros_pieces()
    poly_trajectory = PolyOrderTrajectory(durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs)
    total_duration = float(sum(durations))

    # Trajectory start position: evaluate each axis at t=0 (constant coefficient).
    traj_start_x = float(x_coeffs[0][-1])  # polyval ordering
    traj_start_y = float(y_coeffs[0][-1])
    traj_start_z = float(z_coeffs[0][-1])
    traj_start_yaw = float(yaw_coeffs[0][-1])

    if parsed.relative is None:
        if abs(traj_start_z) < 0.01:
            relative = True
        elif traj_start_z > 0.5:
            relative = False
        else:
            parser.error(
                f'trajectory starts at z={traj_start_z:.3f}; specify --relative or --no-relative'
            )
    else:
        relative = parsed.relative
    print(
        f'Trajectory start: x={traj_start_x:.3f}, y={traj_start_y:.3f}, '
        f'z={traj_start_z:.3f}, yaw={math.degrees(traj_start_yaw):.1f}°; '
        f"execution mode={'relative' if relative else 'absolute'}",
        flush=True,
    )

    rclpy.init(args=ros_args)
    demo = TrajectoryDemo()
    if not parsed.no_visualize_trajectory:
        demo.publish_planned_trajectory(
            poly_trajectory,
            samples=parsed.visualization_samples,
            topic=parsed.visualization_topic,
            period_sec=parsed.visualization_period_sec,
        )

    try:
        failure = None
        cleanup_errors = []
        try:
            print('Waiting for initial pose...', flush=True)
            demo.wait_for_pose()

            demo.upload_trajectory(pieces)
            demo.arm(timeout_sec=parsed.timeout_sec)
            demo.takeoff(timeout_sec=parsed.timeout_sec)
            demo._spin_sleep(1.0)

            try:
                demo.go_to(
                    traj_start_x, traj_start_y, traj_start_z, traj_start_yaw,
                    parsed.go_to_duration, 'trajectory start',
                    relative
                )
                demo._spin_sleep(0.5)
                demo.execute_trajectory(
                    relative=relative,
                    timescale=parsed.timescale,
                    timeout_sec=total_duration / parsed.timescale + 30.0,
                )
                if parsed.execute_in_reverse:
                    demo._spin_sleep(0.5)
                    demo.execute_trajectory(
                        relative=relative,
                        timescale=parsed.timescale,
                        reversed_=True,
                        timeout_sec=total_duration / parsed.timescale + 30.0,
                    )
            except Exception as e:
                demo.get_logger().error(f'[{demo._ns}] error during trajectory: {e}')
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
            print_failure_summary('trajectory_demo', demo._ns, failure, cleanup_errors)
            return 1
        return 0
    finally:
        demo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

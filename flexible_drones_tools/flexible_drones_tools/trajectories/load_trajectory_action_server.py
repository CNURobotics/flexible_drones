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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from flexible_drones_msgs.action import GetSampledTrajectory, GetTrajectory
from flexible_drones_msgs.msg import SampledTrajectory, Trajectory, TrajectoryPolynomialPiece
from flexible_drones_tools.ros_shutdown import spin_node
from flexible_drones_tools.trajectories.utilities.conversion import (
    coefficients_from_pieces,
    sampled_trajectory_from_pieces,
    trajectory_from_pieces,
)
from flexible_drones_tools.trajectories.utilities.io import (
    default_workspace_roots,
    load_trajectory_pieces,
    resolve_existing_file,
    resolve_package_file,
)
from flexible_drones_tools.trajectories.utilities.model import PolyOrderTrajectory


@dataclass(frozen=True)
class LoadedTrajectory:
    name: str
    package: str
    folder: str
    trajectory_file: str
    path: Path
    pieces: tuple[TrajectoryPolynomialPiece, ...]


class LoadTrajectoryActionServer(Node):
    MIN_SAMPLE_DT = 0.01
    MAX_SAMPLE_DT = 1.0

    def __init__(self):
        super().__init__('load_trajectory_action_server')
        self.declare_parameter('config', '')
        self.declare_parameter('plot_trajectory', False)
        self.declare_parameter('plot_samples_per_segment', 100)

        self._trajectories = {}
        self._load_config()

        self._sampled_action_server = ActionServer(
            self,
            GetSampledTrajectory,
            'load_sampled_trajectory',
            self.execute_sampled_trajectory_callback,
        )
        self._trajectory_action_server = ActionServer(
            self,
            GetTrajectory,
            'load_trajectory',
            self.execute_trajectory_callback,
        )
        self.get_logger().info(f'Loaded {len(self._trajectories)} named trajectories; waiting for requests.')

    @staticmethod
    def _default_trajectory_name(trajectory_id):
        return f'Trajectory {int(trajectory_id)}'

    def _load_config(self):
        config_value = self.get_parameter('config').get_parameter_value().string_value.strip()
        if not config_value:
            self.get_logger().warning('No trajectory config provided. Set config:=/path/to/trajectories.yaml.')
            return

        config_path = self._resolve_config_path(config_value)
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError('PyYAML is required to load trajectory configs.') from exc

        with config_path.open('r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle) or {}

        entries = config.get('trajectories', config) if isinstance(config, dict) else config
        if not isinstance(entries, list):
            raise ValueError("Trajectory config must be a list or a mapping with a 'trajectories' list.")

        for index, entry in enumerate(entries):
            record = self._load_entry(entry, index)
            if record.name in self._trajectories:
                raise ValueError(f"Duplicate trajectory name '{record.name}' in {config_path}.")
            self._trajectories[record.name] = record
            self.get_logger().info(
                f"Loaded trajectory '{record.name}' from '{record.path}' " f'with {len(record.pieces)} pieces.'
            )

    def _resolve_config_path(self, config_value):
        return resolve_existing_file(config_value, extra_roots=default_workspace_roots())

    def _load_entry(self, entry, index):
        if not isinstance(entry, dict):
            raise ValueError(f'Trajectory config entry {index} must be a mapping.')

        required_fields = ('name', 'package', 'folder', 'trajectory_file')
        missing_fields = [field for field in required_fields if not str(entry.get(field, '')).strip()]
        if missing_fields:
            raise ValueError(f"Trajectory config entry {index} is missing: {', '.join(missing_fields)}")

        name = str(entry['name']).strip()
        package = str(entry['package']).strip()
        folder = str(entry['folder']).strip()
        trajectory_file = str(entry['trajectory_file']).strip()
        trajectory_path = self._resolve_trajectory_path(package, folder, trajectory_file)
        pieces = tuple(load_trajectory_pieces(trajectory_path))
        if not pieces:
            raise ValueError(f"Trajectory '{name}' from '{trajectory_path}' has no pieces.")

        return LoadedTrajectory(
            name=name,
            package=package,
            folder=folder,
            trajectory_file=trajectory_file,
            path=trajectory_path,
            pieces=pieces,
        )

    def _resolve_trajectory_path(self, package, folder, trajectory_file):
        return resolve_package_file(
            package,
            folder,
            trajectory_file,
            extra_roots=default_workspace_roots(),
        )

    def _lookup_record(self, goal):
        name = goal.name.strip() if goal.name and goal.name.strip() else self._default_trajectory_name(goal.trajectory_id)
        record = self._trajectories.get(name)
        if record is None:
            known_names = ', '.join(sorted(self._trajectories)) or '<none>'
            raise KeyError(f"Trajectory '{name}' not found. Known trajectories: {known_names}")
        return name, record

    def _empty_trajectory_result(self, message):
        result = GetTrajectory.Result()
        result.success = False
        result.message = message
        result.trajectory = Trajectory()
        result.trajectory.trajectory_id = 255
        result.trajectory.pieces = []
        return result

    def _empty_sampled_result(self, trajectory_id, name, message):
        result = GetSampledTrajectory.Result()
        result.success = False
        result.message = message
        result.trajectory = SampledTrajectory()
        result.trajectory.trajectory_id = int(trajectory_id)
        result.trajectory.name = name
        result.trajectory.points = []
        return result

    @staticmethod
    def _publish_loaded_feedback(goal_handle, feedback_type):
        feedback = feedback_type()
        feedback.remaining_sec = 0.0
        feedback.status = 'Loaded trajectory ready.'
        goal_handle.publish_feedback(feedback)

    async def execute_trajectory_callback(self, goal_handle):
        goal = goal_handle.request
        try:
            name, record = self._lookup_record(goal)
        except KeyError as exc:
            message = str(exc).strip("'")
            self.get_logger().warning(message)
            goal_handle.abort()
            return self._empty_trajectory_result(message)

        trajectory = trajectory_from_pieces(goal.trajectory_id, record.pieces)
        message = f"Loaded trajectory '{name}' from '{record.path}'."
        self._maybe_plot_record(record)
        self._publish_loaded_feedback(goal_handle, GetTrajectory.Feedback)
        goal_handle.succeed()

        result = GetTrajectory.Result()
        result.success = True
        result.message = message
        result.trajectory = trajectory
        self.get_logger().info(f"Returning GetTrajectory '{name}' with {len(trajectory.pieces)} pieces.")
        return result

    async def execute_sampled_trajectory_callback(self, goal_handle):
        goal = goal_handle.request
        name = goal.name.strip() if goal.name and goal.name.strip() else self._default_trajectory_name(goal.trajectory_id)
        if not self.MIN_SAMPLE_DT <= float(goal.sample_dt) <= self.MAX_SAMPLE_DT:
            message = f'sample_dt must be between {self.MIN_SAMPLE_DT} and {self.MAX_SAMPLE_DT} seconds.'
            self.get_logger().warning(message)
            goal_handle.abort()
            return self._empty_sampled_result(goal.trajectory_id, name, message)

        try:
            name, record = self._lookup_record(goal)
        except KeyError as exc:
            message = str(exc).strip("'")
            self.get_logger().warning(message)
            goal_handle.abort()
            return self._empty_sampled_result(goal.trajectory_id, name, message)

        trajectory = sampled_trajectory_from_pieces(
            goal.trajectory_id,
            name,
            record.pieces,
            goal.sample_dt,
        )
        message = f"Loaded sampled trajectory '{name}' from '{record.path}'."
        self._maybe_plot_record(record)
        self._publish_loaded_feedback(goal_handle, GetSampledTrajectory.Feedback)
        goal_handle.succeed()

        result = GetSampledTrajectory.Result()
        result.success = True
        result.message = message
        result.trajectory = trajectory
        self.get_logger().info(f"Returning GetSampledTrajectory '{name}' with {len(trajectory.points)} points.")
        return result

    def _maybe_plot_record(self, record):
        if not self.get_parameter('plot_trajectory').get_parameter_value().bool_value:
            self.get_logger().debug('Skipping trajectory plot; set plot_trajectory:=true to enable plotting.')
            return

        try:
            import matplotlib.pyplot as plt

            samples_per_segment = self.get_parameter('plot_samples_per_segment').get_parameter_value().integer_value
            durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs = coefficients_from_pieces(
                record.pieces,
                coefficient_order='numpy',
            )
            x_coeffs = [np.asarray(coeffs, dtype=np.float64) for coeffs in x_coeffs]
            y_coeffs = [np.asarray(coeffs, dtype=np.float64) for coeffs in y_coeffs]
            z_coeffs = [np.asarray(coeffs, dtype=np.float64) for coeffs in z_coeffs]
            yaw_coeffs = [np.asarray(coeffs, dtype=np.float64) for coeffs in yaw_coeffs]

            trajectory = PolyOrderTrajectory(durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs)
            trajectory.plot_trajectory(samples_per_segment=max(2, int(samples_per_segment)))
            plt.show(block=False)
            plt.pause(0.001)
        except Exception as exc:
            self.get_logger().warning(f"Failed to plot trajectory '{record.name}': {exc}")


def main(args=None):
    rclpy.init(args=args)
    server = LoadTrajectoryActionServer()
    try:
        spin_node(server)
    finally:
        server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

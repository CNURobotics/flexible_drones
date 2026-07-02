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

import numpy as np
from flexible_drones_tools.trajectories.utilities.conversion import pieces_from_coefficients
from flexible_drones_tools.trajectories.utilities.io import save_trajectory_csv


class GenerateDroneFormat():
    def __init__(self, durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs, write=False, path=None):
        self.durations = durations
        self.end_times = [0.0] + list(np.cumsum(durations))
        self.x_coeffs = x_coeffs
        self.y_coeffs = y_coeffs
        self.z_coeffs = z_coeffs
        self.yaw_coeffs = yaw_coeffs
        # For csv
        self.write = write
        self.path = path

    """Turn the coefficient values into drone formatted messages."""

    def to_ros_pieces(self):
        return pieces_from_coefficients(
            self.durations,
            self.x_coeffs,
            self.y_coeffs,
            self.z_coeffs,
            self.yaw_coeffs,
            coefficient_order='numpy',
        )

    """Create and call an object to write the trajectory to a csv."""

    def write_to_csv(self):
        if not self.write:
            return
        else:
            save_trajectory_csv(
                self.path,
                self.durations,
                self.x_coeffs,
                self.y_coeffs,
                self.z_coeffs,
                self.yaw_coeffs,
                coefficient_order='numpy',
            )

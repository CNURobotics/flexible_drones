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

import argparse
from pathlib import Path

from flexible_drones_tools.trajectories.utilities.io import load_trajectory_csv


def load_trajectory_data(filename):
    return load_trajectory_csv(filename, coefficient_order='numpy')


def main():
    import matplotlib.pyplot as plt

    from flexible_drones_tools.trajectories.utilities.model import PolyOrderTrajectory

    parser = argparse.ArgumentParser(description='Plot trajectory from polynomial coefficients.')
    parser.add_argument('filename', type=str, help='Path to the CSV file with trajectory data.')
    args = parser.parse_args()

    trajectory_path = Path(args.filename).expanduser()
    if not trajectory_path.exists():
        print(f"Error: File '{trajectory_path}' not found.")
        return

    durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs = load_trajectory_data(trajectory_path)

    traj = PolyOrderTrajectory(durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs)
    traj.plot_trajectory(samples_per_segment=200)
    plt.show()


if __name__ == '__main__':
    main()

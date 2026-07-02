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

"""
Load a drone-order trajectory CSV and plot it with matplotlib.

Produces a 3D path, the xy/xz/yz projections, and position/velocity/acceleration/
jerk/snap versus time for x/y/z/yaw. The trajectory argument follows the same
lookup rules as ``trajectory_demo`` / ``visualize_trajectory``: a bare filename
is resolved in ``$FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE/trajectories/``; relative
or absolute paths are used directly.
"""

import argparse
import sys

from .utilities.io import load_trajectory_csv
from .utilities.model import PolyOrderTrajectory
from .utilities.plotting import TrajectoryPlotter
from .utilities.trajectory_path import resolve_deployment_trajectory_path


def load_drone_order_trajectory(file_path):
    """Load a drone-order CSV into a NumPy-evaluable ``PolyOrderTrajectory``."""
    data = load_trajectory_csv(file_path, coefficient_order='numpy')
    return PolyOrderTrajectory(*data)


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Load a drone-order trajectory CSV and plot it (3D, projections, x/y/z/yaw kinematics through snap).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  export FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE=<deployment_package>
  ros2 run flexible_drones_tools plot_trajectory figure8.csv
  ros2 run flexible_drones_tools plot_trajectory cnu_sail0.csv --samples 800
  ros2 run flexible_drones_tools plot_trajectory ./my_traj.csv
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
    parser.add_argument('--samples', type=int, default=400,
                        help='Number of samples along the trajectory (default: 400)')
    parsed, _ros_args = parser.parse_known_args(args=args)

    if parsed.samples < 2:
        parser.error('--samples must be at least 2')

    try:
        trajectory_path = resolve_deployment_trajectory_path(parsed.trajectory_file)
        print(f"Loading trajectory from '{trajectory_path}' ...", flush=True)
        trajectory = load_drone_order_trajectory(trajectory_path)
    except Exception as exc:
        print(f'Failed to load trajectory: {exc}', file=sys.stderr, flush=True)
        return 1

    import matplotlib.pyplot as plt

    TrajectoryPlotter.plot_trajectory(
        trajectory, samples=parsed.samples, title=str(trajectory_path))
    plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(main())

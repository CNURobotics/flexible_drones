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


class GenerateTrajectory():
    """Base class for trajectory generators."""

    def generate_trajectory(self, num_points, wz):
        pass

    def generate_to_file(self, path, num_points=None, num_segments=20):
        """
        Generate the shape, fit polynomial segments, and write a drone-order CSV.

        Subclasses only implement :meth:`generate_trajectory`; this runs the shared
        sample -> polyfit -> save pipeline and returns the output ``path``. The CSV
        is written in drone coefficient order via the shared I/O helpers.
        """
        from flexible_drones_tools.trajectories.utilities.split_trajectory import SegmentFit

        sample_args = () if num_points is None else (num_points,)
        t, x, y, z, yaw = self.generate_trajectory(*sample_args)
        fit = SegmentFit(t, x, y, z, yaw)
        segments, durations = fit.split_trajectory(num_segments)
        drone_format = fit.polyfit_traj_for_translating(durations, segments, write=True, path=str(path))
        drone_format.write_to_csv()
        return path

    # ----static plotting methods (delegate to the shared TrajectoryPlotter)----

    @staticmethod
    def plot_points_3d(x, y, z):
        from flexible_drones_tools.trajectories.utilities.plotting import TrajectoryPlotter
        TrajectoryPlotter.plot_path_3d(x, y, z, title='X vs Y vs Z; Raw Values')

    @staticmethod
    def plot_points_xy(x, y):
        from flexible_drones_tools.trajectories.utilities.plotting import TrajectoryPlotter
        TrajectoryPlotter.line_plot(x, y, xlabel='X', ylabel='Y', title='X vs Y; Raw Values', color='r')

    @staticmethod
    def plot_points_xt(x, t):
        from flexible_drones_tools.trajectories.utilities.plotting import TrajectoryPlotter
        TrajectoryPlotter.line_plot(t, x, xlabel='Time', ylabel='X', title='X vs Time; Raw Values', color='b')

    @staticmethod
    def plot_points_yt(y, t):
        from flexible_drones_tools.trajectories.utilities.plotting import TrajectoryPlotter
        TrajectoryPlotter.line_plot(t, y, xlabel='Time', ylabel='Y', title='Y vs Time; Raw Values', color='g')

    """Turn trajectory 180 degrees so a drone can run the opposite direction."""

    def turn_trajectory_180(self, x, y):
        xr = -x
        yr = -y
        return xr, yr

    """Automatically run all static plots for easy checking."""

    def run_all_plots(self, t=None, x=None, y=None, z=None, yaw=None):
        if not all(n is not None for n in [t, x, y, z, yaw]):
            print('Missing one or more parameters in run_all_plots!')
            print(f'T: {t}, X: {x}, Y: {y}, Z: {z}, Yaw: {yaw}')
            return
        self.plot_points_3d(x, y, z)
        self.plot_points_xy(x, y)
        self.plot_points_xt(x, t)
        self.plot_points_yt(y, t)

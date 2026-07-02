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

from flexible_drones_tools.trajectories.utilities import generate_drone_format
from flexible_drones_tools.trajectories.utilities.model import PolyOrderTrajectory


class SegmentFit():
    def __init__(self, t, x, y, z, yaws):
        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaws

    """
        Takes a number of desired segments and breaks the trajectory points
        into that number of chunks.
        Returns:
            A list of arrays with the t, x, y, z, yaw segments.
            A list of durations per segment.
    """

    def split_trajectory(self, num_segments=20):
        if num_segments < 1:
            raise ValueError('num_segments must be at least 1')
        if len(self.x) < num_segments:
            raise ValueError('num_segments cannot exceed the number of trajectory points')
        indices = np.array_split(np.arange(len(self.x)), num_segments)
        segments = [(self.t[index], self.x[index],
                     self.y[index], self.z[index],
                     self.yaw[index])
                    for index in indices]
        # for seg in segments:
        #     print(f"segment: {seg}")
        return segments, ([seg[0][-1] - seg[0][0] for seg in segments])

    """
        Takes segments from a trajectory and find the 7th degree polynomial
        of best fit to maximise the smoothness of flight.
        Returns:
            A reference to GenerateDroneFormat
    """

    def polyfit_traj_for_translating(self, durations, segments, write: bool, path: str):
        polynomials = self.segs_to_poly(segments)
        x_coeffs, y_coeffs, z_coeffs, yaw_coeffs = self.poly_to_coeffs(polynomials)
        return generate_drone_format.GenerateDroneFormat(durations, x_coeffs, y_coeffs, z_coeffs,
                                                         yaw_coeffs, write, path)

    """
        Takes segments from a trajectory and find the 7th degree polynomial
        of best fit to maximise the smoothness of flight.
        Returns:
            A reference to PolyOrderTrajectory in numpy format
    """

    def polyfit_trajectory_np(self, durations, segments):
        polynomials = self.segs_to_poly(segments)
        x_coeffs, y_coeffs, z_coeffs, yaw_coeffs = self.poly_to_coeffs(polynomials)
        return PolyOrderTrajectory(durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs)

    """
        Find polynomials that work with segments using polyfit.
    """
    @staticmethod
    def segs_to_poly(segments):
        polynomials = []
        for segment in segments:
            t_segment, x_segment, y_segment, z_segment, yaw_segment = segment
            t_local = t_segment - t_segment[0]
            p_x = np.polyfit(t_local, x_segment, 7)
            p_y = np.polyfit(t_local, y_segment, 7)
            p_z = np.polyfit(t_local, z_segment, 7)
            p_yaw = np.polyfit(t_local, yaw_segment, 7)

            polynomials.append((p_x, p_y, p_z, p_yaw))
        return polynomials

    """
        Sort out the coefficients into groups of x, y, z, and yaw to allow for
        easy writing and plotting.
    """
    @staticmethod
    def poly_to_coeffs(polynomials):
        x_coeffs = []
        y_coeffs = []
        z_coeffs = []
        yaw_coeffs = []
        for i, (c_x, c_y, c_z, c_yaw) in enumerate(polynomials):
            x_coeffs.append(c_x)
            y_coeffs.append(c_y)
            z_coeffs.append(c_z)
            yaw_coeffs.append(c_yaw)
        return x_coeffs, y_coeffs, z_coeffs, yaw_coeffs

    """
        Automatically run scripts to split the data into segments, find the
        polynomial fo best fit, plot the result, and write it to a file
    """

    def split_fit_and_plot(self, write, path):
        seg, dur = self.split_trajectory()
        poly_order = self.polyfit_trajectory_np(dur, seg)
        poly_order.plot_trajectory()

        poly_format = self.polyfit_traj_for_translating(dur, seg, write, path)
        poly_format.to_ros_pieces()
        poly_format.write_to_csv()

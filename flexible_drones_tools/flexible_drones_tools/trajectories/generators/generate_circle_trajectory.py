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

# Run from source (workspace sourced); see generators/README.md. Example:
#   ./generate_circle_trajectory.py --num-points 800 --num-segments 30 --ramp-duration 0.0
# Output defaults to generators/output/circle_trajectory_r<radius>m_z<z>m.csv

from flexible_drones_tools.trajectories.generators.generate_trajectory import GenerateTrajectory
from typing import override
import numpy as np


DEFAULT_RADIUS = 1.0
DEFAULT_Z = 0.0
DEFAULT_DURATION = 10.0
DEFAULT_RAMP_DURATION = 1.0
MIN_POLYFIT_POINTS_PER_SEGMENT = 8


def _meters_part(value):
    return f'{float(value):.2f}'.replace('.', '_')


def circle_filename(radius, z):
    """Return the filename with radius and height encoded in meters."""
    return f'circle_trajectory_r{_meters_part(radius)}m_z{_meters_part(z)}m.csv'


class GenerateCircleTrajectory(GenerateTrajectory):

    def __init__(
        self,
        radius=DEFAULT_RADIUS,
        z=DEFAULT_Z,
        duration=DEFAULT_DURATION,
        ramp_duration=DEFAULT_RAMP_DURATION,
    ):
        self.radius = float(radius)
        self.z = float(z)
        self.duration = float(duration)
        self.ramp_duration = float(ramp_duration)
        self._validate_inputs(self.radius, self.duration, self.ramp_duration)

    @staticmethod
    def _validate_inputs(radius, duration, ramp_duration):
        if radius <= 0.0:
            raise ValueError('radius must be greater than 0.0 meters')
        if duration <= 0.0:
            raise ValueError('duration must be greater than 0.0 seconds')
        if ramp_duration < 0.0:
            raise ValueError('ramp_duration must be greater than or equal to 0.0 seconds')
        if 2.0 * ramp_duration > duration:
            raise ValueError('ramp_duration must be no more than half of duration')

    @staticmethod
    def _validate_num_points(num_points):
        normalized = int(num_points)
        if normalized != num_points:
            raise ValueError('num_points must be an integer')
        if normalized < 2:
            raise ValueError('num_points must be at least 2')
        return normalized

    @staticmethod
    def _validate_polyfit_sampling(num_points, num_segments):
        num_points = GenerateCircleTrajectory._validate_num_points(num_points)
        normalized_segments = int(num_segments)
        if normalized_segments != num_segments:
            raise ValueError('num_segments must be an integer')
        if normalized_segments < 1:
            raise ValueError('num_segments must be at least 1')
        if num_points < normalized_segments * MIN_POLYFIT_POINTS_PER_SEGMENT:
            raise ValueError(
                'num_points must be at least 8 times num_segments '
                'for 7th-degree polynomial fitting'
            )
        return num_points, normalized_segments

    @staticmethod
    def _angle_from_time(t, duration, ramp_duration):
        if ramp_duration == 0.0:
            return 2.0 * np.pi * t / duration

        peak_wz = 2.0 * np.pi / (duration - ramp_duration)
        alpha = peak_wz / ramp_duration
        angle = np.empty_like(t)

        accel = t < ramp_duration
        decel_start = duration - ramp_duration
        cruise = (t >= ramp_duration) & (t <= decel_start)
        decel = t > decel_start

        angle[accel] = 0.5 * alpha * t[accel] ** 2
        angle[cruise] = (
            0.5 * peak_wz * ramp_duration
            + peak_wz * (t[cruise] - ramp_duration)
        )

        decel_t = t[decel] - decel_start
        angle_at_decel_start = peak_wz * (duration - 1.5 * ramp_duration)
        angle[decel] = (
            angle_at_decel_start
            + peak_wz * decel_t
            - 0.5 * alpha * decel_t ** 2
        )
        return angle

    @override
    def generate_trajectory(
        self,
        num_points=500,
        wz=None,
        radius=None,
        z=None,
        duration=None,
        ramp_duration=None,
    ):
        """Generate a circular trajectory around the origin in the XY-plane."""
        num_points = self._validate_num_points(num_points)
        radius = self.radius if radius is None else float(radius)
        z = self.z if z is None else float(z)
        if wz is not None and duration is not None:
            raise ValueError('provide either wz or duration, not both')
        duration = self.duration if duration is None else float(duration)
        if wz is not None:
            wz = float(wz)
            if wz <= 0.0:
                raise ValueError('wz must be greater than 0.0 radians per second')
            duration = 2.0 * np.pi / wz
        ramp_duration = (
            self.ramp_duration if ramp_duration is None else float(ramp_duration)
        )
        self._validate_inputs(radius, duration, ramp_duration)

        t = np.linspace(0.0, duration, num_points)
        angles = self._angle_from_time(t, duration, ramp_duration)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.full_like(x, z)

        yaw = np.unwrap(angles + 0.5 * np.pi)

        return t, x, y, z, yaw

    def generate_to_file(self, path, num_points=None, num_segments=20):
        num_points = 500 if num_points is None else num_points
        num_points, num_segments = self._validate_polyfit_sampling(
            num_points,
            num_segments,
        )
        return super().generate_to_file(path, num_points, num_segments)


def main(args=None):
    import argparse

    from flexible_drones_tools.trajectories.generators import default_output_path

    parser = argparse.ArgumentParser(
        description='Generate a circle trajectory CSV (drone coefficient order).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('path', nargs='?', default=None,
                        help='Output CSV path (default: generators/output/circle_trajectory_r<radius>m_z<z>m.csv).')
    parser.add_argument('--num-points', type=int, default=500,
                        help='Number of sampled points to generate before polynomial fitting.')
    parser.add_argument('--num-segments', type=int, default=20,
                        help='Number of polynomial segments to fit.')
    parser.add_argument('--radius', type=float, default=DEFAULT_RADIUS,
                        help='Circle radius in meters.')
    parser.add_argument('--z', type=float, default=DEFAULT_Z,
                        help='Constant trajectory height in meters.')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION,
                        help='Total circle duration in seconds.')
    parser.add_argument(
        '--ramp-duration',
        type=float,
        default=DEFAULT_RAMP_DURATION,
        help='Seconds to accelerate at the start and decelerate at the end; use 0.0 for a continuous loop.',
    )
    opts = parser.parse_args(args)
    if opts.radius <= 0.0:
        parser.error('--radius must be greater than 0.0')
    if opts.duration <= 0.0:
        parser.error('--duration must be greater than 0.0')
    if opts.ramp_duration < 0.0:
        parser.error('--ramp-duration must be greater than or equal to 0.0')
    if 2.0 * opts.ramp_duration > opts.duration:
        parser.error('--ramp-duration must be no more than half of --duration')
    if opts.num_points < 2:
        parser.error('--num-points must be at least 2')
    if opts.num_segments < 1:
        parser.error('--num-segments must be at least 1')
    if opts.num_points < opts.num_segments * MIN_POLYFIT_POINTS_PER_SEGMENT:
        parser.error(
            '--num-points must be at least 8 times --num-segments '
            'for 7th-degree polynomial fitting'
        )
    path = opts.path or default_output_path(circle_filename(opts.radius, opts.z))
    out = GenerateCircleTrajectory(
        radius=opts.radius,
        z=opts.z,
        duration=opts.duration,
        ramp_duration=opts.ramp_duration,
    ).generate_to_file(
        path, num_points=opts.num_points, num_segments=opts.num_segments)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

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
#   ./generate_figure8_trajectory.py --radius 1.5 --z-floor 1.0 --z-ceil 1.8 --duration 20 --ramp-duration 2
# Output defaults to generators/output/figure8_trajectory_<radius>m_z<floor>m.csv
# The radius is the bounding-circle radius of the whole figure eight, not the
# radius of the arcs that make up the 8.

import numpy as np
from pathlib import Path

from flexible_drones_tools.trajectories.generators.generate_trajectory import GenerateTrajectory


DEFAULT_DURATION = 20.0
DEFAULT_RAMP_DURATION = 2.0
DEFAULT_Z_FLOOR = 1.0
DEFAULT_Z_CEIL = 1.0
MIN_POLYFIT_POINTS_PER_SEGMENT = 8


def default_figure8_output_path(filename):
    """Return the figure-eight output path beside this generator script."""
    output_dir = Path(__file__).resolve().parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _meters_part(value):
    return f'{float(value):.2f}'.replace('.', '_')


def figure8_filename(radius, z_floor=None, z_ceil=None):
    """Return the filename with radius and height encoded in meters."""
    radius_part = f'{float(radius):.2f}'.replace('.', '_')
    if z_floor is None and z_ceil is None:
        return f'figure8_trajectory_{radius_part}m.csv'
    if z_floor is None or z_ceil is None:
        raise ValueError('z_floor and z_ceil must be provided together')

    z_floor_part = _meters_part(z_floor)
    if np.isclose(float(z_floor), float(z_ceil)):
        z_part = f'z{z_floor_part}m'
    else:
        z_ceil_part = _meters_part(z_ceil)
        z_part = f'z{z_floor_part}m_to_{z_ceil_part}m'
    return f'figure8_trajectory_{radius_part}m_{z_part}.csv'


class GenerateFigure8Trajectory(GenerateTrajectory):
    """
    Generate a horizontal figure-eight trajectory with tangent-following yaw.

    The radius is the bounding-circle radius of the whole figure eight, not the
    radius of the arcs that make up the 8.
    """

    def __init__(self, radius=1.0, z_floor=DEFAULT_Z_FLOOR,
                 z_ceil=DEFAULT_Z_CEIL, duration=DEFAULT_DURATION,
                 ramp_duration=DEFAULT_RAMP_DURATION, z=None):
        self.radius = float(radius)
        if z is not None:
            z_floor = z
            z_ceil = z
        self.z_floor = float(z_floor)
        self.z_ceil = float(z_ceil)
        self.duration = float(duration)
        self.ramp_duration = float(ramp_duration)
        self._validate_radius(self.radius)
        self._validate_timing(self.duration, self.ramp_duration)

    @staticmethod
    def _validate_radius(radius):
        if radius <= 0.0:
            raise ValueError('radius must be greater than 0.0 meters')

    @staticmethod
    def _validate_timing(duration, ramp_duration):
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
        num_points = GenerateFigure8Trajectory._validate_num_points(num_points)
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
    def _theta_from_time(t, duration, ramp_duration):
        if ramp_duration == 0.0:
            return 2 * np.pi * t / duration

        peak_wz = 2 * np.pi / (duration - ramp_duration)
        alpha = peak_wz / ramp_duration
        theta = np.empty_like(t)

        accel = t < ramp_duration
        decel_start = duration - ramp_duration
        cruise = (t >= ramp_duration) & (t <= decel_start)
        decel = t > decel_start

        theta[accel] = 0.5 * alpha * t[accel] ** 2
        theta[cruise] = (
            0.5 * peak_wz * ramp_duration
            + peak_wz * (t[cruise] - ramp_duration)
        )

        decel_t = t[decel] - decel_start
        theta_at_decel_start = peak_wz * (duration - 1.5 * ramp_duration)
        theta[decel] = (
            theta_at_decel_start
            + peak_wz * decel_t
            - 0.5 * alpha * decel_t ** 2
        )
        return theta

    def generate_trajectory(
        self,
        num_points=500,
        duration=None,
        ramp_duration=None,
        z_floor=None,
        z_ceil=None,
    ):
        """Generate a Gerono lemniscate centered on the origin."""
        num_points = self._validate_num_points(num_points)
        duration = self.duration if duration is None else float(duration)
        ramp_duration = (
            self.ramp_duration if ramp_duration is None else float(ramp_duration)
        )
        z_floor = self.z_floor if z_floor is None else float(z_floor)
        z_ceil = self.z_ceil if z_ceil is None else float(z_ceil)
        self._validate_timing(duration, ramp_duration)
        t = np.linspace(0.0, duration, num_points)
        theta = self._theta_from_time(t, duration, ramp_duration)

        x = self.radius * np.sin(theta)
        y = self.radius * np.sin(theta) * np.cos(theta)
        z_progress = np.sin(theta) ** 2
        z = z_floor + (z_ceil - z_floor) * z_progress

        dx_dtheta = self.radius * np.cos(theta)
        dy_dtheta = self.radius * np.cos(2 * theta)
        yaw = np.unwrap(np.arctan2(dy_dtheta, dx_dtheta))

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

    parser = argparse.ArgumentParser(
        description='Generate a figure-eight trajectory CSV (drone coefficient order).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'path',
        nargs='?',
        default=None,
        help='Output CSV path (default: generators/output/figure8_trajectory_<radius>m_z<floor>m.csv).',
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=1.0,
        help=(
            'Bounding-circle radius of the whole figure eight in meters; '
            'not the radius of the arcs that make up the 8.'
        ),
    )
    parser.add_argument(
        '--z-floor',
        type=float,
        default=DEFAULT_Z_FLOOR,
        help='Height at the center crossing of the figure eight in meters.',
    )
    parser.add_argument(
        '--z-ceil',
        type=float,
        default=DEFAULT_Z_CEIL,
        help='Height at the outermost point of each loop in meters.',
    )
    parser.add_argument(
        '--z',
        type=float,
        default=None,
        help='Set a constant figure-eight height in meters; overrides --z-floor and --z-ceil.',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=DEFAULT_DURATION,
        help='Total figure-eight duration in seconds.',
    )
    parser.add_argument(
        '--ramp-duration',
        type=float,
        default=DEFAULT_RAMP_DURATION,
        help='Seconds to accelerate at the start and decelerate at the end.',
    )
    parser.add_argument('--num-points', type=int, default=500,
                        help='Number of sampled points to generate before polynomial fitting.')
    parser.add_argument('--num-segments', type=int, default=20,
                        help='Number of polynomial segments to fit.')
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
    z_floor = opts.z_floor if opts.z is None else opts.z
    z_ceil = opts.z_ceil if opts.z is None else opts.z
    path = opts.path or default_figure8_output_path(
        figure8_filename(opts.radius, z_floor, z_ceil)
    )

    out = GenerateFigure8Trajectory(
        radius=opts.radius,
        z_floor=z_floor,
        z_ceil=z_ceil,
        duration=opts.duration,
        ramp_duration=opts.ramp_duration,
    ).generate_to_file(
        path,
        num_points=opts.num_points,
        num_segments=opts.num_segments,
    )
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

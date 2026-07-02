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
#   ./generate_x_frequency_sweep_trajectory.py --duration 20 --start-hz 0.1 --end-hz 1.0
# Output defaults to generators/output/x_frequency_sweep_trajectory.csv

import numpy as np

from flexible_drones_tools.trajectories.generators.generate_trajectory import GenerateTrajectory


DEFAULT_DURATION = 20.0
DEFAULT_START_HZ = 0.1
DEFAULT_END_HZ = 1.0


class GenerateXFrequencySweepTrajectory(GenerateTrajectory):
    """Generate a constant-altitude X-axis chirp trajectory."""

    def __init__(
        self,
        amplitude=1.0,
        y=0.0,
        z=1.0,
        duration=DEFAULT_DURATION,
        start_hz=DEFAULT_START_HZ,
        end_hz=DEFAULT_END_HZ,
    ):
        self.amplitude = float(amplitude)
        self.y = float(y)
        self.z = float(z)
        self.duration = float(duration)
        self.start_hz = float(start_hz)
        self.end_hz = float(end_hz)
        self._validate()

    def _validate(self):
        if self.amplitude <= 0.0:
            raise ValueError('amplitude must be greater than 0.0 meters')
        if self.duration <= 0.0:
            raise ValueError('duration must be greater than 0.0 seconds')
        if self.start_hz < 0.0:
            raise ValueError('start_hz must be greater than or equal to 0.0')
        if self.end_hz < self.start_hz:
            raise ValueError('end_hz must be greater than or equal to start_hz')

    def generate_trajectory(self, num_points=500):
        """Sweep X back and forth while Y and Z stay locked."""
        if num_points < 2:
            raise ValueError('num_points must be at least 2')

        t = np.linspace(0.0, self.duration, num_points)
        chirp_rate = (self.end_hz - self.start_hz) / self.duration
        phase = np.pi * (self.start_hz * t + 0.5 * chirp_rate * t ** 2)

        x = self.amplitude * np.cos(phase)
        y = np.full_like(x, self.y)
        z = np.full_like(x, self.z)
        yaw = np.zeros_like(x)

        return t, x, y, z, yaw


def main(args=None):
    import argparse

    from flexible_drones_tools.trajectories.generators import default_output_path

    parser = argparse.ArgumentParser(
        description='Generate an X-axis frequency-sweep trajectory CSV (drone coefficient order).'
    )
    parser.add_argument(
        'path',
        nargs='?',
        default=None,
        help='Output CSV path (default: generators/output/x_frequency_sweep_trajectory.csv).',
    )
    parser.add_argument('--amplitude', type=float, default=1.0,
                        help='X displacement amplitude in meters.')
    parser.add_argument('--y', type=float, default=0.0,
                        help='Constant Y position in meters.')
    parser.add_argument('--z', type=float, default=1.0,
                        help='Constant Z position in meters.')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION,
                        help='Total trajectory duration in seconds.')
    parser.add_argument('--start-hz', type=float, default=DEFAULT_START_HZ,
                        help='Starting oscillation frequency in Hz.')
    parser.add_argument('--end-hz', type=float, default=DEFAULT_END_HZ,
                        help='Ending oscillation frequency in Hz.')
    parser.add_argument('--num-points', type=int, default=500)
    parser.add_argument('--num-segments', type=int, default=20)
    opts = parser.parse_args(args)

    try:
        generator = GenerateXFrequencySweepTrajectory(
            amplitude=opts.amplitude,
            y=opts.y,
            z=opts.z,
            duration=opts.duration,
            start_hz=opts.start_hz,
            end_hz=opts.end_hz,
        )
        path = opts.path or default_output_path('x_frequency_sweep_trajectory.csv')
        out = generator.generate_to_file(
            path,
            num_points=opts.num_points,
            num_segments=opts.num_segments,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

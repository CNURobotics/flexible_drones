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
#   ./generate_spiral_trajectory.py --num-points 800 --num-segments 30 --max-vx 0.75
# Output defaults to generators/output/spiral_trajectory_r<radius>m_z<floor>m.csv

import numpy as np
from typing import override

from flexible_drones_tools.trajectories.generators.generate_trajectory import GenerateTrajectory


# A nonzero ramp prevents the drone from trying to jump instantly to full speed.
# Reverse mode clamps the top turnaround to at least 1s slowdown, 1s turn,
# and 1s speedup even when the normal ramp/turn durations are lower.
DEFAULT_RAMP_DURATION = 1.0
DEFAULT_MAX_VX = 0.75
DEFAULT_MAX_WZ = np.pi / 2
DEFAULT_RADIUS = 2.5
DEFAULT_Z_FLOOR = 0.75
DEFAULT_Z_CEIL = 2.5
DEFAULT_MAX_ANGLE = 4 * np.pi
DEFAULT_YAW_TURN_DURATION = 1.0
DEFAULT_REVERSE = True
MIN_REVERSE_RAMP_DURATION = 1.0
MIN_REVERSE_YAW_TURN_DURATION = 1.0
MIN_POLYFIT_POINTS_PER_SEGMENT = 8


def spiral_filename(radius, z_floor, z_ceil):
    """Return the filename with max radius and height range encoded in meters."""
    radius_part = f'{float(radius):.2f}'.replace('.', '_')
    z_floor_part = f'{float(z_floor):.2f}'.replace('.', '_')
    if np.isclose(float(z_floor), float(z_ceil)):
        z_part = f'z{z_floor_part}m'
    else:
        z_ceil_part = f'{float(z_ceil):.2f}'.replace('.', '_')
        z_part = f'z{z_floor_part}m_to_{z_ceil_part}m'
    return f'spiral_trajectory_r{radius_part}m_{z_part}.csv'


class GenerateSpiralTrajectory(GenerateTrajectory):

    def __init__(
        self,
        max_vx=DEFAULT_MAX_VX,
        max_wz=DEFAULT_MAX_WZ,
        radius=DEFAULT_RADIUS,
        z_floor=DEFAULT_Z_FLOOR,
        z_ceil=DEFAULT_Z_CEIL,
        ramp_duration=DEFAULT_RAMP_DURATION,
        yaw_turn_duration=DEFAULT_YAW_TURN_DURATION,
        reverse=DEFAULT_REVERSE,
    ):
        self.max_vx = float(max_vx)
        self.max_wz = float(max_wz)
        self.radius = float(radius)
        self.z_floor = float(z_floor)
        self.z_ceil = float(z_ceil)
        self.ramp_duration = float(ramp_duration)
        self.yaw_turn_duration = float(yaw_turn_duration)
        self.reverse = bool(reverse)
        self._validate_inputs(
            self.max_vx,
            self.max_wz,
            self.radius,
            self.ramp_duration,
            self.yaw_turn_duration,
        )

    @staticmethod
    def _validate_inputs(max_vx, max_wz, radius, ramp_duration, yaw_turn_duration):
        if max_vx <= 0.0:
            raise ValueError('max_vx must be greater than 0.0 meters per second')
        if max_wz <= 0.0:
            raise ValueError('max_wz must be greater than 0.0 radians per second')
        if radius <= 0.0:
            raise ValueError('radius must be greater than 0.0 meters')
        if ramp_duration < 0.0:
            raise ValueError('ramp_duration must be greater than or equal to 0.0 seconds')
        if yaw_turn_duration < 0.0:
            raise ValueError('yaw_turn_duration must be greater than or equal to 0.0 seconds')

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
        num_points = GenerateSpiralTrajectory._validate_num_points(num_points)
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
    def _path_length_from_points(x, y, z):
        """Measure cumulative 3D distance along a sampled curve."""
        segment_lengths = np.linalg.norm(
            np.column_stack((
                np.diff(x),
                np.diff(y),
                np.diff(z),
            )),
            axis=1,
        )
        return np.concatenate(([0.0], np.cumsum(segment_lengths)))

    @staticmethod
    def _time_from_distance(path_distance, theta, max_vx, max_wz, ramp_duration):
        return GenerateSpiralTrajectory._time_from_distance_with_ramps(
            path_distance,
            theta,
            max_vx,
            max_wz,
            ramp_duration,
            ramp_duration,
        )

    @staticmethod
    def _time_from_distance_with_ramps(
        path_distance,
        theta,
        max_vx,
        max_wz,
        start_ramp_duration,
        end_ramp_duration,
    ):
        """
        Build a time schedule that obeys both path-speed and angular-speed caps.

        ds/dtheta says how many meters the drone travels for each radian of
        spiral angle.  max_wz * ds/dtheta converts the angular rate limit into a
        local path-speed limit.  Near the center ds/dtheta is small, so max_wz is
        usually the tighter limit; farther out max_vx usually becomes tighter.
        """
        path_length = path_distance[-1]
        ds_dtheta = np.gradient(path_distance, theta)
        local_speed_limit = np.minimum(max_vx, max_wz * ds_dtheta)

        accel_speed = np.full_like(path_distance, np.inf)
        if start_ramp_duration > 0.0:
            # This acceleration gives a max_vx-sized ramp when the path is not
            # already slowed by the angular-rate limit.
            start_acceleration = max_vx / start_ramp_duration
            accel_speed = np.sqrt(2.0 * start_acceleration * path_distance)

        decel_speed = np.full_like(path_distance, np.inf)
        if end_ramp_duration > 0.0:
            end_acceleration = max_vx / end_ramp_duration
            decel_speed = np.sqrt(2.0 * end_acceleration * (path_length - path_distance))

        speed = np.minimum(local_speed_limit, np.minimum(accel_speed, decel_speed))

        # The endpoints are allowed to be zero speed.  Time between dense samples
        # uses average segment speed, so the first and last intervals stay finite.
        segment_distance = np.diff(path_distance)
        segment_speed = 0.5 * (speed[:-1] + speed[1:])
        segment_speed = np.maximum(segment_speed, np.finfo(float).eps)
        segment_time = segment_distance / segment_speed
        return np.concatenate(([0.0], np.cumsum(segment_time)))

    @staticmethod
    def _distance_from_time(t, path_distance, path_time):
        """Look up distance along the path for each requested time sample."""
        return np.interp(
            np.clip(t, path_time[0], path_time[-1]),
            path_time,
            path_distance,
        )

    @override
    def generate_trajectory(
        self,
        num_points=500,
        max_vx=None,
        max_wz=None,
        radius=None,
        z_floor=None,
        z_ceil=None,
        ramp_duration=None,
        yaw_turn_duration=None,
        reverse=None,
    ):

        num_points = self._validate_num_points(num_points)

        # Allow callers to override settings for one generated path
        # without permanently changing the generator object.
        max_vx = self.max_vx if max_vx is None else float(max_vx)
        max_wz = self.max_wz if max_wz is None else float(max_wz)
        radius = self.radius if radius is None else float(radius)
        z_floor = self.z_floor if z_floor is None else float(z_floor)
        z_ceil = self.z_ceil if z_ceil is None else float(z_ceil)
        ramp_duration = (
            self.ramp_duration if ramp_duration is None else float(ramp_duration)
        )
        yaw_turn_duration = (
            self.yaw_turn_duration
            if yaw_turn_duration is None
            else float(yaw_turn_duration)
        )
        reverse = self.reverse if reverse is None else bool(reverse)
        self._validate_inputs(max_vx, max_wz, radius, ramp_duration, yaw_turn_duration)
        turn_ramp_duration = (
            max(ramp_duration, MIN_REVERSE_RAMP_DURATION)
            if reverse
            else ramp_duration
        )
        turn_yaw_duration = (
            max(yaw_turn_duration, MIN_REVERSE_YAW_TURN_DURATION)
            if reverse
            else yaw_turn_duration
        )

        # First build a dense copy of the outbound 3D spiral. This is just for
        # measuring distance along the curve and for interpolation.
        dense_points = max(2000, 10 * num_points)
        theta_dense = np.linspace(0.0, DEFAULT_MAX_ANGLE, dense_points)

        # Archimedean spiral equation: radius grows linearly with angle.
        # The scale makes the final point land at the requested max radius.
        radii_dense = radius * theta_dense / DEFAULT_MAX_ANGLE

        # Convert polar coordinates (radius, angle) into Cartesian x/y points.
        x_dense = radii_dense * np.cos(theta_dense)
        y_dense = radii_dense * np.sin(theta_dense)

        # z is tied to the same spiral progress, then timing is applied through
        # distance below. That makes z accelerate and decelerate with x/y.
        progress_dense = theta_dense / DEFAULT_MAX_ANGLE
        z_dense = z_floor + (z_ceil - z_floor) * progress_dense

        # Measure true 3D path length, including the climb. This lets max_vx
        # limit body/path speed while max_wz limits angular speed near the center.
        path_distance_dense = self._path_length_from_points(x_dense, y_dense, z_dense)
        path_time_dense = self._time_from_distance_with_ramps(
            path_distance_dense,
            theta_dense,
            max_vx,
            max_wz,
            ramp_duration,
            turn_ramp_duration,
        )
        leg_duration = path_time_dense[-1]
        t_out = np.linspace(0.0, leg_duration, num_points)

        # Convert each time into distance traveled, then look up the matching
        # x/y/z point along the dense curve. This keeps speed bounded by both
        # max_vx and max_wz.
        distance_out = self._distance_from_time(
            t_out,
            path_distance_dense,
            path_time_dense,
        )
        x_out = np.interp(distance_out, path_distance_dense, x_dense)
        y_out = np.interp(distance_out, path_distance_dense, y_dense)
        z_out = np.interp(distance_out, path_distance_dense, z_dense)
        theta_out = np.interp(distance_out, path_distance_dense, theta_dense)

        # The inbound leg mirrors the outbound points and descends to z_floor.
        x_in = x_out[::-1]
        y_in = y_out[::-1]
        z_in = z_out[::-1]

        # Yaw follows the tangent of the x/y curve. The outbound tangent comes
        # from differentiating x = r*cos(theta), y = r*sin(theta). The inbound
        # leg travels that same curve backward, so its heading is 180 degrees
        # from the outbound tangent at the matching point.
        dr_dtheta = radius / DEFAULT_MAX_ANGLE
        radii_out = radius * theta_out / DEFAULT_MAX_ANGLE
        dx_dtheta = dr_dtheta * np.cos(theta_out) - radii_out * np.sin(theta_out)
        dy_dtheta = dr_dtheta * np.sin(theta_out) + radii_out * np.cos(theta_out)
        yaw_out = np.arctan2(dy_dtheta, dx_dtheta)
        if not reverse:
            return t_out, x_out, y_out, z_out, np.unwrap(yaw_out)

        yaw_in = yaw_out[::-1] + np.pi

        # Rotate in place at the top before starting the inbound leg. This keeps
        # the drone pointed along the outbound tangent, then smoothly turns it
        # to the inbound tangent while x/y/z remain fixed at the outer point.
        if turn_yaw_duration == 0.0:
            turn_elapsed = np.array([])
            x_turn = np.array([])
            y_turn = np.array([])
            z_turn = np.array([])
            yaw_turn = np.array([])
        else:
            yaw_turn_points = max(
                1,
                int(np.floor(num_points * turn_yaw_duration / leg_duration)),
            )
            turn_elapsed = np.linspace(
                0.0,
                turn_yaw_duration,
                yaw_turn_points + 1,
            )[1:]
            x_turn = np.full_like(turn_elapsed, x_out[-1])
            y_turn = np.full_like(turn_elapsed, y_out[-1])
            z_turn = np.full_like(turn_elapsed, z_out[-1])
            yaw_turn = np.linspace(
                yaw_out[-1],
                yaw_in[0],
                yaw_turn_points + 1,
            )[1:]

        # Concatenate the outbound, in-place yaw turn, and inbound pieces. The
        # inbound arrays skip their first sample because the top/outer point is
        # already represented by the yaw-turn endpoint.
        t = np.concatenate((
            t_out,
            leg_duration + turn_elapsed,
            leg_duration + turn_yaw_duration + t_out[1:],
        ))
        x = np.concatenate((x_out, x_turn, x_in[1:]))
        y = np.concatenate((y_out, y_turn, y_in[1:]))
        z = np.concatenate((z_out, z_turn, z_in[1:]))
        yaw = np.unwrap(np.concatenate((yaw_out, yaw_turn, yaw_in[1:])))
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
        description='Generate a spiral trajectory CSV (drone coefficient order).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'path',
        nargs='?',
        default=None,
        help=(
            'Output CSV path (default: '
            'generators/output/spiral_trajectory_r<radius>m_z<floor>m.csv).'
        ),
    )
    parser.add_argument('--num-points', type=int, default=500,
                        help='Number of sampled points to generate before polynomial fitting.')
    parser.add_argument('--num-segments', type=int, default=20,
                        help='Number of polynomial segments to fit.')
    parser.add_argument(
        '--max-vx',
        '--vx',
        dest='max_vx',
        type=float,
        default=DEFAULT_MAX_VX,
        help='Maximum body/path velocity in meters per second.',
    )
    parser.add_argument(
        '--max-wz',
        type=float,
        default=DEFAULT_MAX_WZ,
        help='Maximum spiral angular rate in radians per second.',
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=DEFAULT_RADIUS,
        help='Maximum spiral radius in meters.',
    )
    parser.add_argument(
        '--z-floor',
        type=float,
        default=DEFAULT_Z_FLOOR,
        help='Starting and ending height in meters.',
    )
    parser.add_argument(
        '--z-ceil',
        type=float,
        default=DEFAULT_Z_CEIL,
        help='Height at the outer/top turnaround in meters.',
    )
    parser.add_argument(
        '--ramp-duration',
        type=float,
        default=DEFAULT_RAMP_DURATION,
        help=(
            'Seconds to accelerate and decelerate. In reverse mode, slowdown '
            'into the turnaround and speedup out of it are clamped to at least 1 second.'
        ),
    )
    parser.add_argument(
        '--yaw-turn-duration',
        type=float,
        default=DEFAULT_YAW_TURN_DURATION,
        help=(
            'Seconds to rotate in place at the top before descending; '
            'clamped to at least 1 second in reverse mode.'
        ),
    )
    parser.add_argument(
        '--reverse',
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REVERSE,
        help='Return from the top back to the center/bottom after an in-place turn.',
    )
    opts = parser.parse_args(args)
    if opts.max_vx <= 0.0:
        parser.error('--max-vx must be greater than 0.0')
    if opts.max_wz <= 0.0:
        parser.error('--max-wz must be greater than 0.0')
    if opts.radius <= 0.0:
        parser.error('--radius must be greater than 0.0')
    if opts.ramp_duration < 0.0:
        parser.error('--ramp-duration must be greater than or equal to 0.0')
    if opts.yaw_turn_duration < 0.0:
        parser.error('--yaw-turn-duration must be greater than or equal to 0.0')
    if opts.num_points < 2:
        parser.error('--num-points must be at least 2')
    if opts.num_segments < 1:
        parser.error('--num-segments must be at least 1')
    if opts.num_points < opts.num_segments * MIN_POLYFIT_POINTS_PER_SEGMENT:
        parser.error(
            '--num-points must be at least 8 times --num-segments '
            'for 7th-degree polynomial fitting'
        )
    path = opts.path or default_output_path(
        spiral_filename(opts.radius, opts.z_floor, opts.z_ceil)
    )
    out = GenerateSpiralTrajectory(
        max_vx=opts.max_vx,
        max_wz=opts.max_wz,
        radius=opts.radius,
        z_floor=opts.z_floor,
        z_ceil=opts.z_ceil,
        ramp_duration=opts.ramp_duration,
        yaw_turn_duration=opts.yaw_turn_duration,
        reverse=opts.reverse,
    ).generate_to_file(
        path, num_points=opts.num_points, num_segments=opts.num_segments)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

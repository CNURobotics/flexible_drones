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
#   ./generate_star_trajectory.py --num-points 800 --num-segments 30 --vx 0.75 --alpha 0.05
# Output defaults to generators/output/star_trajectory_ro<outer>m_ri<inner>m_z<z>m.csv

from typing import override

import numpy as np

from flexible_drones_tools.trajectories.generators.generate_trajectory import GenerateTrajectory


DEFAULT_NUM_TIPS = 5
DEFAULT_OUTER_RADIUS = 1.5
DEFAULT_INNER_RADIUS = 0.75
DEFAULT_Z = 1.0
DEFAULT_MAX_VX = 0.75
DEFAULT_ALPHA = 0.05
DEFAULT_CORNER_SAMPLE_MULTIPLIER = 2.0
MIN_POLYFIT_POINTS_PER_SEGMENT = 8


def _meters_part(value):
    return f'{float(value):.2f}'.replace('.', '_')


def star_filename(outer_radius, inner_radius, z):
    """Return the filename with star radii and height encoded in meters."""
    return (
        f'star_trajectory_ro{_meters_part(outer_radius)}m_'
        f'ri{_meters_part(inner_radius)}m_z{_meters_part(z)}m.csv'
    )


class GenerateStarTrajectory(GenerateTrajectory):

    def __init__(
        self,
        num_tips=DEFAULT_NUM_TIPS,
        outer_radius=DEFAULT_OUTER_RADIUS,
        inner_radius=DEFAULT_INNER_RADIUS,
        z=DEFAULT_Z,
        max_vx=DEFAULT_MAX_VX,
        alpha=DEFAULT_ALPHA,
        corner_sample_multiplier=DEFAULT_CORNER_SAMPLE_MULTIPLIER,
    ):
        self.num_tips = int(num_tips)
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(inner_radius)
        self.z = float(z)
        self.max_vx = float(max_vx)
        self.alpha = float(alpha)
        self.corner_sample_multiplier = float(corner_sample_multiplier)
        self._validate_inputs(
            self.num_tips,
            self.outer_radius,
            self.inner_radius,
            self.max_vx,
            self.alpha,
            self.corner_sample_multiplier,
        )

    @staticmethod
    def _validate_inputs(
        num_tips,
        outer_radius,
        inner_radius,
        max_vx,
        alpha,
        corner_sample_multiplier=DEFAULT_CORNER_SAMPLE_MULTIPLIER,
    ):
        if num_tips < 2:
            raise ValueError('num_tips must be at least 2')
        if outer_radius <= 0.0:
            raise ValueError('outer_radius must be greater than 0.0 meters')
        if inner_radius <= 0.0:
            raise ValueError('inner_radius must be greater than 0.0 meters')
        if inner_radius >= outer_radius:
            raise ValueError('inner_radius must be less than outer_radius')
        if max_vx <= 0.0:
            raise ValueError('max_vx must be greater than 0.0 meters per second')
        if alpha < 0.0 or alpha >= 0.5:
            raise ValueError('alpha must be greater than or equal to 0.0 and less than 0.5')
        if corner_sample_multiplier < 1.0:
            raise ValueError('corner_sample_multiplier must be greater than or equal to 1.0')

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
        num_points = GenerateStarTrajectory._validate_num_points(num_points)
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
    def _vertices(num_tips, outer_radius, inner_radius):
        num_vertices = num_tips * 2
        angles = np.linspace(0.0, 2.0 * np.pi, num_vertices, endpoint=False)
        radii = np.empty_like(angles)
        radii[::2] = outer_radius
        radii[1::2] = inner_radius
        return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

    @classmethod
    def _geometry(cls, num_tips, outer_radius, inner_radius, max_vx, alpha):
        vertices = cls._vertices(num_tips, outer_radius, inner_radius)
        starts = vertices
        ends = np.roll(vertices, -1, axis=0)
        deltas = ends - starts
        lengths = np.linalg.norm(deltas, axis=1)
        headings = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
        durations = np.array([
            cls._segment_duration(length, max_vx, alpha)
            for length in lengths
        ])
        cumulative = np.concatenate(([0.0], np.cumsum(durations)))
        return starts, deltas, lengths, headings, durations, cumulative

    @staticmethod
    def _segment_duration(length, max_vx, alpha):
        if alpha == 0.0:
            return length / max_vx
        return (1.0 + 2.0 * alpha) * length / max_vx

    @staticmethod
    def _segment_progress(local_time, length, max_vx, alpha):
        if alpha == 0.0:
            return np.clip(max_vx * local_time / length, 0.0, 1.0)

        accel_distance = alpha * length
        accel_time = 2.0 * accel_distance / max_vx
        cruise_distance = (1.0 - 2.0 * alpha) * length
        cruise_time = cruise_distance / max_vx
        segment_duration = 2.0 * accel_time + cruise_time
        acceleration = max_vx / accel_time

        local_time = np.asarray(local_time)
        distance = np.empty_like(local_time, dtype=float)
        accel = local_time < accel_time
        cruise = (local_time >= accel_time) & (local_time <= accel_time + cruise_time)
        decel = local_time > accel_time + cruise_time

        distance[accel] = 0.5 * acceleration[accel] * local_time[accel] ** 2
        distance[cruise] = (
            accel_distance[cruise]
            + max_vx * (local_time[cruise] - accel_time[cruise])
        )
        time_remaining = segment_duration[decel] - local_time[decel]
        distance[decel] = (
            length[decel]
            - 0.5 * acceleration[decel] * time_remaining ** 2
        )
        return np.clip(distance / length, 0.0, 1.0)

    @staticmethod
    def _blend_angle(start, end, fraction):
        delta = np.angle(np.exp(1j * (end - start)))
        return start + delta * fraction

    @staticmethod
    def _segment_yaw(local_time, length, max_vx, alpha, segment_index, headings):
        yaw = headings[segment_index].copy()
        if alpha == 0.0:
            return yaw

        accel_time = 2.0 * alpha * length / max_vx
        cruise_time = (1.0 - 2.0 * alpha) * length / max_vx
        decel_start = accel_time + cruise_time
        last_segment = len(headings) - 1

        start_blend = (segment_index > 0) & (local_time < accel_time)
        if np.any(start_blend):
            indices = segment_index[start_blend]
            previous_heading = headings[indices - 1]
            current_heading = headings[indices]
            halfway_heading = GenerateStarTrajectory._blend_angle(
                previous_heading,
                current_heading,
                0.5,
            )
            fraction = np.clip(local_time[start_blend] / accel_time[start_blend], 0.0, 1.0)
            yaw[start_blend] = GenerateStarTrajectory._blend_angle(
                halfway_heading,
                current_heading,
                fraction,
            )

        end_blend = (segment_index < last_segment) & (local_time > decel_start)
        if np.any(end_blend):
            indices = segment_index[end_blend]
            current_heading = headings[indices]
            next_heading = headings[indices + 1]
            halfway_heading = GenerateStarTrajectory._blend_angle(
                current_heading,
                next_heading,
                0.5,
            )
            fraction = np.clip(
                (local_time[end_blend] - decel_start[end_blend]) / accel_time[end_blend],
                0.0,
                1.0,
            )
            yaw[end_blend] = GenerateStarTrajectory._blend_angle(
                current_heading,
                halfway_heading,
                fraction,
            )

        return np.unwrap(yaw)

    @classmethod
    def _sample_at_times(
        cls,
        t,
        starts,
        deltas,
        lengths,
        headings,
        durations,
        cumulative,
        max_vx,
        alpha,
        z,
    ):
        segment_index = np.searchsorted(cumulative, t, side='right') - 1
        segment_index = np.clip(segment_index, 0, len(durations) - 1)
        local_time = t - cumulative[segment_index]
        progress = cls._segment_progress(
            local_time,
            lengths[segment_index],
            max_vx,
            alpha,
        )

        xy = starts[segment_index] + deltas[segment_index] * progress[:, None]
        z_values = np.full(len(t), z)
        yaw = cls._segment_yaw(
            local_time,
            lengths[segment_index],
            max_vx,
            alpha,
            segment_index,
            headings,
        )
        return xy[:, 0], xy[:, 1], z_values, yaw

    @staticmethod
    def _segment_boundaries(cumulative, lengths, max_vx, alpha):
        if alpha == 0.0:
            return cumulative

        accel_time = 2.0 * alpha * lengths / max_vx
        cruise_time = (1.0 - 2.0 * alpha) * lengths / max_vx
        decel_start = accel_time + cruise_time

        boundaries = [cumulative[0], cumulative[0] + decel_start[0]]
        for edge_index in range(1, len(lengths)):
            boundaries.append(cumulative[edge_index] + accel_time[edge_index])
            boundaries.append(cumulative[edge_index] + decel_start[edge_index])
        boundaries.append(cumulative[-1])
        return np.array(boundaries)

    @staticmethod
    def _refine_boundaries(boundaries, requested_segments):
        calculated_segments = len(boundaries) - 1
        if requested_segments == calculated_segments:
            return boundaries

        durations = np.diff(boundaries)
        extra_segments = requested_segments - calculated_segments
        extra_by_segment = np.zeros(calculated_segments, dtype=int)
        order = np.argsort(durations)[::-1]
        for index in order[np.arange(extra_segments) % calculated_segments]:
            extra_by_segment[index] += 1

        refined = [boundaries[0]]
        for index, extra_count in enumerate(extra_by_segment):
            pieces = extra_count + 1
            refined.extend(
                np.linspace(boundaries[index], boundaries[index + 1], pieces + 1)[1:]
            )
        return np.array(refined)

    @staticmethod
    def _segments_from_boundaries(t, x, y, z, yaw, boundaries):
        segments = []
        for start_time, end_time in zip(boundaries[:-1], boundaries[1:]):
            start_index = np.searchsorted(t, start_time, side='left')
            end_index = np.searchsorted(t, end_time, side='left')
            segment = (
                t[start_index:end_index + 1],
                x[start_index:end_index + 1],
                y[start_index:end_index + 1],
                z[start_index:end_index + 1],
                yaw[start_index:end_index + 1],
            )
            if len(segment[0]) < MIN_POLYFIT_POINTS_PER_SEGMENT:
                raise ValueError(
                    'num_points is too low for star corner/straight segmentation; '
                    'increase --num-points'
                )
            segments.append(segment)
        durations = [seg[0][-1] - seg[0][0] for seg in segments]
        return segments, durations

    @staticmethod
    def _sample_counts_for_segments(num_points, num_segments, alpha, corner_sample_multiplier):
        weights = np.ones(num_segments)
        if alpha > 0.0 and num_segments > 2:
            corner_indices = np.arange(1, num_segments - 1, 2)
            weights[corner_indices] = corner_sample_multiplier

        extra_points = max(0, num_points - num_segments * MIN_POLYFIT_POINTS_PER_SEGMENT)
        weighted_extra = extra_points * weights / np.sum(weights)
        counts = MIN_POLYFIT_POINTS_PER_SEGMENT + np.floor(weighted_extra).astype(int)
        remainder = num_points - int(np.sum(counts))
        if remainder > 0:
            order = np.argsort(weighted_extra - np.floor(weighted_extra))[::-1]
            counts[order[:remainder]] += 1
        return counts

    def calculated_num_segments(self):
        starts, deltas, lengths, headings, durations, cumulative = self._geometry(
            self.num_tips,
            self.outer_radius,
            self.inner_radius,
            self.max_vx,
            self.alpha,
        )
        boundaries = self._segment_boundaries(
            cumulative,
            lengths,
            self.max_vx,
            self.alpha,
        )
        return len(boundaries) - 1

    @override
    def generate_trajectory(
        self,
        num_points=500,
        num_tips=None,
        outer_radius=None,
        inner_radius=None,
        z=None,
        max_vx=None,
        alpha=None,
        corner_sample_multiplier=None,
        wz=None,
    ):
        """Generate a star trajectory with speed ramps on each straight edge."""
        if wz is not None:
            raise ValueError('star generation uses max_vx; wz is not supported')
        num_points = self._validate_num_points(num_points)
        num_tips = self.num_tips if num_tips is None else int(num_tips)
        outer_radius = (
            self.outer_radius if outer_radius is None else float(outer_radius)
        )
        inner_radius = (
            self.inner_radius if inner_radius is None else float(inner_radius)
        )
        z = self.z if z is None else float(z)
        max_vx = self.max_vx if max_vx is None else float(max_vx)
        alpha = self.alpha if alpha is None else float(alpha)
        corner_sample_multiplier = (
            self.corner_sample_multiplier
            if corner_sample_multiplier is None
            else float(corner_sample_multiplier)
        )
        self._validate_inputs(
            num_tips,
            outer_radius,
            inner_radius,
            max_vx,
            alpha,
            corner_sample_multiplier,
        )

        starts, deltas, lengths, headings, durations, cumulative = self._geometry(
            num_tips,
            outer_radius,
            inner_radius,
            max_vx,
            alpha,
        )
        total_duration = cumulative[-1]
        t = np.linspace(0.0, total_duration, num_points)
        x, y, z_values, yaw = self._sample_at_times(
            t,
            starts,
            deltas,
            lengths,
            headings,
            durations,
            cumulative,
            max_vx,
            alpha,
            z,
        )

        return t, x, y, z_values, yaw

    def generate_to_file(self, path, num_points=None, num_segments=None):
        from flexible_drones_tools.trajectories.utilities.split_trajectory import SegmentFit

        num_points = 500 if num_points is None else num_points
        num_points = self._validate_num_points(num_points)
        starts, deltas, lengths, headings, durations, cumulative = self._geometry(
            self.num_tips,
            self.outer_radius,
            self.inner_radius,
            self.max_vx,
            self.alpha,
        )
        boundaries = self._segment_boundaries(
            cumulative,
            lengths,
            self.max_vx,
            self.alpha,
        )
        calculated_segments = len(boundaries) - 1
        requested_segments = calculated_segments
        if num_segments is not None:
            normalized_segments = int(num_segments)
            if normalized_segments != num_segments:
                raise ValueError('num_segments must be an integer')
            if normalized_segments < calculated_segments:
                raise ValueError(
                    f'num_segments must be at least {calculated_segments} for the current '
                    'star corner/straight segmentation'
                )
            requested_segments = normalized_segments
        boundaries = self._refine_boundaries(boundaries, requested_segments)
        if num_points < requested_segments * MIN_POLYFIT_POINTS_PER_SEGMENT:
            raise ValueError(
                'num_points must be at least 8 times the calculated star segment count '
                f'({requested_segments}) for 7th-degree polynomial fitting'
            )

        sample_counts = self._sample_counts_for_segments(
            num_points,
            requested_segments,
            self.alpha,
            self.corner_sample_multiplier,
        )
        t = np.unique(np.concatenate([
            np.linspace(start_time, end_time, sample_count)
            for start_time, end_time, sample_count
            in zip(boundaries[:-1], boundaries[1:], sample_counts)
        ]))
        x, y, z_values, yaw = self._sample_at_times(
            t,
            starts,
            deltas,
            lengths,
            headings,
            durations,
            cumulative,
            self.max_vx,
            self.alpha,
            self.z,
        )
        fit = SegmentFit(t, x, y, z_values, yaw)
        segments, segment_durations = self._segments_from_boundaries(
            t,
            x,
            y,
            z_values,
            yaw,
            boundaries,
        )
        drone_format = fit.polyfit_traj_for_translating(
            segment_durations,
            segments,
            write=True,
            path=str(path),
        )
        drone_format.write_to_csv()
        return path


def main(args=None):
    import argparse

    from flexible_drones_tools.trajectories.generators import default_output_path

    parser = argparse.ArgumentParser(
        description='Generate a star trajectory CSV (drone coefficient order).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'path',
        nargs='?',
        default=None,
        help='Output CSV path (default: generators/output/star_trajectory_ro<outer>m_ri<inner>m_z<z>m.csv).',
    )
    parser.add_argument('--num-points', type=int, default=500,
                        help='Number of sampled points to generate before polynomial fitting.')
    parser.add_argument('--num-segments', type=int, default=None,
                        help=('Polynomial segment count; omit to calculate from star corners '
                              'and straights, or exceed it to subdivide pieces.'))
    parser.add_argument('--num-tips', type=int, default=DEFAULT_NUM_TIPS,
                        help='Number of star tips.')
    parser.add_argument('--outer-radius', type=float, default=DEFAULT_OUTER_RADIUS,
                        help='Outer star radius in meters.')
    parser.add_argument('--inner-radius', type=float, default=DEFAULT_INNER_RADIUS,
                        help='Inner star radius in meters.')
    parser.add_argument('--z', type=float, default=DEFAULT_Z,
                        help='Constant trajectory height in meters.')
    parser.add_argument('--vx', '--max-vx', dest='max_vx', type=float,
                        default=DEFAULT_MAX_VX,
                        help='Nominal velocity on the straight sections in meters per second.')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help='Fraction of each straight section used for accel/decel at each end.')
    parser.add_argument('--corner-sample-multiplier', type=float,
                        default=DEFAULT_CORNER_SAMPLE_MULTIPLIER,
                        help='Relative sample density for interior corner transition pieces.')
    opts = parser.parse_args(args)
    if opts.num_tips < 2:
        parser.error('--num-tips must be at least 2')
    if opts.outer_radius <= 0.0:
        parser.error('--outer-radius must be greater than 0.0')
    if opts.inner_radius <= 0.0:
        parser.error('--inner-radius must be greater than 0.0')
    if opts.inner_radius >= opts.outer_radius:
        parser.error('--inner-radius must be less than --outer-radius')
    if opts.max_vx <= 0.0:
        parser.error('--vx must be greater than 0.0')
    if opts.alpha < 0.0 or opts.alpha >= 0.5:
        parser.error('--alpha must be greater than or equal to 0.0 and less than 0.5')
    if opts.corner_sample_multiplier < 1.0:
        parser.error('--corner-sample-multiplier must be greater than or equal to 1.0')
    if opts.num_points < 2:
        parser.error('--num-points must be at least 2')
    generator = GenerateStarTrajectory(
        num_tips=opts.num_tips,
        outer_radius=opts.outer_radius,
        inner_radius=opts.inner_radius,
        z=opts.z,
        max_vx=opts.max_vx,
        alpha=opts.alpha,
        corner_sample_multiplier=opts.corner_sample_multiplier,
    )
    calculated_segments = generator.calculated_num_segments()
    if opts.num_segments is not None and opts.num_segments < calculated_segments:
        parser.error(
            f'--num-segments must be at least {calculated_segments} for the current '
            'star corner/straight segmentation; omit it to calculate automatically'
        )
    requested_segments = calculated_segments if opts.num_segments is None else opts.num_segments
    if opts.num_points < requested_segments * MIN_POLYFIT_POINTS_PER_SEGMENT:
        parser.error(
            '--num-points must be at least 8 times the calculated star segment count '
            f'({requested_segments}) for 7th-degree polynomial fitting'
        )
    path = opts.path or default_output_path(
        star_filename(opts.outer_radius, opts.inner_radius, opts.z)
    )
    out = generator.generate_to_file(
        path, num_points=opts.num_points, num_segments=opts.num_segments)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

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
Reusable matplotlib plotting helpers for trajectories.

Consolidates the 3D / 2D-projection / vs-time plotting that previously lived in
the generators and ``PolyOrderTrajectory``. ``matplotlib`` is imported lazily so
importing this module stays cheap and ROS-safe; call ``plt.show()`` yourself
after building the figures.
"""

import numpy as np


MIN_POSITION_AXIS_SPAN = 0.2


def _heading_from_vxvy(t, vx, vy, yaw):
    """Calculate heading from sampled xy velocity and align it to yaw wrapping."""
    t = np.asarray(t, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    yaw = np.unwrap(np.asarray(yaw, dtype=float))

    heading = np.arctan2(vy, vx)
    turns = np.round((yaw - heading) / (2.0 * np.pi))
    return heading + turns * 2.0 * np.pi


def _finite_extent(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.min(values)), float(np.max(values))


def _expand_limits_for_extent(limits, extent, *, pad_fraction=0.05):
    if extent is None:
        return limits

    lower, upper = float(limits[0]), float(limits[1])
    data_lower, data_upper = extent
    span = max(data_upper - data_lower, upper - lower)
    padding = span * pad_fraction
    return min(lower, data_lower - padding), max(upper, data_upper + padding)


def _limits_with_minimum_span(values, *, minimum_span=MIN_POSITION_AXIS_SPAN, pad_fraction=0.05):
    extent = _finite_extent(values)
    if extent is None:
        half_span = minimum_span / 2.0
        return -half_span, half_span

    lower, upper = extent
    data_span = upper - lower
    span = minimum_span if data_span <= 0.0 else max(data_span * (1.0 + 2.0 * pad_fraction), minimum_span)
    center = (lower + upper) / 2.0
    half_span = span / 2.0
    return center - half_span, center + half_span


def _apply_minimum_position_axis_span(ax, *, x_values=None, y_values=None):
    if x_values is not None:
        ax.set_xlim(_limits_with_minimum_span(x_values))
    if y_values is not None:
        ax.set_ylim(_limits_with_minimum_span(y_values))


def set_equal_3d_bounds(
    ax,
    x,
    y,
    z,
    *,
    xy_limits=(-4.0, 4.0),
    z_limits=(0.0, 3.0),
):
    """
    Apply data-aware 3D limits with equal rendered units.

    The default view is 8 m by 8 m in X/Y and 0-3 m in Z. Limits expand only
    when plotted data falls outside that default.
    """
    xlim = _expand_limits_for_extent(xy_limits, _finite_extent(x))
    ylim = _expand_limits_for_extent(xy_limits, _finite_extent(y))
    zlim = _expand_limits_for_extent(z_limits, _finite_extent(z))

    xy_span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    def centered_limits(limits, span):
        center = (limits[0] + limits[1]) / 2.0
        half_span = span / 2.0
        return center - half_span, center + half_span

    xlim = centered_limits(xlim, xy_span)
    ylim = centered_limits(ylim, xy_span)
    z_span = zlim[1] - zlim[0]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_box_aspect((xy_span, xy_span, z_span))
    return xlim, ylim, zlim


class TrajectoryPlotter:
    """Static matplotlib helpers for plotting trajectory shapes and kinematics."""

    @staticmethod
    def line_plot(a, b, *, xlabel, ylabel, title, color=None, ax=None):
        """Plot ``b`` versus ``a`` on a new (or supplied) 2D axis."""
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots()
        ax.plot(a, b, linewidth=2, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, ls=':', lw=0.6)
        return ax

    @staticmethod
    def plot_path_3d(x, y, z, *, title='3D trajectory', ax=None):
        """Plot a 3D position path."""
        import matplotlib.pyplot as plt

        if ax is None:
            _fig = plt.figure()
            try:
                ax = _fig.add_subplot(projection='3d')
            except Exception:
                plt.close(_fig)  # don't leak an empty figure if 3D is unavailable
                raise
        ax.plot(x, y, z)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(title)
        set_equal_3d_bounds(ax, x, y, z)
        return ax

    @classmethod
    def plot_projections(cls, x, y, z, *, title='2D projections'):
        """Plot xy, xz, and yz position projections side by side."""
        import matplotlib.pyplot as plt

        fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(13, 4))
        cls.line_plot(x, y, xlabel='X [m]', ylabel='Y [m]', title='XY', ax=ax_xy)
        cls.line_plot(x, z, xlabel='X [m]', ylabel='Z [m]', title='XZ', ax=ax_xz)
        cls.line_plot(y, z, xlabel='Y [m]', ylabel='Z [m]', title='YZ', ax=ax_yz)
        _apply_minimum_position_axis_span(ax_xy, x_values=x, y_values=y)
        _apply_minimum_position_axis_span(ax_xz, x_values=x, y_values=z)
        _apply_minimum_position_axis_span(ax_yz, x_values=y, y_values=z)
        for ax in (ax_xy, ax_xz, ax_yz):
            ax.set_aspect('equal', 'box')
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    @classmethod
    def plot_kinematics(
        cls,
        t,
        pos,
        vel,
        acc,
        *,
        jerk=None,
        snap=None,
        heading=None,
        axis_labels=('X', 'Y', 'Z'),
        axis_units=None,
        title='Kinematics vs time',
    ):
        """
        Plot position through snap versus time.

        ``pos``/``vel``/``acc``/``jerk``/``snap`` are arrays shaped ``(N, k)`` with
        one column per axis (only the first ``len(axis_labels)`` columns are
        plotted). Use ``axis_units`` to distinguish mixed linear/angular
        components such as ``('m', 'm', 'm', 'rad')`` for ``[x, y, z, yaw]``.
        ``heading`` may be supplied as a yaw-aligned angle series to overlay on
        the yaw position plot.
        """
        import matplotlib.pyplot as plt

        def unit_for(axis_index):
            if axis_units is None:
                return 'm' if axis_index < 3 else 'rad'
            return axis_units[axis_index]

        def ylabel(quantity, axis_index):
            unit = unit_for(axis_index)
            denominators = {
                'Position': '',
                'Velocity': '/s',
                'Acceleration': '/s^2',
                'Jerk': '/s^3',
                'Snap': '/s^4',
            }
            return f'{quantity} [{unit}{denominators[quantity]}]'

        def values_for_plot(quantity, values, axis_index):
            series = values[:, axis_index]
            if quantity == 'Position' and str(axis_labels[axis_index]).lower() == 'yaw':
                return np.unwrap(series)
            return series

        rows = [
            ('Position', pos),
            ('Velocity', vel),
            ('Acceleration', acc),
        ]
        if jerk is not None:
            rows.append(('Jerk', jerk))
        if snap is not None:
            rows.append(('Snap', snap))

        ncols = len(axis_labels)
        fig_height = 2.6 * len(rows) + 1.0
        fig, axes = plt.subplots(
            len(rows),
            ncols,
            figsize=(4 * ncols, fig_height),
            sharex=True,
            squeeze=False,
        )
        for r, (quantity, values) in enumerate(rows):
            values = np.asarray(values)
            for c in range(ncols):
                ax = axes[r][c]
                ax.plot(t, values_for_plot(quantity, values, c), linewidth=1.5)
                if (
                    heading is not None
                    and quantity == 'Position'
                    and str(axis_labels[c]).lower() == 'yaw'
                ):
                    ax.plot(t, heading, '--', linewidth=1.5, label='heading')
                    ax.legend()
                ax.grid(True, ls=':', lw=0.6)
                if r == 0:
                    ax.set_title(axis_labels[c])
                ax.set_ylabel(ylabel(quantity, c))
                if quantity == 'Position' and unit_for(c) == 'm':
                    _apply_minimum_position_axis_span(
                        ax,
                        y_values=values_for_plot(quantity, values, c),
                    )
                if r == len(rows) - 1:
                    ax.set_xlabel('Time [s]')
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_component_grid(times, rows, *, titles, suptitle='', figsize=(11, 10)):
        """
        Plot a grid of component time histories: one subplot row per quantity.

        ``rows`` is a sequence of arrays shaped ``(ncols, N)`` (one row per
        quantity, e.g. velocity/accel/jerk/snap); ``titles`` is a matching
        ``[row][col]`` list of subplot titles. Returns the figure.
        """
        import matplotlib.pyplot as plt

        nrows = len(rows)
        ncols = len(titles[0])
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.25)
        for i in range(nrows):
            data = np.asarray(rows[i])
            for j in range(ncols):
                sub = fig.add_subplot(grid[i, j])
                sub.plot(times, data[j, :], '-', lw=1.5)
                sub.set_title(titles[i][j])
                sub.grid(True, ls=':', lw=0.6)
        if suptitle:
            fig.suptitle(suptitle, y=0.99)
        return fig

    @staticmethod
    def sample_trajectory(trajectory, samples=400):
        """
        Evaluate a trajectory over its full duration.

        Returns ``(t, pos, vel, acc)`` where ``t`` is shape ``(N,)`` and the
        others are ``(N, 4)`` arrays of ``[x, y, z, yaw]`` (rows past the end of
        the trajectory are dropped).
        """
        total = trajectory.total_duration()
        t = np.linspace(0.0, total, int(samples))
        ts, pos, vel, acc = [], [], [], []
        for ti in t:
            p, v, a = trajectory.evaluate(float(ti))
            if p is None:
                continue
            ts.append(ti)
            pos.append(p)
            vel.append(v)
            acc.append(a)
        return np.asarray(ts), np.asarray(pos), np.asarray(vel), np.asarray(acc)

    @staticmethod
    def _evaluate_axis_derivative(coefficients, local_t, order):
        return np.polyval(np.polyder(coefficients, order), local_t)

    @classmethod
    def _evaluate_high_derivatives(cls, trajectory, segment_index, local_t):
        coeff_sets = (
            trajectory.x_coeffs[segment_index],
            trajectory.y_coeffs[segment_index],
            trajectory.z_coeffs[segment_index],
            trajectory.yaw_coeffs[segment_index],
        )
        jerk = np.array([
            cls._evaluate_axis_derivative(coefficients, local_t, 3)
            for coefficients in coeff_sets
        ])
        snap = np.array([
            cls._evaluate_axis_derivative(coefficients, local_t, 4)
            for coefficients in coeff_sets
        ])
        return jerk, snap

    @classmethod
    def sample_trajectory_with_high_derivatives(cls, trajectory, samples=400):
        """
        Evaluate a trajectory over its full duration through snap.

        Returns ``(t, pos, vel, acc, jerk, snap)`` where ``t`` is shape ``(N,)``
        and the others are ``(N, 4)`` arrays of ``[x, y, z, yaw]``.
        """
        total = trajectory.total_duration()
        t = np.linspace(0.0, total, int(samples))
        ts, pos, vel, acc, jerk, snap = [], [], [], [], [], []
        for ti in t:
            p, v, a = trajectory.evaluate(float(ti))
            if p is None:
                continue
            segment_index = trajectory.last_segment
            local_t = float(ti) - float(trajectory.end_times[segment_index])
            j, s = cls._evaluate_high_derivatives(trajectory, segment_index, local_t)
            ts.append(ti)
            pos.append(p)
            vel.append(v)
            acc.append(a)
            jerk.append(j)
            snap.append(s)
        return (
            np.asarray(ts),
            np.asarray(pos),
            np.asarray(vel),
            np.asarray(acc),
            np.asarray(jerk),
            np.asarray(snap),
        )

    @classmethod
    def plot_trajectory(cls, trajectory, *, samples=400, title=''):
        """Sample a trajectory and produce 3D, projection, and kinematics figures."""
        t, pos, vel, acc, jerk, snap = cls.sample_trajectory_with_high_derivatives(
            trajectory, samples)
        heading = _heading_from_vxvy(t, vel[:, 0], vel[:, 1], pos[:, 3])
        suffix = f' - {title}' if title else ''
        try:
            cls.plot_path_3d(pos[:, 0], pos[:, 1], pos[:, 2], title=f'3D trajectory{suffix}')
        except Exception as exc:  # 3D projection can be unavailable on broken matplotlib installs
            print(f'Skipping 3D plot (matplotlib 3D unavailable): {exc}', flush=True)
        cls.plot_projections(pos[:, 0], pos[:, 1], pos[:, 2], title=f'2D projections{suffix}')
        cls.plot_kinematics(
            t,
            pos,
            vel,
            acc,
            jerk=jerk,
            snap=snap,
            heading=heading,
            axis_labels=('X', 'Y', 'Z', 'Yaw'),
            axis_units=('m', 'm', 'm', 'rad'),
            title=f'Kinematics vs time{suffix}',
        )
        return t, pos, vel, acc

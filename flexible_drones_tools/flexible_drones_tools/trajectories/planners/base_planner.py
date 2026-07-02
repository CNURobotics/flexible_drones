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
Common interface for trajectory planners.

A planner produces a single start->goal trajectory of ``n_segments`` polynomial
pieces. Multi-waypoint paths are stitched manually by the caller. The flight
volume (``bounds``) and the kinematic ``limits`` are supplied as plain
dictionaries so they can be set from ROS parameters or services and serialized
across a process boundary.
"""

from abc import ABC, abstractmethod

# Defaults preserve the historical hard-coded values from the original optimizer.
# yaw_rate_max / yaw_accel_max are conservative *policy* allowances (rad/s, rad/s^2)
# set below the airframe's capability for safety and to keep OptiTrack from losing
# motion-capture lock during fast yaw; tune them up per drone in the limits YAML.
DEFAULT_LIMITS = {
    'v_max': 0.75, 'a_max': 4.0, 'j_max': 10.0, 's_max': 50.0,
    'yaw_rate_max': 2.0, 'yaw_accel_max': 4.0,
}
DEFAULT_BOUNDS = {
    'x_min': -3.0, 'x_max': 3.0,
    'y_min': -3.0, 'y_max': 3.0,
    'z_min': 0.25, 'z_max': 2.5,
}


def point_xyz(value):
    """Read (x, y, z) from a dict ``{'x','y','z'}`` or an object with attributes."""
    if isinstance(value, dict):
        return float(value['x']), float(value['y']), float(value['z'])
    return float(value.x), float(value.y), float(value.z)


def merge_limits(limits):
    merged = dict(DEFAULT_LIMITS)
    if limits:
        merged.update({k: float(v) for k, v in limits.items() if v is not None})
    return merged


def merge_bounds(bounds):
    merged = dict(DEFAULT_BOUNDS)
    if bounds:
        merged.update({k: float(v) for k, v in bounds.items() if v is not None})
    return merged


class TrajectoryPlanner(ABC):
    """Base class: plan a start->goal trajectory as ``n_segments`` polynomial pieces."""

    def __init__(self, limits=None, bounds=None, n_segments=1):
        self.limits = merge_limits(limits)
        self.bounds = merge_bounds(bounds)
        self.n_segments = int(n_segments)

    def resolve(self, bounds, limits, n_segments):
        """Merge per-call overrides over the instance defaults."""
        eff_bounds = merge_bounds(self.bounds)
        if bounds:
            eff_bounds.update({k: float(v) for k, v in bounds.items() if v is not None})
        eff_limits = merge_limits(self.limits)
        if limits:
            eff_limits.update({k: float(v) for k, v in limits.items() if v is not None})
        eff_segments = int(n_segments) if n_segments else self.n_segments
        return eff_bounds, eff_limits, eff_segments

    def build_seed(self, start, goal, obstacles=None, bounds=None, limits=None, n_segments=None):
        """
        Optionally build deterministic initial segments for ``plan``.

        Planners that do not use an explicit initial guess can return ``None``.
        Implementations should avoid expensive solver construction here so callers
        may run seed generation outside an optimizer subprocess.
        """
        return None

    def build_base_seed(self, start, goal, obstacles=None, bounds=None, limits=None, n_segments=None):
        """
        Build the non-optimizer seed used to initialize staged planners.

        Most planners have only one seed builder, so the default delegates to
        ``build_seed``. Planners with an optimizer-backed seed strategy can
        override this to expose the cheaper deterministic initializer.
        """
        return self.build_seed(start, goal, obstacles, bounds, limits, n_segments)

    @abstractmethod
    def plan(
        self,
        start,
        goal,
        obstacles=None,
        bounds=None,
        limits=None,
        n_segments=None,
        initial_segments=None,
    ):
        """
        Plan a trajectory from ``start`` to ``goal``.

        ``start`` / ``goal`` are dicts ``{'position': {x,y,z}, 'velocity': {x,y,z}}``
        (objects with matching attributes are also accepted).
        ``obstacles`` is a list of ``{'type','x','y','radius','height'}`` dicts and
        may be empty/None. ``bounds`` and ``limits`` override the instance defaults.
        ``initial_segments`` may provide a deterministic seed from ``build_seed``.

        Returns a list of segment dicts ``{'T', 'c_x', 'c_y', 'c_z'}`` with
        coefficients in numpy ascending-power order.
        """
        raise NotImplementedError

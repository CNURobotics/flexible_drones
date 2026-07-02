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
Obstacle-free minimum-snap planner.

Solves a closed-form two-segment minimum-snap trajectory (per axis, via KKT)
between ``start`` and ``goal`` with the requested endpoint velocities and zero
endpoint acceleration/jerk/snap. Obstacles are ignored; ``bounds`` are not
enforced as hard constraints (documented limitation -- use the CasADi planner
when the flight volume or obstacles must be respected).

The planner always emits two polynomial segments; ``n_segments`` is accepted for
interface compatibility but does not change the segment count.
"""

import numpy as np

from flexible_drones_tools.trajectories.planners.base_planner import (
    TrajectoryPlanner,
    point_xyz,
)
from flexible_drones_tools.trajectories.utilities.kkt import (
    REG,
    solve_axis_inside_corner_KKT,
)


class KktPlanner(TrajectoryPlanner):
    """Closed-form two-segment minimum-snap planner (no obstacle avoidance)."""

    def __init__(self, limits=None, bounds=None, n_segments=2,
                 rho=0.5, min_total_time=0.5, cruise_fraction=1.0, reg=REG):
        super().__init__(limits=limits, bounds=bounds, n_segments=n_segments)
        self.rho = float(rho)
        self.min_total_time = float(min_total_time)
        self.cruise_fraction = float(cruise_fraction)
        self.reg = float(reg)

    def _total_time(self, start_pos, goal_pos, v_max):
        distance = float(np.linalg.norm(np.asarray(goal_pos) - np.asarray(start_pos)))
        cruise = max(self.cruise_fraction * v_max, 1e-6)
        return max(distance / cruise, self.min_total_time)

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
        _eff_bounds, eff_limits, _eff_segments = self.resolve(bounds, limits, n_segments)

        start_pos = np.array(point_xyz(start['position'] if isinstance(start, dict) else start.position))
        start_vel = np.array(point_xyz(start['velocity'] if isinstance(start, dict) else start.velocity))
        goal_pos = np.array(point_xyz(goal['position'] if isinstance(goal, dict) else goal.position))
        goal_vel = np.array(point_xyz(goal['velocity'] if isinstance(goal, dict) else goal.velocity))

        total_time = self._total_time(start_pos, goal_pos, eff_limits['v_max'])
        T1 = self.rho * total_time
        T2 = (1.0 - self.rho) * total_time

        seg1 = {'T': T1}
        seg2 = {'T': T2}
        for ndx, axis in enumerate(('c_x', 'c_y', 'c_z')):
            c1, c2, _cost = solve_axis_inside_corner_KKT(
                start_pos[ndx], goal_pos[ndx],
                start_vel[ndx], goal_vel[ndx],
                T1, T2, reg=self.reg,
            )
            seg1[axis] = np.asarray(c1, dtype=np.float64)
            seg2[axis] = np.asarray(c2, dtype=np.float64)

        return [seg1, seg2]


if __name__ == '__main__':
    planner = KktPlanner()
    demo_start = {'position': {'x': 0.0, 'y': 0.0, 'z': 1.0},
                  'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0}}
    demo_goal = {'position': {'x': 2.0, 'y': 1.0, 'z': 1.5},
                 'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0}}
    for piece in planner.plan(demo_start, demo_goal):
        print(piece)

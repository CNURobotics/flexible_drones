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

"""Unit tests for PolyOrderTrajectory."""

import numpy as np

from flexible_drones_tools.trajectories.utilities.poly_order_trajectory import PolyOrderTrajectory
from flexible_drones_tools.trajectories.utilities.plotting import TrajectoryPlotter


def _two_segment_traj():
    return PolyOrderTrajectory(
        durations=[2.0, 3.0],
        x_coeffs=[np.array([0.0]), np.array([1.0])],
        y_coeffs=[np.array([0.0]), np.array([0.0])],
        z_coeffs=[np.array([0.0]), np.array([0.0])],
        yaw_coeffs=[np.array([0.0]), np.array([0.0])],
    )


def test_end_times_have_one_boundary_per_segment_plus_start():
    traj = _two_segment_traj()

    assert traj.end_times == [0.0, 2.0, 5.0]
    assert len(traj.end_times) == len(traj.durations) + 1
    assert traj.total_duration() == 5.0


def test_evaluate_exact_end_uses_final_segment_endpoint():
    traj = _two_segment_traj()

    pos, vel, acc = traj.evaluate(traj.total_duration())

    assert pos is not None
    assert pos[0] == 1.0
    assert traj.last_segment == 1


def test_trajectory_plotter_includes_yaw_axis(monkeypatch):
    traj = _two_segment_traj()
    captured = {}

    def fake_path_3d(*_args, **_kwargs):
        return None

    def fake_projections(*_args, **_kwargs):
        return None

    def fake_kinematics(_t, _pos, _vel, _acc, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(TrajectoryPlotter, 'plot_path_3d', fake_path_3d)
    monkeypatch.setattr(TrajectoryPlotter, 'plot_projections', fake_projections)
    monkeypatch.setattr(TrajectoryPlotter, 'plot_kinematics', fake_kinematics)

    TrajectoryPlotter.plot_trajectory(traj, samples=5)

    assert captured['axis_labels'] == ('X', 'Y', 'Z', 'Yaw')
    assert captured['axis_units'] == ('m', 'm', 'm', 'rad')
    assert captured['jerk'].shape == (5, 4)
    assert captured['snap'].shape == (5, 4)


def test_trajectory_plotter_samples_jerk_and_snap_from_coefficients():
    traj = PolyOrderTrajectory(
        durations=[1.0],
        x_coeffs=[np.array([1.0, 0.0, 0.0, 0.0, 0.0])],
        y_coeffs=[np.array([0.0])],
        z_coeffs=[np.array([0.0])],
        yaw_coeffs=[np.array([0.0])],
    )

    t, _pos, _vel, _acc, jerk, snap = (
        TrajectoryPlotter.sample_trajectory_with_high_derivatives(traj, samples=3))

    np.testing.assert_allclose(t, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(jerk[:, 0], [0.0, 12.0, 24.0])
    np.testing.assert_allclose(snap[:, 0], [24.0, 24.0, 24.0])
    np.testing.assert_allclose(jerk[:, 1:], 0.0)
    np.testing.assert_allclose(snap[:, 1:], 0.0)

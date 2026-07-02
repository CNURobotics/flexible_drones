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

import importlib.util
from pathlib import Path

import numpy as np
import pytest


GENERATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / 'flexible_drones_tools'
    / 'trajectories'
    / 'generators'
    / 'generate_figure8_trajectory.py'
)
SPEC = importlib.util.spec_from_file_location('source_generate_figure8_trajectory', GENERATOR_PATH)
generate_figure8_trajectory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_figure8_trajectory)


def test_figure8_filename_encodes_radius_to_two_decimal_places():
    assert generate_figure8_trajectory.figure8_filename(1.0) == 'figure8_trajectory_1_00m.csv'
    assert generate_figure8_trajectory.figure8_filename(0.5) == 'figure8_trajectory_0_50m.csv'
    assert generate_figure8_trajectory.figure8_filename(1.25) == 'figure8_trajectory_1_25m.csv'
    assert generate_figure8_trajectory.figure8_filename(1.234) == 'figure8_trajectory_1_23m.csv'


def test_default_figure8_output_path_uses_script_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    script_dir = tmp_path / 'source_generators'
    script_dir.mkdir()
    monkeypatch.setattr(
        generate_figure8_trajectory,
        '__file__',
        str(script_dir / 'generate_figure8_trajectory.py'),
    )

    assert generate_figure8_trajectory.default_figure8_output_path('figure8_trajectory_1_50m.csv') == (
        script_dir / 'output' / 'figure8_trajectory_1_50m.csv'
    )
    assert (script_dir / 'output').is_dir()


def test_figure8_radius_scales_generated_points():
    _, x, y, _, _ = generate_figure8_trajectory.GenerateFigure8Trajectory(
        radius=1.5,
    ).generate_trajectory(num_points=501)

    assert np.max(x) == pytest.approx(1.5, abs=1e-3)
    assert np.min(x) == pytest.approx(-1.5, abs=1e-3)
    assert np.max(y) == pytest.approx(0.75, abs=1e-3)
    assert np.min(y) == pytest.approx(-0.75, abs=1e-3)


def test_figure8_default_duration_is_20_seconds():
    t, _, _, _, _ = generate_figure8_trajectory.GenerateFigure8Trajectory().generate_trajectory()

    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(20.0)


def test_figure8_duration_controls_generated_time():
    t, _, _, _, _ = generate_figure8_trajectory.GenerateFigure8Trajectory(
        duration=12.5,
    ).generate_trajectory()

    assert t[-1] == pytest.approx(12.5)


def test_figure8_uses_symmetric_accel_and_decel_ramps():
    duration = 20.0
    ramp_duration = 2.0
    t = np.linspace(0.0, duration, 1001)
    theta = generate_figure8_trajectory.GenerateFigure8Trajectory._theta_from_time(
        t,
        duration,
        ramp_duration,
    )
    theta_dot = np.gradient(theta, t)

    assert theta[0] == pytest.approx(0.0)
    assert theta[-1] == pytest.approx(2 * np.pi)
    assert theta_dot[0] < theta_dot[len(theta_dot) // 2]
    assert theta_dot[-1] < theta_dot[len(theta_dot) // 2]
    assert theta_dot[0] == pytest.approx(theta_dot[-1], rel=1e-6)

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

import pytest

from flexible_drones_tools.trajectories.utilities.io import (
    AXES,
    COEFFICIENT_COUNT,
    load_trajectory_csv,
    save_trajectory_csv,
)


def _valid_row(duration=1.0):
    coeffs = [0.0] * COEFFICIENT_COUNT
    return [str(duration)] + [str(v) for v in coeffs] * len(AXES)


def _write_csv(tmp_path, rows, header=True):
    path = tmp_path / 'trajectory.csv'
    lines = []
    if header:
        columns = ['duration']
        for axis in AXES:
            columns.extend(f'{axis}^{i}' for i in range(COEFFICIENT_COUNT))
        lines.append(','.join(columns))
    lines.extend(','.join(row) for row in rows)
    path.write_text('\n'.join(lines) + '\n')
    return path


def test_load_trajectory_csv_round_trip(tmp_path):
    path = tmp_path / 'round_trip.csv'
    durations = [1.0, 2.0]
    coeffs = [[[float(i)] * COEFFICIENT_COUNT for i in range(len(durations))] for _ in AXES]
    save_trajectory_csv(path, durations, *coeffs)

    loaded_durations, *loaded_coeffs = load_trajectory_csv(path)
    assert loaded_durations == durations
    assert loaded_coeffs == coeffs


def test_load_trajectory_csv_rejects_zero_duration(tmp_path):
    path = _write_csv(tmp_path, [_valid_row(duration=0.0)])
    with pytest.raises(ValueError, match='invalid duration'):
        load_trajectory_csv(path)


def test_load_trajectory_csv_rejects_negative_duration(tmp_path):
    path = _write_csv(tmp_path, [_valid_row(duration=-1.0)])
    with pytest.raises(ValueError, match='invalid duration'):
        load_trajectory_csv(path)


def test_load_trajectory_csv_rejects_nan_coefficient(tmp_path):
    row = _valid_row()
    row[3] = 'nan'
    path = _write_csv(tmp_path, [row])
    with pytest.raises(ValueError, match='non-finite'):
        load_trajectory_csv(path)


def test_load_trajectory_csv_rejects_inf_coefficient(tmp_path):
    row = _valid_row()
    row[3] = 'inf'
    path = _write_csv(tmp_path, [row])
    with pytest.raises(ValueError, match='non-finite'):
        load_trajectory_csv(path)


def test_load_trajectory_csv_rejects_truncated_row_no_header(tmp_path):
    row = _valid_row()[:-1]
    path = _write_csv(tmp_path, [row], header=False)
    with pytest.raises(ValueError, match='columns'):
        load_trajectory_csv(path)


def test_load_trajectory_csv_rejects_extra_columns_no_header(tmp_path):
    row = _valid_row() + ['0.0']
    path = _write_csv(tmp_path, [row], header=False)
    with pytest.raises(ValueError, match='columns'):
        load_trajectory_csv(path)


def test_load_trajectory_csv_rejects_extra_columns_with_header(tmp_path):
    row = _valid_row() + ['0.0']
    path = _write_csv(tmp_path, [row])
    with pytest.raises(ValueError, match='columns'):
        load_trajectory_csv(path)


def test_load_trajectory_csv_rejects_missing_header_column(tmp_path):
    path = tmp_path / 'missing_column.csv'
    path.write_text('duration,x^0\n1.0,0.0\n')
    with pytest.raises(ValueError, match='missing expected column'):
        load_trajectory_csv(path)

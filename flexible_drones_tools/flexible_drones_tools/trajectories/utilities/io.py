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

import csv
import math
import os
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

from flexible_drones_tools.trajectories.utilities.conversion import (
    coefficients_from_pieces,
    pieces_from_coefficients,
)


AXES = ('x', 'y', 'z', 'yaw')
COEFFICIENT_COUNT = 8


def normalize_path_value(path_value):
    """Expand user and environment variables in a path-like value."""
    return Path(os.path.expandvars(os.path.expanduser(str(path_value))))


def resolve_existing_file(path_value, extra_roots=None):
    path = normalize_path_value(path_value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)
        for root in extra_roots or ():
            root_path = normalize_path_value(root)
            candidates.append(root_path / path)
            candidates.append(root_path / 'src' / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f'File not found: {path_value}')


def resolve_package_file(package, folder, filename, extra_roots=None, package_share_lookup=None):
    candidates = []
    package_share_lookup = package_share_lookup or get_package_share_directory
    try:
        candidates.append(Path(package_share_lookup(package)) / folder / filename)
    except PackageNotFoundError:
        pass

    for root in extra_roots or ():
        candidates.append(normalize_path_value(root) / 'src' / package / folder / filename)
    candidates.append(Path.cwd() / package / folder / filename)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"File not found for package='{package}', folder='{folder}', filename='{filename}'."
    )


def default_workspace_roots():
    """Return source-tree search roots from $WORKSPACE_ROOT (the workspace convention)."""
    workspace_root = os.environ.get('WORKSPACE_ROOT')
    if workspace_root:
        return [normalize_path_value(workspace_root)]
    return []


def load_trajectory_csv(path, coefficient_order='drone'):
    durations = []
    x_coeffs = []
    y_coeffs = []
    z_coeffs = []
    yaw_coeffs = []

    with normalize_path_value(path).open(newline='') as csvfile:
        sample = csvfile.read(1024)
        csvfile.seek(0)
        has_header = 'duration' in sample.lower()

        if has_header:
            reader = csv.DictReader(csvfile)
            for line_number, row in enumerate(reader, start=2):
                duration, x_coeff, y_coeff, z_coeff, yaw_coeff = _parse_header_row(
                    row,
                    coefficient_order,
                    line_number,
                )
                durations.append(duration)
                x_coeffs.append(x_coeff)
                y_coeffs.append(y_coeff)
                z_coeffs.append(z_coeff)
                yaw_coeffs.append(yaw_coeff)
        else:
            reader = csv.reader(csvfile)
            for line_number, row in enumerate(reader, start=1):
                if not row:
                    continue
                duration, x_coeff, y_coeff, z_coeff, yaw_coeff = _parse_numeric_row(
                    row,
                    coefficient_order,
                    line_number,
                )
                durations.append(duration)
                x_coeffs.append(x_coeff)
                y_coeffs.append(y_coeff)
                z_coeffs.append(z_coeff)
                yaw_coeffs.append(yaw_coeff)

    if not durations:
        raise ValueError(f"Trajectory CSV '{path}' has no trajectory rows.")
    return durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs


def load_trajectory_pieces(path):
    durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs = load_trajectory_csv(
        path,
        coefficient_order='drone',
    )
    return pieces_from_coefficients(
        durations,
        x_coeffs,
        y_coeffs,
        z_coeffs,
        yaw_coeffs,
        coefficient_order='drone',
    )


def save_trajectory_csv(
    path,
    durations,
    x_coeffs,
    y_coeffs,
    z_coeffs,
    yaw_coeffs,
    coefficient_order='drone',
):
    path = normalize_path_value(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coeffs_by_axis = (x_coeffs, y_coeffs, z_coeffs, yaw_coeffs)

    with path.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['duration']
        for axis, coeffs in zip(AXES, coeffs_by_axis):
            coefficient_count = len(coeffs[0]) if coeffs else COEFFICIENT_COUNT
            header.extend(f'{axis}^{i}' for i in range(coefficient_count))
        writer.writerow(header)

        for index, duration in enumerate(durations):
            row = [f'{float(duration):.6f}']
            for coeffs in coeffs_by_axis:
                drone_order = _to_drone_order(coeffs[index], coefficient_order)
                row.extend(f'{float(value):.9f}' for value in drone_order)
            writer.writerow(row)


def save_trajectory_pieces(path, pieces):
    durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs = coefficients_from_pieces(
        pieces,
        coefficient_order='drone',
    )
    save_trajectory_csv(
        path,
        durations,
        x_coeffs,
        y_coeffs,
        z_coeffs,
        yaw_coeffs,
        coefficient_order='drone',
    )


def _parse_header_row(row, coefficient_order, line_number):
    extra_values = row.get(None)
    if extra_values:
        expected_columns = 1 + len(AXES) * COEFFICIENT_COUNT
        actual_columns = expected_columns + len(extra_values)
        raise ValueError(
            f'Trajectory CSV row {line_number} has {actual_columns} columns; '
            f'expected exactly {expected_columns}.'
        )

    try:
        duration = float(row['duration'])
        coeffs = []
        for axis in AXES:
            values = [float(row[f'{axis}^{i}']) for i in range(COEFFICIENT_COUNT)]
            coeffs.append(_from_drone_order(values, coefficient_order))
    except KeyError as exc:
        raise ValueError(
            f'Trajectory CSV row {line_number} is missing expected column {exc}.'
        ) from exc

    _validate_duration(duration, line_number)
    for axis, axis_coeffs in zip(AXES, coeffs):
        _validate_coefficients(axis_coeffs, axis, line_number)
    return duration, *coeffs


def _parse_numeric_row(row, coefficient_order, line_number):
    expected_columns = 1 + len(AXES) * COEFFICIENT_COUNT
    if len(row) != expected_columns:
        raise ValueError(
            f'Trajectory CSV row {line_number} has {len(row)} columns; '
            f'expected exactly {expected_columns}.'
        )
    values = [float(value) for value in row]

    duration = values[0]
    _validate_duration(duration, line_number)

    coeffs = []
    index = 1
    for axis in AXES:
        axis_coeffs = _from_drone_order(values[index:index + COEFFICIENT_COUNT], coefficient_order)
        _validate_coefficients(axis_coeffs, axis, line_number)
        coeffs.append(axis_coeffs)
        index += COEFFICIENT_COUNT
    return duration, *coeffs


def _validate_duration(duration, line_number):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            f'Trajectory CSV row {line_number} has an invalid duration ({duration}); '
            'must be a finite value greater than zero.'
        )


def _validate_coefficients(coefficients, axis, line_number):
    for value in coefficients:
        if not math.isfinite(value):
            raise ValueError(
                f"Trajectory CSV row {line_number} has a non-finite '{axis}' coefficient "
                f'({value}).'
            )


def _from_drone_order(coefficients, coefficient_order):
    if coefficient_order == 'drone':
        return [float(value) for value in coefficients]
    if coefficient_order == 'numpy':
        return [float(value) for value in reversed(coefficients)]
    raise ValueError(f"Unknown coefficient_order '{coefficient_order}'.")


def _to_drone_order(coefficients, coefficient_order):
    if coefficient_order == 'drone':
        return [float(value) for value in coefficients]
    if coefficient_order == 'numpy':
        return [float(value) for value in reversed(coefficients)]
    raise ValueError(f"Unknown coefficient_order '{coefficient_order}'.")

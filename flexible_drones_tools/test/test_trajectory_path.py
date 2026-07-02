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

from pathlib import Path

import pytest

from flexible_drones_tools.trajectories.utilities import trajectory_path as trajectory_path_module


def test_resolve_trajectory_path_direct_path(tmp_path: Path):
    csv_path = tmp_path / 'figure8.csv'
    csv_path.write_text('header\n', encoding='utf-8')

    resolved = trajectory_path_module.resolve_trajectory_path(
        trajectory_path=str(csv_path),
    )

    assert resolved == str(csv_path)


def test_resolve_trajectory_path_expands_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    csv_path = tmp_path / 'figure8.csv'
    csv_path.write_text('header\n', encoding='utf-8')
    monkeypatch.setenv('FD_TRAJECTORY_TEST_DIR', str(tmp_path))

    resolved = trajectory_path_module.resolve_trajectory_path(
        trajectory_path='$FD_TRAJECTORY_TEST_DIR/figure8.csv',
    )

    assert resolved == str(csv_path)


def test_resolve_trajectory_path_package_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    package_root = tmp_path / 'fake_pkg'
    trajectory_dir = package_root / 'trajectories'
    trajectory_dir.mkdir(parents=True)
    csv_path = trajectory_dir / 'figure8.csv'
    csv_path.write_text('header\n', encoding='utf-8')

    monkeypatch.setattr(
        trajectory_path_module,
        'get_package_share_directory',
        lambda package_name: str(package_root),
    )

    resolved = trajectory_path_module.resolve_trajectory_path(
        trajectory_package='fake_pkg',
        trajectory_folder='trajectories',
        trajectory_file='figure8.csv',
    )

    assert resolved == str(csv_path)


def test_resolve_trajectory_path_empty_package_uses_env_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    package_root = tmp_path / 'deployment_share'
    trajectory_dir = package_root / 'trajectories'
    trajectory_dir.mkdir(parents=True)
    csv_path = trajectory_dir / 'figure8.csv'
    csv_path.write_text('header\n', encoding='utf-8')
    monkeypatch.setenv('FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE', 'some_site_package')
    monkeypatch.setattr(
        trajectory_path_module,
        'get_package_share_directory',
        lambda package_name: str(package_root),
    )

    resolved = trajectory_path_module.resolve_trajectory_path(
        trajectory_file='figure8.csv',
    )

    assert resolved == str(csv_path)


def test_resolve_trajectory_path_empty_package_requires_folder(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv('FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE', raising=False)

    with pytest.raises(ValueError, match='trajectory_path'):
        trajectory_path_module.resolve_trajectory_path(
            trajectory_package='',
            trajectory_folder='',
            trajectory_file='figure8.csv',
        )


def test_resolve_trajectory_path_requires_explicit_source(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv('FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE', raising=False)

    with pytest.raises(ValueError, match='trajectory_path'):
        trajectory_path_module.resolve_trajectory_path()


def test_resolve_deployment_trajectory_path_direct_path(tmp_path: Path):
    csv_path = tmp_path / 'figure8.csv'
    csv_path.write_text('header\n', encoding='utf-8')

    resolved = trajectory_path_module.resolve_deployment_trajectory_path(str(csv_path))

    assert resolved == str(csv_path)


def test_resolve_deployment_trajectory_path_expands_env_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    csv_path = tmp_path / 'figure8.csv'
    csv_path.write_text('header\n', encoding='utf-8')
    monkeypatch.setenv('FD_TRAJECTORY_TEST_DIR', str(tmp_path))

    resolved = trajectory_path_module.resolve_deployment_trajectory_path(
        '$FD_TRAJECTORY_TEST_DIR/figure8.csv',
    )

    assert resolved == str(csv_path)


def test_resolve_deployment_trajectory_path_bare_filename_package_lookup(tmp_path: Path):
    package_root = tmp_path / 'deployment_share'
    trajectory_dir = package_root / 'trajectories'
    trajectory_dir.mkdir(parents=True)
    csv_path = trajectory_dir / 'figure8.csv'
    csv_path.write_text('header\n', encoding='utf-8')

    resolved = trajectory_path_module.resolve_deployment_trajectory_path(
        'figure8.csv',
        package='chris_serc_deployment',
        package_share_lookup=lambda package_name: str(package_root),
    )

    assert resolved == str(csv_path)


def test_resolve_deployment_trajectory_path_bare_filename_uses_env_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    package_root = tmp_path / 'deployment_share'
    trajectory_dir = package_root / 'trajectories'
    trajectory_dir.mkdir(parents=True)
    csv_path = trajectory_dir / 'figure8.csv'
    csv_path.write_text('header\n', encoding='utf-8')
    monkeypatch.setenv('FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE', 'some_site_package')

    captured = {}

    def _lookup(package_name):
        captured['package'] = package_name
        return str(package_root)

    resolved = trajectory_path_module.resolve_deployment_trajectory_path(
        'figure8.csv',
        package_share_lookup=_lookup,
    )

    assert resolved == str(csv_path)
    assert captured['package'] == 'some_site_package'


def test_resolve_deployment_trajectory_path_bare_filename_requires_package(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv('FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE', raising=False)

    with pytest.raises(ValueError, match='trajectory package is required'):
        trajectory_path_module.resolve_deployment_trajectory_path('figure8.csv')


def test_resolve_trajectory_write_path_explicit_dir(tmp_path: Path):
    output_dir = tmp_path / 'generated' / 'trajectories'

    resolved = trajectory_path_module.resolve_trajectory_write_path(
        trajectory_dir=str(output_dir),
        trajectory_file='saved.csv',
    )

    assert resolved == str(output_dir / 'saved.csv')
    assert output_dir.is_dir()


def test_resolve_trajectory_write_path_expands_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('FD_TRAJECTORY_OUTPUT_DIR', str(tmp_path / 'generated'))

    resolved = trajectory_path_module.resolve_trajectory_write_path(
        trajectory_dir='$FD_TRAJECTORY_OUTPUT_DIR/trajectories',
        trajectory_file='saved.csv',
    )

    assert resolved == str(tmp_path / 'generated' / 'trajectories' / 'saved.csv')
    assert (tmp_path / 'generated' / 'trajectories').is_dir()


def test_resolve_trajectory_write_path_package_fallback(tmp_path: Path):
    package_root = tmp_path / 'fake_pkg_share'

    resolved = trajectory_path_module.resolve_trajectory_write_path(
        trajectory_package='fake_pkg',
        trajectory_folder='trajectories',
        trajectory_file='saved.csv',
        package_share_lookup=lambda package_name: str(package_root),
    )

    assert resolved == str(package_root / 'trajectories' / 'saved.csv')
    assert (package_root / 'trajectories').is_dir()

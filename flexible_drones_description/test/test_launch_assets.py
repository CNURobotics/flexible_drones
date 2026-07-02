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

"""Unit tests for launch_assets model discovery."""

import warnings

import pytest

from flexible_drones_description.launch_assets import (
    GATE_MODEL_NAME,
    default_drone_model_xacro,
)

# Non-existent package name prevents ament share-directory lookup from
# injecting the real installed package as a second candidate root.
_ISOLATED_PKG = 'test-isolated-pkg'


def _make_model(root, name, model_root='models'):
    xacro = root / model_root / name / 'urdf' / f'{name}_model.urdf.xacro'
    xacro.parent.mkdir(parents=True, exist_ok=True)
    xacro.write_text('')
    return root


def test_discovers_single_valid_model(tmp_path):
    _make_model(tmp_path, 'alpha')
    result = default_drone_model_xacro(package_root=tmp_path, package_name=_ISOLATED_PKG)
    assert result == {'alpha': 'models/alpha/urdf/alpha_model.urdf.xacro'}


def test_discovers_multiple_models_in_sorted_order(tmp_path):
    for name in ('zeta', 'alpha', 'beta'):
        _make_model(tmp_path, name)
    result = default_drone_model_xacro(package_root=tmp_path, package_name=_ISOLATED_PKG)
    assert list(result.keys()) == ['alpha', 'beta', 'zeta']


def test_gates_excluded_by_default_without_warning(tmp_path):
    (tmp_path / 'models' / GATE_MODEL_NAME).mkdir(parents=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = default_drone_model_xacro(package_root=tmp_path, package_name=_ISOLATED_PKG)
    assert result == {}


def test_custom_excluded_names_suppress_warning(tmp_path):
    (tmp_path / 'models' / 'ignore_me').mkdir(parents=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = default_drone_model_xacro(
            package_root=tmp_path,
            package_name=_ISOLATED_PKG,
            excluded_model_names={'ignore_me'},
        )
    assert result == {}


def test_warns_about_directory_missing_xacro(tmp_path):
    (tmp_path / 'models' / 'orphan').mkdir(parents=True)
    with pytest.warns(UserWarning, match='orphan'):
        result = default_drone_model_xacro(package_root=tmp_path, package_name=_ISOLATED_PKG)
    assert result == {}


def test_warns_about_directory_with_wrong_xacro_name(tmp_path):
    wrong = tmp_path / 'models' / 'mytype' / 'urdf' / 'wrong_name.urdf.xacro'
    wrong.parent.mkdir(parents=True)
    wrong.write_text('')
    with pytest.warns(UserWarning, match='mytype'):
        result = default_drone_model_xacro(package_root=tmp_path, package_name=_ISOLATED_PKG)
    assert result == {}


def test_files_in_models_dir_are_silently_ignored(tmp_path):
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models' / 'not_a_dir.txt').write_text('')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = default_drone_model_xacro(package_root=tmp_path, package_name=_ISOLATED_PKG)
    assert result == {}


def test_nonexistent_models_dir_returns_empty(tmp_path):
    result = default_drone_model_xacro(
        package_root=tmp_path,
        package_name=_ISOLATED_PKG,
        model_root='nonexistent',
    )
    assert result == {}


def test_empty_models_dir_returns_empty(tmp_path):
    (tmp_path / 'models').mkdir()
    result = default_drone_model_xacro(package_root=tmp_path, package_name=_ISOLATED_PKG)
    assert result == {}


def test_excluded_and_valid_models_coexist_without_warning(tmp_path):
    _make_model(tmp_path, 'alpha')
    (tmp_path / 'models' / GATE_MODEL_NAME).mkdir(parents=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = default_drone_model_xacro(package_root=tmp_path, package_name=_ISOLATED_PKG)
    assert result == {'alpha': 'models/alpha/urdf/alpha_model.urdf.xacro'}


def test_warning_message_names_missing_xacro_path(tmp_path):
    (tmp_path / 'models' / 'newtype').mkdir(parents=True)
    with pytest.warns(UserWarning) as record:
        default_drone_model_xacro(package_root=tmp_path, package_name=_ISOLATED_PKG)
    msg = str(record[0].message)
    assert 'newtype' in msg
    assert 'models/newtype/urdf/newtype_model.urdf.xacro' in msg
    assert 'excluded_model_names' in msg

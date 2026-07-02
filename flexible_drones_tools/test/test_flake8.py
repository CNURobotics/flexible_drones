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

"""Colcon lint hook for Flake8."""

from pathlib import Path

import ament_flake8.main as ament_flake8_main
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    """Run Flake8 against the source package and tests."""
    original_style_guide = ament_flake8_main.get_flake8_style_guide

    def _single_job_style_guide(argv):
        return original_style_guide(list(argv) + ['--jobs=1'])

    ament_flake8_main.get_flake8_style_guide = _single_job_style_guide
    try:
        rc, errors = ament_flake8_main.main_with_errors(
            argv=[
                '--config', str(PACKAGE_ROOT / 'setup.cfg'),
                str(PACKAGE_ROOT / 'flexible_drones_tools'),
                str(PACKAGE_ROOT / 'test'),
            ]
        )
    finally:
        ament_flake8_main.get_flake8_style_guide = original_style_guide

    assert rc == 0, '\n'.join(errors)

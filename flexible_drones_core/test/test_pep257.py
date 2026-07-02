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

"""Colcon lint hook for PEP 257 docstrings."""

from pathlib import Path

from ament_pep257.main import main
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    """Run PEP 257 checks against the lightweight test suite."""
    rc = main(argv=[str(PACKAGE_ROOT / 'test')])
    assert rc == 0

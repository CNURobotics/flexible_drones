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

"""Colcon lint hook for copyright headers."""

from pathlib import Path

from ament_copyright.main import main
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# Generated trajectory artifacts (git-ignored) live under the generators output
# folder; they are build products, not source, so exclude them from linting.
GENERATED_OUTPUT_DIR = (
    PACKAGE_ROOT / 'flexible_drones_tools' / 'trajectories' / 'generators' / 'output'
)


@pytest.mark.copyright
@pytest.mark.linter
def test_copyright():
    """Run copyright checks against package source files."""
    excludes = [str(path) for path in GENERATED_OUTPUT_DIR.rglob('*') if path.is_file()]
    rc = main(argv=[str(PACKAGE_ROOT), '--exclude', *excludes])
    assert rc == 0

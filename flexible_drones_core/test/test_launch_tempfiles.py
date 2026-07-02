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

"""Unit tests for launch temp-file helpers."""

from pathlib import Path

from flexible_drones_core.launch_tempfiles import cleanup_temp_paths, write_temp_text


def test_write_temp_text_creates_file_with_expected_contents():
    """Write text to a temporary file and return its path."""
    path = Path(write_temp_text('hello world', suffix='.txt', prefix='fd_test_'))
    try:
        assert path.exists()
        assert path.read_text(encoding='utf-8') == 'hello world'
    finally:
        path.unlink(missing_ok=True)


def test_cleanup_temp_paths_removes_files():
    """Remove temporary paths and return no launch actions."""
    path = Path(write_temp_text('cleanup me', suffix='.txt', prefix='fd_test_'))

    actions = cleanup_temp_paths([str(path)])

    assert not path.exists()
    assert actions == []

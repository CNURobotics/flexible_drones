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

"""Helpers for launch-managed temporary files."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence

from launch.actions import RegisterEventHandler
from launch.event_handlers import OnShutdown


def write_temp_text(
    content: str,
    *,
    suffix: str,
    prefix: str = 'flexible_drones_',
) -> str:
    """Write text content to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix=suffix,
        prefix=prefix,
        delete=False,
        encoding='utf-8',
    ) as handle:
        handle.write(content)
        return handle.name


def cleanup_temp_paths(paths: Sequence[str]) -> list:
    """Remove temporary files and return an empty launch action list."""
    for path in dict.fromkeys(str(path) for path in paths if path):
        try:
            os.unlink(path)
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return []


def make_temp_file_cleanup_handler(paths: Sequence[str]) -> RegisterEventHandler | None:
    """Build an OnShutdown handler that removes the provided temp files."""
    managed_paths = tuple(dict.fromkeys(str(path) for path in paths if path))
    if not managed_paths:
        return None

    return RegisterEventHandler(OnShutdown(on_shutdown=lambda event, context, paths=managed_paths: cleanup_temp_paths(paths)))

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

"""General-purpose utilities shared across flexible_drones packages."""

from __future__ import annotations

from typing import Any


_MISSING = object()

_TRUE_STRINGS = {'1', 'true', 'yes', 'on'}
_FALSE_STRINGS = {'0', 'false', 'no', 'off'}


def parse_bool_field(value: Any, *, label: str, default: Any = _MISSING) -> bool:
    """Parse a boolean from YAML config or a ROS parameter value.

    Accepts Python bool, int 0/1, and common string representations
    (true/false, yes/no, on/off, 1/0). Raises ValueError for unrecognised
    values. Pass ``default`` to return a fallback when value is None.
    """
    if value is None and default is not _MISSING:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(
        f'{label} must be a boolean (got {value!r}). '
        'Use true/false, yes/no, on/off, or 1/0.'
    )

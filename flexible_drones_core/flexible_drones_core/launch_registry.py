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

"""Launch-facing helpers shared across flexible_drones backends."""

from __future__ import annotations

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

from flexible_drones_core.ansi import (
    BOLD_ORANGE as _ANSI_BRIGHT_ORANGE,
    BOLD_RED as _ANSI_BRIGHT_RED,
    RESET as _ANSI_RESET,
)


def format_registry_error(summary: str, detail: str | None = None) -> str:
    """Format launch-time registry errors with a readable colored prefix."""
    message = f'{_ANSI_BRIGHT_ORANGE}{summary}{_ANSI_RESET}'
    if detail:
        message += f' {_ANSI_BRIGHT_RED}{detail}{_ANSI_RESET}'
    return message


def require_backend_package(manager_type: str, manager_profiles: dict) -> None:
    """Validate a manager profile and ensure its backend package is installed."""
    profile = manager_profiles.get(manager_type)
    if profile is None:
        raise RuntimeError(
            f"No manager profile configured for manager_type '{manager_type}'."
        )

    package_name = str(profile.get('package', '')).strip()
    if not package_name:
        raise RuntimeError(
            f"Manager profile '{manager_type}' does not define a backend package."
        )

    try:
        get_package_share_directory(package_name)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Required backend package '{package_name}' for manager_type '{manager_type}' is not available."
        ) from exc

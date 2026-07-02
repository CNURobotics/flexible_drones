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

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from flexible_drones_core.deployment import DEPLOYMENT_PACKAGE_ENV

from flexible_drones_tools.trajectories.utilities.io import (
    normalize_path_value,
    resolve_existing_file,
    resolve_package_file,
)


def resolve_trajectory_path(
    *,
    trajectory_path: str = '',
    trajectory_package: str = '',
    trajectory_folder: str = 'trajectories',
    trajectory_file: str = '',
) -> str:
    raw_path = str(trajectory_path).strip()
    if raw_path:
        return str(resolve_existing_file(raw_path))

    package_name = (
        str(trajectory_package).strip()
        or os.environ.get(DEPLOYMENT_PACKAGE_ENV, '')
    )
    folder_name = str(trajectory_folder).strip()
    file_name = str(trajectory_file).strip()

    if not file_name:
        raise ValueError(
            "Provide 'trajectory_path' or 'trajectory_file'."
        )

    # Empty package with a folder value: treat folder as a direct filesystem path.
    # When FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE is set, the env-selected package
    # becomes the default lookup root for bare trajectory_file values.
    if not package_name:
        if not folder_name:
            raise ValueError(
                "Provide 'trajectory_path', 'trajectory_package', "
                f'${DEPLOYMENT_PACKAGE_ENV}, or a filesystem trajectory_folder.'
            )
        return str(resolve_existing_file(Path(folder_name) / file_name))

    folder_name = folder_name or 'trajectories'
    return str(
        resolve_package_file(
            package_name,
            folder_name,
            file_name,
            package_share_lookup=get_package_share_directory,
        )
    )


def resolve_deployment_trajectory_path(
    trajectory_arg: str,
    *,
    package: str = '',
    folder: str = 'trajectories',
    package_share_lookup=None,
) -> str:
    """
    Resolve a demo-style trajectory argument.

    Bare filenames are looked up in ``package/folder``. Values that look like
    filesystem paths are resolved directly. When ``package`` is empty it falls
    back to the ``FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE`` environment variable; a
    package must be supplied one way or the other to resolve a bare filename.
    """
    trajectory_arg = str(trajectory_arg).strip()
    if not trajectory_arg:
        raise ValueError('Provide a trajectory CSV filename or path.')

    is_path = (
        os.sep in trajectory_arg
        or '/' in trajectory_arg
        or trajectory_arg.startswith('.')
        or trajectory_arg.startswith('~')
    )
    if is_path:
        resolved = normalize_path_value(trajectory_arg).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f'Trajectory file not found: {resolved}')
        return str(resolved)

    package = str(package).strip() or os.environ.get(DEPLOYMENT_PACKAGE_ENV, '')
    if not package:
        raise ValueError(
            f"A trajectory package is required to resolve bare filename '{trajectory_arg}'. "
            f'Pass package=..., give a full/relative path, or set ${DEPLOYMENT_PACKAGE_ENV}.'
        )

    package_share_lookup = package_share_lookup or get_package_share_directory
    try:
        resolved = Path(package_share_lookup(package)) / folder / trajectory_arg
        if resolved.exists():
            return str(resolved)
    except Exception:
        pass

    raise FileNotFoundError(
        f"'{trajectory_arg}' not found in {package}/{folder}/. "
        'Provide a full or relative path, or install the file in that package.'
    )


def resolve_trajectory_write_path(
    *,
    trajectory_dir: str = '',
    trajectory_package: str = '',
    trajectory_folder: str = 'trajectories',
    trajectory_file: str = '',
    package_share_lookup=None,
) -> str:
    file_name = str(trajectory_file).strip()
    if not file_name:
        raise ValueError("Provide 'trajectory_file' for the trajectory output path.")

    raw_dir = str(trajectory_dir).strip()
    if raw_dir:
        output_dir = normalize_path_value(raw_dir)
    else:
        package_name = str(trajectory_package).strip()
        folder_name = str(trajectory_folder).strip() or 'trajectories'
        if not package_name:
            raise ValueError("Provide 'trajectory_dir' or 'trajectory_package'.")

        package_share_lookup = package_share_lookup or get_package_share_directory
        output_dir = Path(package_share_lookup(package_name)) / folder_name

    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / file_name)

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

"""
Generic roster + per-deployment overlay loading for drone fleets.

Used by both orchestrator-driven swarm launches and plain per-backend bringups
to resolve which drones to run; it does not depend on the orchestrator. This
engine is independent of any particular site or deployment package: the caller
supplies the package name (or an explicit package root) and the deployment
name; site-specific defaults such as the deployment package, the default
deployment, and the manager profiles live in the deployment package that wires
this engine to concrete data.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import os

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

from flexible_drones_core.registry import _KNOWN_SPEC_KEYS, DroneSpec
from flexible_drones_core.utils import parse_bool_field

# Layout conventions used when resolving roster/deployment assets. These are
# engine conventions, not site choices: the roster lives at config/roster.yaml
# and each deployment owns deployments/<name>/setup.yaml.
DEFAULT_ROSTER_REL = 'config/roster.yaml'
DEFAULT_SETUP_FILENAME = 'setup.yaml'

# Environment variables that supply launch-time deployment defaults when the
# operator has not passed them as launch arguments. Backends import these and
# the DEFAULT_* values below so the convention is defined in exactly one place.
DEPLOYMENT_PACKAGE_ENV = 'FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE'
ROSTER_ENV = 'FLEXIBLE_DRONES_ROSTER'
DEPLOYMENT_SETUP_ENV = 'FLEXIBLE_DRONES_DEPLOYMENT_SETUP'

# Env-derived launch defaults; empty unless the operator exports the variable.
DEFAULT_ROSTER = os.environ.get(ROSTER_ENV, '')
DEFAULT_DEPLOYMENT_SETUP = os.environ.get(DEPLOYMENT_SETUP_ENV, '')
DEFAULT_DEPLOYMENT_PACKAGE = os.environ.get(DEPLOYMENT_PACKAGE_ENV, '')


def is_filesystem_path(path: str) -> bool:
    """Return True if path is an absolute, ~, or explicitly relative (./ or /) path."""
    path = path.strip()
    return bool(path) and (path.startswith(('.', '/', '~')) or Path(path).is_absolute())


def validate_deployment_paths(
    *,
    roster: str,
    roster_package: str,
    deployment_setup: str,
    deployment_package: str,
) -> None:
    """
    Fail early with a friendly error when deployment data cannot be resolved.

    A package-relative roster/setup needs a package to resolve against;
    deployment_package is accepted as a fallback for roster_package. Absolute or
    ./ paths are self-contained and need no package.
    """
    effective_roster_package = roster_package.strip() or deployment_package.strip()
    if not roster.strip() and not effective_roster_package:
        raise RuntimeError(
            "Argument 'roster' is required unless 'deployment_package', "
            f"'roster_package', or ${ROSTER_ENV} is set."
        )
    if roster.strip() and not is_filesystem_path(roster) and not effective_roster_package:
        raise RuntimeError(
            "Argument 'deployment_package' or 'roster_package' is required when "
            f"'roster' is package-relative. Pass an absolute/./ path, or set "
            f'${DEPLOYMENT_PACKAGE_ENV}.'
        )
    if not deployment_package.strip() and not is_filesystem_path(deployment_setup):
        raise RuntimeError(
            "Argument 'deployment_package' is required unless 'deployment_setup' "
            'is an absolute or ./ path. '
            f'Alternatively set ${DEPLOYMENT_PACKAGE_ENV} or ${DEPLOYMENT_SETUP_ENV}.'
        )


def require_package_name(
    package_name: str,
    *,
    argument_name: str,
    env_var: str = DEPLOYMENT_PACKAGE_ENV,
    purpose: str,
    example_package: str = 'chris_serc_deployment',
    extra_hint: str = '',
) -> str:
    """Return a stripped package name or raise an actionable launch error."""
    package_name = package_name.strip()
    if package_name:
        return package_name

    message = (
        f'{argument_name} is required to {purpose}. '
        f'Set it with `{argument_name}:=<package_name>` or export '
        f'`{env_var}=<package_name>`. For example: '
        f'`{argument_name}:={example_package}`.'
    )
    extra_hint = extra_hint.strip()
    if extra_hint:
        message += f' {extra_hint}'

    raise RuntimeError(message)


def parse_selected_names(raw: str, *, argument_name: str = 'fliers') -> list[str] | None:
    """
    Parse a YAML-list launch argument into a name filter.

    Return None when raw is empty or null, meaning "all enabled drones".
    Return a list of strings when explicit names are given; the list may be
    empty if the caller explicitly passed [].
    """
    if not raw.strip():
        return None

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"'{argument_name}' must be a YAML list string, e.g. [cf1, cf3]: {exc}"
        ) from exc

    if parsed is None:
        return None
    if not isinstance(parsed, list):
        raise ValueError(f"'{argument_name}' must be a YAML list string, e.g. [cf1, cf3]")
    selected = [str(name).strip() for name in parsed if str(name).strip()]
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in selected:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"Duplicate {argument_name} entries are not allowed: {', '.join(duplicates)}")
    return selected


def _package_share(package_name: str) -> Path | None:
    try:
        return Path(get_package_share_directory(package_name))
    except PackageNotFoundError:
        return None


def _package_root_path(
    package_name: str | None = None,
    package_root: os.PathLike[str] | str | None = None,
) -> Path | None:
    if package_root is not None:
        return Path(package_root)
    if package_name is None:
        return None
    return _package_share(package_name)


def resolve_package_relative_path(
    rel_path: str,
    *,
    package_name: str | None = None,
    package_root: os.PathLike[str] | str | None = None,
) -> str:
    """Resolve a package-relative path against an override root or package share."""
    rel_path = os.path.expanduser(rel_path)
    if os.path.isabs(rel_path):
        return rel_path
    if rel_path.startswith(('.', '/')):
        candidate = Path(rel_path).resolve()
        if candidate.exists():
            return str(candidate)
        raise RuntimeError(f"Could not locate deployment asset '{rel_path}'.")

    root_path = _package_root_path(package_name=package_name, package_root=package_root)
    if root_path is not None:
        candidate = root_path / rel_path
        if candidate.exists():
            return str(candidate)

    raise RuntimeError(
        f"Could not locate deployment asset '{rel_path}' in package '{package_name}'."
    )


def resolve_roster_path(
    roster: str = DEFAULT_ROSTER_REL,
    *,
    package_name: str | None = None,
    package_root: os.PathLike[str] | str | None = None,
) -> str:
    """Resolve the roster path, defaulting to config/roster.yaml."""
    roster = roster.strip() or DEFAULT_ROSTER_REL
    return resolve_package_relative_path(
        roster,
        package_name=package_name,
        package_root=package_root,
    )


def resolve_deployment_setup_path(
    deployment: str,
    *,
    setup_file: str = '',
    package_name: str | None = None,
    package_root: os.PathLike[str] | str | None = None,
) -> str:
    """Resolve a deployment setup file from a deployment name or explicit path."""
    deployment = deployment.strip()
    setup_file = os.path.expanduser(setup_file.strip())
    if setup_file:
        if os.path.isabs(setup_file) or setup_file.startswith(('.', '/')):
            return os.path.abspath(setup_file)
        if '/' in setup_file or setup_file.endswith('.yaml'):
            rel_path = setup_file
        else:
            if not deployment:
                raise ValueError(
                    'A deployment name is required to resolve a bare setup filename.'
                )
            rel_path = os.path.join('deployments', deployment, setup_file)
    else:
        if not deployment:
            raise ValueError(
                'A deployment name or explicit setup_file is required.'
            )
        rel_path = os.path.join('deployments', deployment, DEFAULT_SETUP_FILENAME)

    return resolve_package_relative_path(
        rel_path,
        package_name=package_name,
        package_root=package_root,
    )


def _normalize_manager_type(raw_manager: Any, *, label: str) -> str:
    manager_type = str(raw_manager).strip()
    if manager_type:
        return manager_type
    raise ValueError(f'{label} must define manager_type.')


def _coerce_mapping(entry: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f'{label} must be a mapping.')
    return dict(entry)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _parse_spec(
    entry: dict[str, Any],
    *,
    label: str,
    fallback_name: str | None = None,
) -> DroneSpec:
    name = str(entry.get('name', fallback_name or '')).strip()
    if not name:
        raise ValueError(f"{label} must define a non-empty 'name'.")

    drone_type = str(entry.get('type', '')).strip() or 'unknown'
    ros_namespace = entry.get('ros_namespace')
    if ros_namespace is not None:
        ros_namespace = str(ros_namespace).strip()

    raw_ros_params = entry.get('ros_params', {})
    if raw_ros_params is None:
        raw_ros_params = {}
    if not isinstance(raw_ros_params, dict):
        raise ValueError(f'{label}.ros_params must be a dict')
    ros_params = dict(raw_ros_params)

    extra_args = entry.get('extra_args', []) or []
    if not isinstance(extra_args, list) or any(not isinstance(item, str) for item in extra_args):
        raise ValueError(f'{label}.extra_args must be a list of strings')

    raw_extras = entry.get('extras', {}) or {}
    if not isinstance(raw_extras, dict):
        raise ValueError(f'{label}.extras must be a dict')
    extras = dict(raw_extras)
    top_level_extras = {
        key: value
        for key, value in entry.items()
        if key not in _KNOWN_SPEC_KEYS
    }
    duplicate_extra_keys = sorted(set(extras) & set(top_level_extras))
    if duplicate_extra_keys:
        raise ValueError(
            f'{label}.extras duplicates unknown top-level key(s): '
            + ', '.join(duplicate_extra_keys)
        )
    extras.update(top_level_extras)

    return DroneSpec(
        name=name,
        uri=str(entry.get('uri', '')).strip(),
        type=drone_type,
        manager_type=_normalize_manager_type(
            entry.get('manager_type', entry.get('manager', '')),
            label=label,
        ),
        enabled=parse_bool_field(entry.get('enabled', True), label=f'{label}.enabled'),
        ros_namespace=ros_namespace,
        ros_params=ros_params,
        extra_args=extra_args,
        extras=extras,
    )


def load_roster_specs(roster_path: str) -> list[DroneSpec]:
    """Load roster entries without applying any deployment overlay."""
    with open(roster_path, 'r', encoding='utf-8') as handle:
        document = yaml.safe_load(handle) or {}

    if not isinstance(document, dict):
        raise ValueError('Roster YAML must contain a top-level mapping.')

    raw_drones = document.get('drones', [])
    if isinstance(raw_drones, list):
        return [
            _parse_spec(_coerce_mapping(entry, label=f'drones[{idx}]'), label=f'drones[{idx}]')
            for idx, entry in enumerate(raw_drones)
        ]
    if isinstance(raw_drones, dict):
        return [
            _parse_spec(
                _coerce_mapping(entry, label=f'drones.{name}'),
                label=f'drones.{name}',
                fallback_name=str(name).strip(),
            )
            for name, entry in raw_drones.items()
        ]

    raise ValueError("Roster YAML key 'drones' must be a list or mapping.")


def load_deployment_overlays(setup_path: str) -> dict[str, dict[str, Any]]:
    """Load deployment overlay entries keyed by drone name."""
    with open(setup_path, 'r', encoding='utf-8') as handle:
        document = yaml.safe_load(handle) or {}

    if not isinstance(document, dict):
        raise ValueError('Deployment setup YAML must contain a top-level mapping.')

    raw_drones = document.get('drones', {})
    if not isinstance(raw_drones, dict):
        raise ValueError("Deployment setup YAML key 'drones' must be a mapping.")

    overlays: dict[str, dict[str, Any]] = {}
    for name, entry in raw_drones.items():
        drone_name = str(name).strip()
        if not drone_name:
            continue
        overlays[drone_name] = _coerce_mapping(entry or {}, label=f'drones.{drone_name}')
    return overlays


def load_deployment_specs(
    roster_path: str,
    setup_path: str,
    *,
    selected_names: Sequence[str] | None = None,
) -> list[DroneSpec]:
    """
    Load roster specs, apply deployment overlays, then filter the result.

    The roster provides the base fleet, and the deployment setup overlays
    per-drone fields such as enabled state, manager parameters, and initial
    pose.  Omitted fields inherit from the layer above.  enabled defaults to
    true in the roster, but enabled: false is a monotonic safety gate: once any
    config layer disables a drone, later layers and selected_names cannot
    re-enable it.

    selected_names chooses the final subset of effectively enabled specs.  When
    selected_names is None, all enabled specs listed in the deployment setup are
    returned.  When selected_names is a list, every requested name must exist and
    remain enabled; an explicit empty list therefore selects no drones.
    """
    roster_specs = load_roster_specs(roster_path)
    deployment_overlays = load_deployment_overlays(setup_path)
    roster_names = {spec.name for spec in roster_specs}
    deployment_names = set(deployment_overlays)
    unknown_setup_names = sorted(deployment_names - roster_names)
    if unknown_setup_names:
        raise ValueError(
            'Deployment setup references drones not found in roster: '
            + ', '.join(unknown_setup_names)
        )

    merged_specs: list[DroneSpec] = []
    for roster_spec in roster_specs:
        raw_entry = {
            'name': roster_spec.name,
            'uri': roster_spec.uri,
            'type': roster_spec.type,
            'manager_type': roster_spec.manager_type,
            'enabled': roster_spec.enabled,
            'ros_namespace': roster_spec.ros_namespace,
            'ros_params': dict(roster_spec.ros_params or {}),
            'extra_args': list(roster_spec.extra_args or []),
            **dict(roster_spec.extras or {}),
        }

        overlay = deployment_overlays.get(roster_spec.name)
        if overlay is not None:
            raw_entry = _deep_merge(raw_entry, overlay)
            overlay_enabled = parse_bool_field(
                overlay.get('enabled', roster_spec.enabled),
                label=f'merged.{roster_spec.name}.enabled',
            )
            raw_entry['enabled'] = roster_spec.enabled and overlay_enabled

        merged_specs.append(
            _parse_spec(raw_entry, label=f'merged.{roster_spec.name}', fallback_name=roster_spec.name)
        )

    if selected_names is not None:
        by_name = {spec.name: spec for spec in merged_specs}
        missing = [name for name in selected_names if name not in by_name]
        if missing:
            raise ValueError(
                f"Requested fliers not found in roster/deployment selection: {', '.join(missing)}"
            )
        disabled = [name for name in selected_names if not by_name[name].enabled]
        if disabled:
            disabled_names = ', '.join(disabled)
            raise ValueError(
                f'Requested fliers are disabled by the roster/deployment selection: '
                f'{disabled_names}'
            )
        return [by_name[name] for name in selected_names]

    return [spec for spec in merged_specs if spec.enabled and spec.name in deployment_names]


def apply_manager_overrides(
    specs: Sequence[DroneSpec],
    manager_by_name: Mapping[str, str],
) -> list[DroneSpec]:
    """Return specs with selected manager types overridden by drone name."""
    if not manager_by_name:
        return list(specs)

    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in manager_by_name if name not in by_name]
    if missing:
        raise ValueError(
            'Manager overrides reference drones not in the selected fleet: '
            f"{', '.join(missing)}. Add them to 'fliers' first, or remove them "
            "from the override list such as 'gazebo_fliers'."
        )

    return [
        replace(spec, manager_type=manager_by_name.get(spec.name, spec.manager_type))
        for spec in specs
    ]

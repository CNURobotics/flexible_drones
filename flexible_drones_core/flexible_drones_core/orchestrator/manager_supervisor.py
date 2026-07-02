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

"""Hybrid supervisor for subprocess and embedded drone managers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import os
import threading
import time

from flexible_drones_core.manager.config import (
    ManagerConfig,
    base_manager_config_from_spec,
)

from .process_manager import ProcessManager
from ..registry import DroneSpec

EmbeddedFactory = Callable[[ManagerConfig], 'EmbeddedHandle']


@dataclass
class EmbeddedHandle:
    start: Callable[[], None]
    stop: Callable[[], None]
    is_alive: Callable[[], bool]
    cleanup: Callable[[], None] | None = None


@dataclass
class EmbeddedInfo:
    name: str
    spec: DroneSpec
    handle: EmbeddedHandle
    package: str
    executable: str
    started_at_unix: float = 0.0  # wall clock, for state display only
    restart_count: int = 0
    # Restart scheduling uses the monotonic clock so an NTP/chrony step can
    # never stretch or collapse the backoff window.
    next_restart_earliest_mono: float = 0.0
    last_exit_code: int | None = None
    last_error: str | None = None
    active: bool = False
    ever_started: bool = False


class ManagerSupervisor:
    """Routes specs to either subprocess managers or embedded managers."""

    def __init__(
        self,
        *,
        package: str,
        executable: str,
        manager_profiles: dict[str, dict[str, Any]] | None = None,
        embedded_factories: dict[str, EmbeddedFactory] | None = None,
        python_executable: str | None = None,
        ros_domain_id: int | None = None,
        shutdown_timeouts: tuple[float, float, float] = (3.0, 3.0, 1.0),
        restart_backoff_base_s: float = 1.0,
        restart_backoff_max_s: float = 20.0,
        max_restarts: int = 20,
        logger: Any = None,
    ):
        self._default_package = package
        self._default_executable = executable
        self._manager_profiles = manager_profiles or {}
        self._embedded_factories = embedded_factories or {}
        self._logger = logger
        self._restart_backoff_base_s = restart_backoff_base_s
        self._restart_backoff_max_s = restart_backoff_max_s
        self._max_restarts = max_restarts

        self._lock = threading.RLock()
        self._embedded: dict[str, EmbeddedInfo] = {}

        self._procman = ProcessManager(
            package=package,
            executable=executable,
            manager_profiles=manager_profiles,
            python_executable=python_executable,
            ros_domain_id=ros_domain_id,
            shutdown_timeouts=shutdown_timeouts,
            restart_backoff_base_s=restart_backoff_base_s,
            restart_backoff_max_s=restart_backoff_max_s,
            max_restarts=max_restarts,
            logger=logger,
        )

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            return
        getattr(self._logger, level)(message)

    def _resolve_profile(self, manager_type: str) -> tuple[str, str]:
        profile = self._manager_profiles.get(manager_type, {})
        package = str(profile.get('package', self._default_package)).strip()
        executable = str(profile.get('executable', self._default_executable)).strip()

        if not package or not executable:
            raise ValueError(
                f"Invalid manager profile for '{manager_type}': "
                f"package='{package}' executable='{executable}'"
            )
        return package, executable

    def _is_embedded(self, spec: DroneSpec) -> bool:
        return spec.manager_type in self._embedded_factories

    def _restart_backoff(self, restart_count: int) -> float:
        return min(
            self._restart_backoff_base_s * (2**restart_count),
            self._restart_backoff_max_s,
        )

    def list_states(self) -> dict[str, dict[str, Any]]:
        states = self._procman.list_states()

        with self._lock:
            now_mono = time.monotonic()
            now_unix = time.time()
            for name, info in self._embedded.items():
                if not info.active:
                    state = 'STOPPED'
                elif info.handle.is_alive():
                    state = 'RUNNING'
                elif info.last_error:
                    state = 'WAITING_RETRY'
                else:
                    state = 'STARTING'

                # Scheduling is monotonic; project the remaining backoff onto
                # the wall clock for display in the published state snapshot.
                next_restart_unix = now_unix + max(
                    0.0, info.next_restart_earliest_mono - now_mono
                )
                states[name] = {
                    'state': state,
                    'pid': os.getpid(),
                    'started_at_unix': info.started_at_unix,
                    'restart_count': info.restart_count,
                    'next_restart_earliest_unix': next_restart_unix,
                    'last_exit_code': info.last_exit_code,
                    'last_error': info.last_error,
                    'manager_type': info.spec.manager_type,
                    'package': info.package,
                    'executable': info.executable,
                    'launch_mode': 'embedded',
                    'last_start_cmd': ['embedded', info.package, info.executable],
                    'spec': info.spec.to_dict(),
                }

        return states

    def list_running_names(self) -> list[str]:
        running = set(self._procman.list_running_names())
        with self._lock:
            for name, info in self._embedded.items():
                if info.active and info.handle.is_alive():
                    running.add(name)
        return sorted(running)

    def is_alive(self, name: str) -> bool:
        with self._lock:
            info = self._embedded.get(name)
            if info is not None:
                return info.active and info.handle.is_alive()
        return self._procman.is_alive(name)

    def ensure_running(self, desired: list[DroneSpec]) -> None:
        desired_by_name = {spec.name: spec for spec in desired}
        desired_embedded = {spec.name: spec for spec in desired if self._is_embedded(spec)}
        blocked_names: set[str] = set()

        with self._lock:
            for name, info in list(self._embedded.items()):
                desired_spec = desired_by_name.get(name)
                if desired_spec is None or not self._is_embedded(desired_spec):
                    if not self._remove_embedded_locked(name):
                        blocked_names.add(name)
                    continue
                if info.spec != desired_spec:
                    if not self._remove_embedded_locked(name):
                        blocked_names.add(name)

        desired_proc = [
            spec for spec in desired
            if not self._is_embedded(spec) and spec.name not in blocked_names
        ]
        self._procman.ensure_running(desired_proc)

        with self._lock:
            for name, spec in desired_embedded.items():
                if name in blocked_names:
                    continue
                info = self._embedded.get(name)
                if info is None:
                    try:
                        info = self._create_embedded_locked(spec)
                    except Exception as exc:  # noqa: B902
                        self._log(
                            'error',
                            f'[{name}] Failed to construct embedded manager: {exc}',
                        )
                        continue
                elif info.spec != spec:
                    if not self._remove_embedded_locked(name):
                        continue
                    try:
                        info = self._create_embedded_locked(spec)
                    except Exception as exc:  # noqa: B902
                        self._log(
                            'error',
                            f'[{name}] Failed to reconstruct embedded manager: {exc}',
                        )
                        continue

                self._start_embedded_locked(info)

    def stop_all(self) -> None:
        self._procman.stop_all()
        with self._lock:
            for name in list(self._embedded.keys()):
                self._stop_embedded_locked(name)

    def start_all(self, specs: list[DroneSpec]) -> None:
        for spec in specs:
            self.start_spec(spec)

    def start_spec(self, spec: DroneSpec) -> bool:
        if not self._is_embedded(spec):
            return self._procman.start_spec(spec)

        with self._lock:
            info = self._embedded.get(spec.name)
            if info is None:
                try:
                    info = self._create_embedded_locked(spec)
                except Exception as exc:  # noqa: B902
                    self._log(
                        'error',
                        f'[{spec.name}] Failed to construct embedded manager: {exc}',
                    )
                    return False
            elif info.spec != spec:
                if not self._remove_embedded_locked(spec.name):
                    return False
                try:
                    info = self._create_embedded_locked(spec)
                except Exception as exc:  # noqa: B902
                    self._log(
                        'error',
                        f'[{spec.name}] Failed to reconstruct embedded manager: {exc}',
                    )
                    return False

            return self._start_embedded_locked(info)

    def stop_drone(self, drone_name: str) -> bool:
        with self._lock:
            if drone_name in self._embedded:
                return self._stop_embedded_locked(drone_name)
        return self._procman.stop_drone(drone_name)

    def _create_embedded_locked(self, spec: DroneSpec) -> EmbeddedInfo:
        package, executable = self._resolve_profile(spec.manager_type)
        profile = self._manager_profiles.get(spec.manager_type, {})
        config = base_manager_config_from_spec(spec, profile)
        factory = self._embedded_factories[spec.manager_type]
        handle = factory(config)

        info = EmbeddedInfo(
            name=spec.name,
            spec=spec,
            handle=handle,
            package=package,
            executable=executable,
        )
        self._embedded[spec.name] = info
        return info

    def _start_embedded_locked(self, info: EmbeddedInfo) -> bool:
        if info.active and info.handle.is_alive():
            return True

        now = time.monotonic()
        if info.restart_count >= self._max_restarts:
            self._log(
                'error',
                f'[{info.name}] Reached max embedded restarts '
                f'({self._max_restarts}); no further automatic restarts.',
            )
            return False
        if now < info.next_restart_earliest_mono:
            return False

        restarting_existing_manager = info.ever_started
        try:
            info.handle.start()
            info.active = True
            info.ever_started = True
            info.last_error = None
            info.last_exit_code = None
            if restarting_existing_manager:
                info.restart_count += 1
            info.started_at_unix = time.time()
            info.next_restart_earliest_mono = (
                time.monotonic() + self._restart_backoff(info.restart_count)
            )
            self._log(
                'info',
                f"[{info.name}] Started embedded manager package='{info.package}' "
                f"executable='{info.executable}' "
                f'restart_count={info.restart_count}',
            )
            return True
        except Exception as exc:  # noqa: B902
            info.active = False
            info.last_error = str(exc)
            info.last_exit_code = 1
            info.restart_count += 1
            info.next_restart_earliest_mono = now + self._restart_backoff(info.restart_count)
            self._log('warning', f'[{info.name}] Embedded manager start failed: {exc}')
            return False

    def _stop_embedded_locked(self, name: str) -> bool:
        info = self._embedded.get(name)
        if info is None:
            return False

        if not info.active:
            return True
        if not info.ever_started:
            info.active = False
            return True

        try:
            info.handle.stop()
            info.last_exit_code = 0
            info.last_error = None
        except Exception as exc:  # noqa: B902
            info.last_exit_code = 1
            info.last_error = str(exc)
            self._log('error', f'[{name}] Failed to stop embedded manager: {exc}')
            info.active = False
            return False

        info.active = False
        self._log('info', f'[{name}] Stopped embedded manager exit_code={info.last_exit_code}')
        return True

    def _remove_embedded_locked(self, name: str) -> bool:
        if name not in self._embedded:
            return True
        if not self._stop_embedded_locked(name):
            return False
        try:
            cleanup = self._embedded[name].handle.cleanup
            if cleanup is not None:
                cleanup()
        except Exception as exc:  # noqa: B902
            self._log('error', f'[{name}] Failed to clean up embedded manager: {exc}')
            return False
        del self._embedded[name]
        return True

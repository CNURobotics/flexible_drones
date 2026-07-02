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

"""Subprocess helpers for managing per-drone manager processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import os
import signal
import subprocess
import threading
import time

from flexible_drones_core.manager.config import (
    ManagerConfig,
    base_manager_config_from_spec,
    has_node_remap,
)

from ..registry import DroneSpec


FINAL_REAP_TIMEOUT_S = 2.0


@dataclass
class ProcInfo:
    name: str
    spec: DroneSpec
    popen: subprocess.Popen
    started_at_unix: float  # wall clock, for state display only
    package: str
    executable: str
    started_at_mono: float = 0.0
    last_exit_code: int | None = None
    restart_count: int = 0
    # Restart scheduling uses the monotonic clock so an NTP/chrony step can
    # never stretch or collapse the backoff window.
    next_restart_earliest_mono: float = 0.0
    last_start_cmd: list[str] | None = None
    last_reported_exit_code: int | None = None
    max_restarts_reported: bool = False

    def is_alive(self) -> bool:
        return self.popen.poll() is None

    def state(self) -> str:
        if self.is_alive():
            return 'RUNNING'
        if self.last_exit_code is None:
            return 'EXITED'
        if self.last_exit_code == 0:
            return 'EXITED_OK'
        return f'CRASHED({self.last_exit_code})'


class ProcessManager:
    """
    Supervises per-drone manager node processes.

    It does NOT do ROS calls. It only ensures that for each enabled DroneSpec,
    a corresponding manager process is running, with restart/backoff.
    """

    def __init__(
        self,
        *,
        package: str,
        executable: str,
        manager_profiles: dict[str, dict[str, Any]] | None = None,
        python_executable: str | None = None,
        ros_domain_id: int | None = None,
        shutdown_timeouts: tuple[float, float, float] = (3.0, 3.0, 1.0),
        restart_backoff_base_s: float = 1.0,
        restart_backoff_max_s: float = 20.0,
        max_restarts: int = 20,
        restart_count_reset_after_s: float = 300.0,
        logger: Any = None,
    ):
        """
        Args:
          package/executable: default profile used with
            `ros2 run <package> <executable>`
          manager_profiles: map manager_type -> {package, executable}
          python_executable: optional advanced direct python runner
          ros_domain_id: if set, exported in the child env
          shutdown_timeouts: (SIGINT wait, SIGTERM wait, SIGKILL wait)
          restart_count_reset_after_s: healthy uptime required before the
            automatic-restart counter is reset. Set <= 0 to disable.
          logger: optional object exposing info/warning/error methods (e.g. rclpy logger)
        """
        self._default_package = package
        self._default_executable = executable
        self._manager_profiles = manager_profiles or {}
        self._python_executable = python_executable
        self._ros_domain_id = ros_domain_id
        self._shutdown_timeouts = shutdown_timeouts
        self._logger = logger

        self._restart_backoff_base_s = restart_backoff_base_s
        self._restart_backoff_max_s = restart_backoff_max_s
        self._max_restarts = max_restarts
        self._restart_count_reset_after_s = restart_count_reset_after_s

        self._lock = threading.RLock()
        self._procs: dict[str, ProcInfo] = {}

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            return
        getattr(self._logger, level)(message)

    def list_states(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            now_mono = time.monotonic()
            now_unix = time.time()
            out: dict[str, dict[str, Any]] = {}
            for name, pi in self._procs.items():
                # Scheduling is monotonic; project the remaining backoff onto
                # the wall clock for display in the published state snapshot.
                next_restart_unix = now_unix + max(
                    0.0, pi.next_restart_earliest_mono - now_mono
                )
                out[name] = {
                    'state': pi.state(),
                    'pid': pi.popen.pid,
                    'started_at_unix': pi.started_at_unix,
                    'restart_count': pi.restart_count,
                    'next_restart_earliest_unix': next_restart_unix,
                    'last_exit_code': pi.last_exit_code,
                    'manager_type': pi.spec.manager_type,
                    'package': pi.package,
                    'executable': pi.executable,
                    'launch_mode': 'subprocess',
                    'last_start_cmd': pi.last_start_cmd,
                    'spec': pi.spec.to_dict(),
                }
            return out

    def list_running_names(self) -> list[str]:
        with self._lock:
            return [name for name, pi in self._procs.items() if pi.is_alive()]

    def is_alive(self, name: str) -> bool:
        with self._lock:
            pi = self._procs.get(name)
            return bool(pi and pi.is_alive())

    def ensure_running(self, desired: list[DroneSpec]) -> None:
        """
        Reconcile desired enabled specs with actual processes.
        Start missing, restart crashed (with backoff), stop undesired.
        """
        desired_by_name = {s.name: s for s in desired}
        now = time.monotonic()

        with self._lock:
            # Stop processes that are no longer desired.
            for name in list(self._procs.keys()):
                if name not in desired_by_name:
                    self._stop_locked(name)

            # Start or restart desired processes.
            for name, spec in desired_by_name.items():
                pi = self._procs.get(name)

                if pi is None:
                    try:
                        self._start_locked(spec)
                    except Exception as exc:  # noqa: B902
                        self._log(
                            'error',
                            f'[{spec.name}] Failed to start manager process: {exc}',
                        )
                    continue

                if pi.is_alive():
                    self._maybe_reset_restart_count_locked(pi, now)
                    # Spec changed -> restart to apply new args/params/profile.
                    if pi.spec != spec:
                        if not self._stop_locked(name):
                            continue
                        try:
                            self._start_locked(spec)
                        except Exception as exc:  # noqa: B902
                            self._log(
                                'error',
                                f'[{spec.name}] Failed to restart manager after ' f'spec change: {exc}',
                            )
                    continue

                # Not alive -> consider restart policy/backoff.
                pi.last_exit_code = pi.popen.poll()
                self._maybe_reset_restart_count_locked(pi, now)
                if pi.last_exit_code != pi.last_reported_exit_code:
                    level = 'info' if pi.last_exit_code == 0 else 'warning'
                    self._log(
                        level,
                        f'[{name}] Manager exited with code {pi.last_exit_code}',
                    )
                    pi.last_reported_exit_code = pi.last_exit_code
                if pi.restart_count >= self._max_restarts:
                    if not pi.max_restarts_reported:
                        self._log(
                            'error',
                            f'[{name}] Reached max restarts ' f'({self._max_restarts}); no further automatic ' 'restarts.',
                        )
                        pi.max_restarts_reported = True
                    continue
                if now < pi.next_restart_earliest_mono:
                    continue

                try:
                    self._start_locked(spec, replacing=name, prev=pi)
                except Exception as exc:  # noqa: B902
                    self._log(
                        'error',
                        f'[{spec.name}] Failed to restart manager process: {exc}',
                    )

    def stop_all(self) -> None:
        with self._lock:
            procs = list(self._procs.items())

        self._stop_many(procs)

    def start_all(self, specs: list[DroneSpec]) -> None:
        with self._lock:
            for spec in specs:
                pi = self._procs.get(spec.name)
                if pi is None or not pi.is_alive():
                    try:
                        self._start_locked(spec)
                    except Exception as exc:  # noqa: B902
                        self._log(
                            'error',
                            f'[{spec.name}] Failed to start manager process ' f'from start_all: {exc}',
                        )

    def start_spec(self, spec: DroneSpec) -> bool:
        with self._lock:
            pi = self._procs.get(spec.name)
            if pi and pi.is_alive() and pi.spec == spec:
                return True

            if pi is not None and not self._stop_locked(spec.name):
                return False
            try:
                self._start_locked(spec)
                return True
            except Exception as exc:  # noqa: B902
                self._log(
                    'error',
                    f'[{spec.name}] Failed to start manager process ' f'from start_spec: {exc}',
                )
                return False

    def stop_drone(self, drone_name: str) -> bool:
        with self._lock:
            if drone_name not in self._procs:
                return False
            return self._stop_locked(drone_name)

    def _resolve_profile(self, manager_type: str) -> tuple[str, str]:
        profile = self._manager_profiles.get(manager_type, {})
        package = str(profile.get('package', self._default_package)).strip()
        executable = str(profile.get('executable', self._default_executable)).strip()

        if not package or not executable:
            raise ValueError(f"Invalid manager profile for '{manager_type}': " f"package='{package}' executable='{executable}'")
        return package, executable

    @staticmethod
    def _format_param_value(value: Any) -> str:
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value

        return json.dumps(value)

    @staticmethod
    def _has_node_remap(args: list[str]) -> bool:
        return has_node_remap(args)

    def _build_cmd_from_config(
        self,
        config: ManagerConfig,
        package: str,
        executable: str,
    ) -> tuple[list[str], str, str]:
        extra = list(config.extra_args or ())
        ros_args: list[str] = ['--ros-args', '-r', f'__ns:={config.ros_namespace}']
        if config.node_name:
            ros_args += ['-r', f'__node:={config.node_name}']

        for key, value in config.to_ros_parameters().items():
            if value is None:
                continue
            ros_args += ['-p', f'{key}:={self._format_param_value(value)}']

        if self._python_executable:
            return (
                [self._python_executable, executable] + extra + ros_args,
                package,
                executable,
            )

        return (
            ['ros2', 'run', package, executable] + extra + ros_args,
            package,
            executable,
        )

    def _build_cmd(self, spec: DroneSpec) -> tuple[list[str], str, str]:
        """
        Command line contract for manager node:
          params: drone_name, uri, drone_type, manager_type
        plus any user ros_params.

        Namespace is applied via: --ros-args -r __ns:=<ns>
        """
        package, executable = self._resolve_profile(spec.manager_type)
        profile = self._manager_profiles.get(spec.manager_type, {})
        config = base_manager_config_from_spec(spec, profile)
        return self._build_cmd_from_config(config, package, executable)

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self._ros_domain_id is not None:
            env['ROS_DOMAIN_ID'] = str(int(self._ros_domain_id))
        return env

    def _start_locked(
        self,
        spec: DroneSpec,
        replacing: str | None = None,
        prev: ProcInfo | None = None,
    ) -> None:
        cmd, package, executable = self._build_cmd(spec)
        restart_count = prev.restart_count + 1 if (replacing and prev) else 0

        # Start process in its own process group so we can signal it cleanly.
        # Child stdout/stderr intentionally inherits the orchestrator's streams so
        # manager node output is visible during development. To suppress, restore:
        #   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        # A future forward_output parameter on ProcessManager should control this.
        pop = subprocess.Popen(
            cmd,
            env=self._child_env(),
            text=True,
            start_new_session=True,
        )

        now_mono = time.monotonic()
        now_unix = time.time()

        # Exponential backoff: base * 2^restarts, capped.
        backoff = min(
            self._restart_backoff_base_s * (2**restart_count),
            self._restart_backoff_max_s,
        )
        next_restart_mono = now_mono + backoff

        self._procs[spec.name] = ProcInfo(
            name=spec.name,
            spec=spec,
            popen=pop,
            started_at_unix=now_unix,
            started_at_mono=now_mono,
            package=package,
            executable=executable,
            last_exit_code=None,
            restart_count=restart_count,
            next_restart_earliest_mono=next_restart_mono,
            last_start_cmd=cmd,
        )
        self._log(
            'info',
            f"[{spec.name}] Started manager pid={pop.pid} package='{package}' "
            f"executable='{executable}' "
            f'restart_count={restart_count}',
        )

    def _maybe_reset_restart_count_locked(self, pi: ProcInfo, now_mono: float) -> None:
        if self._restart_count_reset_after_s <= 0.0:
            return
        if pi.restart_count <= 0:
            return
        if pi.started_at_mono <= 0.0:
            return
        healthy_uptime = now_mono - pi.started_at_mono
        if healthy_uptime < self._restart_count_reset_after_s:
            return

        pi.restart_count = 0
        pi.max_restarts_reported = False
        self._log(
            'info',
            f'[{pi.name}] Reset manager restart count after '
            f'{healthy_uptime:.1f}s of uptime.',
        )

    def _stop_locked(self, name: str) -> bool:
        pi = self._procs.get(name)
        if not pi:
            return True

        if pi.is_alive():
            try:
                pgid = os.getpgid(pi.popen.pid)
            except Exception:  # noqa: B902
                pgid = None

            # Step 1: SIGINT (like Ctrl+C)
            self._signal_and_wait(pi, pgid, signal.SIGINT, self._shutdown_timeouts[0])

            # Step 2: SIGTERM
            if pi.is_alive():
                self._signal_and_wait(pi, pgid, signal.SIGTERM, self._shutdown_timeouts[1])

            # Step 3: SIGKILL
            if pi.is_alive():
                self._signal_and_wait(pi, pgid, signal.SIGKILL, self._shutdown_timeouts[2])

        if not self._wait_for_reap_locked(name, pi):
            return False

        self._log('info', f'[{name}] Stopped manager exit_code={pi.last_exit_code}')
        del self._procs[name]
        return True

    def _stop_many(self, procs: list[tuple[str, ProcInfo]]) -> None:
        if not procs:
            return

        pgids = {
            name: self._process_group_id(pi)
            for name, pi in procs
        }
        for sig, timeout_s in (
            (signal.SIGINT, self._shutdown_timeouts[0]),
            (signal.SIGTERM, self._shutdown_timeouts[1]),
            (signal.SIGKILL, self._shutdown_timeouts[2]),
        ):
            alive = [(name, pi) for name, pi in procs if pi.is_alive()]
            if not alive:
                break

            for name, pi in alive:
                self._signal_process(pi, pgids.get(name), sig)
            self._wait_for_group_exit(alive, timeout_s)

        for name, pi in procs:
            with self._lock:
                if self._procs.get(name) is not pi:
                    continue
                if not self._wait_for_reap_locked(name, pi):
                    continue
                self._log('info', f'[{name}] Stopped manager exit_code={pi.last_exit_code}')
                del self._procs[name]

    def _wait_for_reap_locked(self, name: str, pi: ProcInfo) -> bool:
        try:
            pi.last_exit_code = pi.popen.wait(timeout=FINAL_REAP_TIMEOUT_S)
            return True
        except subprocess.TimeoutExpired:
            pi.last_exit_code = pi.popen.poll()
            if pi.last_exit_code is not None:
                return True
            self._log(
                'error',
                f'[{name}] Manager pid={pi.popen.pid} did not exit after SIGKILL; '
                'keeping process entry to prevent a conflicting respawn.',
            )
            return False

    def _process_group_id(self, pi: ProcInfo) -> int | None:
        try:
            return os.getpgid(pi.popen.pid)
        except Exception:  # noqa: B902
            return None

    def _signal_process(self, pi: ProcInfo, pgid: int | None, sig: int) -> None:
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                pi.popen.send_signal(sig)
        except ProcessLookupError:
            return
        except Exception:  # noqa: B902
            try:
                pi.popen.send_signal(sig)
            except Exception:  # noqa: B902
                return

    def _wait_for_group_exit(self, procs: list[tuple[str, ProcInfo]], timeout_s: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if all(not pi.is_alive() for _, pi in procs):
                return
            time.sleep(0.05)

    def _signal_and_wait(self, pi: ProcInfo, pgid: int | None, sig: int, timeout_s: float) -> None:
        if not pi.is_alive():
            return

        self._signal_process(pi, pgid, sig)
        self._wait_for_group_exit([('', pi)], timeout_s)

    def quick_stop_drone(self, drone_id: str) -> bool:
        return self.stop_drone(drone_id)

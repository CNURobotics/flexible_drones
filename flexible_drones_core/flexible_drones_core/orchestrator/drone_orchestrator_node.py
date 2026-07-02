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

from dataclasses import dataclass, field
import json
import threading
import time
from typing import Any, Callable, Sequence

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

import yaml

from std_msgs.msg import String
from std_srvs.srv import Trigger

from builtin_interfaces.msg import Duration as RosDuration
from flexible_drones_msgs.action import (
    GoTo,
    Land,
    Takeoff,
    SwarmGoTo,
    SwarmLand,
    SwarmTakeoff,
)
from flexible_drones_msgs.srv import Arm, SpawnDrone, SwarmArm

from .manager_supervisor import ManagerSupervisor
from ..registry import DroneSpec, Registry

_GOAL_STATUS_CANCELED = 5
_SWARM_RETURN_CODE_BUSY = 4
_SWARM_RETURN_CODE_CANCELED = 5


@dataclass
class _SwarmActionSummary:
    targeted: int = 0
    unavailable: int = 0
    sent: int = 0
    accepted: int = 0
    rejected: int = 0
    goal_response_timeout: int = 0
    succeeded: int = 0
    failed: int = 0
    canceled: int = 0
    result_timeout: int = 0
    canceled_by_request: bool = False

    def return_code(self) -> int:
        if self.canceled_by_request:
            return _SWARM_RETURN_CODE_CANCELED
        if self.targeted == 0:
            return 0
        if self.succeeded == self.targeted:
            return 0
        if self.succeeded > 0:
            return 1
        return 2

    def as_dict(self) -> dict[str, int]:
        return {
            'targeted': self.targeted,
            'unavailable': self.unavailable,
            'sent': self.sent,
            'accepted': self.accepted,
            'rejected': self.rejected,
            'goal_response_timeout': self.goal_response_timeout,
            'succeeded': self.succeeded,
            'failed': self.failed,
            'canceled': self.canceled,
            'result_timeout': self.result_timeout,
            'canceled_by_request': int(self.canceled_by_request),
        }


@dataclass
class _ActiveSwarmAction:
    action_name: str
    cancel_requested: bool = False
    accepted_goal_handles: dict[str, Any] = field(default_factory=dict)
    result_futures: dict[str, Any] = field(default_factory=dict)
    cancel_futures: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _SwarmServiceSummary:
    targeted: int = 0
    unavailable: int = 0
    sent: int = 0
    succeeded: int = 0
    failed: int = 0
    response_timeout: int = 0

    def complete_success(self) -> bool:
        return self.targeted > 0 and self.succeeded == self.targeted

    def as_dict(self) -> dict[str, int]:
        return {
            'targeted': self.targeted,
            'unavailable': self.unavailable,
            'sent': self.sent,
            'succeeded': self.succeeded,
            'failed': self.failed,
            'response_timeout': self.response_timeout,
        }


def _format_swarm_service_message(command: str, summary: _SwarmServiceSummary) -> str:
    return (
        f'{command} completed: {summary.succeeded} succeeded, '
        f'{summary.failed} failed, {summary.unavailable} unavailable, '
        f'{summary.response_timeout} timed out ({summary.targeted} targeted).'
    )


class DroneOrchestratorNode(Node):
    """ROS2 node responsible for supervising multi-manager per-drone processes."""

    def __init__(self):
        super().__init__('drone_orchestrator')

        # Parameters
        self.declare_parameter('registry_path', '')
        self.declare_parameter('selected_drones', [])
        self.declare_parameter('manager_package', 'flexible_drones_core')
        self.declare_parameter('manager_executable', '')
        self.declare_parameter('manager_profiles_yaml', '')
        self.declare_parameter('reconcile_period_s', 1.0)
        self.declare_parameter('ros_domain_id', -1)
        self.declare_parameter('swarm_goal_response_timeout_s', 2.0)
        self.declare_parameter('swarm_cancel_goal_response_grace_s', 0.2)
        self.declare_parameter('swarm_result_timeout_s', 30.0)
        self.declare_parameter('swarm_service_response_timeout_s', 5.0)

        self._selected_drones = self._read_selected_drones_param()

        registry_path = str(self.get_parameter('registry_path').value).strip()
        manager_pkg = str(self.get_parameter('manager_package').value).strip() or 'flexible_drones_core'
        manager_exe = str(self.get_parameter('manager_executable').value).strip()
        reconcile_period = max(0.2, float(self.get_parameter('reconcile_period_s').value))
        self._swarm_goal_response_timeout_s = max(
            0.1,
            float(self.get_parameter('swarm_goal_response_timeout_s').value),
        )
        self._swarm_cancel_goal_response_grace_s = max(
            0.0,
            float(self.get_parameter('swarm_cancel_goal_response_grace_s').value),
        )
        self._swarm_result_timeout_s = max(
            0.1,
            float(self.get_parameter('swarm_result_timeout_s').value),
        )
        self._swarm_service_response_timeout_s = max(
            0.1,
            float(self.get_parameter('swarm_service_response_timeout_s').value),
        )

        ros_domain_id_raw = int(self.get_parameter('ros_domain_id').value)
        ros_domain_id: int | None = None if ros_domain_id_raw < 0 else ros_domain_id_raw

        manager_profiles_yaml = str(self.get_parameter('manager_profiles_yaml').value)
        manager_profiles = self._parse_manager_profiles(manager_profiles_yaml)
        embedded_factories = self._build_embedded_factories(manager_profiles)

        # Desired-state registry
        if not registry_path:
            self.get_logger().warning(
                "Parameter 'registry_path' is empty. Orchestrator will start "
                'with empty registry.'
            )
            self._registry = Registry(specs=[], source_path=None)
        else:
            self._registry = Registry(specs=[], source_path=registry_path)
            try:
                self._registry.reload_from_file(
                    registry_path,
                    selected_names=self._registry_selection_filter(),
                )
                self._warn_missing_selected_drones()
            except Exception as exc:  # noqa: B902
                self.get_logger().error(f"Failed to load registry from '{registry_path}': {exc}")
                self._registry = Registry(specs=[], source_path=registry_path)

        self._procman = ManagerSupervisor(
            package=manager_pkg,
            executable=manager_exe,
            manager_profiles=manager_profiles,
            embedded_factories=embedded_factories,
            ros_domain_id=ros_domain_id,
            logger=self.get_logger(),
        )

        # Runtime desired-state overrides.
        self._force_start: set[str] = set()
        self._force_stop: set[str] = set()
        self._paused = False

        # Outbound client caches for fan-out commands.
        self._arm_clients: dict[str, rclpy.client.Client] = {}
        self._takeoff_clients: dict[str, ActionClient] = {}
        self._land_clients: dict[str, ActionClient] = {}
        self._go_to_clients: dict[str, ActionClient] = {}
        self._client_cache_lock = threading.Lock()
        self._swarm_action_lock = threading.Lock()
        self._active_swarm_context_lock = threading.Lock()
        self._active_swarm_context: _ActiveSwarmAction | None = None
        self._swarm_callback_group = ReentrantCallbackGroup()

        # Publisher: process states snapshot (JSON string).
        self._states_pub = self.create_publisher(String, '/swarm/process_states', 10)

        # Admin services.
        self.create_service(Trigger, '/swarm/reconcile_now', self._srv_reconcile_now)
        self.create_service(Trigger, '/swarm/reload_registry', self._srv_reload_registry)
        self.create_service(Trigger, '/swarm/start_all', self._srv_start_all)
        self.create_service(Trigger, '/swarm/stop_all', self._srv_stop_all)
        self.create_service(Trigger, '/swarm/registry_dump', self._srv_registry_dump)

        # On-demand per-drone process control.
        self.create_service(SpawnDrone, '/swarm/spawn_drone', self._srv_spawn_drone)
        self.create_service(SpawnDrone, '/swarm/stop_drone', self._srv_stop_drone)

        # Swarm-wide fan-out commands.
        # Arm stays as a service — it is instantaneous and needs no progress feedback.
        self.create_service(
            SwarmArm,
            '/swarm/all/arm',
            self._srv_all_arm,
            callback_group=self._swarm_callback_group,
        )
        # Takeoff / land / goto are action servers so callers receive full lifecycle
        # counters and a result code without polling.
        self._swarm_takeoff_server = ActionServer(
            self,
            SwarmTakeoff,
            '/swarm/all/takeoff',
            self._action_all_takeoff,
            callback_group=self._swarm_callback_group,
            cancel_callback=lambda _: self._cancel_swarm_action_callback('takeoff'),
        )
        self._swarm_land_server = ActionServer(
            self,
            SwarmLand,
            '/swarm/all/land',
            self._action_all_land,
            callback_group=self._swarm_callback_group,
            cancel_callback=lambda _: self._cancel_swarm_action_callback('land'),
        )
        self._swarm_go_to_server = ActionServer(
            self,
            SwarmGoTo,
            '/swarm/all/go_to',
            self._action_all_go_to,
            callback_group=self._swarm_callback_group,
            cancel_callback=lambda _: self._cancel_swarm_action_callback('go_to'),
        )

        # Timers.
        self._timer = self.create_timer(reconcile_period, self._on_timer)
        self._shutdown_started = False

        self.get_logger().info(
            'DroneOrchestrator ready. '
            f"registry_path='{registry_path}', selected={self._selected_drones}, "
            f"default_manager='{manager_pkg}/{manager_exe}', "
            f'drones={len(self._registry.list_all())}'
        )

    def _shutdown_managers(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True

        try:
            if rclpy.ok(context=self.context):
                self.get_logger().info('Stopping managed drone manager processes...')
            self._procman.stop_all()
        except Exception as exc:  # noqa: B902
            if rclpy.ok(context=self.context):
                self.get_logger().error(f'Failed to stop all manager processes during shutdown: {exc}')

    def destroy_node(self) -> bool:
        self._shutdown_managers()
        return super().destroy_node()

    # -------------------------
    # Parameter + YAML helpers
    # -------------------------
    def _read_selected_drones_param(self) -> list[str]:
        raw = self.get_parameter('selected_drones').value
        if isinstance(raw, (list, tuple)):
            return [str(x).strip() for x in raw if str(x).strip()]

        if isinstance(raw, str) and raw.strip():
            try:
                parsed = yaml.safe_load(raw)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:  # noqa: B902
                pass
        return []

    def _registry_selection_filter(self) -> list[str] | None:
        """Return the selected_names filter for Registry loads.

        An unset/empty selected_drones parameter means "no filter, load all",
        which the Registry API expects as None — it treats an explicit empty
        list as "select no drones".
        """
        return list(self._selected_drones) if self._selected_drones else None

    def _parse_manager_profiles(self, raw_yaml: str) -> dict[str, dict[str, Any]]:
        if not raw_yaml.strip():
            return {}

        parsed = yaml.safe_load(raw_yaml)
        if not isinstance(parsed, dict):
            raise ValueError('manager_profiles_yaml must decode to a mapping')

        out: dict[str, dict[str, Any]] = {}
        for key, value in parsed.items():
            manager_type = str(key).strip()
            if not manager_type:
                continue
            if not isinstance(value, dict):
                raise ValueError(f"manager profile '{manager_type}' must be a mapping")

            package = str(value.get('package', '')).strip()
            executable = str(value.get('executable', '')).strip()
            if not executable:
                raise ValueError(f"manager profile '{manager_type}' missing 'executable'")
            if not package:
                package = str(self.get_parameter('manager_package').value).strip() or 'flexible_drones_core'
            manager_owns_drone_prefix = bool(value.get('manager_owns_drone_prefix', False))
            node_name_template = str(value.get('node_name_template', '')).strip()
            launch_mode = str(value.get('launch_mode', 'subprocess')).strip().lower() or 'subprocess'
            if launch_mode not in {'subprocess', 'embedded'}:
                raise ValueError(f"manager profile '{manager_type}' has invalid launch_mode " f"'{launch_mode}'")

            out[manager_type] = {
                'package': package,
                'executable': executable,
                'manager_owns_drone_prefix': manager_owns_drone_prefix,
                'node_name_template': node_name_template,
                'launch_mode': launch_mode,
                'embedded_factory_module': str(value.get('embedded_factory_module', '')).strip(),
                'embedded_factory_func': str(value.get('embedded_factory_func', '')).strip(),
            }

        return out

    def _build_embedded_factories(self, manager_profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
        import importlib

        factories: dict[str, Any] = {}
        for manager_type, profile in manager_profiles.items():
            if profile.get('launch_mode') != 'embedded':
                continue

            factory_module = str(profile.get('embedded_factory_module', '')).strip()
            factory_func = str(profile.get('embedded_factory_func', '')).strip()
            if not factory_module or not factory_func:
                raise ValueError(
                    f"Manager profile '{manager_type}' has launch_mode: embedded but is missing "
                    "'embedded_factory_module' and/or 'embedded_factory_func'."
                )

            try:
                mod = importlib.import_module(factory_module)
            except ImportError as exc:
                raise RuntimeError(
                    f"Cannot load embedded factory for '{manager_type}': "
                    f"module '{factory_module}' not found. "
                    f'Ensure the backend package is built and sourced.'
                ) from exc

            fn = getattr(mod, factory_func, None)
            if fn is None:
                raise ValueError(
                    f"Embedded factory module '{factory_module}' has no attribute '{factory_func}'."
                )

            node = self
            factories[manager_type] = lambda config, _fn=fn, _node=node: _fn(config, _node)

        return factories

    def _warn_missing_selected_drones(self) -> None:
        if not self._selected_drones:
            return

        loaded = set(self._registry.names())
        missing = [name for name in self._selected_drones if name not in loaded]
        if missing:
            self.get_logger().warning(f'Selected drones not found in registry YAML: {missing}')

    # -------------------------
    # Timed reconcile + publish
    # -------------------------
    def _desired_specs(self) -> list[DroneSpec]:
        desired_by_name: dict[str, DroneSpec] = {}

        if not self._paused:
            for spec in self._registry.list_enabled():
                desired_by_name[spec.name] = spec

        for name in self._force_start:
            spec = self._registry.get(name)
            if spec is not None:
                desired_by_name[name] = spec.with_enabled(True)

        for name in self._force_stop:
            desired_by_name.pop(name, None)

        return list(desired_by_name.values())

    def _on_timer(self) -> None:
        self._procman.ensure_running(self._desired_specs())

        states = self._procman.list_states()
        msg = String()
        msg.data = json.dumps(
            {
                'stamp_unix': time.time(),
                'registry_source': self._registry.source_path,
                'registry_loaded_at_unix': self._registry.loaded_at_unix,
                'selected_drones': self._selected_drones,
                'paused': self._paused,
                'force_start': sorted(self._force_start),
                'force_stop': sorted(self._force_stop),
                'states': states,
            },
            sort_keys=True,
        )
        self._states_pub.publish(msg)

    # -------------------------
    # Services: admin/process
    # -------------------------
    def _srv_reconcile_now(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            self._on_timer()
            response.success = True
            response.message = 'Reconcile completed.'
        except Exception as exc:  # noqa: B902
            response.success = False
            response.message = f'Reconcile failed: {exc}'
        return response

    def _srv_reload_registry(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            if not self._registry.source_path:
                raise ValueError('No registry_path set; cannot reload.')

            self._selected_drones = self._read_selected_drones_param()
            self._registry.reload_from_file(
                self._registry.source_path,
                selected_names=self._registry_selection_filter(),
            )
            self._warn_missing_selected_drones()
            response.success = True
            response.message = (
                f"Registry reloaded from '{self._registry.source_path}'. " f'Drones={len(self._registry.list_all())}'
            )
        except Exception as exc:  # noqa: B902
            response.success = False
            response.message = f'Reload failed: {exc}'
        return response

    def _srv_start_all(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            self._paused = False
            self._force_stop.clear()
            self._procman.start_all(self._desired_specs())
            response.success = True
            response.message = 'Start-all requested.'
        except Exception as exc:  # noqa: B902
            response.success = False
            response.message = f'Start-all failed: {exc}'
        return response

    def _srv_stop_all(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            self._paused = True
            self._procman.stop_all()
            response.success = True
            response.message = 'Stop-all completed.'
        except Exception as exc:  # noqa: B902
            response.success = False
            response.message = f'Stop-all failed: {exc}'
        return response

    def _srv_registry_dump(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            data = {
                'source_path': self._registry.source_path,
                'loaded_at_unix': self._registry.loaded_at_unix,
                'selected_drones': self._selected_drones,
                'drones': [spec.to_dict() for spec in self._registry.list_all()],
            }
            response.success = True
            response.message = json.dumps(data, sort_keys=True)
        except Exception as exc:  # noqa: B902
            response.success = False
            response.message = f'Dump failed: {exc}'
        return response

    def _srv_spawn_drone(self, request: SpawnDrone.Request, response: SpawnDrone.Response) -> SpawnDrone.Response:
        drone_id = request.drone_id.strip()
        if not drone_id:
            response.success = False
            response.message = 'drone_id is empty'
            return response

        spec = self._registry.get(drone_id)
        if spec is None:
            response.success = False
            response.message = f"Drone '{drone_id}' is not in the active registry"
            return response

        try:
            self._force_stop.discard(drone_id)
            self._force_start.add(drone_id)
            response.success = self._procman.start_spec(spec.with_enabled(True))
            if response.success:
                response.message = f"Spawned drone '{drone_id}' as manager_type='{spec.manager_type}'"
            else:
                response.message = f"Failed to start manager for drone '{drone_id}'"
        except Exception as exc:  # noqa: B902
            response.success = False
            response.message = f"Failed to spawn '{drone_id}': {exc}"

        return response

    def _srv_stop_drone(self, request: SpawnDrone.Request, response: SpawnDrone.Response) -> SpawnDrone.Response:
        drone_id = request.drone_id.strip()
        if not drone_id:
            response.success = False
            response.message = 'drone_id is empty'
            return response

        if self._registry.get(drone_id) is None:
            response.success = False
            response.message = f"Drone '{drone_id}' is not in the active registry"
            return response

        self._force_start.discard(drone_id)
        self._force_stop.add(drone_id)

        stopped = self._procman.stop_drone(drone_id)
        if stopped:
            response.success = True
            response.message = f"Stopped drone '{drone_id}'"
        else:
            # Keep success true if requested stop already holds desired state.
            response.success = True
            response.message = f"Drone '{drone_id}' was not running; marked as force-stopped"
        return response

    # -------------------------
    # Services: swarm-wide fan-out
    # -------------------------
    def _running_drones(self) -> list[str]:
        return sorted(self._procman.list_running_names())

    def _srv_all_arm(self, request: SwarmArm.Request, response: SwarmArm.Response) -> SwarmArm.Response:
        if not self._swarm_action_lock.acquire(blocking=False):
            response.targeted = len(self._running_drones())
            response.unavailable = 0
            response.sent = 0
            response.succeeded = 0
            response.failed = 0
            response.response_timeout = 0
            response.success = False
            response.message = 'arm rejected: another swarm action is active'
            return response

        try:
            drones = self._running_drones()
            summary = self._execute_swarm_service(
                'arm',
                drones,
                self._ensure_arm_client,
                self._build_arm_request,
                self._arm_response_success,
            )
            response.targeted = summary.targeted
            response.unavailable = summary.unavailable
            response.sent = summary.sent
            response.succeeded = summary.succeeded
            response.failed = summary.failed
            response.response_timeout = summary.response_timeout
            response.success = summary.complete_success()
            response.message = _format_swarm_service_message('arm', summary)
        finally:
            self._end_swarm_action()
        return response

    def _action_all_takeoff(self, goal_handle) -> SwarmTakeoff.Result:
        result = SwarmTakeoff.Result()
        if not self._begin_swarm_action('takeoff', goal_handle, result):
            return result

        request = goal_handle.request
        try:
            drones = self._running_drones()
            duration_sec = self._duration_seconds_or_default(request.duration, 3.0)
            summary = self._execute_swarm_action(
                'takeoff',
                drones,
                self._ensure_takeoff_client,
                lambda: self._build_takeoff_goal(request.height, duration_sec),
                SwarmTakeoff.Feedback,
                goal_handle.publish_feedback,
            )

            self._fill_swarm_action_result(result, summary)
            self._finish_swarm_action_goal(goal_handle, summary)
            return result
        finally:
            self._end_swarm_action()

    def _action_all_land(self, goal_handle) -> SwarmLand.Result:
        result = SwarmLand.Result()
        if not self._begin_swarm_action('land', goal_handle, result):
            return result

        # Safety-critical: proceed with commandable drones regardless of
        # unavailable count.
        request = goal_handle.request
        try:
            drones = self._running_drones()
            duration_sec = self._duration_seconds_or_default(request.duration, 3.0)
            land_height = float(request.height)
            summary = self._execute_swarm_action(
                'land',
                drones,
                self._ensure_land_client,
                lambda: self._build_land_goal(land_height, duration_sec),
                SwarmLand.Feedback,
                goal_handle.publish_feedback,
            )

            self._fill_swarm_action_result(result, summary)
            self._finish_swarm_action_goal(goal_handle, summary)
            return result
        finally:
            self._end_swarm_action()

    def _action_all_go_to(self, goal_handle) -> SwarmGoTo.Result:
        request = goal_handle.request
        result = SwarmGoTo.Result()
        if request.frame == 0:
            self.get_logger().warning(
                'SwarmGoTo aborted: FRAME_ABSOLUTE is not supported at the '
                'swarm level. '
                'Use FRAME_RELATIVE_MAP (1) or FRAME_RELATIVE_BODY (2).'
            )
            result.return_code = 3
            goal_handle.abort()
            return result

        if not self._begin_swarm_action('go_to', goal_handle, result):
            return result

        try:
            drones = self._running_drones()
            duration_sec = self._duration_seconds_or_default(request.duration, 3.0)
            summary = self._execute_swarm_action(
                'go_to',
                drones,
                self._ensure_go_to_client,
                lambda: self._build_go_to_goal(
                    request.x,
                    request.y,
                    request.z,
                    request.yaw,
                    duration_sec,
                    request.frame,
                ),
                SwarmGoTo.Feedback,
                goal_handle.publish_feedback,
            )

            self._fill_swarm_action_result(result, summary)
            self._finish_swarm_action_goal(goal_handle, summary)
            return result
        finally:
            self._end_swarm_action()

    def _future_done(self, future) -> bool:
        return bool(future.done())

    def _wait_for_futures(
        self,
        pending: dict[str, Any],
        timeout_s: float,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> set[str]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        completed: set[str] = set()
        remaining = dict(pending)
        completion_event = threading.Event()
        completion_lock = threading.Lock()

        def mark_completed(drone_name: str) -> None:
            with completion_lock:
                if drone_name not in remaining:
                    return
                completed.add(drone_name)
                del remaining[drone_name]
            completion_event.set()

        def make_done_callback(drone_name: str) -> Callable[[Any], None]:
            def _done_callback(_future: Any) -> None:
                mark_completed(drone_name)

            return _done_callback

        for drone_name, future in list(remaining.items()):
            if self._future_done(future):
                mark_completed(drone_name)
                continue

            try:
                future.add_done_callback(make_done_callback(drone_name))
            except Exception:  # noqa: B902
                pass

        while True:
            with completion_lock:
                if not remaining:
                    break

            if cancel_requested is not None and cancel_requested():
                break

            timeout_remaining = deadline - time.monotonic()
            if timeout_remaining <= 0.0:
                break

            completion_event.wait(min(0.1, timeout_remaining))
            completion_event.clear()

            with completion_lock:
                futures_to_check = list(remaining.items())
            for drone_name, future in futures_to_check:
                if self._future_done(future):
                    mark_completed(drone_name)

        with completion_lock:
            futures_to_check = list(remaining.items())
        for drone_name, future in futures_to_check:
            if self._future_done(future):
                mark_completed(drone_name)

        return completed

    def _publish_swarm_action_feedback(
        self,
        summary: _SwarmActionSummary,
        feedback_factory: Callable[[], Any],
        publish_feedback: Callable[[Any], None],
    ) -> None:
        feedback = feedback_factory()
        feedback.targeted = summary.targeted
        feedback.unavailable = summary.unavailable
        feedback.sent = summary.sent
        feedback.accepted = summary.accepted
        feedback.rejected = summary.rejected
        feedback.goal_response_timeout = summary.goal_response_timeout
        feedback.succeeded = summary.succeeded
        feedback.failed = summary.failed
        feedback.canceled = summary.canceled
        feedback.result_timeout = summary.result_timeout
        publish_feedback(feedback)

    def _fill_swarm_action_result(self, result: Any, summary: _SwarmActionSummary) -> None:
        result.targeted = summary.targeted
        result.unavailable = summary.unavailable
        result.sent = summary.sent
        result.accepted = summary.accepted
        result.rejected = summary.rejected
        result.goal_response_timeout = summary.goal_response_timeout
        result.succeeded = summary.succeeded
        result.failed = summary.failed
        result.canceled = summary.canceled
        result.result_timeout = summary.result_timeout
        result.return_code = summary.return_code()
        self.get_logger().info(
            f'Swarm action result {json.dumps(summary.as_dict(), sort_keys=True)} ' f'return_code={result.return_code}'
        )

    def _finish_swarm_action_goal(self, goal_handle, summary: _SwarmActionSummary) -> None:
        return_code = summary.return_code()
        if return_code == _SWARM_RETURN_CODE_CANCELED:
            goal_handle.canceled()
        elif return_code == 2:
            goal_handle.abort()
        else:
            goal_handle.succeed()

    def _begin_swarm_action(self, action_name: str, goal_handle, result: Any) -> bool:
        if self._swarm_action_lock.acquire(blocking=False):
            self._set_active_swarm_context(_ActiveSwarmAction(action_name=action_name))
            return True

        self.get_logger().warning(f'{action_name} rejected: another swarm action is active')
        result.return_code = _SWARM_RETURN_CODE_BUSY
        goal_handle.abort()
        return False

    def _end_swarm_action(self) -> None:
        self._set_active_swarm_context(None)
        self._swarm_action_lock.release()

    def _set_active_swarm_context(self, context: _ActiveSwarmAction | None) -> None:
        with self._active_swarm_context_lock:
            self._active_swarm_context = context

    def _get_active_swarm_context(self) -> _ActiveSwarmAction | None:
        with self._active_swarm_context_lock:
            return self._active_swarm_context

    def _swarm_action_cancel_requested(self, context: _ActiveSwarmAction | None) -> bool:
        if context is None:
            return False
        with context.lock:
            return context.cancel_requested

    def _cancel_swarm_action_callback(self, action_name: str):
        context = self._get_active_swarm_context()
        if context is None or context.action_name != action_name:
            self.get_logger().info(f'Rejecting {action_name} cancel request: no matching active swarm action')
            return CancelResponse.REJECT

        self._request_swarm_action_cancel(context, f'{action_name} cancel requested')
        return CancelResponse.ACCEPT

    def _request_swarm_action_cancel(self, context: _ActiveSwarmAction, reason: str) -> None:
        with context.lock:
            first_request = not context.cancel_requested
            context.cancel_requested = True
            handles = dict(context.accepted_goal_handles)

        if first_request:
            self.get_logger().info(f'Swarm {context.action_name}: {reason}; forwarding cancel to accepted goals')
        self._enqueue_swarm_goal_cancels(context, handles, reason)

    def _record_swarm_goal_handle(self, context: _ActiveSwarmAction | None, drone_name: str, goal_handle: Any) -> None:
        if context is None:
            return
        with context.lock:
            context.accepted_goal_handles[drone_name] = goal_handle
            should_cancel = context.cancel_requested
        if should_cancel:
            self._enqueue_swarm_goal_cancels(
                context,
                {drone_name: goal_handle},
                'goal accepted after swarm cancel request',
            )

    def _record_swarm_result_future(self, context: _ActiveSwarmAction | None, drone_name: str, future: Any) -> None:
        if context is None:
            return
        with context.lock:
            context.result_futures[drone_name] = future

    def _enqueue_swarm_goal_cancels(
        self,
        context: _ActiveSwarmAction | None,
        handles: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        if context is None or not handles:
            return {}

        enqueued: dict[str, Any] = {}
        for drone_name, goal_handle in handles.items():
            with context.lock:
                if drone_name in context.cancel_futures:
                    continue
                context.cancel_futures[drone_name] = None

            try:
                cancel_future = goal_handle.cancel_goal_async()
            except Exception as exc:  # noqa: B902
                with context.lock:
                    context.cancel_futures.pop(drone_name, None)
                self.get_logger().warning(f'[{drone_name}] Failed to request cancel after {reason}: {exc}')
                continue

            with context.lock:
                context.cancel_futures[drone_name] = cancel_future
            enqueued[drone_name] = cancel_future
            self.get_logger().info(f'[{drone_name}] Requested cancel after {reason}')

        return enqueued

    def _wait_for_swarm_cancel_futures(self, context: _ActiveSwarmAction | None) -> set[str]:
        if context is None:
            return set()
        with context.lock:
            cancel_futures = {
                drone_name: future
                for drone_name, future in context.cancel_futures.items()
                if future is not None
            }
        if not cancel_futures:
            return set()
        return self._wait_for_futures(
            cancel_futures,
            self._swarm_goal_response_timeout_s,
        )

    def _execute_swarm_action(
        self,
        command: str,
        drone_names: Sequence[str],
        ensure_client: Callable[[str], ActionClient],
        build_goal: Callable[[], Any],
        feedback_factory: Callable[[], Any],
        publish_feedback: Callable[[Any], None],
    ) -> _SwarmActionSummary:
        summary = _SwarmActionSummary(targeted=len(drone_names))
        goal_futures: dict[str, Any] = {}
        context = self._get_active_swarm_context()

        def cancel_requested():
            return self._swarm_action_cancel_requested(context)

        for drone_name in drone_names:
            if cancel_requested():
                self.get_logger().info(f'Stopping {command} goal dispatch after swarm cancel request')
                break

            client = ensure_client(drone_name)
            if not client.wait_for_server(timeout_sec=0.2):
                summary.unavailable += 1
                self._publish_swarm_action_feedback(summary, feedback_factory, publish_feedback)
                continue

            try:
                goal_futures[drone_name] = client.send_goal_async(build_goal())
                summary.sent += 1
            except Exception as exc:  # noqa: B902
                summary.failed += 1
                self.get_logger().error(f'[{drone_name}] {command} goal send exception: {exc}')
            self._publish_swarm_action_feedback(summary, feedback_factory, publish_feedback)

        completed_goals = self._wait_for_futures(
            goal_futures,
            self._swarm_goal_response_timeout_s,
            cancel_requested=cancel_requested,
        )
        if cancel_requested():
            pending_goal_responses = {
                drone_name: future
                for drone_name, future in goal_futures.items()
                if drone_name not in completed_goals
            }
            if pending_goal_responses:
                self.get_logger().info(
                    f'Swarm {command}: draining goal responses for '
                    f'{self._swarm_cancel_goal_response_grace_s:.2f}s after cancel request'
                )
                completed_goals.update(
                    self._wait_for_futures(
                        pending_goal_responses,
                        self._swarm_cancel_goal_response_grace_s,
                    )
                )
        result_futures: dict[str, Any] = {}

        for drone_name, future in goal_futures.items():
            if drone_name not in completed_goals:
                if cancel_requested():
                    continue
                summary.goal_response_timeout += 1
                self.get_logger().warning(f'[{drone_name}] {command} goal response timeout')
                self._publish_swarm_action_feedback(summary, feedback_factory, publish_feedback)
                continue

            try:
                goal_handle = future.result()
            except Exception as exc:  # noqa: B902
                summary.failed += 1
                self.get_logger().error(f'[{drone_name}] {command} goal response exception: {exc}')
                self._publish_swarm_action_feedback(summary, feedback_factory, publish_feedback)
                continue

            if not goal_handle.accepted:
                summary.rejected += 1
                self.get_logger().warning(f'[{drone_name}] {command} goal rejected')
                self._publish_swarm_action_feedback(summary, feedback_factory, publish_feedback)
                continue

            summary.accepted += 1
            self.get_logger().info(f'[{drone_name}] {command} goal accepted')
            self._record_swarm_goal_handle(context, drone_name, goal_handle)
            result_futures[drone_name] = goal_handle.get_result_async()
            self._record_swarm_result_future(context, drone_name, result_futures[drone_name])
            self._publish_swarm_action_feedback(summary, feedback_factory, publish_feedback)

        completed_results = self._wait_for_futures(
            result_futures,
            self._swarm_result_timeout_s,
            cancel_requested=cancel_requested,
        )

        if cancel_requested():
            summary.canceled_by_request = True
            handles_to_cancel: dict[str, Any] = {}
            if context is not None:
                with context.lock:
                    handles_to_cancel = dict(context.accepted_goal_handles)
            self._enqueue_swarm_goal_cancels(
                context,
                handles_to_cancel,
                f'swarm {command} cancellation',
            )
            self._wait_for_swarm_cancel_futures(context)

        for drone_name, future in result_futures.items():
            if drone_name not in completed_results:
                if cancel_requested():
                    continue
                summary.result_timeout += 1
                self.get_logger().warning(f'[{drone_name}] {command} result timeout')
                self._publish_swarm_action_feedback(summary, feedback_factory, publish_feedback)
                continue

            try:
                wrapped = future.result()
                if wrapped.status == _GOAL_STATUS_CANCELED:
                    summary.canceled += 1
                    self.get_logger().warning(f'[{drone_name}] {command} result canceled')
                    self._publish_swarm_action_feedback(summary, feedback_factory, publish_feedback)
                    continue

                result = wrapped.result
                return_code = int(result.return_code)
                if return_code == 0:
                    summary.succeeded += 1
                    self.get_logger().info(f'[{drone_name}] {command} result success')
                else:
                    summary.failed += 1
                    self.get_logger().warning(
                        f'[{drone_name}] {command} result failure '
                        f'(return_code={return_code})'
                    )
            except Exception as exc:  # noqa: B902
                summary.failed += 1
                self.get_logger().error(f'[{drone_name}] {command} result exception: {exc}')

            self._publish_swarm_action_feedback(summary, feedback_factory, publish_feedback)

        if not cancel_requested():
            timed_out_handles: dict[str, Any] = {}
            if context is not None:
                with context.lock:
                    for drone_name, goal_handle in context.accepted_goal_handles.items():
                        if drone_name in result_futures and drone_name not in completed_results:
                            timed_out_handles[drone_name] = goal_handle
            if timed_out_handles:
                self._enqueue_swarm_goal_cancels(
                    context,
                    timed_out_handles,
                    f'swarm {command} result timeout',
                )
                self._wait_for_swarm_cancel_futures(context)
                self._publish_swarm_action_feedback(summary, feedback_factory, publish_feedback)

        return summary

    def _execute_swarm_service(
        self,
        command: str,
        drone_names: Sequence[str],
        ensure_client: Callable[[str], Any],
        build_request: Callable[[], Any],
        response_success: Callable[[Any], bool],
    ) -> _SwarmServiceSummary:
        summary = _SwarmServiceSummary(targeted=len(drone_names))
        futures: dict[str, Any] = {}

        for drone_name in drone_names:
            client = ensure_client(drone_name)
            if not client.wait_for_service(timeout_sec=0.2):
                summary.unavailable += 1
                continue

            try:
                futures[drone_name] = client.call_async(build_request())
                summary.sent += 1
            except Exception as exc:  # noqa: B902
                summary.failed += 1
                self.get_logger().error(f'[{drone_name}] {command} service send exception: {exc}')

        completed = self._wait_for_futures(
            futures,
            self._swarm_service_response_timeout_s,
        )

        for drone_name, future in futures.items():
            if drone_name not in completed:
                summary.response_timeout += 1
                self.get_logger().warning(f'[{drone_name}] {command} service response timeout')
                continue

            try:
                response = future.result()
                if response_success(response):
                    summary.succeeded += 1
                    self.get_logger().info(f'[{drone_name}] {command} OK')
                else:
                    summary.failed += 1
                    self.get_logger().warning(f'[{drone_name}] {command} failed')
            except Exception as exc:  # noqa: B902
                summary.failed += 1
                self.get_logger().error(f'[{drone_name}] {command} service call exception: {exc}')

        return summary

    # -------------------------
    # Fan-out dispatch helpers
    # -------------------------
    def _duration_seconds_or_default(self, duration: RosDuration, default_sec: float) -> float:
        duration_sec = float(duration.sec) + float(duration.nanosec) / 1_000_000_000.0
        return duration_sec if duration_sec > 0.0 else float(default_sec)

    def _seconds_to_duration(self, duration_sec: float) -> RosDuration:
        dur = RosDuration()
        dur.sec = int(duration_sec)
        dur.nanosec = int((duration_sec % 1.0) * 1_000_000_000)
        return dur

    def _build_arm_request(self) -> Arm.Request:
        request = Arm.Request()
        request.arm = True
        request.timeout_sec = float(self._swarm_service_response_timeout_s)
        return request

    def _arm_response_success(self, response: Arm.Response) -> bool:
        return bool(response.success)

    def _ensure_arm_client(self, drone_name: str):
        with self._client_cache_lock:
            client = self._arm_clients.get(drone_name)
            if client is None:
                client = self.create_client(
                    Arm,
                    f'/{drone_name}/arm',
                    callback_group=self._swarm_callback_group,
                )
                self._arm_clients[drone_name] = client
            return client

    def _ensure_takeoff_client(self, drone_name: str) -> ActionClient:
        with self._client_cache_lock:
            client = self._takeoff_clients.get(drone_name)
            if client is None:
                client = ActionClient(
                    self,
                    Takeoff,
                    f'/{drone_name}/takeoff',
                    callback_group=self._swarm_callback_group,
                )
                self._takeoff_clients[drone_name] = client
            return client

    def _build_takeoff_goal(self, height: float, duration_sec: float) -> Takeoff.Goal:
        goal = Takeoff.Goal()
        goal.height = float(height)
        goal.duration = self._seconds_to_duration(duration_sec)
        return goal

    def _ensure_land_client(self, drone_name: str) -> ActionClient:
        with self._client_cache_lock:
            client = self._land_clients.get(drone_name)
            if client is None:
                client = ActionClient(
                    self,
                    Land,
                    f'/{drone_name}/land',
                    callback_group=self._swarm_callback_group,
                )
                self._land_clients[drone_name] = client
            return client

    def _build_land_goal(self, height: float, duration_sec: float) -> Land.Goal:
        goal = Land.Goal()
        goal.height = height
        goal.duration = self._seconds_to_duration(duration_sec)
        return goal

    def _ensure_go_to_client(self, drone_name: str) -> ActionClient:
        with self._client_cache_lock:
            client = self._go_to_clients.get(drone_name)
            if client is None:
                client = ActionClient(
                    self,
                    GoTo,
                    f'/{drone_name}/go_to',
                    callback_group=self._swarm_callback_group,
                )
                self._go_to_clients[drone_name] = client
            return client

    def _build_go_to_goal(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        duration_sec: float,
        frame: int,
    ) -> GoTo.Goal:
        goal = GoTo.Goal()
        goal.goal.x = float(x)
        goal.goal.y = float(y)
        goal.goal.z = float(z)
        goal.yaw = float(yaw)
        goal.frame = int(frame)
        goal.duration = self._seconds_to_duration(duration_sec)
        return goal


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DroneOrchestratorNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok(context=node.context):
            rclpy.shutdown(context=node.context)

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

"""Unit tests for orchestrator helpers that do not need a ROS runtime."""

import importlib
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from flexible_drones_core.registry import DroneSpec, Registry


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _install_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    if '.' in name:
        parent_name, child_name = name.rsplit('.', 1)
        parent = sys.modules.get(parent_name)
        if parent is None:
            parent = _install_module(parent_name)
        setattr(parent, child_name, module)
    return module


def _make_slotted(name, slots):
    """Return a class whose instances reject attributes not in slots."""
    return type(name, (), {'__slots__': list(slots)})


def _dummy_service(name: str):
    return type(
        name,
        (),
        {
            'Request': type(f'{name}Request', (), {}),
            'Response': type(f'{name}Response', (), {}),
        },
    )


class _GoToGoal:
    """Strict GoTo.Goal stub with a pre-initialized nested point."""

    __slots__ = ['group_mask', 'frame', 'goal', 'yaw', 'duration']

    def __init__(self):
        self.goal = SimpleNamespace(x=0.0, y=0.0, z=0.0)


def _load_orchestrator_module():
    sys.modules.pop('flexible_drones_core.orchestrator.drone_orchestrator_node', None)

    _SWARM_RESULT = [
        'targeted',
        'unavailable',
        'sent',
        'accepted',
        'rejected',
        'goal_response_timeout',
        'succeeded',
        'failed',
        'canceled',
        'result_timeout',
        'return_code',
    ]
    _SWARM_FEEDBACK = [
        'targeted',
        'unavailable',
        'sent',
        'accepted',
        'rejected',
        'goal_response_timeout',
        'succeeded',
        'failed',
        'canceled',
        'result_timeout',
    ]

    _install_module(
        'rclpy',
        ok=lambda context=None: True,
        init=lambda args=None: None,
        shutdown=lambda context=None: None,
    )
    _install_module(
        'rclpy.action',
        ActionClient=type('ActionClient', (), {}),
        ActionServer=type(
            'ActionServer',
            (),
            {
                '__init__': lambda self, *a, **kw: None,
            },
        ),
        CancelResponse=SimpleNamespace(REJECT=1, ACCEPT=2),
    )
    _install_module('rclpy.node', Node=type('Node', (), {}))
    _install_module(
        'rclpy.callback_groups',
        ReentrantCallbackGroup=type('ReentrantCallbackGroup', (), {}),
    )
    _install_module('std_msgs.msg', String=type('String', (), {}))
    _install_module('std_srvs.srv', Trigger=_dummy_service('Trigger'))
    # Duration with slots so tests catch writes to non-existent fields.
    _install_module(
        'builtin_interfaces.msg',
        Duration=_make_slotted('Duration', ['sec', 'nanosec']),
    )

    # Strict Arm.Request: only the two fields Arm.srv defines.
    _ArmRequest = _make_slotted('ArmRequest', ['arm', 'timeout_sec'])
    _ArmResponse = _make_slotted('ArmResponse', ['success'])
    _Arm = type('Arm', (), {'Request': _ArmRequest, 'Response': _ArmResponse})

    _install_module(
        'flexible_drones_msgs.action',
        GoTo=type(
            'GoTo',
            (),
            {
                'Goal': _GoToGoal,
                'Result': _make_slotted('GoToResult', ['return_code']),
                'Feedback': _make_slotted(
                    'GoToFeedback',
                    ['current_position', 'current_yaw'],
                ),
            },
        ),
        Land=type(
            'Land',
            (),
            {
                'Goal': _make_slotted('LandGoal', ['group_mask', 'height', 'duration']),
                'Result': _make_slotted('LandResult', ['return_code']),
                'Feedback': _make_slotted('LandFeedback', ['current_height']),
            },
        ),
        Takeoff=type(
            'Takeoff',
            (),
            {
                'Goal': _make_slotted('TakeoffGoal', ['group_mask', 'height', 'duration']),
                'Result': _make_slotted('TakeoffResult', ['return_code']),
                'Feedback': _make_slotted('TakeoffFeedback', ['current_height']),
            },
        ),
        SwarmTakeoff=type(
            'SwarmTakeoff',
            (),
            {
                'Goal': _make_slotted('SwarmTakeoffGoal', ['height', 'duration']),
                'Result': _make_slotted('SwarmTakeoffResult', _SWARM_RESULT),
                'Feedback': _make_slotted('SwarmTakeoffFeedback', _SWARM_FEEDBACK),
            },
        ),
        SwarmLand=type(
            'SwarmLand',
            (),
            {
                'Goal': _make_slotted('SwarmLandGoal', ['height', 'duration']),
                'Result': _make_slotted('SwarmLandResult', _SWARM_RESULT),
                'Feedback': _make_slotted('SwarmLandFeedback', _SWARM_FEEDBACK),
            },
        ),
        SwarmGoTo=type(
            'SwarmGoTo',
            (),
            {
                'Goal': _make_slotted(
                    'SwarmGoToGoal',
                    ['x', 'y', 'z', 'yaw', 'duration', 'frame'],
                ),
                'Result': _make_slotted('SwarmGoToResult', _SWARM_RESULT),
                'Feedback': _make_slotted('SwarmGoToFeedback', _SWARM_FEEDBACK),
            },
        ),
    )
    _install_module(
        'flexible_drones_msgs.srv',
        Arm=_Arm,
        SpawnDrone=_dummy_service('SpawnDrone'),
        SwarmArm=type(
            'SwarmArm',
            (),
            {
                'Request': type('SwarmArmRequest', (), {}),
                'Response': _make_slotted(
                    'SwarmArmResponse',
                    [
                        'success',
                        'message',
                        'targeted',
                        'unavailable',
                        'sent',
                        'succeeded',
                        'failed',
                        'response_timeout',
                    ],
                ),
            },
        ),
    )

    return importlib.import_module('flexible_drones_core.orchestrator.drone_orchestrator_node')


# ---------------------------------------------------------------------------
# Existing orchestrator-logic tests
# ---------------------------------------------------------------------------


def test_read_selected_drones_param_accepts_list_and_yaml_string():
    """Accept both list-valued and YAML-string selected_drones parameters."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)

    node.get_parameter = lambda name: SimpleNamespace(value=['cf1', ' cf2 ', ''])
    assert node._read_selected_drones_param() == ['cf1', 'cf2']

    node.get_parameter = lambda name: SimpleNamespace(value='[cf3, cf4]')
    assert node._read_selected_drones_param() == ['cf3', 'cf4']


def test_registry_selection_filter_maps_empty_selection_to_none():
    """Map an unset selected_drones parameter to None so the registry loads all."""
    # The Registry API treats an explicit empty selected_names list as "select
    # no drones", so the node must pass None (load all) when no selection is set.
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)

    node._selected_drones = []
    assert node._registry_selection_filter() is None

    node._selected_drones = ['cf1', 'cf2']
    assert node._registry_selection_filter() == ['cf1', 'cf2']


def test_reload_registry_without_selection_loads_all_drones(tmp_path):
    """Reloading with no selected_drones set must load every registry entry."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)

    registry_yaml = tmp_path / 'registry.yaml'
    registry_yaml.write_text(
        """
        drones:
          - name: cf1
            uri: radio://cf1
            type: crazyflie
            manager_type: cf_manager
          - name: cf2
            uri: radio://cf2
            type: crazyflie
            manager_type: cf_manager
        """,
        encoding='utf-8',
    )

    node._registry = Registry(specs=[], source_path=str(registry_yaml))
    node.get_parameter = lambda name: SimpleNamespace(value=[])
    node.get_logger = lambda: SimpleNamespace(
        info=lambda msg: None,
        warning=lambda msg: None,
        error=lambda msg: None,
    )

    response = SimpleNamespace(success=False, message='')
    node._srv_reload_registry(SimpleNamespace(), response)

    assert response.success is True, response.message
    assert node._registry.names() == ['cf1', 'cf2']


def test_parse_manager_profiles_and_desired_specs_work_without_ros_runtime():
    """Exercise profile parsing and desired-state logic without a ROS node."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    node.get_parameter = lambda name: SimpleNamespace(value='fallback_pkg')

    profiles = node._parse_manager_profiles(
        """
        cf_manager:
          package: fallback_pkg
          executable: cf_manager_node
          launch_mode: embedded
          embedded_factory_module: test_backend.factory
          embedded_factory_func: create_handle
        pihawk_manager:
          package: pihawk_pkg
          executable: pihawk_manager_node
          node_name_template: "{name}_px_manager"
        """
    )

    assert profiles['cf_manager']['package'] == 'fallback_pkg'
    assert profiles['cf_manager']['launch_mode'] == 'embedded'
    assert profiles['cf_manager']['embedded_factory_module'] == 'test_backend.factory'
    assert profiles['cf_manager']['embedded_factory_func'] == 'create_handle'
    assert profiles['pihawk_manager']['package'] == 'pihawk_pkg'
    assert profiles['pihawk_manager']['node_name_template'] == '{name}_px_manager'

    cf1 = DroneSpec(name='cf1', uri='radio://cf1', type='crazyflie', manager_type='cf_manager', enabled=True)
    cf2 = DroneSpec(name='cf2', uri='radio://cf2', type='crazyflie', manager_type='cf_manager', enabled=False)
    node._registry = Registry(specs=[cf1, cf2])
    node._paused = False
    node._force_start = {'cf2'}
    node._force_stop = {'cf1'}

    desired = node._desired_specs()

    assert [spec.name for spec in desired] == ['cf2']
    assert desired[0].enabled is True


def test_registry_reload_swaps_specs_after_new_snapshot_is_ready(monkeypatch):
    """Readers should keep seeing the old registry until reload has a full replacement."""
    cf1 = DroneSpec(name='cf1', uri='radio://cf1', type='crazyflie', manager_type='cf_manager', enabled=True)
    cf2 = DroneSpec(name='cf2', uri='radio://cf2', type='crazyflie', manager_type='cf_manager', enabled=True)
    registry = Registry(specs=[cf1], source_path='/tmp/fleet.yaml')

    build_started = threading.Event()
    finish_build = threading.Event()
    original_build = Registry._build_specs_by_name

    def load_specs(path, selected_names=None):
        return [cf2]

    def slow_build(specs):
        build_started.set()
        assert finish_build.wait(timeout=1.0)
        return original_build(specs)

    monkeypatch.setattr(Registry, 'load_specs_from_yaml', staticmethod(load_specs))
    monkeypatch.setattr(Registry, '_build_specs_by_name', staticmethod(slow_build))

    thread = threading.Thread(target=registry.reload_from_file)
    thread.start()
    assert build_started.wait(timeout=1.0)
    assert registry.names() == ['cf1']

    finish_build.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert registry.names() == ['cf2']


def test_embedded_profile_requires_factory_fields():
    """Embedded manager profiles must declare the backend factory explicitly."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    node.get_logger = lambda: SimpleNamespace(warning=lambda msg: None)

    with pytest.raises(ValueError, match="missing 'embedded_factory_module'"):
        node._build_embedded_factories(
            {
                'custom_manager': {'launch_mode': 'embedded'},
            }
        )


def test_spawn_drone_does_not_clear_global_pause():
    """Starting one drone should not resume all enabled drones after stop_all."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)

    cf1 = DroneSpec(name='cf1', uri='radio://cf1', type='crazyflie', manager_type='cf_manager', enabled=True)
    node._registry = Registry(specs=[cf1])
    node._paused = True
    node._force_start = set()
    node._force_stop = {'cf1'}

    started = []
    node._procman = SimpleNamespace(start_spec=lambda spec: started.append(spec) or True)

    request = SimpleNamespace(drone_id='cf1')
    response = SimpleNamespace(success=False, message='')
    node._srv_spawn_drone(request, response)

    assert response.success is True
    assert node._paused is True
    assert node._force_start == {'cf1'}
    assert node._force_stop == set()
    assert [spec.name for spec in started] == ['cf1']


def test_spawn_drone_reports_start_failure():
    """Spawn service failure should reflect manager startup failure."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)

    cf1 = DroneSpec(name='cf1', uri='radio://cf1', type='crazyflie', manager_type='cf_manager', enabled=True)
    node._registry = Registry(specs=[cf1])
    node._force_start = set()
    node._force_stop = {'cf1'}
    node._procman = SimpleNamespace(start_spec=lambda spec: False)

    request = SimpleNamespace(drone_id='cf1')
    response = SimpleNamespace(success=True, message='')
    node._srv_spawn_drone(request, response)

    assert response.success is False
    assert response.message == "Failed to start manager for drone 'cf1'"
    assert node._force_start == {'cf1'}
    assert node._force_stop == set()


def test_stop_drone_rejects_unknown_drone():
    """Stop service should fail fast for names absent from the registry."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)

    cf1 = DroneSpec(name='cf1', uri='radio://cf1', type='crazyflie', manager_type='cf_manager', enabled=True)
    node._registry = Registry(specs=[cf1])
    node._force_start = set()
    node._force_stop = set()
    node._procman = SimpleNamespace(stop_drone=lambda name: True)

    request = SimpleNamespace(drone_id='ghost')
    response = SimpleNamespace(success=True, message='')
    node._srv_stop_drone(request, response)

    assert response.success is False
    assert response.message == "Drone 'ghost' is not in the active registry"
    assert node._force_start == set()
    assert node._force_stop == set()


def test_stop_drone_known_not_running_is_idempotent_success():
    """Known drones that are already stopped should remain force-stopped."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)

    cf1 = DroneSpec(name='cf1', uri='radio://cf1', type='crazyflie', manager_type='cf_manager', enabled=True)
    node._registry = Registry(specs=[cf1])
    node._force_start = {'cf1'}
    node._force_stop = set()
    node._procman = SimpleNamespace(stop_drone=lambda name: False)

    request = SimpleNamespace(drone_id='cf1')
    response = SimpleNamespace(success=False, message='')
    node._srv_stop_drone(request, response)

    assert response.success is True
    assert response.message == "Drone 'cf1' was not running; marked as force-stopped"
    assert node._force_start == set()
    assert node._force_stop == {'cf1'}


# ---------------------------------------------------------------------------
# Swarm action-callback tests (fake clients)
# ---------------------------------------------------------------------------


class _FakeActionClient:
    """Records goals sent and controls wait_for_server outcome."""

    def __init__(
        self,
        available=True,
        *,
        accepted=True,
        return_code=0,
        goal_done=True,
        result_done=True,
    ):
        self._available = available
        self._accepted = accepted
        self._return_code = return_code
        self._goal_done = goal_done
        self._result_done = result_done
        self.sent_goals = []
        self.goal_handles = []

    def wait_for_server(self, timeout_sec):
        return self._available

    def send_goal_async(self, goal):
        self.sent_goals.append(goal)
        if not self._goal_done:
            return _FakeFuture(done=False)
        goal_handle = _FakeGoalHandle(
            accepted=self._accepted,
            return_code=self._return_code,
            result_done=self._result_done,
        )
        self.goal_handles.append(goal_handle)
        return _FakeFuture(result=goal_handle)


class _FakeFuture:
    """Small rclpy-like future for pure logic tests."""

    def __init__(self, *, result=None, done=True, exception=None):
        self._result = result
        self._done = done
        self._exception = exception

    def done(self):
        return self._done

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result

    def add_done_callback(self, cb):
        pass


class _CallbackFuture:
    """Future stub that can complete through registered callbacks."""

    def __init__(self, result=None):
        self._done = False
        self._result = result
        self.callbacks = []

    def done(self):
        return self._done

    def result(self):
        return self._result

    def add_done_callback(self, cb):
        self.callbacks.append(cb)

    def complete(self, result=None):
        if result is not None:
            self._result = result
        self._done = True
        for cb in list(self.callbacks):
            cb(self)


class _FakeGoalHandle:
    """Small action goal handle for pure logic tests."""

    def __init__(self, *, accepted=True, return_code=0, result_done=True):
        self.accepted = accepted
        self._return_code = return_code
        self._result_done = result_done
        self.cancel_requests = 0

    def get_result_async(self):
        return _FakeFuture(
            done=self._result_done,
            result=SimpleNamespace(
                status=0,
                result=SimpleNamespace(return_code=self._return_code),
            ),
        )

    def cancel_goal_async(self):
        self.cancel_requests += 1
        return _FakeFuture(result=SimpleNamespace())


class _FakeServiceClient:
    """Records requests sent and controls wait_for_service outcome."""

    def __init__(self, available=True):
        self._available = available
        self.sent_requests = []

    def wait_for_service(self, timeout_sec):
        return self._available

    def call_async(self, req):
        self.sent_requests.append(req)
        return _FakeFuture(result=SimpleNamespace(success=True))


def _fake_goal_handle(request):
    published = []
    terminal_states = []
    return (
        SimpleNamespace(
            request=request,
            publish_feedback=lambda fb: published.append(fb),
            succeed=lambda: terminal_states.append('succeed'),
            abort=lambda: terminal_states.append('abort'),
            canceled=lambda: terminal_states.append('canceled'),
            terminal_states=terminal_states,
        ),
        published,
    )


def _duration(seconds):
    return SimpleNamespace(
        sec=int(seconds),
        nanosec=int((float(seconds) % 1.0) * 1_000_000_000),
    )


def _configure_swarm_action_node(node):
    node._swarm_goal_response_timeout_s = 0.01
    node._swarm_cancel_goal_response_grace_s = 0.01
    node._swarm_result_timeout_s = 0.01
    node._swarm_service_response_timeout_s = 0.01
    node._client_cache_lock = threading.Lock()
    node._swarm_action_lock = threading.Lock()
    node._active_swarm_context_lock = threading.Lock()
    node._active_swarm_context = None
    node.get_logger = lambda: SimpleNamespace(
        info=lambda msg: None,
        warning=lambda msg: None,
        error=lambda msg: None,
    )


def test_wait_for_futures_wakes_from_done_callback():
    """_wait_for_futures should use future callbacks instead of timeout polling."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)

    future = _CallbackFuture()

    def complete_future():
        future.complete()

    timer = threading.Timer(0.02, complete_future)
    timer.start()
    try:
        completed = node._wait_for_futures({'cf1': future}, timeout_s=1.0)
    finally:
        timer.cancel()

    assert completed == {'cf1'}
    assert future.callbacks


def test_action_all_takeoff_reads_height_and_duration():
    """_action_all_takeoff reads both height and duration from the swarm goal."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)

    node._running_drones = lambda: ['cf1', 'cf2']
    node._takeoff_clients = {
        'cf1': _FakeActionClient(),
        'cf2': _FakeActionClient(),
    }

    goal_handle, _ = _fake_goal_handle(SimpleNamespace(height=1.2, duration=_duration(4.0)))
    result = node._action_all_takeoff(goal_handle)

    sent_goal = node._takeoff_clients['cf1'].sent_goals[0]
    assert sent_goal.height == pytest.approx(1.2)
    assert sent_goal.duration.sec == 4
    assert result.accepted == 2
    assert result.unavailable == 0
    assert result.return_code == 0


def test_action_all_takeoff_defaults_duration_when_zero():
    """Duration of 0 in the swarm goal falls back to the 3.0 s default."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)

    node._running_drones = lambda: ['cf1']
    node._takeoff_clients = {'cf1': _FakeActionClient()}

    goal_handle, _ = _fake_goal_handle(SimpleNamespace(height=0.5, duration=_duration(0.0)))
    node._action_all_takeoff(goal_handle)

    sent_goal = node._takeoff_clients['cf1'].sent_goals[0]
    assert sent_goal.duration.sec == 3


def test_swarm_action_summary_empty_swarm_is_success():
    """No targeted drones is distinct from all targeted drones failing."""
    module = _load_orchestrator_module()

    summary = module._SwarmActionSummary(targeted=0)

    assert summary.return_code() == 0


def test_swarm_action_summary_canceled_request_returns_code_5():
    """Explicit top-level swarm cancellation has its own return code."""
    module = _load_orchestrator_module()

    summary = module._SwarmActionSummary(targeted=2, accepted=2, canceled_by_request=True)

    assert summary.return_code() == 5


def test_build_arm_request_sets_timeout_sec():
    """Swarm arm requests should send an explicit per-drone timeout."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    node._swarm_service_response_timeout_s = 7.5

    request = node._build_arm_request()

    assert request.arm is True
    assert request.timeout_sec == pytest.approx(7.5)


def test_srv_all_arm_rejects_when_swarm_action_active():
    """Swarm arm should not run concurrently with another top-level swarm action."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)
    node._running_drones = lambda: ['cf1', 'cf2']

    node._swarm_action_lock.acquire()
    try:
        response = SimpleNamespace()
        node._srv_all_arm(SimpleNamespace(), response)
    finally:
        node._swarm_action_lock.release()

    assert response.success is False
    assert response.message == 'arm rejected: another swarm action is active'
    assert response.targeted == 2
    assert response.sent == 0
    assert response.succeeded == 0


def test_srv_all_arm_releases_swarm_action_lock_after_success():
    """Swarm arm should release the top-level action lock after fan-out completes."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)

    node._running_drones = lambda: ['cf1']
    node._arm_clients = {'cf1': _FakeServiceClient()}

    response = SimpleNamespace()
    node._srv_all_arm(SimpleNamespace(), response)

    assert response.success is True
    assert response.targeted == 1
    assert response.succeeded == 1
    assert node._swarm_action_lock.acquire(blocking=False)
    node._swarm_action_lock.release()


def test_swarm_service_message_is_prose_not_json():
    """Swarm service messages are human-facing; counts live in response fields."""
    module = _load_orchestrator_module()

    message = module._format_swarm_service_message(
        'arm',
        module._SwarmServiceSummary(
            targeted=3,
            unavailable=1,
            succeeded=2,
            failed=0,
            response_timeout=0,
        ),
    )

    assert message == ('arm completed: 2 succeeded, 0 failed, 1 unavailable, 0 timed out ' '(3 targeted).')
    assert not message.startswith('{')


def test_embedded_factory_is_loaded_from_profile():
    """Embedded manager profiles should load backend factories by module/function name."""
    calls = []

    def create_handle(spec, node):
        calls.append((spec, node))
        return 'handle'

    _install_module('test_backend')
    _install_module('test_backend.factory', create_handle=create_handle)
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    profiles = {
        'custom_manager': {
            'launch_mode': 'embedded',
            'embedded_factory_module': 'test_backend.factory',
            'embedded_factory_func': 'create_handle',
        }
    }

    factories = node._build_embedded_factories(profiles)
    spec = DroneSpec(name='d1', type='demo', uri='sim://d1', manager_type='custom_manager')

    assert factories['custom_manager'](spec) == 'handle'
    assert calls == [(spec, node)]


def test_action_all_go_to_reads_frame_not_relative():
    """_action_all_go_to must read request.frame (not request.relative)."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)

    node._running_drones = lambda: ['cf1']
    node._go_to_clients = {'cf1': _FakeActionClient()}

    # Request has only the fields SwarmGoTo.action defines.  Accessing
    # request.relative from old code would raise AttributeError here.
    request = SimpleNamespace(
        x=0.5,
        y=0.0,
        z=0.3,
        yaw=0.1,
        duration=_duration(3.0),
        frame=2,
    )
    goal_handle, _ = _fake_goal_handle(request)
    result = node._action_all_go_to(goal_handle)

    sent_goal = node._go_to_clients['cf1'].sent_goals[0]
    assert sent_goal.frame == 2
    assert result.accepted == 1
    assert result.return_code == 0
    assert goal_handle.terminal_states == ['succeed']


def test_action_all_go_to_partial_sets_return_code_1():
    """Partial dispatch (some drones unavailable) sets return_code=1, not 0."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)

    node._running_drones = lambda: ['cf1', 'cf2']
    node._go_to_clients = {
        'cf1': _FakeActionClient(),
        'cf2': _FakeActionClient(available=False),
    }

    goal_handle, _ = _fake_goal_handle(
        SimpleNamespace(
            x=0.0,
            y=0.0,
            z=1.0,
            yaw=0.0,
            duration=_duration(3.0),
            frame=1,
        )
    )
    result = node._action_all_go_to(goal_handle)

    assert result.accepted == 1
    assert result.unavailable == 1
    assert result.return_code == 1
    assert goal_handle.terminal_states == ['succeed']


def test_action_all_takeoff_rejects_when_swarm_action_active():
    """Only one top-level swarm action may run at a time."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)

    node._swarm_action_lock.acquire()
    try:
        goal_handle, _ = _fake_goal_handle(
            SimpleNamespace(
                height=1.0,
                duration=_duration(3.0),
            )
        )
        result = node._action_all_takeoff(goal_handle)
    finally:
        node._swarm_action_lock.release()

    assert result.return_code == 4
    assert goal_handle.terminal_states == ['abort']


def test_action_all_land_rejected_goal_counts_as_none_succeeded():
    """Rejected per-drone goals should set rejected=1, not unavailable."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)

    node._running_drones = lambda: ['cf1']
    node._land_clients = {'cf1': _FakeActionClient(accepted=False)}

    goal_handle, _ = _fake_goal_handle(SimpleNamespace(height=0.1, duration=_duration(5.5)))
    result = node._action_all_land(goal_handle)

    sent_goal = node._land_clients['cf1'].sent_goals[0]
    assert sent_goal.duration.sec == 5
    assert sent_goal.duration.nanosec == 500_000_000
    assert result.unavailable == 0
    assert result.rejected == 1
    assert result.accepted == 0
    assert result.return_code == 2
    assert goal_handle.terminal_states == ['abort']


def test_action_all_land_result_failure_counts_as_none_succeeded():
    """Accepted goals with nonzero result codes should make the swarm action fail."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)

    node._running_drones = lambda: ['cf1']
    node._land_clients = {'cf1': _FakeActionClient(return_code=7)}

    goal_handle, _ = _fake_goal_handle(SimpleNamespace(height=0.1, duration=_duration(3.0)))
    result = node._action_all_land(goal_handle)

    assert result.accepted == 1
    assert result.failed == 1
    assert result.unavailable == 0
    assert result.return_code == 2
    assert goal_handle.terminal_states == ['abort']


def test_action_all_land_result_timeout_counts_as_none_succeeded():
    """Accepted goals that never return should be result timeouts."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)

    node._running_drones = lambda: ['cf1']
    node._land_clients = {'cf1': _FakeActionClient(result_done=False)}

    goal_handle, _ = _fake_goal_handle(SimpleNamespace(height=0.1, duration=_duration(3.0)))
    result = node._action_all_land(goal_handle)

    assert result.accepted == 1
    assert result.result_timeout == 1
    assert result.unavailable == 0
    assert result.return_code == 2
    assert goal_handle.terminal_states == ['abort']
    assert node._land_clients['cf1'].goal_handles[0].cancel_requests == 1


def test_action_all_takeoff_cancel_fans_out_to_accepted_goals():
    """Top-level swarm cancellation should cancel all accepted per-drone goals."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)
    node._swarm_result_timeout_s = 1.0

    client = _FakeActionClient(result_done=False)
    node._running_drones = lambda: ['cf1']
    node._takeoff_clients = {'cf1': client}

    goal_handle, _ = _fake_goal_handle(SimpleNamespace(height=1.0, duration=_duration(3.0)))
    result_box = {}
    thread = threading.Thread(
        target=lambda: result_box.setdefault('result', node._action_all_takeoff(goal_handle))
    )
    thread.start()
    try:
        deadline = time.time() + 1.0
        while not client.goal_handles and time.time() < deadline:
            time.sleep(0.01)
        assert client.goal_handles, 'per-drone goal should be accepted before cancel'

        response = node._cancel_swarm_action_callback('takeoff')
        thread.join(timeout=1.0)
    finally:
        if thread.is_alive():
            thread.join(timeout=1.0)

    assert response == module.CancelResponse.ACCEPT
    assert not thread.is_alive()
    assert client.goal_handles[0].cancel_requests == 1
    assert result_box['result'].return_code == 5
    assert goal_handle.terminal_states == ['canceled']


def test_action_all_land_cancel_fans_out_to_accepted_goals():
    """Top-level swarm land cancellation should cancel all accepted per-drone goals."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)
    node._swarm_result_timeout_s = 1.0

    client = _FakeActionClient(result_done=False)
    node._running_drones = lambda: ['cf1']
    node._land_clients = {'cf1': client}

    goal_handle, _ = _fake_goal_handle(SimpleNamespace(height=0.1, duration=_duration(3.0)))
    result_box = {}
    thread = threading.Thread(
        target=lambda: result_box.setdefault('result', node._action_all_land(goal_handle))
    )
    thread.start()
    try:
        deadline = time.time() + 1.0
        while not client.goal_handles and time.time() < deadline:
            time.sleep(0.01)
        assert client.goal_handles, 'per-drone goal should be accepted before cancel'

        response = node._cancel_swarm_action_callback('land')
        thread.join(timeout=1.0)
    finally:
        if thread.is_alive():
            thread.join(timeout=1.0)

    assert response == module.CancelResponse.ACCEPT
    assert not thread.is_alive()
    assert client.goal_handles[0].cancel_requests == 1
    assert result_box['result'].return_code == 5
    assert goal_handle.terminal_states == ['canceled']


def test_record_swarm_goal_handle_cancels_late_accept_after_swarm_cancel():
    """A goal accepted after top-level cancel should be canceled immediately."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)
    context = module._ActiveSwarmAction(action_name='takeoff', cancel_requested=True)
    goal_handle = _FakeGoalHandle()

    node._record_swarm_goal_handle(context, 'cf1', goal_handle)

    assert goal_handle.cancel_requests == 1


def test_swarm_cancel_drains_late_goal_response_during_grace():
    """A goal response that arrives just after swarm cancel should still be canceled."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)
    _configure_swarm_action_node(node)
    node._swarm_goal_response_timeout_s = 1.0
    node._swarm_cancel_goal_response_grace_s = 0.2
    node._swarm_result_timeout_s = 0.01

    goal_response = _CallbackFuture(result=_FakeGoalHandle(result_done=False))

    class _DelayedGoalClient:
        def __init__(self):
            self.sent_goals = []

        def wait_for_server(self, timeout_sec):
            return True

        def send_goal_async(self, goal):
            self.sent_goals.append(goal)
            return goal_response

    client = _DelayedGoalClient()
    node._running_drones = lambda: ['cf1']
    node._takeoff_clients = {'cf1': client}

    goal_handle, _ = _fake_goal_handle(SimpleNamespace(height=1.0, duration=_duration(3.0)))
    result_box = {}
    thread = threading.Thread(
        target=lambda: result_box.setdefault('result', node._action_all_takeoff(goal_handle))
    )
    thread.start()
    try:
        deadline = time.time() + 1.0
        while not client.sent_goals and time.time() < deadline:
            time.sleep(0.01)
        assert client.sent_goals, 'per-drone goal request should be sent before cancel'

        response = node._cancel_swarm_action_callback('takeoff')
        timer = threading.Timer(0.02, goal_response.complete)
        timer.start()
        try:
            thread.join(timeout=1.0)
        finally:
            timer.cancel()
    finally:
        if thread.is_alive():
            thread.join(timeout=1.0)

    assert response == module.CancelResponse.ACCEPT
    assert not thread.is_alive()
    assert goal_response.result().cancel_requests == 1
    assert result_box['result'].return_code == 5
    assert goal_handle.terminal_states == ['canceled']


def test_action_all_go_to_frame_absolute_aborts_with_return_code_3():
    """FRAME_ABSOLUTE goals must be aborted with return_code=3."""
    module = _load_orchestrator_module()
    node = module.DroneOrchestratorNode.__new__(module.DroneOrchestratorNode)

    warnings = []
    node.get_logger = lambda: SimpleNamespace(
        info=lambda msg: None,
        warning=lambda msg: warnings.append(msg),
        error=lambda msg: None,
    )

    goal_handle, _ = _fake_goal_handle(SimpleNamespace(frame=0))
    result = node._action_all_go_to(goal_handle)

    assert result.return_code == 3
    assert goal_handle.terminal_states == ['abort']
    assert warnings, 'A warning must be logged when FRAME_ABSOLUTE is aborted'

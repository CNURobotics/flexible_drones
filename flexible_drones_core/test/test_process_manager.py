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

"""Unit tests for ProcessManager — command construction, backoff, and reconciliation."""

import signal
import time
from unittest.mock import MagicMock, patch

from flexible_drones_core.orchestrator.process_manager import ProcInfo, ProcessManager
from flexible_drones_core.registry import DroneSpec


# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_manager(**kwargs) -> ProcessManager:
    """Build a ProcessManager with compact defaults."""
    defaults = {'package': 'test_pkg', 'executable': 'test_exe'}
    defaults.update(kwargs)
    return ProcessManager(**defaults)


def _make_spec(name='cf1', manager_type='cf_manager', **kwargs) -> DroneSpec:
    """Build a DroneSpec with compact defaults."""
    defaults = {
        'name': name,
        'uri': f'radio://{name}',
        'type': 'crazyflie',
        'manager_type': manager_type,
    }
    defaults.update(kwargs)
    return DroneSpec(**defaults)


def _fake_popen(alive=True, exit_code=1):
    """Return a Popen-like mock with configurable liveness."""
    mock = MagicMock()
    mock.pid = 9999
    mock.poll.return_value = None if alive else exit_code
    return mock


class _FakeLogger:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info_messages = []

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.info_messages.append(message)


def test_build_cmd_uses_root_namespace_when_manager_owns_prefix():
    """Use the root namespace when the manager already prefixes topics."""
    manager = ProcessManager(
        package='fallback_pkg',
        executable='fallback_exe',
        manager_profiles={
            'cf_manager': {
                'package': 'flexible_drones_crazyflie',
                'executable': 'cf_manager_node',
                'manager_owns_drone_prefix': True,
            }
        },
    )
    spec = DroneSpec(
        name='cf1',
        uri='radio://cf1',
        type='crazyflie',
        manager_type='cf_manager',
        ros_params={'group_mask': 3, 'enabled': True},
    )

    cmd, package, executable = manager._build_cmd(spec)

    assert package == 'flexible_drones_crazyflie'
    assert executable == 'cf_manager_node'
    assert cmd[:4] == ['ros2', 'run', 'flexible_drones_crazyflie', 'cf_manager_node']
    assert '__ns:=/' in cmd
    assert 'drone_name:=cf1' in cmd
    assert 'uri:=radio://cf1' in cmd
    assert 'drone_type:=crazyflie' in cmd
    assert 'manager_type:=cf_manager' in cmd
    assert 'type_drone:=crazyflie' not in cmd
    assert 'group_mask:=3' in cmd
    assert 'enabled:=true' in cmd


def test_build_cmd_applies_profile_node_name_template():
    """Manager profiles can assign stable per-drone ROS node names."""
    manager = ProcessManager(
        package='fallback_pkg',
        executable='fallback_exe',
        manager_profiles={
            'gazebo_manager': {
                'package': 'flexible_drones_gazebo',
                'executable': 'gazebo_drone_manager',
                'manager_owns_drone_prefix': True,
                'node_name_template': '{name}_gz_manager',
            }
        },
    )
    spec = _make_spec(name='cf1', manager_type='gazebo_manager')

    cmd, _, _ = manager._build_cmd(spec)

    assert '__ns:=/' in cmd
    assert '__node:=cf1_gz_manager' in cmd


def test_build_cmd_preserves_explicit_node_remap():
    """Do not add a templated node name when the spec already supplies one."""
    manager = ProcessManager(
        package='fallback_pkg',
        executable='fallback_exe',
        manager_profiles={
            'gazebo_manager': {
                'package': 'flexible_drones_gazebo',
                'executable': 'gazebo_drone_manager',
                'node_name_template': '{name}_gz_manager',
            }
        },
    )
    spec = _make_spec(name='cf1', manager_type='gazebo_manager', extra_args=['--ros-args', '-r', '__node:=custom'])

    cmd, _, _ = manager._build_cmd(spec)

    assert '__node:=custom' in cmd
    assert '__node:=cf1_gz_manager' not in cmd


def test_build_cmd_preserves_explicit_namespace_and_serializes_complex_params():
    """Keep explicit namespaces and serialize non-scalar ROS params."""
    manager = ProcessManager(
        package='fallback_pkg',
        executable='fallback_exe',
        manager_profiles={
            'cf_manager': {
                'package': 'flexible_drones_crazyflie',
                'executable': 'cf_manager_node',
                'manager_owns_drone_prefix': True,
            }
        },
    )
    spec = DroneSpec(
        name='cf1',
        uri='radio://cf1',
        type='crazyflie',
        manager_type='cf_manager',
        ros_namespace='/custom_ns',
        ros_params={'gains': [1, 2], 'limits': {'max_vel': 3.0}},
        extra_args=['--debug'],
    )

    cmd, _, _ = manager._build_cmd(spec)

    assert '__ns:=/custom_ns' in cmd
    assert 'gains:=[1, 2]' in cmd
    assert 'limits:={"max_vel": 3.0}' in cmd
    assert cmd.index('--debug') < cmd.index('--ros-args')


def test_build_cmd_places_extra_args_before_generated_ros_args():
    """Keep executable args out of ROS argument parsing."""
    manager = _make_manager()
    spec = _make_spec(extra_args=['--debug'])

    cmd, _, _ = manager._build_cmd(spec)

    assert cmd[:5] == ['ros2', 'run', 'test_pkg', 'test_exe', '--debug']
    assert cmd[5] == '--ros-args'


def test_child_env_injects_ros_domain_id():
    """Export ROS_DOMAIN_ID into the child process environment."""
    manager = ProcessManager(
        package='pkg',
        executable='exe',
        ros_domain_id=42,
    )

    env = manager._child_env()

    assert env['ROS_DOMAIN_ID'] == '42'


def test_list_states_marks_subprocess_launch_mode():
    """Subprocess state snapshots should expose launch_mode like embedded snapshots."""
    manager = _make_manager()
    spec = _make_spec()
    manager._procs['cf1'] = ProcInfo(
        name='cf1',
        spec=spec,
        popen=_fake_popen(),
        started_at_unix=123.0,
        package='pkg',
        executable='exe',
    )

    states = manager.list_states()

    assert states['cf1']['launch_mode'] == 'subprocess'


# ── _format_param_value ───────────────────────────────────────────────────────


def test_format_param_value_booleans():
    """Format boolean values as ROS-compatible lowercase strings."""
    assert ProcessManager._format_param_value(True) == 'true'
    assert ProcessManager._format_param_value(False) == 'false'


def test_format_param_value_numbers():
    """Format numeric values with their string representation."""
    assert ProcessManager._format_param_value(42) == '42'
    assert ProcessManager._format_param_value(1.5) == '1.5'


def test_format_param_value_string_passthrough():
    """Leave string parameter values unchanged."""
    assert ProcessManager._format_param_value('hello') == 'hello'


def test_format_param_value_complex_types_produce_json():
    """Serialize complex values as single-argument JSON."""
    assert ProcessManager._format_param_value([1, 2]) == '[1, 2]'
    result = ProcessManager._format_param_value({'a': 1, 'nested': {'b': [2, 3]}})
    assert result == '{"a": 1, "nested": {"b": [2, 3]}}'
    assert '\n' not in result


# ── _resolve_profile ──────────────────────────────────────────────────────────


def test_resolve_profile_falls_back_to_defaults_for_unknown_type():
    """Use default package/executable when no manager profile matches."""
    mgr = _make_manager(package='default_pkg', executable='default_exe')
    package, executable = mgr._resolve_profile('unknown_manager')
    assert package == 'default_pkg'
    assert executable == 'default_exe'


def test_resolve_profile_uses_registered_profile():
    """Resolve package/executable from a registered manager profile."""
    mgr = _make_manager(
        manager_profiles={
            'gz_manager': {'package': 'gz_pkg', 'executable': 'gz_exe'},
        }
    )
    package, executable = mgr._resolve_profile('gz_manager')
    assert package == 'gz_pkg'
    assert executable == 'gz_exe'


# ── backoff formula ───────────────────────────────────────────────────────────


def test_backoff_doubles_each_restart():
    """next_restart_earliest_mono encodes base * 2^restart_count from start time."""
    mgr = _make_manager(restart_backoff_base_s=1.0, restart_backoff_max_s=60.0)
    spec = _make_spec()

    with patch(
        'flexible_drones_core.orchestrator.process_manager.subprocess.Popen',
        return_value=_fake_popen(),
    ):
        mgr._start_locked(spec)
        pi0 = mgr._procs['cf1']
        # First start: restart_count=0, backoff = 1.0 * 2^0 = 1.0
        assert abs((pi0.next_restart_earliest_mono - time.monotonic()) - 1.0) < 0.1

        prev = pi0
        mgr._start_locked(spec, replacing='cf1', prev=prev)
        pi1 = mgr._procs['cf1']
        # restart_count=1, backoff = 1.0 * 2^1 = 2.0
        assert abs((pi1.next_restart_earliest_mono - time.monotonic()) - 2.0) < 0.1

        mgr._start_locked(spec, replacing='cf1', prev=pi1)
        pi2 = mgr._procs['cf1']
        # restart_count=2, backoff = 1.0 * 2^2 = 4.0
        assert abs((pi2.next_restart_earliest_mono - time.monotonic()) - 4.0) < 0.1


def test_backoff_capped_at_max():
    """Clamp restart backoff at the configured maximum."""
    mgr = _make_manager(restart_backoff_base_s=1.0, restart_backoff_max_s=5.0)
    spec = _make_spec()

    # Simulate many restarts by constructing a fake prev with high restart_count.
    with patch(
        'flexible_drones_core.orchestrator.process_manager.subprocess.Popen',
        return_value=_fake_popen(),
    ):
        mgr._start_locked(spec)
        fake_prev = mgr._procs['cf1']
        object.__setattr__(fake_prev, 'restart_count', 9)

        mgr._start_locked(spec, replacing='cf1', prev=fake_prev)
        pi = mgr._procs['cf1']
        backoff = pi.next_restart_earliest_mono - time.monotonic()
        assert abs(backoff - 5.0) < 0.1


# ── ensure_running reconciliation ─────────────────────────────────────────────


def test_ensure_running_starts_process_for_new_spec():
    """Start a process for each desired spec that is not already running."""
    mgr = _make_manager()
    spec = _make_spec()

    with patch(
        'flexible_drones_core.orchestrator.process_manager.subprocess.Popen',
        return_value=_fake_popen(),
    ):
        mgr.ensure_running([spec])

    assert mgr.is_alive('cf1')


def test_ensure_running_stops_process_not_in_desired():
    """Stop a managed process when it disappears from the desired roster."""
    mgr = _make_manager(shutdown_timeouts=(0.0, 0.0, 0.0))
    spec = _make_spec()

    with patch(
        'flexible_drones_core.orchestrator.process_manager.subprocess.Popen',
        return_value=_fake_popen(),
    ):
        mgr.ensure_running([spec])

    # Remove from desired; process dies on stop (mock poll returns None then 0).
    mgr._procs['cf1'].popen.poll.return_value = 0
    mgr.ensure_running([])

    assert 'cf1' not in mgr._procs


def test_ensure_running_does_not_restart_within_backoff():
    """Skip restarting a dead process while its backoff window is active."""
    mgr = _make_manager(restart_backoff_base_s=60.0, restart_backoff_max_s=60.0)
    spec = _make_spec()
    dying = _fake_popen(alive=False, exit_code=1)

    with patch(
        'flexible_drones_core.orchestrator.process_manager.subprocess.Popen',
        return_value=dying,
    ):
        mgr.ensure_running([spec])

    # Process is already dead; backoff window is 60 s, so ensure_running should skip.
    with patch('flexible_drones_core.orchestrator.process_manager.subprocess.Popen') as mock_popen:
        mgr.ensure_running([spec])
        mock_popen.assert_not_called()


def test_ensure_running_stops_at_max_restarts():
    """Stop respawning after the configured restart limit is reached."""
    logger = _FakeLogger()
    mgr = _make_manager(
        max_restarts=2,
        restart_backoff_base_s=0.0,
        restart_backoff_max_s=0.0,
        logger=logger,
    )
    spec = _make_spec()

    spawn_calls = []

    def _popen_factory(cmd, **kwargs):
        p = _fake_popen(alive=False, exit_code=1)
        spawn_calls.append(p)
        return p

    with patch(
        'flexible_drones_core.orchestrator.process_manager.subprocess.Popen',
        side_effect=_popen_factory,
    ):
        for _ in range(10):
            mgr.ensure_running([spec])

    # First spawn + max_restarts (2) restarts = 3 total; anything beyond is suppressed.
    assert len(spawn_calls) == 3
    assert logger.errors.count(
        '[cf1] Reached max restarts (2); no further automatic restarts.'
    ) == 1


def test_ensure_running_resets_restart_count_after_healthy_uptime():
    """Allow automatic restarts again after a process ran long enough."""
    logger = _FakeLogger()
    mgr = _make_manager(
        max_restarts=1,
        restart_backoff_base_s=0.0,
        restart_backoff_max_s=0.0,
        restart_count_reset_after_s=10.0,
        logger=logger,
    )
    spec = _make_spec()
    old_proc = _fake_popen(alive=False, exit_code=1)
    mgr._procs['cf1'] = ProcInfo(
        name='cf1',
        spec=spec,
        popen=old_proc,
        started_at_unix=time.time() - 60.0,
        package='pkg',
        executable='exe',
        started_at_mono=time.monotonic() - 60.0,
        restart_count=1,
    )

    with patch(
        'flexible_drones_core.orchestrator.process_manager.subprocess.Popen',
        return_value=_fake_popen(),
    ) as mock_popen:
        mgr.ensure_running([spec])

    mock_popen.assert_called_once()
    assert mgr._procs['cf1'].restart_count == 1
    assert logger.errors == []
    assert any('Reset manager restart count' in msg for msg in logger.info_messages)


def test_stop_all_broadcasts_signal_phase_before_waiting():
    """Signal every process in a phase before waiting for that phase to settle."""
    mgr = _make_manager()
    first = _fake_popen()
    first.pid = 1001
    second = _fake_popen()
    second.pid = 1002
    mgr._procs['cf1'] = ProcInfo(
        name='cf1',
        spec=_make_spec(name='cf1'),
        popen=first,
        started_at_unix=1.0,
        package='pkg',
        executable='exe',
    )
    mgr._procs['cf2'] = ProcInfo(
        name='cf2',
        spec=_make_spec(name='cf2'),
        popen=second,
        started_at_unix=1.0,
        package='pkg',
        executable='exe',
    )
    events = []

    def _record_signal(pi, _pgid, sig):
        events.append(('signal', pi.popen.pid, sig))

    def _record_wait(procs, _timeout_s):
        events.append(('wait', [pi.popen.pid for _, pi in procs]))

    with patch.object(mgr, '_process_group_id', return_value=None), \
            patch.object(mgr, '_signal_process', side_effect=_record_signal), \
            patch.object(mgr, '_wait_for_group_exit', side_effect=_record_wait), \
            patch.object(mgr, '_wait_for_reap_locked', return_value=True):
        mgr.stop_all()

    assert events[:3] == [
        ('signal', 1001, signal.SIGINT),
        ('signal', 1002, signal.SIGINT),
        ('wait', [1001, 1002]),
    ]
    assert 'cf1' not in mgr._procs
    assert 'cf2' not in mgr._procs

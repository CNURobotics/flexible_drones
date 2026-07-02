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

"""Unit tests for manager-supervisor routing behavior."""

from flexible_drones_core.orchestrator.manager_supervisor import (
    EmbeddedHandle,
    ManagerSupervisor,
)
from flexible_drones_core.registry import DroneSpec


class _FakeEmbeddedManager:

    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.cleaned = 0
        self.alive = False

    def start(self):
        self.started += 1
        self.alive = True

    def stop(self):
        self.stopped += 1
        self.alive = False

    def is_alive(self):
        return self.alive

    def cleanup(self):
        self.cleaned += 1


class _FailingEmbeddedManager(_FakeEmbeddedManager):

    def start(self):
        self.started += 1
        raise RuntimeError('boom')


def test_ensure_running_routes_embedded_and_subprocess_specs():
    """Route embedded specs locally and pass subprocess specs downstream."""
    fake_manager = _FakeEmbeddedManager()
    proc_calls = []
    embedded_configs = []
    supervisor = ManagerSupervisor(
        package='fallback_pkg',
        executable='fallback_exe',
        manager_profiles={
            'cf_manager': {'package': 'cf_pkg', 'executable': 'cf_exe'},
            'pihawk_manager': {'package': 'pihawk_pkg', 'executable': 'pihawk_exe'},
        },
        embedded_factories={
            'cf_manager': lambda config: (
                embedded_configs.append(config)
                or EmbeddedHandle(
                    start=fake_manager.start,
                    stop=fake_manager.stop,
                    is_alive=fake_manager.is_alive,
                )
            )
        },
    )
    supervisor._procman.ensure_running = lambda specs: proc_calls.append(list(specs))
    supervisor._procman.list_running_names = lambda: ['px01']

    embedded = DroneSpec(
        name='cf1',
        uri='radio://cf1',
        type='crazyflie',
        manager_type='cf_manager',
    )
    subprocess = DroneSpec(
        name='px01',
        uri='udp://px01',
        type='pihawk',
        manager_type='pihawk_manager',
    )

    supervisor.ensure_running([embedded, subprocess])

    assert proc_calls == [[subprocess]]
    assert fake_manager.started == 1
    assert embedded_configs[0].drone_name == 'cf1'
    assert embedded_configs[0].drone_type == 'crazyflie'
    assert supervisor.list_running_names() == ['cf1', 'px01']


def test_start_spec_restarts_embedded_spec_changes():
    """Embedded spec changes should recreate the embedded manager handle."""
    fake_managers = []

    def make_handle(_config):
        fake_manager = _FakeEmbeddedManager()
        fake_managers.append(fake_manager)
        return EmbeddedHandle(
            start=fake_manager.start,
            stop=fake_manager.stop,
            is_alive=fake_manager.is_alive,
            cleanup=fake_manager.cleanup,
        )

    supervisor = ManagerSupervisor(
        package='fallback_pkg',
        executable='fallback_exe',
        manager_profiles={'cf_manager': {'package': 'cf_pkg', 'executable': 'cf_exe'}},
        embedded_factories={'cf_manager': make_handle},
    )

    original = DroneSpec(
        name='cf1',
        uri='radio://cf1',
        type='crazyflie',
        manager_type='cf_manager',
    )
    changed = DroneSpec(
        name='cf1',
        uri='radio://cf1',
        type='crazyflie',
        manager_type='cf_manager',
        ros_params={'group_mask': 1},
    )

    assert supervisor.start_spec(original) is True
    assert supervisor.start_spec(changed) is True
    assert len(fake_managers) == 2
    assert fake_managers[0].started == 1
    assert fake_managers[0].stopped == 1
    assert fake_managers[0].cleaned == 1
    assert fake_managers[1].started == 1


def test_ensure_running_stops_embedded_before_starting_subprocess_replacement():
    """A launch-mode change must not overlap old embedded and new subprocess owners."""
    events = []
    fake_manager = _FakeEmbeddedManager()
    supervisor = ManagerSupervisor(
        package='fallback_pkg',
        executable='fallback_exe',
        manager_profiles={
            'cf_manager': {'package': 'cf_pkg', 'executable': 'cf_exe'},
            'gazebo_manager': {'package': 'gz_pkg', 'executable': 'gz_exe'},
        },
        embedded_factories={
            'cf_manager': lambda config: EmbeddedHandle(
                start=lambda: (events.append('embedded_start'), fake_manager.start())[1],
                stop=lambda: (events.append('embedded_stop'), fake_manager.stop())[1],
                is_alive=fake_manager.is_alive,
            )
        },
    )

    def record_proc_specs(specs):
        events.append(('proc_ensure', [spec.name for spec in specs]))

    supervisor._procman.ensure_running = record_proc_specs

    embedded = DroneSpec(
        name='d1',
        uri='radio://d1',
        type='crazyflie',
        manager_type='cf_manager',
    )
    subprocess = DroneSpec(
        name='d1',
        uri='sim://d1',
        type='crazyflie',
        manager_type='gazebo_manager',
    )

    supervisor.ensure_running([embedded])
    supervisor.ensure_running([subprocess])

    assert events == [
        ('proc_ensure', []),
        'embedded_start',
        'embedded_stop',
        ('proc_ensure', ['d1']),
    ]
    assert supervisor.list_states() == {}


def test_ensure_running_restarts_changed_embedded_spec():
    """Registry reconciliation should stop, recreate, and restart on embedded spec change."""
    fake_managers = []

    def make_handle(_config):
        fake_manager = _FakeEmbeddedManager()
        fake_managers.append(fake_manager)
        return EmbeddedHandle(
            start=fake_manager.start,
            stop=fake_manager.stop,
            is_alive=fake_manager.is_alive,
            cleanup=fake_manager.cleanup,
        )

    supervisor = ManagerSupervisor(
        package='fallback_pkg',
        executable='fallback_exe',
        manager_profiles={'cf_manager': {'package': 'cf_pkg', 'executable': 'cf_exe'}},
        embedded_factories={'cf_manager': make_handle},
    )
    supervisor._procman.ensure_running = lambda specs: None

    original = DroneSpec(
        name='cf1',
        uri='radio://cf1',
        type='crazyflie',
        manager_type='cf_manager',
    )
    changed = DroneSpec(
        name='cf1',
        uri='radio://cf1',
        type='crazyflie',
        manager_type='cf_manager',
        ros_params={'group_mask': 1},
    )

    supervisor.ensure_running([original])
    supervisor.ensure_running([changed])

    assert len(fake_managers) == 2
    assert fake_managers[0].stopped == 1
    assert fake_managers[0].cleaned == 1
    assert fake_managers[1].started == 1
    assert supervisor.list_states()['cf1']['spec']['ros_params'] == {'group_mask': 1}


def test_failed_embedded_start_does_not_call_stop():
    """A never-started embedded manager should not receive stop after start failure."""
    fake_manager = _FailingEmbeddedManager()
    supervisor = ManagerSupervisor(
        package='fallback_pkg',
        executable='fallback_exe',
        manager_profiles={'cf_manager': {'package': 'cf_pkg', 'executable': 'cf_exe'}},
        embedded_factories={
            'cf_manager': lambda config: EmbeddedHandle(
                start=fake_manager.start,
                stop=fake_manager.stop,
                is_alive=fake_manager.is_alive,
            )
        },
    )

    spec = DroneSpec(
        name='cf1',
        uri='radio://cf1',
        type='crazyflie',
        manager_type='cf_manager',
    )

    assert supervisor.start_spec(spec) is False
    assert supervisor.list_states()['cf1']['state'] == 'STOPPED'
    assert supervisor.list_running_names() == []
    assert supervisor.is_alive('cf1') is False
    assert supervisor.stop_drone('cf1') is True
    assert fake_manager.started == 1
    assert fake_manager.stopped == 0

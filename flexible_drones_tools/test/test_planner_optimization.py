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

"""Unit tests for the out-of-process optimizer runner."""

import queue
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.modules.setdefault('casadi', SimpleNamespace())

from flexible_drones_tools.trajectories.planners import optimization  # noqa: E402
from flexible_drones_tools.trajectories.planners.errors import (  # noqa: E402
    TrajectoryPlanningCanceled,
    TrajectoryPlanningError,
    TrajectoryPlanningTimeout,
)


def _check_cancel(cancel_event, deadline):
    """Mirror of the node's cancel/timeout guard, injected into the runner."""
    if cancel_event is not None and cancel_event.is_set():
        raise TrajectoryPlanningCanceled('Trajectory planning canceled.')
    if deadline is not None and time.monotonic() > deadline:
        raise TrajectoryPlanningTimeout('Trajectory planning timed out.')


def test_optimizer_wait_progress_interpolates_within_optimizer_band():
    optimizer_start = 100.0
    deadline = 140.0

    assert optimization.optimizer_wait_progress(
        optimizer_start, deadline, now=optimizer_start,
    ) == pytest.approx(optimization.OPTIMIZER_PROGRESS_START)
    assert optimization.optimizer_wait_progress(
        optimizer_start, deadline, now=120.0,
    ) == pytest.approx(0.67)
    assert optimization.optimizer_wait_progress(
        optimizer_start, deadline, now=deadline + 10.0,
    ) == pytest.approx(optimization.OPTIMIZER_PROGRESS_END)


class _EmptyQueue:
    def get_nowait(self):
        raise queue.Empty

    def close(self):
        pass


class _ExitedProcess:
    def __init__(self, *args, **kwargs):
        self.join_calls = 0
        self.terminated = False

    def start(self):
        pass

    def is_alive(self):
        return False

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        self.join_calls += 1


class _EmptyResultContext:
    def Queue(self):
        return _EmptyQueue()

    def Process(self, *args, **kwargs):
        return _ExitedProcess(*args, **kwargs)


def test_run_optimizer_process_reports_missing_queue_result():
    with pytest.raises(TrajectoryPlanningError, match='without result'):
        optimization.run_optimizer_process(
            _EmptyResultContext(),
            {},
            threading.Event(),
            time.monotonic() + 1.0,
            _check_cancel,
        )


class _NeverEndingProcess:
    def __init__(self, *args, **kwargs):
        self.join_calls = 0
        self.terminated = False

    def start(self):
        pass

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        self.join_calls += 1


class _TimeoutContext:
    def __init__(self):
        self.process = None

    def Queue(self):
        return _EmptyQueue()

    def Process(self, *args, **kwargs):
        self.process = _NeverEndingProcess(*args, **kwargs)
        return self.process


def test_run_optimizer_process_terminates_and_joins_on_timeout():
    context = _TimeoutContext()

    with pytest.raises(TrajectoryPlanningTimeout):
        optimization.run_optimizer_process(
            context,
            {},
            threading.Event(),
            time.monotonic() - 1.0,
            _check_cancel,
        )

    assert context.process.terminated is True
    assert context.process.join_calls >= 1


class _ListQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _QueuedResultQueue:
    def __init__(self, items):
        self.items = list(items)

    def get_nowait(self):
        if self.items:
            return self.items.pop(0)
        raise queue.Empty

    def close(self):
        pass


class _QueuedPipelineContext:
    def __init__(self, items):
        self.queue = _QueuedResultQueue(items)
        self.process = None

    def Queue(self):
        return self.queue

    def Process(self, *args, **kwargs):
        self.process = _NeverEndingProcess(*args, **kwargs)
        return self.process


def _serialized_seed():
    return [
        {
            'T': 1.0,
            'c_x': [0.0, 1.0],
            'c_y': [0.0, 0.0],
            'c_z': [1.0, 0.0],
        }
    ]


def _pipeline_inputs():
    return {
        'planner': 'fake',
        'planner_kwargs': {},
        'limits': {'v_max': 1.0, 'a_max': 4.0, 'j_max': 10.0, 's_max': 50.0},
        'bounds': {'x_min': -3.0, 'x_max': 3.0, 'y_min': -3.0, 'y_max': 3.0, 'z_min': 0.25, 'z_max': 2.5},
        'n_segments': 1,
        'start_position': {'x': 0.0, 'y': 0.0, 'z': 1.0},
        'target_position': {'x': 1.0, 'y': 0.0, 'z': 1.0},
        'start_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'target_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'obstacles': [{'type': 'cylinder', 'x': 0.5, 'y': 0.0, 'radius': 0.2, 'height': 2.5}],
        'initial_segments': _serialized_seed(),
        'artifact_root': '',
        'deadline': time.monotonic() + 10.0,
    }


def test_pipeline_message_includes_trajectory_metrics():
    message = optimization.pipeline_message(
        'constrained',
        'Candidate ready.',
        trajectory=optimization.deserialize_optimizer_segments(_serialized_seed()),
        duration_sec=1.25,
        cost=42.5,
    )

    assert message['duration_sec'] == pytest.approx(1.25)
    assert message['cost'] == pytest.approx(42.5)
    assert message['trajectory'][0]['T'] == pytest.approx(1.0)


def test_run_planner_pipeline_process_cancel_does_not_return_best_valid():
    valid_message = optimization.pipeline_message(
        'constrained',
        'Candidate ready.',
        valid=True,
        trajectory=optimization.deserialize_optimizer_segments(_serialized_seed()),
    )
    context = _QueuedPipelineContext([('message', valid_message)])
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(TrajectoryPlanningCanceled):
        optimization.run_planner_pipeline_process(
            context,
            _pipeline_inputs(),
            cancel_event,
            time.monotonic() + 10.0,
            _check_cancel,
        )

    assert context.process.terminated is True
    assert context.process.join_calls >= 1


def test_run_planner_pipeline_process_error_returns_best_valid():
    valid_message = optimization.pipeline_message(
        'constrained',
        'Candidate ready.',
        valid=True,
        trajectory=optimization.deserialize_optimizer_segments(_serialized_seed()),
    )
    context = _QueuedPipelineContext([
        ('message', valid_message),
        ('error', 'optional refinement failed'),
    ])

    result = optimization.run_planner_pipeline_process(
        context,
        _pipeline_inputs(),
        threading.Event(),
        time.monotonic() + 10.0,
        _check_cancel,
    )

    assert optimization.serialize_optimizer_segments(result) == _serialized_seed()
    assert context.process.terminated is True
    assert context.process.join_calls >= 1


def test_constrained_stage_ipopt_schedule_uses_full_iter_budget_and_tightens_tolerance():
    schedule = optimization.constrained_stage_ipopt_schedule({
        'ipopt.max_iter': 123,
        'ipopt.tol': 1e-6,
        'ipopt.constr_viol_tol': 2e-6,
        'ipopt.acceptable_tol': 1e-5,
        'ipopt.acceptable_constr_viol_tol': 2e-5,
    })

    multipliers = [multiplier for multiplier, _options in schedule]
    options = [options for _multiplier, options in schedule]

    assert multipliers == [100.0, 10.0, 1.0]
    assert [option['ipopt.max_iter'] for option in options] == [123, 123, 123]
    assert [option['ipopt.tol'] for option in options] == pytest.approx(
        [1e-4, 1e-5, 1e-6])
    assert [option['ipopt.constr_viol_tol'] for option in options] == pytest.approx(
        [2e-4, 2e-5, 2e-6])
    assert [option['ipopt.acceptable_tol'] for option in options] == pytest.approx(
        [1e-3, 1e-4, 1e-5])


def test_diagnostic_continuity_issue_reports_worst_junction():
    segments = optimization.deserialize_optimizer_segments([
        {
            'T': 1.0,
            'c_x': [0.0, 1.000002],
            'c_y': [0.0],
            'c_z': [1.0],
        },
        {
            'T': 1.0,
            'c_x': [1.000002, 1.0],
            'c_y': [0.0],
            'c_z': [1.0],
        },
    ])

    issue = optimization.diagnostic_continuity_issue(segments)

    assert issue is not None
    assert issue['axis'] == 'c_x'
    assert issue['segment_index'] == 0
    assert issue['derivative_order'] == 1
    assert issue['junction_time'] == pytest.approx(1.0)
    assert issue['diff'] == pytest.approx(2e-6)
    text = optimization.diagnostic_continuity_issue_text(issue)
    assert 'c_x segment 0->1 order 1' in text
    assert 'junction_t=1.0000s' in text
    assert 'point=(1.0000, 0.0000, 1.0000)' in text
    assert 'diff=2e-06' in text
    assert 'tol=1e-06' in text


def test_pipeline_writes_request_scoped_artifacts_and_metrics(monkeypatch):
    seed = optimization.deserialize_optimizer_segments(_serialized_seed())
    captured_stages = []

    class FakePlanner:
        last_solve_status = 'Solve_Succeeded'

        def __init__(self, *args, **kwargs):
            self.last_objective_value = None

        def _rescale_segment_to_real_time(self, segment):
            return dict(segment)

        def _rescale_segment_to_normalized_time(self, segment):
            return dict(segment)

        def kinematic_time_dilation_factors(self, segments, **kwargs):
            return {}

        def plan(self, *args, **kwargs):
            self.last_objective_value = 10.0 if kwargs['obstacles'] == [] else 20.0
            return seed

    def fake_write_stage_artifact(planner, stage, segments, artifact_root):
        captured_stages.append(stage)
        return f'{artifact_root}/{stage}.csv'

    inputs = _pipeline_inputs()
    inputs['artifact_root'] = '/tmp'
    inputs['artifact_request_id'] = 'request_123'
    monkeypatch.setitem(optimization.PLANNERS, 'fake', FakePlanner)
    monkeypatch.setattr(optimization, 'has_diagnostic_continuity_issue', lambda _segments: True)
    validate_calls = []

    def fake_dense_validate_segments(*args, **kwargs):
        validate_calls.append(args[0])
        if len(validate_calls) == 1:
            raise RuntimeError('candidate still intersects an obstacle')

    monkeypatch.setattr(optimization, 'dense_validate_segments', fake_dense_validate_segments)
    monkeypatch.setattr(optimization, 'write_stage_artifact_if_enabled', fake_write_stage_artifact)

    result_queue = _ListQueue()
    optimization.planner_pipeline_process_main(inputs, result_queue)

    assert captured_stages == [
        'request_123_001_seed',
        'request_123_002_constrained_invalid',
        'request_123_003_constrained_candidate',
        'request_123_004_constrained_candidate',
    ]
    messages = [payload for status, payload in result_queue.items if status == 'message']
    valid_messages = [message for message in messages if message['valid']]
    assert valid_messages[-1]['duration_sec'] == pytest.approx(1.0)
    assert valid_messages[-1]['cost'] == pytest.approx(20.0)
    assert valid_messages[-1]['artifact_path'].endswith(
        'request_123_004_constrained_candidate.csv')


def test_pipeline_skips_warm_start_and_solves_obstacles_from_geometric_seed(monkeypatch):
    seed = optimization.deserialize_optimizer_segments(_serialized_seed())
    constrained_calls = []

    class FakePlanner:
        last_solve_status = 'Solve_Succeeded'

        def __init__(self, *args, **kwargs):
            self.last_objective_value = None

        def _rescale_segment_to_real_time(self, segment):
            return dict(segment)

        def _rescale_segment_to_normalized_time(self, segment):
            return dict(segment)

        def plan(self, *args, **kwargs):
            constrained_calls.append(kwargs)
            self.last_objective_value = 20.0
            return seed

    monkeypatch.setitem(optimization.PLANNERS, 'fake', FakePlanner)
    monkeypatch.setattr(optimization, 'dense_validate_segments', lambda *args, **kwargs: None)
    monkeypatch.setattr(optimization, 'has_diagnostic_continuity_issue', lambda _segments: True)

    result_queue = _ListQueue()
    optimization.planner_pipeline_process_main(_pipeline_inputs(), result_queue)

    assert len(constrained_calls) == 3
    assert constrained_calls[0]['obstacles']
    assert optimization.serialize_optimizer_segments(constrained_calls[0]['initial_segments']) == (
        optimization.serialize_optimizer_segments(seed)
    )
    messages = [payload for status, payload in result_queue.items if status == 'message']
    assert any(
        'Skipping bounds/kinematics warm-start solve' in message['msg']
        for message in messages
    )
    assert result_queue.items[-1][0] == 'ok'


def test_pipeline_skip_warm_start_does_not_publish_stage_obstacle_markers(monkeypatch):
    seed = optimization.deserialize_optimizer_segments(_serialized_seed())
    constrained_calls = []

    class FakePlanner:
        last_solve_status = 'Solve_Succeeded'

        def __init__(self, *args, **kwargs):
            self.last_objective_value = 10.0

        def _rescale_segment_to_real_time(self, segment):
            return dict(segment)

        def _rescale_segment_to_normalized_time(self, segment):
            return dict(segment)

        def kinematic_time_dilation_factors(self, segments, **kwargs):
            return {}

        def plan(self, *args, **kwargs):
            constrained_calls.append(kwargs)
            return seed

    monkeypatch.setitem(optimization.PLANNERS, 'fake', FakePlanner)
    monkeypatch.setattr(optimization, 'dense_validate_segments', lambda *args, **kwargs: None)
    monkeypatch.setattr(optimization, 'has_diagnostic_continuity_issue', lambda _segments: True)

    result_queue = _ListQueue()
    optimization.planner_pipeline_process_main(_pipeline_inputs(), result_queue)

    assert len(constrained_calls) == 3
    messages = [payload for status, payload in result_queue.items if status == 'message']
    skip_message = next(
        message for message in messages
        if message['step'] == optimization.PIPELINE_STEP_OBSTACLE_FREE
    )
    assert 'obstacles' not in skip_message


def test_pipeline_reuses_invalid_constrained_candidate_for_next_tolerance_pass(monkeypatch):
    seed = optimization.deserialize_optimizer_segments(_serialized_seed())
    first_constrained = optimization.deserialize_optimizer_segments([
        {
            'T': 1.4,
            'c_x': [0.0, 1.4],
            'c_y': [0.0, 0.0],
            'c_z': [1.0, 0.0],
        }
    ])
    second_constrained = optimization.deserialize_optimizer_segments(_serialized_seed())
    constrained_initial_seeds = []
    constrained_max_iters = []
    constrained_tolerances = []

    class FakePlanner:
        last_solve_status = 'Maximum_Iterations_Exceeded'

        def __init__(self, *args, **kwargs):
            self.last_objective_value = None

        def _rescale_segment_to_real_time(self, segment):
            return dict(segment)

        def _rescale_segment_to_normalized_time(self, segment):
            return dict(segment)

        def kinematic_time_dilation_factors(self, segments, **kwargs):
            return {}

        def plan(self, *args, **kwargs):
            if kwargs['obstacles'] == []:
                self.last_solve_status = 'Solve_Succeeded'
                return seed
            constrained_initial_seeds.append(kwargs['initial_segments'])
            constrained_max_iters.append(kwargs['ipopt_options']['ipopt.max_iter'])
            constrained_tolerances.append(kwargs['ipopt_options']['ipopt.tol'])
            if len(constrained_initial_seeds) == 1:
                self.last_solve_status = 'Solve_Succeeded'
                return first_constrained
            self.last_solve_status = 'Solve_Succeeded'
            return second_constrained

    validate_calls = []

    def fake_dense_validate_segments(*args, **kwargs):
        validate_calls.append(args[0])
        if len(validate_calls) == 1:
            raise RuntimeError('still clips obstacle')

    monkeypatch.setitem(optimization.PLANNERS, 'fake', FakePlanner)
    monkeypatch.setattr(optimization, 'dense_validate_segments', fake_dense_validate_segments)
    monkeypatch.setattr(optimization, 'has_diagnostic_continuity_issue', lambda _segments: True)

    result_queue = _ListQueue()
    optimization.planner_pipeline_process_main(_pipeline_inputs(), result_queue)

    assert len(constrained_initial_seeds) == 3
    assert constrained_max_iters == [800, 800, 800]
    assert constrained_tolerances == pytest.approx([1e-4, 1e-5, 1e-6])
    assert optimization.serialize_optimizer_segments(constrained_initial_seeds[1]) == (
        optimization.serialize_optimizer_segments(first_constrained)
    )
    messages = [payload for status, payload in result_queue.items if status == 'message']
    finished_messages = [
        message for message in messages
        if 'finished with solver status' in message['msg']
    ]
    assert any(
        'Obstacle-constrained solve pass 1/3 finished with solver status '
        'Solve_Succeeded' in message['msg']
        for message in finished_messages
    )
    # The intermediate "finished; running dense validation" message reports metrics
    # but no longer re-serializes the trajectory (it is published once per pass with
    # the dense-validation result, plus the final 'ok').
    assert finished_messages
    assert all(message.get('trajectory') is None for message in finished_messages)
    assert any('failed dense validation' in message['msg'] for message in messages)
    assert any(
        'failed dense validation; retriggering dense refinement pass 2/3 '
        'from that candidate (tol_scale=10x)' in message['msg']
        for message in messages
    )
    assert any(
        'Dense-validation refinement pass 2/3 started' in message['msg']
        for message in messages
    )
    # Step labels come from the explicit ``step`` field, not message wording.
    assert any(
        message.get('step') == optimization.PIPELINE_STEP_DENSE_REFINE
        for message in messages
    )
    assert any(message['valid'] for message in messages)
    assert result_queue.items[-1][0] == 'ok'

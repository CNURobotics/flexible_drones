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
Out-of-process optimizer execution for the trajectory planner.

The planner solve runs in a forked subprocess so a single misbehaving optimization
cannot block or crash the action server. This module owns the planner registry, the
(de)serialization across the process boundary, the subprocess entry point, and the
lifecycle/progress loop. The ROS-parameter -> ``numeric_inputs`` assembly stays in
the node and is passed in.
"""

import queue
import time
from types import SimpleNamespace

import numpy as np

from flexible_drones_tools.trajectories.planners import validation
from flexible_drones_tools.trajectories.planners.casadi_obstacle_planner import CasadiObstaclePlanner
from flexible_drones_tools.trajectories.planners.errors import (
    TrajectoryPlanningCanceled,
    TrajectoryPlanningError,
    TrajectoryPlanningTimeout,
)
from flexible_drones_tools.trajectories.planners.kkt_planner import KktPlanner

# Registry of selectable planners, keyed by the 'planner' parameter value.
PLANNERS = {
    'casadi_obstacle': CasadiObstaclePlanner,
    'kkt': KktPlanner,
}

# Pipeline step identifiers carried in queue payloads so the action server can label
# major steps without string-matching message text. See PlanTrajectoryActionServer.
PIPELINE_STEP_SEED = 'seed'
PIPELINE_STEP_OBSTACLE_FREE = 'obstacle_free'
PIPELINE_STEP_OBSTACLE_SOLVE = 'obstacle_solve'
PIPELINE_STEP_DENSE_CHECK = 'dense_check'
PIPELINE_STEP_DENSE_REFINE = 'dense_refine'

# Progress fractions reported while the optimizer subprocess runs.
OPTIMIZER_PROGRESS_START = 0.55
OPTIMIZER_PROGRESS_END = 0.79
UNCONSTRAINED_PROGRESS = 0.62
CONSTRAINED_PROGRESS = 0.82
REFINEMENT_PROGRESS = 0.90
CONSTRAINED_STAGE_TOLERANCE_MULTIPLIERS = (100.0, 10.0, 1.0)
CONSTRAINED_STAGE_TOLERANCE_KEYS = (
    'ipopt.tol',
    'ipopt.constr_viol_tol',
    'ipopt.acceptable_tol',
    'ipopt.acceptable_constr_viol_tol',
)

CONSTRAINED_STAGE_IPOPT_OPTIONS = {
    'ipopt.max_iter': 800,
    'ipopt.tol': 1e-6,
    'ipopt.constr_viol_tol': 1e-6,
    'ipopt.acceptable_tol': 1e-5,
    'ipopt.acceptable_constr_viol_tol': 1e-5,
    'ipopt.acceptable_iter': 5,
}
ACCEPTED_FINAL_STATUSES = {'Solve_Succeeded', 'Maximum_Iterations_Exceeded'}


def serialize_optimizer_segments(segments):
    """Convert planner segments to JSON/pickle-safe lists for the result queue."""
    serialized = []
    for segment in segments:
        serialized.append(
            {
                'T': float(segment['T']),
                'c_x': [float(value) for value in segment['c_x']],
                'c_y': [float(value) for value in segment['c_y']],
                'c_z': [float(value) for value in segment['c_z']],
            }
        )
    return serialized


def deserialize_optimizer_segments(segments):
    """Restore numpy coefficient arrays from the serialized subprocess result."""
    return [
        {
            'T': float(segment['T']),
            'c_x': np.asarray(segment['c_x'], dtype=np.float64),
            'c_y': np.asarray(segment['c_y'], dtype=np.float64),
            'c_z': np.asarray(segment['c_z'], dtype=np.float64),
        }
        for segment in segments
    ]


def normalized_seed_from_real_time(planner, segments):
    return [
        planner._rescale_segment_to_normalized_time(segment)
        for segment in segments
    ]


def vector_from_dict(values):
    return SimpleNamespace(
        x=float(values['x']),
        y=float(values['y']),
        z=float(values['z']),
    )


def dense_validate_segments(segments, numeric_inputs):
    validation.validate_boundary_constraints(
        segments,
        vector_from_dict(numeric_inputs['start_position']),
        vector_from_dict(numeric_inputs['target_position']),
        vector_from_dict(numeric_inputs['start_velocity']),
        vector_from_dict(numeric_inputs['target_velocity']),
    )
    validation.validate_kinematic_limits(segments, numeric_inputs['limits'])
    validation.validate_position_bounds(segments, numeric_inputs['bounds'])
    validation.validate_continuity(segments)
    validation.validate_obstacle_clearance(segments, numeric_inputs['obstacles'])


def diagnostic_continuity_issue(segments):
    """Return the worst tight continuity issue for a shippable candidate, if any."""
    starts = validation.segment_start_times(segments)
    worst = None
    for derivative_order, tolerance in sorted(validation.DIAGNOSTIC_CONTINUITY_TOLERANCES.items()):
        for axis in ('c_x', 'c_y', 'c_z'):
            for segment_index in range(len(segments) - 1):
                diff, left, right = validation.continuity_difference(
                    segments,
                    axis,
                    segment_index,
                    derivative_order,
                )
                if abs(diff) > float(tolerance):
                    score = abs(diff) / float(tolerance)
                    if worst is None or score > worst['score']:
                        junction_time = (
                            starts[segment_index] + float(segments[segment_index]['T'])
                        )
                        point = validation.evaluate_position(
                            segments,
                            segment_index,
                            float(segments[segment_index]['T']),
                        )
                        worst = {
                            'axis': axis,
                            'segment_index': segment_index,
                            'derivative_order': derivative_order,
                            'left': left,
                            'right': right,
                            'diff': abs(diff),
                            'tolerance': float(tolerance),
                            'score': score,
                            'junction_time': junction_time,
                            'point': point,
                        }
    return worst


def has_diagnostic_continuity_issue(segments):
    """Return True when a shippable candidate still has tight continuity warnings."""
    return diagnostic_continuity_issue(segments) is not None


def diagnostic_continuity_issue_text(issue):
    """Format a diagnostic continuity issue for status/log messages."""
    if issue is None:
        return ''
    point = issue['point']
    axis = issue['axis']
    segment_index = issue['segment_index']
    next_segment_index = segment_index + 1
    derivative_order = issue['derivative_order']
    junction_time = issue['junction_time']
    left = issue['left']
    right = issue['right']
    diff = issue['diff']
    tolerance = issue['tolerance']
    return (
        f'{axis} segment {segment_index}->{next_segment_index} '
        f'order {derivative_order}: '
        f'junction_t={junction_time:.4f}s, '
        f'point=({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}), '
        f'left={left:.9g}, right={right:.9g}, '
        f'diff={diff:.4g}, tol={tolerance:.4g}'
    )


def pipeline_message(
    stage,
    msg,
    *,
    step=None,
    valid=False,
    progress=None,
    deadline=None,
    started_at=None,
    timing=None,
    trajectory=None,
    artifact_path=None,
    duration_sec=None,
    cost=None,
    obstacles=None,
):
    now = time.monotonic()
    payload = {
        'stage': str(stage),
        'step': str(step) if step is not None else None,
        'msg': str(msg),
        'valid': bool(valid),
        'progress': float(progress) if progress is not None else None,
        'elapsed_sec': float(now - started_at) if started_at is not None else None,
        'remaining_sec': float(max(0.0, deadline - now)) if deadline is not None else None,
        'timing': dict(timing or {}),
        'trajectory': serialize_optimizer_segments(trajectory) if trajectory is not None else None,
        'duration_sec': float(duration_sec) if duration_sec is not None else None,
        'cost': float(cost) if cost is not None else None,
    }
    if artifact_path is not None:
        payload['artifact_path'] = str(artifact_path)
    # Obstacle dicts are already plain float/str values (JSON/pickle-safe as-is),
    # unlike 'trajectory' which needs numpy-array serialization. Only set when the
    # stage wants to (re)publish a specific obstacle set for visualization -- see
    # PlanTrajectoryActionServer's pipeline_message callback.
    if obstacles is not None:
        payload['obstacles'] = list(obstacles)
    return payload


def put_pipeline_message(result_queue, *args, **kwargs):
    result_queue.put(('message', pipeline_message(*args, **kwargs)))


def check_child_deadline(deadline):
    if deadline is not None and time.monotonic() > float(deadline):
        raise TrajectoryPlanningTimeout('Trajectory planning child reached its deadline.')


def write_stage_artifact_if_enabled(planner, stage, segments, artifact_root):
    if not artifact_root:
        return None
    return planner.write_stage_artifact(stage, segments, artifact_root)


def trajectory_total_duration(segments):
    """Return the real-time duration represented by optimizer segments."""
    return sum(float(segment.get('T', 0.0)) for segment in segments)


def trajectory_message_metrics(planner, segments):
    """Collect trajectory metrics for queue feedback and logs."""
    cost = getattr(planner, 'last_objective_value', None)
    return {
        'duration_sec': trajectory_total_duration(segments),
        'cost': cost,
    }


def planner_solve_timing_text(planner):
    solve_sec = getattr(planner, 'last_solve_sec', None)
    overall_sec = getattr(planner, 'last_overall_sec', None)
    if solve_sec is None and overall_sec is None:
        return ''
    parts = []
    if solve_sec is not None:
        parts.append(f'solve={float(solve_sec):.3f}s')
    if overall_sec is not None:
        parts.append(f'overall={float(overall_sec):.3f}s')
    return ', '.join(parts)


def constrained_stage_ipopt_schedule(base_options):
    """Return full-budget constrained IPOPT passes with progressively tighter tolerances."""
    options = dict(base_options)
    options['ipopt.max_iter'] = max(
        1,
        int(options.get(
            'ipopt.max_iter',
            CONSTRAINED_STAGE_IPOPT_OPTIONS['ipopt.max_iter'],
        )),
    )
    schedule = []
    for multiplier in CONSTRAINED_STAGE_TOLERANCE_MULTIPLIERS:
        pass_options = dict(options)
        for key in CONSTRAINED_STAGE_TOLERANCE_KEYS:
            if key in pass_options:
                pass_options[key] = float(pass_options[key]) * multiplier
        schedule.append((multiplier, pass_options))
    return schedule


def real_time_segments_from_seed(planner, segments):
    """Convert normalized seed segments into real-time segments for publication."""
    if not hasattr(planner, '_rescale_segment_to_real_time'):
        return [dict(segment) for segment in segments]
    return [planner._rescale_segment_to_real_time(segment) for segment in segments]


def optimizer_process_main(numeric_inputs, result_queue):
    """Subprocess entry point: build the selected planner, solve, return segments."""
    try:
        planner_cls = PLANNERS[numeric_inputs['planner']]
        planner = planner_cls(
            limits=numeric_inputs['limits'],
            bounds=numeric_inputs['bounds'],
            n_segments=numeric_inputs['n_segments'],
            **numeric_inputs.get('planner_kwargs', {}),
        )
        start = {
            'position': numeric_inputs['start_position'],
            'velocity': numeric_inputs['start_velocity'],
        }
        goal = {
            'position': numeric_inputs['target_position'],
            'velocity': numeric_inputs['target_velocity'],
        }
        if numeric_inputs.get('start_tangent') is not None:
            start['travel_tangent'] = numeric_inputs['start_tangent']
        if numeric_inputs.get('target_tangent') is not None:
            goal['travel_tangent'] = numeric_inputs['target_tangent']
        initial_segments = numeric_inputs.get('initial_segments')
        plan_kwargs = {}
        if initial_segments is not None:
            plan_kwargs['initial_segments'] = deserialize_optimizer_segments(initial_segments)
        if numeric_inputs.get('gate_transition_targets'):
            plan_kwargs['gate_transition_targets'] = numeric_inputs['gate_transition_targets']
        segments = planner.plan(
            start,
            goal,
            obstacles=numeric_inputs['obstacles'],
            bounds=numeric_inputs['bounds'],
            limits=numeric_inputs['limits'],
            n_segments=numeric_inputs['n_segments'],
            **plan_kwargs,
        )
        result_queue.put(('ok', serialize_optimizer_segments(segments)))
    except Exception as exc:  # noqa: B902
        result_queue.put(('error', f'{type(exc).__name__}: {exc}'))


def planner_pipeline_process_main(numeric_inputs, result_queue):
    """Subprocess entry point for staged anytime trajectory optimization."""
    started_at = time.monotonic()
    deadline = numeric_inputs.get('deadline')
    try:
        planner_cls = PLANNERS[numeric_inputs['planner']]
        planner = planner_cls(
            limits=numeric_inputs['limits'],
            bounds=numeric_inputs['bounds'],
            n_segments=numeric_inputs['n_segments'],
            **numeric_inputs.get('planner_kwargs', {}),
        )
        start = {
            'position': numeric_inputs['start_position'],
            'velocity': numeric_inputs['start_velocity'],
        }
        goal = {
            'position': numeric_inputs['target_position'],
            'velocity': numeric_inputs['target_velocity'],
        }
        if numeric_inputs.get('start_tangent') is not None:
            start['travel_tangent'] = numeric_inputs['start_tangent']
        if numeric_inputs.get('target_tangent') is not None:
            goal['travel_tangent'] = numeric_inputs['target_tangent']
        seed = deserialize_optimizer_segments(numeric_inputs['initial_segments'])
        if numeric_inputs.get('initial_segments_are_real_time'):
            seed = normalized_seed_from_real_time(planner, seed)
        artifact_root = str(numeric_inputs.get('artifact_root', '') or '').strip()
        artifact_request_id = str(numeric_inputs.get('artifact_request_id', '') or '').strip()
        artifact_index = 0
        timing = {}

        def write_artifact(stage, segments):
            nonlocal artifact_index
            artifact_index += 1
            stage_label = f'{artifact_index:03d}_{stage}'
            if artifact_request_id:
                stage_label = f'{artifact_request_id}_{stage_label}'
            return write_stage_artifact_if_enabled(
                planner, stage_label, segments, artifact_root)

        seed_real_time = real_time_segments_from_seed(planner, seed)
        seed_artifact = write_artifact('seed', seed_real_time)
        artifact_text = f'; wrote {seed_artifact}' if seed_artifact else ''
        put_pipeline_message(
            result_queue,
            'seed',
            f'Initial seed ready{artifact_text}.',
            step=PIPELINE_STEP_SEED,
            progress=UNCONSTRAINED_PROGRESS,
            deadline=deadline,
            started_at=started_at,
            timing=timing,
            trajectory=seed_real_time,
            artifact_path=seed_artifact,
            **trajectory_message_metrics(planner, seed_real_time),
        )

        # The geometric seed now goes straight to the obstacle-constrained solve.
        # Keep the old pipeline step for user feedback/progress continuity.
        put_pipeline_message(
            result_queue,
            'unconstrained',
            'Skipping bounds/kinematics warm-start solve; using geometric seed for obstacle solve.',
            step=PIPELINE_STEP_OBSTACLE_FREE,
            progress=UNCONSTRAINED_PROGRESS,
            deadline=deadline,
            started_at=started_at,
            timing=timing,
        )
        check_child_deadline(deadline)
        timing['unconstrained_sec'] = 0.0
        constrained_seed = seed
        put_pipeline_message(
            result_queue,
            'constrained',
            'Starting obstacle-constrained solve.',
            step=PIPELINE_STEP_OBSTACLE_SOLVE,
            progress=CONSTRAINED_PROGRESS,
            deadline=deadline,
            started_at=started_at,
            timing=timing,
        )
        constrained_stage_started_at = time.monotonic()
        constrained_options = dict(
            numeric_inputs.get('constrained_ipopt_options', CONSTRAINED_STAGE_IPOPT_OPTIONS))
        constrained_schedule = constrained_stage_ipopt_schedule(constrained_options)
        constrained = None
        constrained_validation_error = None
        dense_refinement_started = False
        for pass_index, (tolerance_multiplier, pass_options) in enumerate(
            constrained_schedule, start=1,
        ):
            check_child_deadline(deadline)
            pass_label = (
                'Dense-validation refinement'
                if dense_refinement_started else 'Obstacle-constrained solve'
            )
            solve_step = (
                PIPELINE_STEP_DENSE_REFINE
                if dense_refinement_started else PIPELINE_STEP_OBSTACLE_SOLVE
            )
            check_step = (
                PIPELINE_STEP_DENSE_REFINE
                if dense_refinement_started else PIPELINE_STEP_DENSE_CHECK
            )
            put_pipeline_message(
                result_queue,
                'constrained',
                (
                    f'{pass_label} pass {pass_index}/{len(constrained_schedule)} started '
                    f'(max_iter={pass_options["ipopt.max_iter"]}, '
                    f'tol_scale={tolerance_multiplier:g}x).'
                ),
                step=solve_step,
                progress=CONSTRAINED_PROGRESS,
                deadline=deadline,
                started_at=started_at,
                timing=timing,
            )
            pass_started_at = time.monotonic()
            constrained = planner.plan(
                start,
                goal,
                obstacles=numeric_inputs['obstacles'],
                bounds=numeric_inputs['bounds'],
                limits=numeric_inputs['limits'],
                n_segments=len(constrained_seed),
                initial_segments=constrained_seed,
                gate_transition_targets=numeric_inputs.get('gate_transition_targets'),
                ipopt_options=pass_options,
                accepted_statuses=ACCEPTED_FINAL_STATUSES,
            )
            timing[f'constrained_pass_{pass_index}_sec'] = time.monotonic() - pass_started_at
            timing['constrained_sec'] = time.monotonic() - constrained_stage_started_at
            solve_timing = planner_solve_timing_text(planner)
            solve_timing_text = f', {solve_timing}' if solve_timing else ''
            put_pipeline_message(
                result_queue,
                'constrained',
                (
                    f'{pass_label} pass {pass_index}/{len(constrained_schedule)} finished '
                    f'with solver status {planner.last_solve_status}{solve_timing_text}; '
                    f'running dense validation.'
                ),
                step=check_step,
                progress=CONSTRAINED_PROGRESS,
                deadline=deadline,
                started_at=started_at,
                timing=timing,
                **trajectory_message_metrics(planner, constrained),
            )
            if planner.last_solve_status != 'Solve_Succeeded':
                # Accepted only via the looser ACCEPTED_FINAL_STATUSES (e.g. IPOPT
                # hit max_iter without truly converging); flag this distinctly so
                # it doesn't read like a routine converged pass.
                put_pipeline_message(
                    result_queue,
                    'constrained',
                    (
                        f'{pass_label} pass {pass_index}/{len(constrained_schedule)} did not '
                        f'converge (status={planner.last_solve_status}); proceeding with dense '
                        'validation on a best-effort candidate, not a verified optimum.'
                    ),
                    step=check_step,
                    valid=False,
                    progress=CONSTRAINED_PROGRESS,
                    deadline=deadline,
                    started_at=started_at,
                    timing=timing,
                )
            try:
                dense_validate_segments(constrained, numeric_inputs)
            except Exception as exc:  # noqa: B902
                constrained_validation_error = exc
                constrained_artifact = write_artifact('constrained_invalid', constrained)
                artifact_text = f'; wrote {constrained_artifact}' if constrained_artifact else ''
                dense_refinement_started = planner.last_solve_status == 'Solve_Succeeded'
                put_pipeline_message(
                    result_queue,
                    'constrained',
                    (
                        f'{pass_label} pass {pass_index}/{len(constrained_schedule)} '
                        f'failed dense validation{artifact_text}: {exc}'
                    ),
                    step=check_step,
                    valid=False,
                    progress=CONSTRAINED_PROGRESS,
                    deadline=deadline,
                    started_at=started_at,
                    timing=timing,
                    trajectory=constrained,
                    artifact_path=constrained_artifact,
                    **trajectory_message_metrics(planner, constrained),
                )
                # Only warm-start the next pass from this candidate if IPOPT actually
                # converged to it -- a non-converged/diverged solve can carry NaN or
                # wildly-off coefficients, and warm-starting from that tends to
                # repeat or worsen the failure instead of refining toward a fix.
                # Otherwise keep retrying from the seed already used this pass.
                if pass_index < len(constrained_schedule):
                    next_multiplier = constrained_schedule[pass_index][0]
                    if planner.last_solve_status == 'Solve_Succeeded':
                        constrained_seed = normalized_seed_from_real_time(planner, constrained)
                        seed_note = 'from that candidate'
                    else:
                        seed_note = (
                            f'from the pass {pass_index} seed '
                            f'(solver status {planner.last_solve_status}, not reused)'
                        )
                    put_pipeline_message(
                        result_queue,
                        'constrained',
                        (
                            f'{pass_label} pass {pass_index}/{len(constrained_schedule)} '
                            'failed dense validation; retriggering dense refinement '
                            f'pass {pass_index + 1}/{len(constrained_schedule)} '
                            f'{seed_note} (tol_scale={next_multiplier:g}x).'
                        ),
                        step=PIPELINE_STEP_DENSE_REFINE,
                        valid=False,
                        progress=CONSTRAINED_PROGRESS,
                        deadline=deadline,
                        started_at=started_at,
                        timing=timing,
                    )
                continue

            constrained_artifact = write_artifact('constrained_candidate', constrained)
            artifact_text = f'; wrote {constrained_artifact}' if constrained_artifact else ''
            put_pipeline_message(
                result_queue,
                'constrained',
                (
                    f'{pass_label} pass {pass_index}/{len(constrained_schedule)} '
                    f'dense-valid ({planner.last_solve_status}){artifact_text}.'
                ),
                step=check_step,
                valid=True,
                progress=REFINEMENT_PROGRESS,
                deadline=deadline,
                started_at=started_at,
                timing=timing,
                trajectory=constrained,
                artifact_path=constrained_artifact,
                **trajectory_message_metrics(planner, constrained),
            )
            continuity_issue = diagnostic_continuity_issue(constrained)
            continuity_needs_refinement = (
                continuity_issue is not None or has_diagnostic_continuity_issue(constrained)
            )
            out_of_time = deadline is not None and time.monotonic() >= float(deadline)
            if (
                pass_index == len(constrained_schedule)
                or not continuity_needs_refinement
                or out_of_time
            ):
                result_queue.put(('ok', serialize_optimizer_segments(constrained)))
                return
            put_pipeline_message(
                result_queue,
                'constrained',
                (
                    f'{pass_label} pass {pass_index}/{len(constrained_schedule)} '
                    'is dense-valid but has diagnostic continuity warnings; '
                    f'{diagnostic_continuity_issue_text(continuity_issue) or "details unavailable"}; '
                    'continuing with tighter tolerances.'
                ),
                step=check_step,
                valid=True,
                progress=REFINEMENT_PROGRESS,
                deadline=deadline,
                started_at=started_at,
                timing=timing,
            )
            constrained_seed = normalized_seed_from_real_time(planner, constrained)

        if constrained_validation_error is not None:
            raise constrained_validation_error
        raise TrajectoryPlanningError('Constrained optimizer exhausted its tolerance schedule.')
    except Exception as exc:  # noqa: B902
        result_queue.put(('error', f'{type(exc).__name__}: {exc}'))


def optimizer_wait_progress(optimizer_start, deadline, now=None):
    """Interpolate progress between START and END across the optimizer time budget."""
    if deadline is None:
        return OPTIMIZER_PROGRESS_START

    now = time.monotonic() if now is None else float(now)
    optimizer_budget = max(1e-6, float(deadline) - float(optimizer_start))
    elapsed = max(0.0, now - float(optimizer_start))
    fraction = min(1.0, elapsed / optimizer_budget)
    return OPTIMIZER_PROGRESS_START + (
        OPTIMIZER_PROGRESS_END - OPTIMIZER_PROGRESS_START
    ) * fraction


def run_optimizer_process(
    mp_context,
    numeric_inputs,
    cancel_event,
    deadline,
    check_cancel,
    progress_callback=None,
):
    """
    Run the planner in a subprocess, polling ``check_cancel`` and reporting progress.

    ``check_cancel(cancel_event, deadline)`` must raise TrajectoryPlanningCanceled /
    TrajectoryPlanningTimeout when appropriate. Returns deserialized segments.
    """
    result_queue = mp_context.Queue()
    process = mp_context.Process(
        target=optimizer_process_main,
        args=(numeric_inputs, result_queue),
        daemon=True,
    )
    process.start()
    optimizer_start = time.monotonic()
    try:
        while process.is_alive():
            try:
                check_cancel(cancel_event, deadline)
            except (TrajectoryPlanningCanceled, TrajectoryPlanningTimeout):
                process.terminate()
                raise
            if progress_callback:
                progress_callback(optimizer_wait_progress(optimizer_start, deadline))
            time.sleep(0.05)

        process.join(timeout=2.0)
        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty as exc:
            raise TrajectoryPlanningError('optimizer process exited without result') from exc

        if status == 'error':
            raise TrajectoryPlanningError(payload)
        return deserialize_optimizer_segments(payload)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2.0)
        result_queue.close()


def run_planner_pipeline_process(
    mp_context,
    numeric_inputs,
    cancel_event,
    deadline,
    check_cancel,
    message_callback=None,
    progress_callback=None,
):
    """
    Run the staged planner pipeline in one spawned subprocess.

    Queue payloads are dictionaries with ``msg``, ``valid`` and optional serialized
    ``trajectory`` fields. The caller can retain the latest dense-valid candidate
    and terminate the process on cancel/timeout.
    """
    numeric_inputs = dict(numeric_inputs)
    numeric_inputs['deadline'] = deadline
    result_queue = mp_context.Queue()
    process = mp_context.Process(
        target=planner_pipeline_process_main,
        args=(numeric_inputs, result_queue),
        daemon=True,
    )
    process.start()
    best_valid = None
    latest_candidate = None
    try:
        while process.is_alive():
            while True:
                try:
                    status, payload = result_queue.get_nowait()
                except queue.Empty:
                    break
                if status == 'message':
                    if payload.get('trajectory') is not None:
                        latest_candidate = deserialize_optimizer_segments(payload['trajectory'])
                        if payload.get('valid'):
                            best_valid = latest_candidate
                    if progress_callback and payload.get('progress') is not None:
                        progress_callback(payload['progress'])
                    if message_callback:
                        message_callback(payload)
                elif status == 'error':
                    if best_valid is not None:
                        return best_valid
                    raise TrajectoryPlanningError(payload)
                elif status == 'ok':
                    latest_candidate = deserialize_optimizer_segments(payload)
                    best_valid = latest_candidate
            try:
                check_cancel(cancel_event, deadline)
            except TrajectoryPlanningCanceled:
                process.terminate()
                raise
            except TrajectoryPlanningTimeout:
                process.terminate()
                if best_valid is not None:
                    return best_valid
                raise
            time.sleep(0.05)

        process.join(timeout=2.0)
        while True:
            try:
                status, payload = result_queue.get_nowait()
            except queue.Empty:
                break
            if status == 'message':
                if payload.get('trajectory') is not None:
                    latest_candidate = deserialize_optimizer_segments(payload['trajectory'])
                    if payload.get('valid'):
                        best_valid = latest_candidate
                if progress_callback and payload.get('progress') is not None:
                    progress_callback(payload['progress'])
                if message_callback:
                    message_callback(payload)
            elif status == 'error':
                if best_valid is not None:
                    return best_valid
                raise TrajectoryPlanningError(payload)
            elif status == 'ok':
                latest_candidate = deserialize_optimizer_segments(payload)
                best_valid = latest_candidate

        if best_valid is not None:
            return best_valid
        raise TrajectoryPlanningError('planner pipeline process exited without result')
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2.0)
        result_queue.close()

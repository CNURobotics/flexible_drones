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

"""Unit tests for the trajectory planner action-server coordination helpers."""

import math
import sys
import threading
from types import SimpleNamespace

from numpy.polynomial.polynomial import Polynomial
import pytest

sys.modules.setdefault('casadi', SimpleNamespace())

from flexible_drones_tools.trajectories import plan_trajectory_action_server as planner_module  # noqa: E402
from flexible_drones_tools.trajectories.utilities.io import load_trajectory_csv  # noqa: E402

_BOUNDS = {'x_min': -3.0, 'x_max': 3.0, 'y_min': -3.0, 'y_max': 3.0, 'z_min': 0.25, 'z_max': 2.5}
_LIMITS = {'v_max': 1.0, 'a_max': 4.0, 'j_max': 10.0, 's_max': 50.0}


class _FakeLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.infos.append(message)

    def debug(self, message):
        pass


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _point(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def _bare_server():
    server = planner_module.PlanTrajectoryActionServer.__new__(
        planner_module.PlanTrajectoryActionServer
    )
    server._planning_worker_lock = threading.Lock()
    server._planning_worker = None
    server._planning_reserved = False
    server._state_lock = threading.Lock()
    server._planner_type = 'kkt'
    server._limits = dict(_LIMITS)
    server.get_parameter = lambda name: SimpleNamespace(value=1)
    server.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: SimpleNamespace())
    )
    server._logger = _FakeLogger()
    server.get_logger = lambda: server._logger
    # Bare obstacle manager (state + delegation) sharing the server as its node.
    manager = planner_module.ObstacleManager.__new__(planner_module.ObstacleManager)
    manager.node = server
    manager._state_lock = server._state_lock
    manager._obstacles = []
    manager._default_capsules = []
    manager._obstacle_marker_publisher = None
    manager.clear_markers = lambda planning_frame: None
    server.obstacles = manager
    return server


def _double_parameter(value):
    return SimpleNamespace(
        get_parameter_value=lambda: SimpleNamespace(double_value=float(value))
    )


def _segment(duration=1.0, c_x=None):
    return {
        'T': duration,
        'c_x': [0.0, 1.0] if c_x is None else c_x,
        'c_y': [0.0],
        'c_z': [1.0],
    }


def _goal():
    return SimpleNamespace(
        start=SimpleNamespace(
            frame_id='map',
            pose=SimpleNamespace(
                position=_point(),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            twist=SimpleNamespace(linear=_point(), angular=_point()),
        ),
        target=SimpleNamespace(
            frame_id='map',
            pose=SimpleNamespace(
                position=_point(1.0, 0.0, 0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            twist=SimpleNamespace(linear=_point(), angular=_point()),
        ),
    )


def _casadi_parameter_values():
    return {
        'seed_strategy': 'obstacle_boundary_subdivision',
        'min_segment_time': 0.5,
        'max_duration': 60.0,
        'gate_transition_distance': 0.5,
        'gate_transition_position_weight': 10.0,
        'gate_transition_velocity_weight': 1.0,
        'gate_transition_velocity_threshold': 0.01,
        'n_eval': 10,
        'obstacle_base_samples': 41,
        'obstacle_refine_multiplier': 5,
        'obstacle_refine_margin': 0.03,
        'obstacle_refine_radius_fraction': 0.25,
        'obstacle_refine_min_spacing': 0.01,
        'obstacle_refine_max_spacing': 0.25,
        'boundary_bias_margin': 0.5,
        'boundary_bias_weight': 50.0,
        'w_time': 10.0,
        'w_acc': 1.0,
        'w_jerk': 0.0,
        'w_snap': 1.0,
        'kinematic_constraint_scale': 0.975,
        'w_yaw_rate': 0.0,
        'w_yaw_acc': 2.0,
        'w_yaw_align': 1.0,
        'w_yaw_smooth': 0.01,
        'w_yaw_accel_limit': 1.0,
        'w_yaw_snap': 0.0,
        'backtrack_objective_weight': 0.0,
        'behind_start_objective_weight': 0.0,
        'planner_artifact_dir': '/tmp',
        'initial_ipopt_max_iter': 800,
        'initial_ipopt_tol': 1e-4,
        'initial_ipopt_constr_viol_tol': 1e-4,
        'initial_ipopt_acceptable_tol': 1e-3,
        'initial_ipopt_acceptable_constr_viol_tol': 1e-3,
        'initial_ipopt_acceptable_iter': 5,
        'constrained_ipopt_max_iter': 800,
        'constrained_ipopt_tol': 1e-6,
        'constrained_ipopt_constr_viol_tol': 1e-6,
        'constrained_ipopt_acceptable_tol': 1e-5,
        'constrained_ipopt_acceptable_constr_viol_tol': 1e-5,
        'constrained_ipopt_acceptable_iter': 5,
        'n_segments': 6,
        'yaw_follow_tangent': True,
        'trajectory_visualization_samples': 160,
    }


def test_main_suppresses_keyboard_interrupt_and_cleans_up(monkeypatch):
    """Console-script entry point handles Ctrl+C without relying on __main__."""
    events = []

    class _FakeServer:
        def destroy_node(self):
            events.append('destroy_node')

    class _FakeExecutor:
        def add_node(self, node):
            events.append(('add_node', node))

        def spin(self):
            events.append('spin')
            raise KeyboardInterrupt

        def shutdown(self):
            events.append('executor_shutdown')

    server = _FakeServer()
    executor = _FakeExecutor()
    monkeypatch.setattr(planner_module.rclpy, 'init', lambda args=None: events.append(('init', args)))
    monkeypatch.setattr(planner_module.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(planner_module.rclpy, 'shutdown', lambda: events.append('rclpy_shutdown'))
    monkeypatch.setattr(planner_module, 'PlanTrajectoryActionServer', lambda: server)
    monkeypatch.setattr(planner_module, 'MultiThreadedExecutor', lambda: executor)

    planner_module.main(args=['--ros-args'])

    assert events == [
        ('init', ['--ros-args']),
        ('add_node', server),
        'spin',
        'executor_shutdown',
        'destroy_node',
        'rclpy_shutdown',
    ]


def test_planning_goal_callback_warns_until_obstacles_configured():
    server = _bare_server()
    server.obstacles._obstacles = None

    assert server._planning_goal_callback(None) == planner_module.GoalResponse.ACCEPT

    assert 'no service obstacle update yet' in server._logger.warnings[0]
    assert server._planning_reserved is True


def test_planning_goal_callback_reserves_slot_and_rejects_busy_goal():
    server = _bare_server()

    assert server._planning_goal_callback(None) == planner_module.GoalResponse.ACCEPT
    assert server._planning_goal_callback(None) == planner_module.GoalResponse.REJECT


def test_planning_timeout_defaults_to_server_max_without_goal_request():
    server = _bare_server()
    server.get_parameter = lambda name: _double_parameter(300.0)

    assert server._planning_timeout_sec() == pytest.approx(300.0)
    assert server._planning_timeout_sec(SimpleNamespace(timeout_sec=0.0)) == pytest.approx(300.0)


def test_planning_timeout_clamps_positive_goal_request_to_server_max():
    server = _bare_server()
    server.get_parameter = lambda name: _double_parameter(300.0)

    assert server._planning_timeout_sec(SimpleNamespace(timeout_sec=180.0)) == pytest.approx(180.0)
    assert server._planning_timeout_sec(SimpleNamespace(timeout_sec=600.0)) == pytest.approx(300.0)


def test_planning_feedback_period_reads_parameter():
    server = _bare_server()
    server.get_parameter = lambda name: _double_parameter(0.5)

    assert server._planning_feedback_period_sec() == pytest.approx(0.5)


def test_planning_feedback_period_has_floor():
    server = _bare_server()
    server.get_parameter = lambda name: _double_parameter(0.01)

    assert server._planning_feedback_period_sec() == pytest.approx(0.1)


def test_trajectory_ready_message_includes_segments_and_duration():
    pieces = [
        planner_module.polynomial_piece_from_coefficients(
            1.25, [0.0], [0.0], [1.0], [0.0]),
        planner_module.polynomial_piece_from_coefficients(
            2.5, [1.0], [0.0], [1.0], [0.0]),
    ]
    trajectory = planner_module.trajectory_from_pieces(42, pieces)

    assert planner_module.PlanTrajectoryActionServer._trajectory_ready_message(
        trajectory,
        'Generated trajectory',
    ) == 'Generated trajectory: 2 segments, duration=3.750s.'


def test_trajectory_ready_message_can_include_planning_time():
    pieces = [
        planner_module.polynomial_piece_from_coefficients(
            1.25, [0.0], [0.0], [1.0], [0.0]),
        planner_module.polynomial_piece_from_coefficients(
            2.5, [1.0], [0.0], [1.0], [0.0]),
    ]
    trajectory = planner_module.trajectory_from_pieces(42, pieces)

    assert planner_module.PlanTrajectoryActionServer._trajectory_ready_message(
        trajectory,
        'Generated trajectory',
        planning_duration_sec=12.3456,
    ) == 'Generated trajectory: 2 segments, duration=3.750s, planning_time=12.346s.'


def test_save_cache_enabled_defaults_to_save_parameter():
    server = _bare_server()
    values = {'use_cache': False, 'save_cache': True}
    server.get_parameter = lambda name: SimpleNamespace(value=values[name])

    assert server._save_cache_enabled() is True


def test_use_cache_forces_save_cache_enabled():
    server = _bare_server()
    values = {'use_cache': True, 'save_cache': False}
    server.get_parameter = lambda name: SimpleNamespace(value=values[name])

    assert server._save_cache_enabled() is True


def test_cache_saving_can_be_disabled_when_not_loading_cache():
    server = _bare_server()
    values = {'use_cache': False, 'save_cache': False}
    server.get_parameter = lambda name: SimpleNamespace(value=values[name])

    assert server._save_cache_enabled() is False


def test_log_planning_boundaries_reports_requested_and_transformed_values():
    server = _bare_server()
    goal = SimpleNamespace(
        start=SimpleNamespace(
            frame_id='gate_A_target_1',
            pose=SimpleNamespace(position=_point(1.0, 2.0, 3.0)),
            twist=SimpleNamespace(linear=_point(0.1, 0.2, 0.3)),
        ),
        target=SimpleNamespace(
            frame_id='gate_B_target_2',
            pose=SimpleNamespace(position=_point(4.0, 5.0, 6.0)),
            twist=SimpleNamespace(linear=_point(0.4, 0.5, 0.6)),
        ),
    )
    base_start_pose = SimpleNamespace(pose=SimpleNamespace(position=_point(1.5, 2.5, 3.5)))
    base_start_twist = SimpleNamespace(twist=SimpleNamespace(linear=_point(0.7, 0.8, 0.9)))
    base_target_pose = SimpleNamespace(pose=SimpleNamespace(position=_point(4.5, 5.5, 6.5)))
    base_target_twist = SimpleNamespace(twist=SimpleNamespace(linear=_point(1.0, 1.1, 1.2)))

    server._log_planning_boundaries(
        goal,
        'map',
        base_start_pose,
        base_start_twist,
        base_target_pose,
        base_target_twist,
    )

    assert len(server._logger.infos) == 2
    assert "start_frame='gate_A_target_1'" in server._logger.infos[0]
    assert 'start_pos=(1.000, 2.000, 3.000)' in server._logger.infos[0]
    assert "preferred_frame='map'" in server._logger.infos[1]
    assert 'target_pos=(4.500, 5.500, 6.500)' in server._logger.infos[1]


def test_fit_output_yaw_follows_straight_line_heading():
    from flexible_drones_tools.trajectories.planners import yaw_planning

    segments = [
        {
            'T': 2.0,
            'c_x': [0.0, 1.0],
            'c_y': [0.0, 1.0],
            'c_z': [1.0],
        },
    ]

    # Constant +45 deg tangent with no explicit boundary pins -> constant yaw.
    yaw_coefficients = yaw_planning.fit_yaw_coefficients(
        segments,
        start_yaw=None,
        target_yaw=None,
    )

    assert len(yaw_coefficients) == 1
    assert yaw_coefficients[0][0] == pytest.approx(0.785398163, abs=1e-5)
    assert yaw_coefficients[0][1:] == pytest.approx([0.0] * 7, abs=1e-5)


def test_fit_output_yaw_uses_short_boundary_winding_near_hover():
    from flexible_drones_tools.trajectories.planners import yaw_planning

    segments = [
        {
            'T': 2.0,
            'c_x': [1.0],
            'c_y': [1.0],
            'c_z': [1.0],
        },
    ]

    start_yaw = 170.0 * 3.141592653589793 / 180.0
    target_yaw = -170.0 * 3.141592653589793 / 180.0
    yaw_coefficients = yaw_planning.fit_yaw_coefficients(
        segments,
        start_yaw=start_yaw,
        target_yaw=target_yaw,
        w_align=0.0,
    )

    yaw_poly = Polynomial(yaw_coefficients[0])
    assert yaw_poly(0.0) == pytest.approx(start_yaw, abs=1e-6)
    assert yaw_poly(2.0) == pytest.approx(start_yaw + 20.0 * 3.141592653589793 / 180.0, abs=1e-6)


def test_fit_output_yaw_pins_boundary_acceleration_and_jerk_to_zero():
    from flexible_drones_tools.trajectories.planners import yaw_planning

    segments = [
        {
            'T': 2.0,
            'c_x': [0.0],
            'c_y': [0.0],
            'c_z': [1.0],
        },
    ]

    yaw_coefficients = yaw_planning.fit_yaw_coefficients(
        segments,
        start_yaw=0.0,
        target_yaw=math.pi / 2.0,
        w_align=0.0,
    )

    yaw_poly = Polynomial(yaw_coefficients[0])
    assert len(yaw_coefficients[0]) == 8
    for endpoint in (0.0, 2.0):
        assert yaw_poly.deriv(2)(endpoint) == pytest.approx(0.0, abs=1e-8)
        assert yaw_poly.deriv(3)(endpoint) == pytest.approx(0.0, abs=1e-8)
    assert yaw_poly.deriv(4)(0.0) != pytest.approx(0.0, abs=1e-8)


def test_fit_output_yaw_interior_segment_boundaries_are_continuity_only():
    from flexible_drones_tools.trajectories.planners import yaw_planning

    segments = [
        {
            'T': 1.0,
            'c_x': [0.0, 1.0],
            'c_y': [0.0, 0.0],
            'c_z': [1.0],
        },
        {
            'T': 1.0,
            'c_x': [1.0, 1.0],
            'c_y': [0.0, 1.0],
            'c_z': [1.0],
        },
    ]

    yaw_coefficients = yaw_planning.fit_yaw_coefficients(
        segments,
        start_yaw=0.0,
        target_yaw=math.pi / 2.0,
        w_align=0.0,
    )

    left = Polynomial(yaw_coefficients[0])
    right = Polynomial(yaw_coefficients[1])
    for order in range(5):
        assert left.deriv(order)(1.0) == pytest.approx(right.deriv(order)(0.0), abs=1e-8)
    assert left.deriv(3)(1.0) != pytest.approx(0.0, abs=1e-8)


def test_fit_output_yaw_accel_limit_weight_softens_acceleration():
    from flexible_drones_tools.trajectories.planners import yaw_planning

    segments = [
        {
            'T': 0.5,
            'c_x': [0.0, 0.0],
            'c_y': [0.0, 1.0],
            'c_z': [1.0],
        },
        {
            'T': 0.5,
            'c_x': [0.0, 0.0],
            'c_y': [0.5, 1.0],
            'c_z': [1.0],
        },
    ]

    loose = yaw_planning.fit_yaw_coefficients(
        segments,
        start_yaw=0.0,
        target_yaw=math.pi / 2.0,
        w_align=10.0,
        w_smooth=0.0,
        w_snap=1e-4,
        w_accel_limit=0.0,
        yaw_accel_max=2.0,
    )
    softened = yaw_planning.fit_yaw_coefficients(
        segments,
        start_yaw=0.0,
        target_yaw=math.pi / 2.0,
        w_align=10.0,
        w_smooth=0.0,
        w_snap=1e-4,
        w_accel_limit=10.0,
        yaw_accel_max=2.0,
    )

    sample_times = [index / 200.0 for index in range(101)]
    loose_peak = max(
        abs(float(Polynomial(coeffs).deriv(2)(t)))
        for coeffs in loose
        for t in sample_times
    )
    softened_peak = max(
        abs(float(Polynomial(coeffs).deriv(2)(t)))
        for coeffs in softened
        for t in sample_times
    )

    assert softened_peak < loose_peak


def test_output_yaw_warning_is_accel_only():
    server = _bare_server()
    server._limits = {
        **server._limits,
        'yaw_rate_max': 0.5,
        'yaw_accel_max': 1.0,
    }

    server._warn_on_output_yaw_accel_limit([1.0], [[0.0, 2.0]])
    assert server._logger.warnings == []

    server._warn_on_output_yaw_accel_limit([1.0], [[0.0, 0.0, 2.0]])
    assert len(server._logger.warnings) == 1
    assert 'yaw acceleration' in server._logger.warnings[0]


def test_cache_filename_ignores_yaw_follow_tangent():
    server = _bare_server()
    values = {
        'n_segments': 6,
        'seed_strategy': 'obstacle_boundary_subdivision',
        'yaw_follow_tangent': True,
    }
    server.get_parameter = lambda name: SimpleNamespace(value=values.get(name, True))
    goal = _goal()

    with_yaw = server._cache_filename(
        goal,
        'start',
        'target',
        'map',
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
    )
    values['yaw_follow_tangent'] = False
    without_yaw = server._cache_filename(
        goal,
        'start',
        'target',
        'map',
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
    )

    assert with_yaw == without_yaw


def test_cache_filename_ignores_seed_strategy():
    server = _bare_server()
    values = {
        'n_segments': 6,
        'seed_strategy': 'obstacle_boundary_subdivision',
        'yaw_follow_tangent': True,
    }
    server.get_parameter = lambda name: SimpleNamespace(value=values.get(name, True))
    goal = _goal()

    obstacle_boundary = server._cache_filename(
        goal,
        'start',
        'target',
        'map',
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
    )
    values['seed_strategy'] = 'straight_line'
    straight_line = server._cache_filename(
        goal,
        'start',
        'target',
        'map',
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
    )

    assert obstacle_boundary == straight_line


def test_cache_filename_ignores_casadi_objective_weight():
    server = _bare_server()
    server._planner_type = 'casadi_obstacle'
    values = _casadi_parameter_values()
    server.get_parameter = lambda name: SimpleNamespace(value=values[name])
    goal = _goal()

    low_weight = server._cache_filename(
        goal,
        'start',
        'target',
        'map',
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
    )
    values['w_snap'] = 4.0
    high_weight = server._cache_filename(
        goal,
        'start',
        'target',
        'map',
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
    )

    assert low_weight == high_weight


def test_cache_filename_ignores_limits():
    server = _bare_server()
    values = _casadi_parameter_values()
    server._planner_type = 'casadi_obstacle'
    server.get_parameter = lambda name: SimpleNamespace(value=values[name])
    goal = _goal()

    low_limit = server._cache_filename(
        goal,
        'start',
        'target',
        'map',
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
    )
    server._limits = {**server._limits, 'v_max': 0.5}
    high_limit = server._cache_filename(
        goal,
        'start',
        'target',
        'map',
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
    )

    assert low_limit == high_limit


def test_cached_pieces_dense_validation_returns_seed_segments():
    server = _bare_server()
    piece = server._build_polynomial_piece(
        1.0,
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0],
    )

    segments = server._cached_pieces_pass_validation(
        [piece],
        _point(0.0, 0.0, 1.0),
        _point(1.0, 0.0, 1.0),
        _point(1.0, 0.0, 0.0),
        _point(1.0, 0.0, 0.0),
        _BOUNDS,
        [],
    )

    assert len(segments) == 1
    assert segments[0]['T'] == pytest.approx(1.0)
    assert segments[0]['c_x'][1] == pytest.approx(1.0)


def test_run_optimizer_process_passes_seed_strategy_parameter(monkeypatch):
    server = _bare_server()
    values = _casadi_parameter_values()
    server._planner_type = 'casadi_obstacle'
    server._mp_context = object()
    server.get_parameter = lambda name: SimpleNamespace(value=values[name])
    captured = {}

    def fake_run_planner_pipeline_process(mp_context, numeric_inputs, *args, **kwargs):
        captured['mp_context'] = mp_context
        captured['numeric_inputs'] = numeric_inputs
        return []

    monkeypatch.setattr(
        planner_module.optimization,
        'run_planner_pipeline_process',
        fake_run_planner_pipeline_process,
    )

    server._run_optimizer_process(
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
        cancel_event=None,
        deadline=None,
    )

    assert captured['mp_context'] is server._mp_context
    assert captured['numeric_inputs']['n_segments'] == 3
    assert captured['numeric_inputs']['planner_kwargs']['seed_strategy'] == 'obstacle_boundary_subdivision'
    assert captured['numeric_inputs']['planner_kwargs']['w_yaw_acc'] == 2.0
    assert captured['numeric_inputs']['planner_kwargs']['kinematic_constraint_scale'] == pytest.approx(
        0.975)
    assert captured['numeric_inputs']['artifact_root'] == '/tmp'
    assert captured['numeric_inputs']['artifact_request_id'].startswith('request_')
    assert captured['numeric_inputs']['initial_ipopt_options']['ipopt.max_iter'] == 800
    assert captured['numeric_inputs']['initial_ipopt_options']['ipopt.acceptable_iter'] == 5
    assert captured['numeric_inputs']['constrained_ipopt_options']['ipopt.tol'] == pytest.approx(1e-6)
    assert len(captured['numeric_inputs']['initial_segments']) == 3


def test_run_optimizer_process_forwards_start_and_target_up_axes(monkeypatch):
    server = _bare_server()
    values = _casadi_parameter_values()
    server._planner_type = 'casadi_obstacle'
    server._mp_context = object()
    server.get_parameter = lambda name: SimpleNamespace(value=values[name])
    captured = {}
    server._publish_optimizing_path = lambda *args, **kwargs: None

    def fake_build_initial_seed(*args, **kwargs):
        captured['seed_kwargs'] = kwargs
        return [_segment()]

    def fake_run_planner_pipeline_process(mp_context, numeric_inputs, *args, **kwargs):
        del mp_context, numeric_inputs, args, kwargs
        return []

    monkeypatch.setattr(server, '_build_initial_seed', fake_build_initial_seed)
    monkeypatch.setattr(
        planner_module.optimization,
        'run_planner_pipeline_process',
        fake_run_planner_pipeline_process,
    )

    half_root = math.sqrt(0.5)
    server._run_optimizer_process(
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(1.0, 0.0, 0.0),
        _point(1.0, 0.0, 0.0),
        [],
        _BOUNDS,
        cancel_event=None,
        deadline=None,
        start_orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        target_orientation=SimpleNamespace(x=half_root, y=0.0, z=0.0, w=half_root),
    )

    assert captured['seed_kwargs']['z_axis'] == pytest.approx([0.0, 0.0, 1.0])
    assert captured['seed_kwargs']['goal_z_axis'] == pytest.approx([0.0, -1.0, 0.0])


def test_pipeline_status_callback_excludes_nested_remaining_time(monkeypatch):
    server = _bare_server()
    values = _casadi_parameter_values()
    server._planner_type = 'casadi_obstacle'
    server._mp_context = object()
    server.get_parameter = lambda name: SimpleNamespace(value=values[name])
    statuses = []

    def fake_run_planner_pipeline_process(mp_context, numeric_inputs, *args, **kwargs):
        del mp_context, numeric_inputs, args
        kwargs['message_callback']({
            'stage': 'constrained',
            'msg': 'Starting obstacle-constrained solve.',
            'remaining_sec': 12.3,
        })
        return []

    monkeypatch.setattr(
        planner_module.optimization,
        'run_planner_pipeline_process',
        fake_run_planner_pipeline_process,
    )

    server._run_optimizer_process(
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
        cancel_event=None,
        deadline=None,
        status_callback=statuses.append,
    )

    # Action feedback is a clean single line (the \n\n\n>>> decoration is console-only),
    # and the stageless payload falls back to its stage label, never message wording.
    assert statuses == ['Obstacle solve']


def test_pipeline_message_republishes_stage_obstacle_markers(monkeypatch):
    server = _bare_server()
    values = _casadi_parameter_values()
    server._planner_type = 'casadi_obstacle'
    server._mp_context = object()
    server.get_parameter = lambda name: SimpleNamespace(value=values[name])
    marker_calls = []
    server.obstacles.publish_markers_if_subscribed = (
        lambda published_obstacles, frame: marker_calls.append((published_obstacles, frame)))

    def fake_run_planner_pipeline_process(mp_context, numeric_inputs, *args, **kwargs):
        del mp_context, numeric_inputs, args
        kwargs['message_callback']({
            'stage': 'unconstrained',
            'msg': 'Starting bounds/kinematics warm-start solve with 1 gate-adjacent obstacle(s).',
            'obstacles': [{'type': 'cylinder', 'x': 0.5, 'y': 0.0, 'radius': 0.2, 'height': 2.5}],
        })
        kwargs['message_callback']({
            'stage': 'constrained',
            'msg': 'Starting obstacle-constrained solve.',
            # No 'obstacles' key on this one -- should not trigger another publish.
        })
        return []

    monkeypatch.setattr(
        planner_module.optimization,
        'run_planner_pipeline_process',
        fake_run_planner_pipeline_process,
    )

    server._run_optimizer_process(
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
        cancel_event=None,
        deadline=None,
    )

    assert len(marker_calls) == 1
    published_obstacles, frame = marker_calls[0]
    assert frame == 'map'
    assert published_obstacles[0]['x'] == pytest.approx(0.5)


def test_pipeline_status_prefix_distinguishes_major_steps():
    cls = planner_module.PlanTrajectoryActionServer
    status = cls._format_pipeline_status
    log = cls._format_pipeline_log
    opt = planner_module.optimization

    # The explicit step drives the label; feedback form carries no decoration.
    assert status('seed', 'Initial seed ready.', opt.PIPELINE_STEP_SEED) == (
        'Seed | Initial seed ready.')
    assert status('unconstrained', 'Starting solve.', opt.PIPELINE_STEP_OBSTACLE_FREE) == (
        'Geometric seed | Starting solve.')
    assert status('constrained', 'pass 1/4 started.', opt.PIPELINE_STEP_OBSTACLE_SOLVE) == (
        'Obstacle solve | pass 1/4 started.')
    assert status('constrained', 'running dense validation.', opt.PIPELINE_STEP_DENSE_CHECK) == (
        'Dense check | running dense validation.')
    assert status('constrained', 'refinement pass 2/4.', opt.PIPELINE_STEP_DENSE_REFINE) == (
        'Dense refinement | refinement pass 2/4.')

    # Console form keeps the blank-line + marker decoration for terminal scanning.
    assert log('seed', 'Initial seed ready.', opt.PIPELINE_STEP_SEED) == (
        '\n\n\n>>> Seed | Initial seed ready.')

    # Without an explicit step, labeling falls back to the stage and ignores message
    # wording (so a "refinement" message in the constrained stage is not mislabeled 05).
    assert status('constrained', 'Dense-validation refinement pass 2/4.') == (
        'Obstacle solve | Dense-validation refinement pass 2/4.')


def test_pipeline_feedback_is_compact_but_keeps_metrics():
    cls = planner_module.PlanTrajectoryActionServer
    opt = planner_module.optimization

    assert cls._compact_pipeline_feedback({
        'stage': 'unconstrained',
        'step': opt.PIPELINE_STEP_OBSTACLE_FREE,
        'msg': 'Obstacle-free candidate dense-valid with obstacles (Solve_Succeeded); wrote /tmp/long.csv.',
        'valid': True,
        'duration_sec': 3.745,
        'cost': 2.96562,
    }) == 'Geometric seed | dense-valid (T=3.75s, cost=2.97)'
    assert cls._compact_pipeline_feedback({
        'stage': 'constrained',
        'step': opt.PIPELINE_STEP_DENSE_CHECK,
        'msg': 'Obstacle-constrained solve pass 1/4 failed dense validation; wrote /tmp/long.csv: details',
        'duration_sec': 3.73,
        'cost': 2.95824,
    }) == 'Dense check | dense validation failed (T=3.73s, cost=2.96)'
    assert cls._compact_pipeline_feedback({
        'stage': 'constrained',
        'step': opt.PIPELINE_STEP_OBSTACLE_SOLVE,
        'msg': 'Starting obstacle-constrained solve.',
    }) == 'Obstacle solve'


def test_plan_returns_result_when_timeout_expires_before_optional_outputs(tmp_path, monkeypatch):
    server = _bare_server()
    values = {
        'limits': '',
        'preferred_frame': 'map',
        'planner_artifact_dir': str(tmp_path),
        'output_dir': str(tmp_path / 'pkg' / 'trajectories'),
        'use_cache': False,
        'save_cache': True,
        'yaw_follow_tangent': False,
        'min_segment_time': 0.5,
        'max_duration': 60.0,
        'gate_transition_distance': 0.5,
        'gate_transition_position_weight': 10.0,
        'gate_transition_velocity_weight': 1.0,
        'gate_transition_velocity_threshold': 0.01,
    }
    server.get_parameter = lambda name: _Parameter(values[name])
    server._parse_limits = lambda limits_yaml: {
        **dict(_LIMITS),
        'yaw_rate_max': 2.0,
        'yaw_accel_max': 6.0,
    }
    server.obstacles = SimpleNamespace(
        obstacle_planning_frame=lambda preferred_frame: preferred_frame,
        snapshot=lambda: (dict(_BOUNDS), []),
        effective_obstacles=lambda service_obstacles, bounds, frame: [],
        publish_markers_if_subscribed=lambda obstacles, frame: None,
        clear_markers=lambda frame: None,
    )
    server._should_use_in_place_scurve = lambda *args, **kwargs: False
    server._run_optimizer_process = lambda *args, **kwargs: [
        _segment(duration=1.0, c_x=[0.0, 1.0])
    ]
    server._check_cancel_or_timeout = lambda cancel_event, deadline: None
    monkeypatch.setattr(planner_module.time, 'monotonic', lambda: 2.0)
    monkeypatch.setattr(
        planner_module.validation,
        'validate_boundary_constraints',
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        planner_module.validation,
        'validate_kinematic_limits',
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        planner_module.validation,
        'validate_position_bounds',
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        planner_module.validation,
        'validate_obstacle_clearance',
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        planner_module.validation,
        'warn_on_discontinuity',
        lambda *args, **kwargs: None,
    )
    server._write_final_trajectory_artifact = lambda *args, **kwargs: pytest.fail(
        'optional artifact write should be skipped after timeout')
    server.save_to_csv = lambda *args, **kwargs: pytest.fail(
        'optional cache write should be skipped after timeout')
    server._publish_planned_path = lambda *args, **kwargs: pytest.fail(
        'optional visualization should be skipped after timeout')

    trajectory, message = server._plan_polynomial_trajectory(
        _goal(),
        traj_id=7,
        deadline=1.0,
    )

    assert message == 'Generated trajectory.'
    assert trajectory.trajectory_id == 7
    assert len(trajectory.pieces) == 1
    assert any('Skipping final trajectory artifact' in msg for msg in server._logger.infos)
    assert any('Skipping trajectory cache write' in msg for msg in server._logger.infos)
    assert any('Skipping planned trajectory visualization' in msg for msg in server._logger.infos)


def test_run_optimizer_process_uses_cached_seed_as_real_time(monkeypatch):
    server = _bare_server()
    values = _casadi_parameter_values()
    server._planner_type = 'casadi_obstacle'
    server._mp_context = object()
    server.get_parameter = lambda name: SimpleNamespace(value=values[name])
    cached_seed = [{
        'T': 1.0,
        'c_x': [0.0, 1.0],
        'c_y': [0.0],
        'c_z': [0.0],
    }]
    captured = {}

    def fake_run_planner_pipeline_process(mp_context, numeric_inputs, *args, **kwargs):
        captured['numeric_inputs'] = numeric_inputs
        return []

    monkeypatch.setattr(
        planner_module.optimization,
        'run_planner_pipeline_process',
        fake_run_planner_pipeline_process,
    )

    server._run_optimizer_process(
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
        cancel_event=None,
        deadline=None,
        seed_segments=cached_seed,
        seed_segments_are_real_time=True,
    )

    assert captured['numeric_inputs']['initial_segments_are_real_time'] is True
    assert captured['numeric_inputs']['initial_segments'][0]['c_x'] == [0.0, 1.0]


def test_run_optimizer_process_publishes_returned_pipeline_segments(monkeypatch):
    server = _bare_server()
    values = _casadi_parameter_values()
    server._planner_type = 'casadi_obstacle'
    server._mp_context = object()
    server.get_parameter = lambda name: SimpleNamespace(value=values[name])
    publisher = _FakePublisher()
    server._optimizing_path_publisher = publisher
    server._optimizing_path_topic = '/optimizing_trajectory'
    returned_segments = [_segment(duration=1.0, c_x=[0.0, 2.0])]

    def fake_run_planner_pipeline_process(mp_context, numeric_inputs, *args, **kwargs):
        del mp_context, numeric_inputs, args, kwargs
        return returned_segments

    monkeypatch.setattr(
        planner_module.optimization,
        'run_planner_pipeline_process',
        fake_run_planner_pipeline_process,
    )

    result = server._run_optimizer_process(
        _point(),
        _point(1.0, 0.0, 0.0),
        _point(),
        _point(),
        [],
        _BOUNDS,
        cancel_event=None,
        deadline=None,
    )

    assert result == returned_segments
    assert len(publisher.messages) >= 1
    path_msg = publisher.messages[-1]
    assert path_msg.header.frame_id == 'map'
    assert path_msg.poses[-1].pose.position.x == pytest.approx(2.0)


def test_optimizing_path_publish_samples_seed_segments():
    server = _bare_server()
    publisher = _FakePublisher()
    server._optimizing_path_publisher = publisher
    server._optimizing_path_topic = '/optimizing_trajectory'

    server._publish_optimizing_path([_segment(duration=2.0, c_x=[0.0, 1.0])], 'map')

    assert len(publisher.messages) == 1
    path_msg = publisher.messages[0]
    assert path_msg.header.frame_id == 'map'
    assert len(path_msg.poses) == 2
    assert path_msg.poses[0].pose.position.x == pytest.approx(0.0)
    assert path_msg.poses[-1].pose.position.x == pytest.approx(1.0)


def test_empty_optimizing_path_publish_clears_visualization():
    server = _bare_server()
    publisher = _FakePublisher()
    server._optimizing_path_publisher = publisher
    server._optimizing_path_topic = '/optimizing_trajectory'

    server._publish_empty_optimizing_path('map')

    assert len(publisher.messages) == 1
    path_msg = publisher.messages[0]
    assert path_msg.header.frame_id == 'map'
    assert path_msg.poses == []


def test_planned_path_publish_samples_polynomial_pieces():
    server = _bare_server()
    publisher = _FakePublisher()
    server._planned_path_publisher = publisher
    server._planned_path_topic = '/planned_trajectory'
    piece = planner_module.PlanTrajectoryActionServer._build_polynomial_piece(
        1.0,
        [0.0, 1.0],
        [0.0],
        [1.0],
        [0.0],
    )

    server._publish_planned_path([piece], 'map')

    assert len(publisher.messages) == 1
    path_msg = publisher.messages[0]
    assert path_msg.header.frame_id == 'map'
    assert len(path_msg.poses) == 2
    assert path_msg.poses[0].pose.position.x == pytest.approx(0.0)
    assert path_msg.poses[-1].pose.position.x == pytest.approx(1.0)


def test_log_seed_diagnostics_reports_subdivision_and_moves():
    server = _bare_server()
    diagnostics = {
        'base_segments': 3,
        'returned_segments': 6,
        'subdivided': True,
        'moves': [
            {
                'label': 'pA',
                'obstacle_index': 0,
                'obstacle_type': 'cylinder',
                'distance_before': 0.0,
                'distance_after': 0.95,
                'point_before': (2.0, 0.0, 1.0),
                'point_after': (1.05, 0.0, 1.0),
            },
        ],
    }

    server._log_seed_diagnostics(diagnostics)

    assert 'Seed subdivision: 3 -> 6 segment(s).' in server._logger.infos[0]
    assert 'pA moved for obstacle 0' in server._logger.infos[1]
    assert 'distance 0.000 -> 0.950' in server._logger.infos[1]


def test_log_seed_diagnostics_reports_clear_seed():
    server = _bare_server()

    server._log_seed_diagnostics({
        'base_segments': 3,
        'returned_segments': 3,
        'subdivided': False,
        'moves': [],
    })

    assert 'no guide point intersections' in server._logger.infos[0]


def test_obstacle_signature_ignores_marker_endpoint_flags():
    base_obstacle = planner_module.obstacles.capsule_dict(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        0.25,
        source='urdf',
        name='same-planning-shape',
    )
    flagged_obstacle = dict(base_obstacle)
    flagged_obstacle['a_endpoint_contained'] = True
    flagged_obstacle['b_endpoint_contained'] = True

    assert planner_module.PlanTrajectoryActionServer._obstacle_signature([base_obstacle]) == (
        planner_module.PlanTrajectoryActionServer._obstacle_signature([flagged_obstacle])
    )


def test_save_to_csv_commits_with_atomic_replace(tmp_path, monkeypatch):
    server = _bare_server()
    final_path = tmp_path / 'trajectory.csv'

    def fake_save(path, pieces):
        assert path != final_path
        path.write_text('duration\n1.0\n', encoding='utf-8')

    monkeypatch.setattr(planner_module, 'save_trajectory_pieces', fake_save)

    server.save_to_csv([], final_path)

    assert final_path.read_text(encoding='utf-8') == 'duration\n1.0\n'
    assert not list(tmp_path.glob('*.tmp'))


def test_write_final_trajectory_artifact_includes_yaw_coefficients(tmp_path):
    server = _bare_server()
    piece = server._build_polynomial_piece(
        1.0,
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    path = server._write_final_trajectory_artifact(
        [piece],
        tmp_path,
        'request:123',
    )

    assert path is not None
    assert path.name.startswith('flexible_drones_planner_request_123_final_with_yaw_')
    durations, _x_coeffs, _y_coeffs, _z_coeffs, yaw_coeffs = load_trajectory_csv(path)
    assert durations == [pytest.approx(1.0)]
    assert yaw_coeffs[0][:2] == pytest.approx([0.25, 0.5])


def _params(values):
    return lambda name: SimpleNamespace(value=values[name])


def _set_param(name, value):
    return SimpleNamespace(name=name, value=value)


class _Parameter:
    def __init__(self, value):
        self.value = value

    def get_parameter_value(self):
        try:
            double_value = float(self.value)
        except (TypeError, ValueError):
            double_value = 0.0
        return SimpleNamespace(
            string_value=str(self.value),
            double_value=double_value,
        )


# --- Two-mode dispatch predicate -------------------------------------------------

def test_should_use_in_place_scurve_selects_short_hover_moves():
    server = _bare_server()
    server.get_parameter = _params(
        {'mode_b_distance_threshold': 0.2, 'mode_b_speed_threshold': 0.05})
    zero = _point(0.0, 0.0, 0.0)
    start = _point(0.0, 0.0, 1.0)

    near = _point(0.05, 0.0, 1.0)
    assert server._should_use_in_place_scurve(start, near, zero, zero) is True


def test_should_use_in_place_scurve_rejects_long_moves():
    server = _bare_server()
    server.get_parameter = _params(
        {'mode_b_distance_threshold': 0.2, 'mode_b_speed_threshold': 0.05})
    zero = _point(0.0, 0.0, 0.0)
    start = _point(0.0, 0.0, 1.0)

    far = _point(1.0, 0.0, 1.0)
    assert server._should_use_in_place_scurve(start, far, zero, zero) is False


def test_should_use_in_place_scurve_rejects_moving_boundaries():
    server = _bare_server()
    server.get_parameter = _params(
        {'mode_b_distance_threshold': 0.2, 'mode_b_speed_threshold': 0.05})
    zero = _point(0.0, 0.0, 0.0)
    start = _point(0.0, 0.0, 1.0)
    near = _point(0.05, 0.0, 1.0)

    fast = _point(0.5, 0.0, 0.0)  # boundary speed above threshold
    assert server._should_use_in_place_scurve(start, near, fast, zero) is False


# --- Mode B candidate validation / fallback trigger ------------------------------

def test_pieces_pass_validation_accepts_clear_in_place_move():
    from flexible_drones_tools.trajectories.planners import scurve

    server = _bare_server()
    server._limits = dict(_LIMITS)
    server._limits.update({'yaw_rate_max': 2.0, 'yaw_accel_max': 4.0})
    segments = scurve.plan_in_place_scurve(
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        start_yaw=0.0, target_yaw=1.0, limits=server._limits)

    start = _point(0.0, 0.0, 1.0)
    assert server._pieces_pass_validation(
        segments, start, start, _point(), _point(), _BOUNDS, []) is True


def test_pieces_pass_validation_rejects_blocked_path():
    from flexible_drones_tools.trajectories.planners import scurve

    server = _bare_server()
    server._limits = dict(_LIMITS)
    server._limits.update({'yaw_rate_max': 2.0, 'yaw_accel_max': 4.0})
    segments = scurve.plan_in_place_scurve(
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        start_yaw=0.0, target_yaw=1.0, limits=server._limits)

    start = _point(0.0, 0.0, 1.0)
    blocking = [{'type': 'cylinder', 'x': 0.0, 'y': 0.0, 'radius': 0.5, 'height': 3.0}]
    assert server._pieces_pass_validation(
        segments, start, start, _point(), _point(), _BOUNDS, blocking) is False
    assert any('rejected' in message for message in server._logger.infos)


# --- Runtime parameter validation callback ---------------------------------------

def test_validate_parameter_update_rejects_negative_weight():
    server = _bare_server()
    result = server._validate_parameter_update([_set_param('w_yaw_rate', -1.0)])
    assert result.successful is False
    assert 'w_yaw_rate' in result.reason


def test_validate_parameter_update_accepts_nonnegative_weight():
    server = _bare_server()
    assert server._validate_parameter_update(
        [_set_param('w_yaw_rate', 2.0)]).successful is True


def test_validate_parameter_update_rejects_nonpositive_counts():
    server = _bare_server()
    assert server._validate_parameter_update(
        [_set_param('n_eval', 0)]).successful is False
    assert server._validate_parameter_update(
        [_set_param('n_segments', -3)]).successful is False


def test_validate_parameter_update_checks_n_segments_for_seed_strategy():
    server = _bare_server()
    server.get_parameter = _params(
        {'seed_strategy': 'obstacle_boundary_subdivision', 'n_segments': 6})
    assert server._validate_parameter_update(
        [_set_param('n_segments', 4)]).successful is False
    assert server._validate_parameter_update(
        [_set_param('n_segments', 3)]).successful is True
    assert server._validate_parameter_update(
        [_set_param('n_segments', 6)]).successful is True


def test_validate_parameter_update_validates_limits_yaml():
    server = _bare_server()
    assert server._validate_parameter_update(
        [_set_param('limits', '{yaw_rate_max: 1.5}')]).successful is True
    assert server._validate_parameter_update(
        [_set_param('limits', '{yaw_rate_max: -1.0}')]).successful is False
    assert server._validate_parameter_update(
        [_set_param('limits', '[not, a, mapping]')]).successful is False


def test_validate_parameter_update_rejects_invalid_kinematic_constraint_scale():
    server = _bare_server()

    assert server._validate_parameter_update(
        [_set_param('kinematic_constraint_scale', 0.5)]).successful is True
    assert server._validate_parameter_update(
        [_set_param('kinematic_constraint_scale', 0.0)]).successful is False
    assert server._validate_parameter_update(
        [_set_param('kinematic_constraint_scale', 1.1)]).successful is False

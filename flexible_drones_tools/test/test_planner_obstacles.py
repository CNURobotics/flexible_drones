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

"""Unit tests for obstacle/bounds geometry and the ObstacleManager."""

import threading
from types import SimpleNamespace

import pytest
from rclpy.parameter import Parameter
from visualization_msgs.msg import Marker

from flexible_drones_tools.trajectories.planners import obstacles
from flexible_drones_tools.trajectories.planners.obstacles import ObstacleManager

_BOUNDS = {'x_min': -3.0, 'x_max': 3.0, 'y_min': -3.0, 'y_max': 3.0, 'z_min': 0.25, 'z_max': 2.5}


class _FakeLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.infos.append(message)


class _FakePublisher:
    def __init__(self, subscription_count=1):
        self.messages = []
        self._subscription_count = subscription_count

    def get_subscription_count(self):
        return self._subscription_count

    def publish(self, message):
        self.messages.append(message)


def _obstacle(**kwargs):
    values = {'type': 'cylinder', 'x': 0.0, 'y': 0.0, 'radius': 0.25, 'height': 0.0}
    values.update(kwargs)
    return SimpleNamespace(**values)


def _bare_manager(obstacles_state=None, publisher=None):
    manager = ObstacleManager.__new__(ObstacleManager)
    manager._state_lock = threading.Lock()
    manager._obstacles = obstacles_state
    manager._default_capsules = []
    manager._obstacle_marker_publisher = publisher
    manager._last_obstacle_marker_keys = []
    logger = _FakeLogger()
    manager.node = SimpleNamespace(
        get_logger=lambda: logger,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: SimpleNamespace())
        ),
    )
    manager.logger = logger
    return manager


# --- service callbacks -------------------------------------------------------

def test_set_obstacles_accepts_empty_list_as_configured():
    manager = _bare_manager(obstacles_state=None)
    request = SimpleNamespace(obstacles=[])
    response = SimpleNamespace(success=False, message='')

    result = manager.set_obstacles_callback(request, response)

    assert result.success is True
    assert result.message == 'Updated obstacles: 0 cylinder(s).'
    assert manager._obstacles == []
    assert manager.obstacles_configured() is True


def test_set_obstacles_rejects_invalid_radius_without_updating(capsys):
    existing = [{'type': 'cylinder', 'x': 1.0, 'y': 2.0, 'radius': 0.3, 'height': 0.0}]
    manager = _bare_manager(obstacles_state=list(existing))
    request = SimpleNamespace(obstacles=[_obstacle(radius=0.0)])
    response = SimpleNamespace(success=True, message='')

    result = manager.set_obstacles_callback(request, response)
    captured = capsys.readouterr()

    assert result.success is False
    assert 'radius must be > 0.0' in result.message
    assert '\033[31m' in captured.out
    assert manager._obstacles == existing


def test_set_obstacles_rejects_non_finite_value_without_updating():
    manager = _bare_manager(obstacles_state=None)
    request = SimpleNamespace(obstacles=[_obstacle(x=float('nan'))])
    response = SimpleNamespace(success=True, message='')

    result = manager.set_obstacles_callback(request, response)

    assert result.success is False
    assert 'non-finite' in result.message
    assert manager._obstacles is None


# --- parameter parsing -------------------------------------------------------

def test_string_array_parameter_value_accepts_typed_empty_default():
    parameter = Parameter('obstacle_description_topics', Parameter.Type.STRING_ARRAY, [])
    assert obstacles.string_array_parameter_value(parameter) == []


def test_string_array_parameter_value_ignores_blank_initialized_default():
    parameter = Parameter('obstacle_description_topics', Parameter.Type.STRING_ARRAY, [''])
    assert obstacles.string_array_parameter_value(parameter) == []


def test_string_array_parameter_value_returns_topic_list():
    parameter = Parameter(
        'obstacle_description_topics',
        Parameter.Type.STRING_ARRAY,
        ['/gate_A_description', '/gate_B_description'],
    )
    assert obstacles.string_array_parameter_value(parameter) == [
        '/gate_A_description',
        '/gate_B_description',
    ]


# --- capsule geometry --------------------------------------------------------

def test_capsule_endpoint_containment_flags_only_inside_other_capsules():
    capsules = [
        obstacles.capsule_dict([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.2, source='urdf', name='contained-end'),
        obstacles.capsule_dict([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], 0.4, source='urdf', name='containing'),
    ]

    obstacles.mark_contained_capsule_endpoints(capsules)

    assert capsules[0]['a_endpoint_contained'] is False
    assert capsules[0]['b_endpoint_contained'] is False
    assert capsules[1]['a_endpoint_contained'] is True
    assert capsules[1]['b_endpoint_contained'] is False


def test_capsule_endpoint_containment_suppresses_endpoint_inside_capsule_midspan():
    capsules = [
        obstacles.capsule_dict([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.2, source='urdf', name='contained-mid'),
        obstacles.capsule_dict([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], 0.4, source='urdf', name='containing-mid'),
    ]

    obstacles.mark_contained_capsule_endpoints(capsules)

    assert capsules[0]['a_endpoint_contained'] is False
    assert capsules[0]['b_endpoint_contained'] is True
    assert capsules[1]['a_endpoint_contained'] is True
    assert capsules[1]['b_endpoint_contained'] is False


def test_merge_colinear_overlapping_capsules_reduces_capsule_chain():
    capsules = [
        obstacles.capsule_dict([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.25, source='urdf', name='first'),
        obstacles.capsule_dict([1.4, 0.0, 0.0], [2.0, 0.0, 0.0], 0.25, source='urdf', name='second'),
        obstacles.capsule_dict([2.4, 0.0, 0.0], [3.0, 0.0, 0.0], 0.25, source='urdf', name='third'),
    ]

    merged = obstacles.merge_colinear_overlapping_capsules(capsules)

    assert len(merged) == 1
    assert merged[0]['ax'] == pytest.approx(0.0)
    assert merged[0]['bx'] == pytest.approx(3.0)
    assert merged[0]['radius'] == pytest.approx(0.25)


def test_merge_colinear_overlapping_capsules_keeps_separated_capsules():
    capsules = [
        obstacles.capsule_dict([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.25),
        obstacles.capsule_dict([1.6, 0.0, 0.0], [2.0, 0.0, 0.0], 0.25),
    ]
    assert len(obstacles.merge_colinear_overlapping_capsules(capsules)) == 2


def test_merge_colinear_overlapping_capsules_keeps_angled_capsules():
    capsules = [
        obstacles.capsule_dict([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.25),
        obstacles.capsule_dict([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 0.25),
    ]
    assert len(obstacles.merge_colinear_overlapping_capsules(capsules)) == 2


def _urdf_capsule_template(link_name):
    return {
        'topic': '/gate_description',
        'link_name': link_name,
        'collision_index': 0,
        'a_local': [0.0, 0.0, 0.0],
        'b_local': [0.0, 0.0, 1.0],
        'raw_radius': 0.25,
        'a_endpoint_contained': False,
        'b_endpoint_contained': False,
    }


def test_merge_fixed_urdf_capsule_templates_merges_across_fixed_links():
    robot = SimpleNamespace(
        links=[SimpleNamespace(name='gate_root'), SimpleNamespace(name='gate_child')],
        joints=[
            SimpleNamespace(
                type='fixed', parent='gate_root', child='gate_child',
                origin=SimpleNamespace(xyz=[0.0, 0.0, 1.4], rpy=[0.0, 0.0, 0.0]),
            ),
        ],
    )
    capsules = [_urdf_capsule_template('gate_root'), _urdf_capsule_template('gate_child')]

    merged = obstacles.merge_fixed_urdf_capsule_templates('/gate_description', robot, capsules)

    assert len(merged) == 1
    assert merged[0]['link_name'] == 'gate_root'
    assert merged[0]['a_local'][2] == pytest.approx(0.0)
    assert merged[0]['b_local'][2] == pytest.approx(2.4)
    assert merged[0]['raw_radius'] == pytest.approx(0.25)


def test_merge_fixed_urdf_capsule_templates_keeps_non_fixed_links_separate():
    robot = SimpleNamespace(
        links=[SimpleNamespace(name='gate_root'), SimpleNamespace(name='gate_child')],
        joints=[
            SimpleNamespace(
                type='revolute', parent='gate_root', child='gate_child',
                origin=SimpleNamespace(xyz=[0.0, 0.0, 1.4], rpy=[0.0, 0.0, 0.0]),
            ),
        ],
    )
    capsules = [_urdf_capsule_template('gate_root'), _urdf_capsule_template('gate_child')]

    merged = obstacles.merge_fixed_urdf_capsule_templates('/gate_description', robot, capsules)

    assert len(merged) == 2
    assert [template['link_name'] for template in merged] == ['gate_root', 'gate_child']


def test_capsule_inside_service_cylinder_uses_in_bounds_z_portion():
    capsule = obstacles.capsule_dict([0.0, 2.5, 0.25], [0.0, 2.5, 1.0], 0.2)
    service_obstacle = {'x': 0.0, 'y': 2.5, 'radius': 0.6, 'height': 3.0}
    bounds = dict(_BOUNDS, z_min=0.4, z_max=3.0)

    assert obstacles.capsule_inside_service_cylinder(capsule, service_obstacle, bounds) is True


def test_capsule_inside_service_cylinder_rejects_in_bounds_z_protrusion():
    capsule = obstacles.capsule_dict([0.0, 2.5, 1.0], [0.0, 2.5, 2.95], 0.2)
    service_obstacle = {'x': 0.0, 'y': 2.5, 'radius': 0.6, 'height': 2.8}
    bounds = dict(_BOUNDS, z_min=0.4, z_max=3.0)

    assert obstacles.capsule_inside_service_cylinder(capsule, service_obstacle, bounds) is False


def test_numeric_obstacle_strips_capsule_and_cylinder_fields():
    capsule = obstacles.capsule_dict([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.25, name='x')
    numeric = obstacles.numeric_obstacle(capsule)
    assert set(numeric) == {'type', 'ax', 'ay', 'az', 'bx', 'by', 'bz', 'radius'}
    cyl = obstacles.numeric_obstacle({'type': 'cylinder', 'x': 1.0, 'y': 2.0, 'radius': 0.3})
    assert cyl == {'type': 'cylinder', 'x': 1.0, 'y': 2.0, 'radius': 0.3, 'height': 0.0}


# --- ObstacleManager logging + markers ---------------------------------------

def test_log_service_obstacle_filtering_reports_incoming_and_subsumed_capsules():
    manager = _bare_manager()
    service_obstacles = [{'x': 0.0, 'y': 2.5, 'radius': 0.6, 'height': 3.0}]
    subsumed = [
        [obstacles.capsule_dict([0.0, 2.5, 0.5], [0.0, 2.5, 2.0], 0.25, source='urdf', name='/gate:test:0')],
    ]

    manager._log_service_obstacle_filtering(service_obstacles, subsumed, _BOUNDS)

    assert 'Incoming service obstacle 0: x=0.000, y=2.500' in manager.logger.infos[0]
    assert 'effective_z=(0.250, 3.000)' in manager.logger.infos[0]
    assert 'subsumes 1 default capsule(s): /gate:test:0' in manager.logger.infos[1]


def test_obstacle_marker_array_has_unique_marker_keys():
    publisher = _FakePublisher()
    manager = _bare_manager(publisher=publisher)
    capsules = [
        {'type': 'capsule', 'ax': 0.0, 'ay': 0.0, 'az': 0.0,
         'bx': 0.0, 'by': 0.0, 'bz': 1.0, 'radius': 0.25},
    ]

    manager.publish_markers_if_subscribed(capsules, 'map')

    # Published as two separate messages: a DELETEALL-only array, then the markers.
    assert len(publisher.messages) == 2
    clear_keys = [(marker.ns, marker.id) for marker in publisher.messages[0].markers]
    assert clear_keys == [('planner_obstacles_clear', -1)]
    marker_keys = [(marker.ns, marker.id) for marker in publisher.messages[1].markers]
    assert marker_keys == [
        ('planner_obstacles', 0),
        ('planner_obstacles', 1),
        ('planner_obstacles', 2),
    ]
    assert len(marker_keys) == len(set(marker_keys))


def test_clear_markers_publishes_deleteall_array():
    publisher = _FakePublisher(subscription_count=0)
    manager = _bare_manager(publisher=publisher)
    manager._last_obstacle_marker_keys = [('planner_obstacles', 2)]

    manager.clear_markers('map')

    assert len(publisher.messages) == 1
    markers = publisher.messages[0].markers
    assert len(markers) == 1
    assert markers[0].header.frame_id == 'map'
    assert markers[0].action == Marker.DELETEALL
    assert manager._last_obstacle_marker_keys == []


def test_obstacle_marker_redraw_deletes_prior_keys_before_deleteall():
    publisher = _FakePublisher()
    manager = _bare_manager(publisher=publisher)
    first_capsules = [
        {'type': 'capsule', 'ax': 0.0, 'ay': 0.0, 'az': 0.0,
         'bx': 0.0, 'by': 0.0, 'bz': 1.0, 'radius': 0.25},
    ]
    second_capsules = [
        {'type': 'capsule', 'ax': 1.0, 'ay': 0.0, 'az': 0.0,
         'bx': 1.0, 'by': 0.0, 'bz': 1.0, 'radius': 0.25,
         'b_endpoint_contained': True},
    ]

    manager.publish_markers_if_subscribed(first_capsules, 'map')
    manager.publish_markers_if_subscribed(second_capsules, 'map')

    assert len(publisher.messages) == 5
    prior_delete_markers = publisher.messages[2].markers
    assert [(marker.ns, marker.id, marker.action) for marker in prior_delete_markers] == [
        ('planner_obstacles', 0, Marker.DELETE),
        ('planner_obstacles', 1, Marker.DELETE),
        ('planner_obstacles', 2, Marker.DELETE),
    ]
    assert all(marker.header.frame_id == 'map' for marker in prior_delete_markers)

    deleteall_markers = publisher.messages[3].markers
    assert len(deleteall_markers) == 1
    assert deleteall_markers[0].action == Marker.DELETEALL
    assert deleteall_markers[0].header.frame_id == 'map'

    new_markers = publisher.messages[4].markers
    assert [(marker.ns, marker.id, marker.action) for marker in new_markers] == [
        ('planner_obstacles', 0, Marker.ADD),
        ('planner_obstacles', 1, Marker.ADD),
    ]
    assert manager._last_obstacle_marker_keys == [
        ('planner_obstacles', 0),
        ('planner_obstacles', 1),
    ]


def test_obstacle_marker_array_renders_pipeline_cylinder_payloads():
    publisher = _FakePublisher()
    manager = _bare_manager(publisher=publisher)
    cylinder = {
        'type': 'cylinder',
        'x': 0.5,
        'y': -0.25,
        'radius': 0.2,
        'height': 2.5,
    }

    manager.publish_markers_if_subscribed([cylinder], 'map')

    assert len(publisher.messages) == 2
    markers = publisher.messages[1].markers
    assert len(markers) == 1
    marker = markers[0]
    assert marker.ns == 'planner_obstacles'
    assert marker.id == 0
    assert marker.type == Marker.CYLINDER
    assert marker.pose.position.x == pytest.approx(0.5)
    assert marker.pose.position.y == pytest.approx(-0.25)
    assert marker.pose.position.z == pytest.approx(1.25)
    assert marker.scale.x == pytest.approx(0.4)
    assert marker.scale.y == pytest.approx(0.4)
    assert marker.scale.z == pytest.approx(2.5)


def test_obstacle_marker_array_skips_contained_endpoint_cap_spheres():
    publisher = _FakePublisher()
    manager = _bare_manager(publisher=publisher)
    capsules = [
        obstacles.capsule_dict(
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.25,
            source='urdf', name='one-contained-cap', b_endpoint_contained=True,
        ),
    ]

    manager.publish_markers_if_subscribed(capsules, 'map')

    markers = publisher.messages[1].markers
    marker_keys = [(marker.ns, marker.id) for marker in markers]
    assert marker_keys == [
        ('planner_obstacles', 0),
        ('planner_obstacles', 1),
    ]
    assert markers[-1].type == Marker.SPHERE
    assert markers[-1].pose.position.z == pytest.approx(0.0)


def test_obstacle_marker_array_keeps_one_shared_corner_cap_sphere():
    publisher = _FakePublisher()
    manager = _bare_manager(publisher=publisher)
    capsules = [
        obstacles.capsule_dict([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.25),
        obstacles.capsule_dict([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 0.25),
    ]
    obstacles.mark_contained_capsule_endpoints(capsules)

    manager.publish_markers_if_subscribed(capsules, 'map')

    markers = publisher.messages[1].markers
    corner_spheres = [
        marker for marker in markers
        if (
            marker.type == Marker.SPHERE
            and marker.pose.position.x == pytest.approx(1.0)
            and marker.pose.position.y == pytest.approx(0.0)
            and marker.pose.position.z == pytest.approx(0.0)
        )
    ]
    assert len(corner_spheres) == 1

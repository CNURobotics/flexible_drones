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
Obstacle and flight-bounds handling for the trajectory planner.

Capsules are the unifying primitive (cylinders/spheres/boxes from URDF collisions
are converted to bounding capsules). This module owns the pure capsule geometry
(parse/merge/bounds/markers) as module functions, and the stateful ``ObstacleManager``
that holds the runtime bounds, service obstacles, and TF-resolved default capsules.
See README.md (Obstacles and boundaries).
"""

import math
import time
from copy import deepcopy

import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.validate_topic_name import validate_topic_name
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String
from tf2_ros import TransformException
from visualization_msgs.msg import Marker, MarkerArray

from flexible_drones_tools.trajectories.planners.base_planner import DEFAULT_BOUNDS
from flexible_drones_tools.trajectories.planners.geometry import (
    origin_to_matrix,
    point_capsule_distance,
    transform_point,
    transform_to_matrix,
)


def string_array_parameter_value(parameter):
    """Read a STRING_ARRAY parameter, falling back to comma-splitting a string value."""
    value = parameter.get_parameter_value()
    strings = [item.strip() for item in value.string_array_value if item.strip()]
    if strings:
        return strings
    if value.string_value:
        return [item.strip() for item in value.string_value.split(',') if item.strip()]
    return []


def capsule_dict(
    a,
    b,
    radius,
    source='service',
    name='',
    a_endpoint_contained=False,
    b_endpoint_contained=False,
):
    """Build a capsule obstacle dict from endpoints a/b and a radius."""
    return {
        'type': 'capsule',
        'ax': float(a[0]), 'ay': float(a[1]), 'az': float(a[2]),
        'bx': float(b[0]), 'by': float(b[1]), 'bz': float(b[2]),
        'radius': float(radius),
        'source': source,
        'name': name,
        'a_endpoint_contained': bool(a_endpoint_contained),
        'b_endpoint_contained': bool(b_endpoint_contained),
    }


def geometry_to_capsule_endpoints(geometry, geometry_types):
    """
    Return (a, b, radius) of a capsule bounding the geometry, or None.

    Cylinders map exactly. Spheres become a degenerate (a == b) capsule.
    Boxes are wrapped by the tightest capsule whose axis runs along the
    box's longest dimension; its radius is half the diagonal of the other
    two dimensions, which contains the whole box (with some inflation).
    Meshes are intentionally unsupported and skipped by the caller.
    """
    Box, Cylinder, Sphere = geometry_types
    if isinstance(geometry, Cylinder):
        radius = float(geometry.radius)
        length = float(geometry.length)
        if radius <= 0.0 or length < 0.0:
            return None
        return [0.0, 0.0, -0.5 * length], [0.0, 0.0, 0.5 * length], radius
    if isinstance(geometry, Sphere):
        radius = float(geometry.radius)
        if radius <= 0.0:
            return None
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], radius
    if isinstance(geometry, Box):
        size = [float(value) for value in geometry.size]
        if len(size) != 3 or any(value <= 0.0 for value in size):
            return None
        axis = int(np.argmax(size))
        others = [size[i] for i in range(3) if i != axis]
        radius = 0.5 * math.hypot(others[0], others[1])
        half_length = 0.5 * size[axis]
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        a[axis] = -half_length
        b[axis] = half_length
        return a, b, radius
    return None


def urdf_link_name(value):
    """Resolve a URDF joint parent/child reference to a link-name string."""
    return getattr(value, 'link', value)


def template_raw_capsule_count(capsules):
    """Return the count of capsules parsed before colinear merging (kept on the first)."""
    if not capsules:
        return 0
    return int(capsules[0].get('raw_capsule_count', len(capsules)))


def parse_capsules_from_urdf(topic, urdf_xml):
    """Parse a URDF string into merged capsule obstacle templates (link-local)."""
    try:
        from urdf_parser_py.urdf import Box, Cylinder, Sphere, URDF
    except ImportError as exc:
        raise ImportError('urdf_parser_py is required to parse obstacle robot_description topics') from exc

    geometry_types = (Box, Cylinder, Sphere)
    robot = URDF.from_xml_string(urdf_xml)
    capsules = []
    for link in robot.links:
        for collision_index, collision in enumerate(link.collisions):
            geometry = getattr(collision, 'geometry', None)
            # Meshes are intentionally not supported as obstacle primitives.
            endpoints = geometry_to_capsule_endpoints(geometry, geometry_types)
            if endpoints is None:
                continue
            local_geom_a, local_geom_b, radius = endpoints
            collision_tf = origin_to_matrix(collision.origin)
            local_a = transform_point(collision_tf, local_geom_a)
            local_b = transform_point(collision_tf, local_geom_b)
            capsules.append({
                'topic': topic,
                'link_name': link.name,
                'collision_index': collision_index,
                'a_local': local_a,
                'b_local': local_b,
                'raw_radius': radius,
                'a_endpoint_contained': False,
                'b_endpoint_contained': False,
            })
    raw_count = len(capsules)
    merged = merge_fixed_urdf_capsule_templates(topic, robot, capsules)
    if merged:
        merged[0]['raw_capsule_count'] = raw_count
    return merged


def merge_fixed_urdf_capsule_templates(topic, robot, capsules):
    """Merge colinear capsules that are rigidly (fixed-joint) connected into one template."""
    if not capsules:
        return []

    link_order = [link.name for link in robot.links]
    fixed_graph = {link_name: [] for link_name in link_order}
    for joint in getattr(robot, 'joints', []):
        if str(getattr(joint, 'type', 'fixed')) != 'fixed':
            continue
        parent = urdf_link_name(getattr(joint, 'parent', None))
        child = urdf_link_name(getattr(joint, 'child', None))
        if parent not in fixed_graph or child not in fixed_graph:
            continue
        parent_to_child = origin_to_matrix(getattr(joint, 'origin', None))
        child_to_parent = np.linalg.inv(parent_to_child)
        fixed_graph[parent].append((child, parent_to_child))
        fixed_graph[child].append((parent, child_to_parent))

    components = {}
    transforms = {}
    visited = set()
    for root in link_order:
        if root in visited:
            continue
        stack = [(root, np.eye(4, dtype=np.float64))]
        visited.add(root)
        while stack:
            link_name, root_to_link = stack.pop()
            components[link_name] = root
            transforms[link_name] = root_to_link
            for neighbor, link_to_neighbor in fixed_graph.get(link_name, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append((neighbor, root_to_link @ link_to_neighbor))

    grouped = {}
    for template in capsules:
        root = components.get(template['link_name'], template['link_name'])
        grouped.setdefault(root, []).append(template)

    merged_templates = []
    for root in link_order:
        group = grouped.get(root, [])
        if not group:
            continue
        merge_inputs = []
        for template in group:
            root_to_link = transforms.get(template['link_name'], np.eye(4, dtype=np.float64))
            merge_inputs.append(
                capsule_dict(
                    transform_point(root_to_link, template['a_local']),
                    transform_point(root_to_link, template['b_local']),
                    template['raw_radius'],
                    source='urdf',
                    name=f"{template['topic']}:{template['link_name']}:{template['collision_index']}",
                )
            )
        for collision_index, merged in enumerate(merge_colinear_overlapping_capsules(merge_inputs)):
            merged_templates.append({
                'topic': topic,
                'link_name': root,
                'collision_index': collision_index,
                'a_local': np.asarray([merged['ax'], merged['ay'], merged['az']], dtype=np.float64),
                'b_local': np.asarray([merged['bx'], merged['by'], merged['bz']], dtype=np.float64),
                'raw_radius': float(merged['radius']),
                'a_endpoint_contained': False,
                'b_endpoint_contained': False,
            })
    return merged_templates


def mark_contained_capsule_endpoints(capsules):
    """Flag capsule endpoints that lie inside another capsule (to skip duplicate caps)."""
    marked_corner_points = []
    for index, capsule in enumerate(capsules):
        for prefix in ('a', 'b'):
            point = (
                float(capsule[f'{prefix}x']),
                float(capsule[f'{prefix}y']),
                float(capsule[f'{prefix}z']),
            )
            capsule_radius = float(capsule['radius'])
            containing_others = [
                other for other_index, other in enumerate(capsules)
                if (
                    other_index != index
                    and point_capsule_distance(point, other) <= float(other['radius']) + 1e-9
                )
            ]
            if not containing_others:
                capsule[f'{prefix}_endpoint_contained'] = False
                continue
            if endpoint_overlaps_capsule_endpoint(point, capsule, containing_others):
                point_vec = np.asarray(point, dtype=np.float64)
                if not any(
                    np.linalg.norm(point_vec - marked_point) <= capsule_radius + marked_radius + 1e-9
                    for marked_point, marked_radius in marked_corner_points
                ):
                    marked_corner_points.append((point_vec, capsule_radius))
                    capsule[f'{prefix}_endpoint_contained'] = False
                    continue
            capsule[f'{prefix}_endpoint_contained'] = True


def endpoint_overlaps_capsule_endpoint(point, capsule, others):
    """Return True if ``point`` is within combined radii of any endpoint of ``others``."""
    point_vec = np.asarray(point, dtype=np.float64)
    capsule_radius = float(capsule['radius'])
    return any(
        np.linalg.norm(
            point_vec - np.asarray(
                [other[f'{prefix}x'], other[f'{prefix}y'], other[f'{prefix}z']],
                dtype=np.float64,
            )
        ) <= capsule_radius + float(other['radius']) + 1e-9
        for other in others
        for prefix in ('a', 'b')
    )


def merge_colinear_overlapping_capsules(capsules):
    """Greedily fuse equal-radius, colinear, overlapping capsules into longer ones."""
    merged = []
    for capsule in capsules:
        candidate = deepcopy(capsule)
        while True:
            for index, existing in enumerate(merged):
                combined = merged_colinear_capsule(existing, candidate)
                if combined is None:
                    continue
                candidate = combined
                merged.pop(index)
                break
            else:
                merged.append(candidate)
                break
    return merged


def merged_colinear_capsule(first, second):
    """Return a single capsule spanning two colinear equal-radius capsules, or None."""
    radius_a = float(first['radius'])
    radius_b = float(second['radius'])
    if not math.isclose(radius_a, radius_b, rel_tol=1e-6, abs_tol=1e-6):
        return None

    a0 = np.asarray([first['ax'], first['ay'], first['az']], dtype=np.float64)
    a1 = np.asarray([first['bx'], first['by'], first['bz']], dtype=np.float64)
    b0 = np.asarray([second['ax'], second['ay'], second['az']], dtype=np.float64)
    b1 = np.asarray([second['bx'], second['by'], second['bz']], dtype=np.float64)
    axis_a = a1 - a0
    axis_b = b1 - b0
    length_a = float(np.linalg.norm(axis_a))
    length_b = float(np.linalg.norm(axis_b))
    if length_a <= 1e-9 or length_b <= 1e-9:
        return None

    unit_a = axis_a / length_a
    unit_b = axis_b / length_b
    if abs(float(np.dot(unit_a, unit_b))) < 1.0 - 1e-6:
        return None

    line_tolerance = max(1e-4, 1e-3 * max(radius_a, radius_b))
    if (
        float(np.linalg.norm(np.cross(b0 - a0, unit_a))) > line_tolerance
        or float(np.linalg.norm(np.cross(b1 - a0, unit_a))) > line_tolerance
    ):
        return None

    first_interval = sorted((0.0, length_a))
    second_interval = sorted((
        float(np.dot(b0 - a0, unit_a)),
        float(np.dot(b1 - a0, unit_a)),
    ))
    gap = max(first_interval[0], second_interval[0]) - min(first_interval[1], second_interval[1])
    if gap > radius_a + radius_b + line_tolerance:
        return None

    endpoints = [a0, a1, b0, b1]
    projections = [float(np.dot(point - a0, unit_a)) for point in endpoints]
    min_index = int(np.argmin(projections))
    max_index = int(np.argmax(projections))
    return capsule_dict(
        endpoints[min_index],
        endpoints[max_index],
        radius_a,
        source=first.get('source', second.get('source', 'urdf')),
        name='+'.join(
            name for name in (first.get('name', ''), second.get('name', ''))
            if name
        ),
    )


def capsule_intersects_bounds(capsule, bounds):
    """Return True if the capsule's bounding box overlaps the flight bounds."""
    radius = float(capsule['radius'])
    for axis, min_key, max_key in (
        ('x', 'x_min', 'x_max'),
        ('y', 'y_min', 'y_max'),
        ('z', 'z_min', 'z_max'),
    ):
        lo = min(float(capsule[f'a{axis}']), float(capsule[f'b{axis}'])) - radius
        hi = max(float(capsule[f'a{axis}']), float(capsule[f'b{axis}'])) + radius
        if hi < float(bounds[min_key]) or lo > float(bounds[max_key]):
            return False
    return True


def service_cylinder_to_capsule(obstacle, bounds):
    """Convert a SetTrajectoryObstacles cylinder into a vertical capsule within bounds."""
    x = float(obstacle['x'])
    y = float(obstacle['y'])
    z_min = float(bounds['z_min'])
    height = float(obstacle.get('height', 0.0))
    z_top = height if height > 0.0 and math.isfinite(height) else float(bounds['z_max'])
    z_low = min(z_min, z_top)
    z_high = max(z_min, z_top)
    return capsule_dict(
        [x, y, z_low],
        [x, y, z_high],
        float(obstacle['radius']),
        source='service',
        name='set_trajectory_obstacles',
    )


def capsule_inside_service_cylinder(capsule, service_obstacle, bounds):
    """Return True if a default capsule is fully subsumed by a service cylinder in bounds."""
    service = service_cylinder_to_capsule(service_obstacle, bounds)
    service_radius = float(service['radius'])
    capsule_radius = float(capsule['radius'])
    if capsule_radius > service_radius:
        return False

    z_low = min(float(service['az']), float(service['bz']))
    z_high = max(float(service['az']), float(service['bz']))
    bounds_z_low = float(bounds['z_min'])
    bounds_z_high = float(bounds['z_max'])
    for prefix in ('a', 'b'):
        x = float(capsule[f'{prefix}x'])
        y = float(capsule[f'{prefix}y'])
        z = float(capsule[f'{prefix}z'])
        xy_distance = math.hypot(x - float(service['ax']), y - float(service['ay']))
        if xy_distance + capsule_radius > service_radius:
            return False
        in_bounds_z_low = max(z - capsule_radius, bounds_z_low)
        in_bounds_z_high = min(z + capsule_radius, bounds_z_high)
        if in_bounds_z_low > in_bounds_z_high:
            continue
        if in_bounds_z_low < z_low or in_bounds_z_high > z_high:
            return False
    return True


def numeric_obstacle(obstacle):
    """Strip an obstacle dict to the numeric fields the planner subprocess needs."""
    otype = str(obstacle.get('type', 'cylinder'))
    if otype == 'capsule':
        return {
            'type': 'capsule',
            'ax': float(obstacle['ax']),
            'ay': float(obstacle['ay']),
            'az': float(obstacle['az']),
            'bx': float(obstacle['bx']),
            'by': float(obstacle['by']),
            'bz': float(obstacle['bz']),
            'radius': float(obstacle['radius']),
        }
    return {
        'type': 'cylinder',
        'x': float(obstacle['x']),
        'y': float(obstacle['y']),
        'radius': float(obstacle['radius']),
        'height': float(obstacle.get('height', 0.0)),
    }


def rotation_from_z_axis(unit_vector):
    """Rotation taking +z onto ``unit_vector`` (used to orient cylinder markers)."""
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    target = np.asarray(unit_vector, dtype=np.float64)
    dot = float(np.clip(np.dot(z_axis, target), -1.0, 1.0))
    cross = np.cross(z_axis, target)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm <= 1e-12:
        if dot >= 0.0:
            return R.identity()
        return R.from_rotvec(np.pi * np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    axis = cross / cross_norm
    angle = math.acos(dot)
    return R.from_rotvec(angle * axis)


RGBA_OBSTACLE = (1.0, 0.4, 0.05, 0.2)


def _obstacle_marker_style(obstacle):
    """Pick the marker namespace/color for an obstacle."""
    del obstacle
    return 'planner_obstacles', RGBA_OBSTACLE


def capsule_cap_marker(position, radius, planning_frame, stamp, marker_id, ns='planner_obstacles', color=RGBA_OBSTACLE):
    """Build a sphere marker for one rounded cap of a capsule obstacle."""
    marker = Marker()
    marker.header.frame_id = planning_frame
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = int(marker_id)
    marker.action = Marker.ADD
    marker.type = Marker.SPHERE
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = float(position[2])
    marker.pose.orientation.w = 1.0
    diameter = max(0.0, 2.0 * float(radius))
    marker.scale.x = diameter
    marker.scale.y = diameter
    marker.scale.z = diameter
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    return marker


def capsule_markers(obstacle, planning_frame, stamp, marker_id):
    """Build markers (body cylinder/sphere plus uncontained end caps) for a capsule."""
    a_vec = np.asarray([obstacle['ax'], obstacle['ay'], obstacle['az']], dtype=np.float64)
    b_vec = np.asarray([obstacle['bx'], obstacle['by'], obstacle['bz']], dtype=np.float64)
    delta = b_vec - a_vec
    length = float(np.linalg.norm(delta))
    midpoint = 0.5 * (a_vec + b_vec)
    radius = float(obstacle['radius'])
    ns, color = _obstacle_marker_style(obstacle)

    marker = Marker()
    marker.header.frame_id = planning_frame
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = int(marker_id)
    marker.action = Marker.ADD
    marker.pose.position.x = float(midpoint[0])
    marker.pose.position.y = float(midpoint[1])
    marker.pose.position.z = float(midpoint[2])
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color

    diameter = max(0.0, 2.0 * radius)
    marker.scale.x = diameter
    marker.scale.y = diameter
    if length <= 1e-9:
        marker.type = Marker.SPHERE
        marker.scale.z = diameter
        marker.pose.orientation.w = 1.0
        return [marker]

    marker.type = Marker.CYLINDER
    marker.scale.z = length
    quat = rotation_from_z_axis(delta / length).as_quat()
    marker.pose.orientation.x = float(quat[0])
    marker.pose.orientation.y = float(quat[1])
    marker.pose.orientation.z = float(quat[2])
    marker.pose.orientation.w = float(quat[3])
    markers = [marker]
    if radius > 0.05:
        if not obstacle.get('a_endpoint_contained', False):
            markers.append(
                capsule_cap_marker(
                    a_vec, radius, planning_frame, stamp, marker_id + len(markers), ns=ns, color=color)
            )
        if not obstacle.get('b_endpoint_contained', False):
            markers.append(
                capsule_cap_marker(
                    b_vec, radius, planning_frame, stamp, marker_id + len(markers), ns=ns, color=color)
            )
    return markers


def cylinder_markers(obstacle, planning_frame, stamp, marker_id):
    """Build a vertical cylinder marker from a normalized planner cylinder."""
    ns, color = _obstacle_marker_style(obstacle)
    radius = float(obstacle['radius'])
    height = max(0.0, float(obstacle.get('height', 0.0)))

    marker = Marker()
    marker.header.frame_id = planning_frame
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = int(marker_id)
    marker.action = Marker.ADD
    marker.type = Marker.CYLINDER
    marker.pose.position.x = float(obstacle['x'])
    marker.pose.position.y = float(obstacle['y'])
    marker.pose.position.z = 0.5 * height
    marker.pose.orientation.w = 1.0
    marker.scale.x = max(0.0, 2.0 * radius)
    marker.scale.y = max(0.0, 2.0 * radius)
    marker.scale.z = height
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    return [marker]


def obstacle_markers(obstacle, planning_frame, stamp, marker_id):
    """Build RViz markers for any obstacle shape emitted by the planner pipeline."""
    if str(obstacle.get('type', 'cylinder')).lower() == 'capsule':
        return capsule_markers(obstacle, planning_frame, stamp, marker_id)
    return cylinder_markers(obstacle, planning_frame, stamp, marker_id)


def color_red(message):
    """Wrap a message in ANSI red for console emphasis."""
    return f'\033[31m{message}\033[0m'


class ObstacleManager:
    """
    Owns runtime flight bounds, service obstacles, and default URDF capsules.

    Holds a reference to the owning rclpy node for TF lookups, logging, parameters,
    and subscription/publisher creation. Pure capsule geometry lives in the module
    functions above. See README.md (Obstacles and boundaries).
    """

    def __init__(self, node):
        self.node = node
        self._state_lock = node._state_lock
        self._bounds = {
            key: float(node.get_parameter(f'bounds.{key}').value) for key in DEFAULT_BOUNDS
        }
        # None means the operator has not provided/cleared obstacle state yet.
        # A configured empty list is valid and means "plan with no obstacles."
        self._obstacles = None
        self._default_capsules = []
        self._last_default_capsules = {}
        self._last_obstacle_tf_warning = {}
        self._obstacle_description_subscriptions = []
        self._obstacle_marker_publisher = None
        self._last_obstacle_marker_keys = []
        self._setup_obstacle_description_subscriptions()
        self._setup_obstacle_marker_publisher()

    def _setup_obstacle_description_subscriptions(self):
        topics = string_array_parameter_value(self.node.get_parameter('obstacle_description_topics'))
        if not topics:
            return

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        for topic in topics:
            subscription = self.node.create_subscription(
                String,
                topic,
                lambda msg, topic_name=topic: self._obstacle_description_callback(topic_name, msg),
                qos,
            )
            self._obstacle_description_subscriptions.append(subscription)
        self.node.get_logger().info(
            'Subscribed to default obstacle descriptions: ' + ', '.join(topics)
        )

    def _setup_obstacle_marker_publisher(self):
        topic = self.node.get_parameter(
            'planner_obstacles_marker_topic').get_parameter_value().string_value.strip()
        try:
            validate_topic_name(topic)
        except Exception as exc:  # noqa: B902
            self.node.get_logger().warning(
                f"Not publishing planner obstacle markers: invalid topic name '{topic}': {exc}"
            )
            return

        # Depth > 1 so the delete/clear messages published just before the new
        # marker array (publish_markers_if_subscribed) are not dropped from the
        # history cache by a subscriber that has not pulled them yet.
        self._obstacle_marker_publisher = self.node.create_publisher(MarkerArray, topic, 5)
        self.node.get_logger().info(f"Planner obstacle markers enabled on '{topic}'.")

    def _obstacle_description_callback(self, topic, msg):
        try:
            capsules = parse_capsules_from_urdf(topic, msg.data)
        except Exception as exc:  # noqa: B902
            self.node.get_logger().warning(
                f"Failed to parse default obstacles from '{topic}': {type(exc).__name__}: {exc}"
            )
            return

        with self._state_lock:
            self._default_capsules = [
                capsule for capsule in self._default_capsules
                if capsule['topic'] != topic
            ] + capsules
            self._default_capsules.sort(key=lambda capsule: (
                capsule['topic'],
                capsule['link_name'],
                capsule['collision_index'],
            ))
        self.node.get_logger().info(
            f"Parsed {template_raw_capsule_count(capsules)} default capsule obstacle(s) from '{topic}'."
        )
        raw_count = template_raw_capsule_count(capsules)
        if capsules and len(capsules) < raw_count:
            self.node.get_logger().info(
                f"Merged default capsule templates from '{topic}': "
                f'{raw_count} parsed capsule obstacle(s) condensed to '
                f'{len(capsules)} stored capsule template(s).'
            )

    def obstacle_planning_frame(self, preferred_frame):
        """Frame for obstacle TF resolution: the override param, else ``preferred_frame``."""
        frame = self.node.get_parameter(
            'obstacle_planning_frame').get_parameter_value().string_value.strip()
        return frame or preferred_frame

    def resolve_default_capsules(self, planning_frame, bounds=None):
        """TF-resolve stored URDF templates into ``planning_frame``, merge, and bounds-cull."""
        with self._state_lock:
            templates = deepcopy(self._default_capsules)
        if not templates:
            return []

        resolved = []
        safety_margin = max(0.0, float(self.node.get_parameter('obstacle_safety_margin').value))
        now = time.monotonic()
        for template in templates:
            key = (template['topic'], template['link_name'], template['collision_index'])
            try:
                tf = self.node.tf_buffer.lookup_transform(
                    planning_frame,
                    template['link_name'],
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2),
                )
                matrix = transform_to_matrix(tf)
                a_world = transform_point(matrix, template['a_local'])
                b_world = transform_point(matrix, template['b_local'])
                capsule = capsule_dict(
                    a_world,
                    b_world,
                    float(template['raw_radius']) + safety_margin,
                    source='urdf',
                    name=f"{template['topic']}:{template['link_name']}:{template['collision_index']}",
                )
                self._last_default_capsules[key] = capsule
            except TransformException as exc:
                capsule = self._last_default_capsules.get(key)
                last_warning = self._last_obstacle_tf_warning.get(key, 0.0)
                if now - last_warning > 2.0:
                    self._last_obstacle_tf_warning[key] = now
                    fallback = 'using last good pose' if capsule is not None else 'dropping obstacle'
                    self.node.get_logger().warning(
                        f"Failed to transform default obstacle '{template['link_name']}' "
                        f"to '{planning_frame}': {exc}; {fallback}."
                    )
                if capsule is None:
                    continue
            if bounds is not None and not capsule_intersects_bounds(capsule, bounds):
                continue
            resolved.append(deepcopy(capsule))
        unmerged_count = len(resolved)
        resolved = merge_colinear_overlapping_capsules(resolved)
        if len(resolved) < unmerged_count:
            self.node.get_logger().info(
                f'Merged {unmerged_count} default capsule obstacle(s) '
                f'into {len(resolved)} colinear capsule obstacle(s).'
            )
        mark_contained_capsule_endpoints(resolved)
        return resolved

    def effective_obstacles(self, service_obstacles, bounds, planning_frame):
        """Merge TF-resolved default capsules with service cylinders for one request."""
        service_obstacles = service_obstacles or []
        default_capsules = self.resolve_default_capsules(planning_frame, bounds)
        service_subsumed_defaults = [
            [
                capsule for capsule in default_capsules
                if capsule_inside_service_cylinder(capsule, service_obstacle, bounds)
            ]
            for service_obstacle in service_obstacles
        ]
        subsumed_default_ids = {
            id(capsule)
            for subsumed_defaults in service_subsumed_defaults
            for capsule in subsumed_defaults
        }
        filtered_defaults = [
            capsule for capsule in default_capsules
            if id(capsule) not in subsumed_default_ids
        ]
        self._log_service_obstacle_filtering(
            service_obstacles,
            service_subsumed_defaults,
            bounds,
        )
        subsumed_default_count = len(default_capsules) - len(filtered_defaults)
        service_capsules = [
            service_cylinder_to_capsule(obstacle, bounds)
            for obstacle in service_obstacles
        ]
        with self._state_lock:
            base_default_count = len(self._default_capsules)
        self.node.get_logger().info(
            'Obstacle set for planning request: '
            f'default capsules {len(filtered_defaults)}/{len(default_capsules)} '
            f'(resolved from {base_default_count} stored template(s), '
            f'{subsumed_default_count} subsumed by service obstacles), '
            f'service obstacles {len(service_capsules)}, '
            f'total {len(filtered_defaults) + len(service_capsules)}.'
        )
        return filtered_defaults + service_capsules

    def _log_service_obstacle_filtering(self, service_obstacles, service_subsumed_defaults, bounds):
        if not service_obstacles:
            return

        for index, service_obstacle in enumerate(service_obstacles):
            service = service_cylinder_to_capsule(service_obstacle, bounds)
            z_low = min(float(service['az']), float(service['bz']))
            z_high = max(float(service['az']), float(service['bz']))
            self.node.get_logger().info(
                'Incoming service obstacle '
                f'{index}: x={float(service_obstacle["x"]):.3f}, '
                f'y={float(service_obstacle["y"]):.3f}, '
                f'radius={float(service_obstacle["radius"]):.3f}, '
                f'height={float(service_obstacle.get("height", 0.0)):.3f}, '
                f'effective_z=({z_low:.3f}, {z_high:.3f})'
            )
            subsumed = service_subsumed_defaults[index]
            if not subsumed:
                continue
            names = ', '.join(
                capsule.get('name') or f'default_{subsumed_index}'
                for subsumed_index, capsule in enumerate(subsumed)
            )
            self.node.get_logger().info(
                f'Incoming service obstacle {index} subsumes '
                f'{len(subsumed)} default capsule(s): {names}'
            )

    def publish_markers_if_subscribed(self, obstacles, planning_frame):
        """Publish RViz markers for ``obstacles`` if the marker topic has subscribers."""
        publisher = self._obstacle_marker_publisher
        if publisher is None or publisher.get_subscription_count() <= 0:
            return

        stamp = self.node.get_clock().now().to_msg()

        # Publish deletes as standalone messages before the new markers. RViz can
        # apply a single MarkerArray as one visual update, so bundling DELETEALL
        # with ADDs can leave stale markers behind on some redraws.
        prior_keys = list(getattr(self, '_last_obstacle_marker_keys', []))
        if prior_keys:
            delete_array = MarkerArray()
            for ns, marker_id in prior_keys:
                delete_marker = Marker()
                delete_marker.header.frame_id = planning_frame
                delete_marker.header.stamp = stamp
                delete_marker.ns = ns
                delete_marker.id = int(marker_id)
                delete_marker.action = Marker.DELETE
                delete_array.markers.append(delete_marker)
            publisher.publish(delete_array)

        clear_marker = Marker()
        clear_marker.header.frame_id = planning_frame
        clear_marker.header.stamp = stamp
        clear_marker.ns = 'planner_obstacles_clear'
        clear_marker.id = -1
        clear_marker.action = Marker.DELETEALL
        publisher.publish(MarkerArray(markers=[clear_marker]))

        marker_array = MarkerArray()
        marker_id = 0
        for obstacle in obstacles:
            markers = obstacle_markers(obstacle, planning_frame, stamp, marker_id)
            marker_array.markers.extend(markers)
            marker_id += len(markers)

        publisher.publish(marker_array)
        self._last_obstacle_marker_keys = [
            (marker.ns, int(marker.id))
            for marker in marker_array.markers
        ]

    def clear_markers(self, planning_frame):
        """Publish a DELETEALL marker array for planner obstacle visualizations."""
        publisher = self._obstacle_marker_publisher
        if publisher is None:
            return
        stamp = self.node.get_clock().now().to_msg()
        clear_marker = Marker()
        clear_marker.header.frame_id = planning_frame
        clear_marker.header.stamp = stamp
        clear_marker.ns = 'planner_obstacles_clear'
        clear_marker.id = -1
        clear_marker.action = Marker.DELETEALL
        publisher.publish(MarkerArray(markers=[clear_marker]))
        self._last_obstacle_marker_keys = []

    def set_bounds_callback(self, request, response):
        """Handle the set-bounds service request (validate and store flight bounds)."""
        b = request.bounds
        if not (b.x_min < b.x_max and b.y_min < b.y_max and b.z_min < b.z_max):
            response.success = False
            response.message = 'Invalid bounds: each axis requires min < max.'
            self.node.get_logger().warning(response.message)
            return response
        new_bounds = {
            'x_min': float(b.x_min), 'x_max': float(b.x_max),
            'y_min': float(b.y_min), 'y_max': float(b.y_max),
            'z_min': float(b.z_min), 'z_max': float(b.z_max),
        }
        with self._state_lock:
            self._bounds = new_bounds
        response.success = True
        response.message = f'Updated bounds: {new_bounds}'
        self.node.get_logger().info(response.message)
        return response

    def set_obstacles_callback(self, request, response):
        """Handle the set-obstacles service request (validate and store cylinders)."""
        obstacles = []
        for index, obs in enumerate(request.obstacles):
            otype = (obs.type or 'cylinder').lower()
            if otype != 'cylinder':
                response.success = False
                response.message = (
                    f'Invalid obstacle {index}: unsupported type {obs.type}; '
                    'only cylinder is supported.'
                )
                self.warn_rejected(response.message)
                return response
            values = {
                'x': float(obs.x),
                'y': float(obs.y),
                'radius': float(obs.radius),
                'height': float(obs.height),
            }
            invalid_fields = [
                field for field, value in values.items()
                if not math.isfinite(value)
            ]
            if invalid_fields:
                response.success = False
                response.message = (
                    f'Invalid obstacle {index}: non-finite value(s) for '
                    f'{", ".join(invalid_fields)}.'
                )
                self.warn_rejected(response.message)
                return response
            if values['radius'] <= 0.0:
                response.success = False
                response.message = (
                    f'Invalid obstacle {index}: radius must be > 0.0 '
                    f'(got {values["radius"]}).'
                )
                self.warn_rejected(response.message)
                return response
            if values['height'] < 0.0:
                response.success = False
                response.message = (
                    f'Invalid obstacle {index}: height must be >= 0.0, or 0.0 '
                    f'to use the planner z_max (got {values["height"]}).'
                )
                self.warn_rejected(response.message)
                return response
            obstacles.append({
                'type': 'cylinder',
                **values,
            })
        with self._state_lock:
            self._obstacles = obstacles
        response.success = True
        response.message = f'Updated obstacles: {len(obstacles)} cylinder(s).'
        self.node.get_logger().info(response.message)
        return response

    def warn_rejected(self, message):
        """Log a rejected request prominently (red console + node warning)."""
        formatted = color_red(f'[trajectory planner rejected] {message}')
        print(formatted, flush=True)
        self.node.get_logger().warning(message)

    def obstacles_configured(self):
        """Return True once obstacles have been configured (even as an empty list)."""
        with self._state_lock:
            return self._obstacles is not None

    def snapshot(self):
        """Return deep copies of the current (bounds, service obstacles)."""
        with self._state_lock:
            return deepcopy(self._bounds), deepcopy(self._obstacles)

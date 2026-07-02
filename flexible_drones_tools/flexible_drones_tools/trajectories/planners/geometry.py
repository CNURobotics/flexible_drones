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

"""Small numpy geometry helpers shared by the obstacle and validation modules."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_point(matrix, point):
    """Apply a 4x4 homogeneous transform to a 3D point, returning a length-3 array."""
    homogeneous = matrix @ np.asarray(
        [float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=np.float64)
    return homogeneous[:3]


def origin_to_matrix(origin):
    """Convert a URDF origin (xyz/rpy) to a 4x4 homogeneous transform."""
    matrix = np.eye(4, dtype=np.float64)
    if origin is None:
        return matrix

    xyz = getattr(origin, 'xyz', None) or [0.0, 0.0, 0.0]
    rpy = getattr(origin, 'rpy', None) or [0.0, 0.0, 0.0]
    matrix[:3, :3] = R.from_euler('xyz', [float(value) for value in rpy]).as_matrix()
    matrix[:3, 3] = np.asarray([float(value) for value in xyz], dtype=np.float64)
    return matrix


def transform_to_matrix(tf_stamped):
    """Convert a geometry_msgs TransformStamped to a 4x4 homogeneous transform."""
    transform = tf_stamped.transform
    matrix = np.eye(4, dtype=np.float64)
    q = transform.rotation
    matrix[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    t = transform.translation
    matrix[:3, 3] = [float(t.x), float(t.y), float(t.z)]
    return matrix


def point_capsule_distance(point, capsule):
    """Euclidean distance from a 3D point to a capsule's core segment (a->b)."""
    point_vec = np.asarray(point, dtype=np.float64)
    a_vec = np.asarray(
        [capsule['ax'], capsule['ay'], capsule['az']],
        dtype=np.float64,
    )
    b_vec = np.asarray(
        [capsule['bx'], capsule['by'], capsule['bz']],
        dtype=np.float64,
    )
    segment = b_vec - a_vec
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        closest = a_vec
    else:
        t = float(np.dot(point_vec - a_vec, segment) / length_sq)
        closest = a_vec + max(0.0, min(1.0, t)) * segment
    return float(np.linalg.norm(point_vec - closest))

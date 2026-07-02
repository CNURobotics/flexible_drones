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

"""Unit tests for the Mode B in-place S-curve generator and yaw-slew integration."""

import math

from numpy.polynomial.polynomial import Polynomial
import pytest

from flexible_drones_tools.trajectories.planners import scurve, validation, yaw_planning

_LIMITS = {
    'v_max': 1.0, 'a_max': 4.0, 'j_max': 10.0, 's_max': 50.0,
    'yaw_rate_max': 2.0, 'yaw_accel_max': 4.0,
}


def test_in_place_rotation_holds_position_and_sizes_for_yaw():
    segments = scurve.plan_in_place_scurve(
        (1.0, 1.0, 1.5), (1.0, 1.0, 1.5),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        start_yaw=0.0, target_yaw=math.radians(170.0), limits=_LIMITS,
    )
    assert len(segments) == 1
    segment = segments[0]
    duration = segment['T']

    # Position is held (it is a pure rotation) with zero endpoint velocity.
    for key in ('c_x', 'c_y', 'c_z'):
        poly = Polynomial(segment[key])
        assert poly(0.0) == pytest.approx(poly(duration), abs=1e-9)
        assert poly.deriv(1)(0.0) == pytest.approx(0.0, abs=1e-9)
        assert poly.deriv(1)(duration) == pytest.approx(0.0, abs=1e-9)

    # Duration is sized by the yaw slew, not the (zero) translation.
    expected = scurve._yaw_slew_duration(math.radians(170.0), 2.0, 4.0)
    assert duration == pytest.approx(expected, rel=1e-6)


def test_short_translation_honours_endpoints_and_limits():
    segments = scurve.plan_in_place_scurve(
        (0.0, 0.0, 1.0), (0.15, 0.0, 1.0),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        start_yaw=0.0, target_yaw=0.0, limits=_LIMITS,
    )
    segment = segments[0]
    duration = segment['T']
    poly_x = Polynomial(segment['c_x'])
    assert poly_x(0.0) == pytest.approx(0.0, abs=1e-9)
    assert poly_x(duration) == pytest.approx(0.15, abs=1e-9)

    peak_v, peak_a, peak_j, peak_s = scurve._peak_derivatives(segment['c_x'], duration)
    assert peak_v <= _LIMITS['v_max'] + 1e-3
    assert peak_a <= _LIMITS['a_max'] + 1e-3
    assert peak_j <= _LIMITS['j_max'] + 1e-3
    assert peak_s <= _LIMITS['s_max'] + 1e-3


def test_in_place_yaw_slew_respects_yaw_limits():
    target = math.radians(170.0)
    segments = scurve.plan_in_place_scurve(
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        start_yaw=0.0, target_yaw=target, limits=_LIMITS,
    )
    yaw_coefficients = yaw_planning.fit_yaw_coefficients(
        segments, start_yaw=0.0, target_yaw=target,
    )

    duration = segments[0]['T']
    yaw_poly = Polynomial(yaw_coefficients[0])
    assert yaw_poly(0.0) == pytest.approx(0.0, abs=1e-6)
    assert yaw_poly(duration) == pytest.approx(target, abs=1e-6)

    # The realized yaw slew stays within the policy yaw limits (no advisory warnings).
    warnings = validation.find_yaw_limit_warnings(
        [duration], yaw_coefficients,
        _LIMITS['yaw_rate_max'], _LIMITS['yaw_accel_max'],
    )
    assert warnings == []

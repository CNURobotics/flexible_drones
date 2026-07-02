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
Numeric checks for the limit/time-normalized control-effort objective terms.

Uses real CasADi (skips if unavailable or stubbed by another test module).
"""

from importlib.util import find_spec

import numpy as np
import pytest

HAS_CASADI = find_spec('casadi') is not None
pytestmark = pytest.mark.skipif(not HAS_CASADI, reason='real casadi unavailable')

if HAS_CASADI:
    import flexible_drones_tools.trajectories.planners.casadi_obstacle_planner as planner_module
    from flexible_drones_tools.trajectories.planners.casadi_obstacle_planner import CasadiObstaclePlanner

    ca = planner_module.ca
else:
    CasadiObstaclePlanner = None
    ca = None


P_LIM = {'x': (-3.0, 3.0), 'y': (-3.0, 3.0), 'z': (-2.0, 2.0)}
S_VALS = np.linspace(0.0, 1.0, 10)
DS = 1.0 / (len(S_VALS) - 1)


def _planner(a_max=4.0, **weights):
    limits = {'v_max': 1.0, 'a_max': a_max, 'j_max': 10.0, 's_max': 50.0}
    kw = {'w_time': 10.0, 'w_acc': 0.0, 'w_jerk': 0.0, 'w_snap': 0.0}
    kw.update(weights)
    return CasadiObstaclePlanner(limits=limits, **kw)


def _const_accel_coeffs(accel, T):
    # x(s) = 0.5*accel*(T*s)^2  ->  constant physical acceleration 'accel' in x.
    c = [0.0] * 8
    c[2] = 0.5 * accel * T**2
    return ca.DM(c), ca.DM([0.0] * 8), ca.DM([0.0] * 8)


def test_reference_time_is_box_diagonal_over_half_vmax():
    p = _planner()  # v_max = 1.0
    # diagonal = sqrt(6^2 + 6^2 + 4^2), half v_max = 0.5
    diagonal = np.sqrt(6.0**2 + 6.0**2 + 4.0**2)
    assert p._reference_time(P_LIM) == pytest.approx(diagonal / 0.5)


def test_accel_effort_matches_normalized_integral():
    a_max = 4.0
    T, t_ref = 2.0, 12.0
    p = _planner(a_max=a_max, w_acc=1.0)
    cx, cy, cz = _const_accel_coeffs(a_max, T)  # accel exactly at the limit
    cost = float(p.casadi_effort_terms(cx, cy, cz, T, t_ref, S_VALS, DS))
    # (|a|/a_max)^2 = 1 at every sample; integral weight sums to len*ds.
    expected = 1.0 * (T / t_ref) * (len(S_VALS) * DS)
    assert cost == pytest.approx(expected, rel=1e-9)


def test_limit_normalization_makes_weights_comparable():
    # Acceleration at half its limit costs the same as snap at half its limit,
    # given equal weights -> the per-limit normalization is what makes w_acc and
    # w_snap a common currency.
    T, t_ref = 2.0, 12.0
    p_acc = _planner(w_acc=1.0)          # a_max = 4.0
    cx, cy, cz = _const_accel_coeffs(2.0, T)  # accel = 2.0 = 0.5 * a_max
    cost = float(p_acc.casadi_effort_terms(cx, cy, cz, T, t_ref, S_VALS, DS))
    expected = (0.5**2) * (T / t_ref) * (len(S_VALS) * DS)
    assert cost == pytest.approx(expected, rel=1e-9)


def test_cost_scales_inversely_with_reference_time():
    T = 2.0
    p = _planner(w_acc=1.0)
    cx, cy, cz = _const_accel_coeffs(4.0, T)
    c1 = float(p.casadi_effort_terms(cx, cy, cz, T, 6.0, S_VALS, DS))
    c2 = float(p.casadi_effort_terms(cx, cy, cz, T, 12.0, S_VALS, DS))
    assert c1 == pytest.approx(2.0 * c2, rel=1e-9)


def test_term_is_dimensionless_under_uniform_scaling():
    # Scale positions by k and the limit by k -> identical (dimensionless) cost.
    T, t_ref, k = 2.0, 12.0, 3.0
    base = _planner(a_max=4.0, w_acc=1.0)
    scaled = _planner(a_max=4.0 * k, w_acc=1.0)
    cb = float(base.casadi_effort_terms(*_const_accel_coeffs(4.0, T), T, t_ref, S_VALS, DS))
    cs = float(scaled.casadi_effort_terms(*_const_accel_coeffs(4.0 * k, T), T, t_ref, S_VALS, DS))
    assert cs == pytest.approx(cb, rel=1e-9)


def test_zero_weights_give_zero_cost():
    p = _planner()  # all effort weights 0
    cx, cy, cz = _const_accel_coeffs(4.0, 2.0)
    assert float(p.casadi_effort_terms(cx, cy, cz, 2.0, 12.0, S_VALS, DS)) == pytest.approx(0.0)

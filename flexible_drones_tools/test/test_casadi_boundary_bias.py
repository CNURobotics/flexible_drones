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
Numeric checks for the soft flight-volume boundary bias.

Uses real CasADi (skips if unavailable or stubbed by another test module's
import) to evaluate ``casadi_boundary_bias`` as a function of position.
"""

from importlib.util import find_spec
from types import SimpleNamespace

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


MARGIN = 0.5
WEIGHT = 50.0
P_LIM = {'x': (-3.0, 3.0), 'y': (-3.0, 3.0), 'z': (-3.0, 3.0)}


def _bias_fn(margin=MARGIN, weight=WEIGHT):
    fake_self = SimpleNamespace(boundary_bias_margin=margin, boundary_bias_weight=weight)
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    z = ca.MX.sym('z')
    expr = CasadiObstaclePlanner.casadi_boundary_bias(fake_self, x, y, z, P_LIM)
    return ca.Function('bias', [x, y, z], [expr])


def _at(fn, x, y=0.0, z=0.0):
    # y, z held at the box centre so only the x walls are in play.
    return float(fn(x, y, z))


def test_bias_is_zero_in_the_interior():
    fn = _bias_fn()
    # Centre of the box: every wall is 3 m away, well outside the 0.5 m band.
    assert _at(fn, 0.0) == pytest.approx(0.0, abs=1e-9)
    # Exactly at the margin edge (0.5 m inside the +x wall) -> zero.
    assert _at(fn, 3.0 - MARGIN) == pytest.approx(0.0, abs=1e-9)
    # Just beyond the band (0.6 m inside) -> still zero.
    assert _at(fn, 3.0 - 0.6) == pytest.approx(0.0, abs=1e-9)


def test_bias_reaches_weight_at_the_wall():
    fn = _bias_fn()
    # At the wall, encroachment is full (=1), so cost == weight for that one wall.
    assert _at(fn, 3.0) == pytest.approx(WEIGHT, rel=1e-6)
    assert _at(fn, -3.0) == pytest.approx(WEIGHT, rel=1e-6)
    # Halfway into the band: encroach = 0.5 -> weight * 0.25.
    assert _at(fn, 3.0 - 0.25 * MARGIN) == pytest.approx(WEIGHT * (1 - 0.25) ** 2, rel=1e-6)


def test_bias_increases_monotonically_toward_the_wall():
    fn = _bias_fn()
    distances = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]  # distance to the +x wall, shrinking
    costs = [_at(fn, 3.0 - d) for d in distances]
    assert all(b > a for a, b in zip(costs, costs[1:])), costs
    # C1 onset: just inside the band the cost is tiny (quadratic), not a step.
    assert _at(fn, 3.0 - 0.49) < 0.1


def test_two_adjacent_walls_sum():
    fn = _bias_fn()
    # Corner-ish point near both the +x and +y walls (each 0.1 m away).
    expected = 2.0 * WEIGHT * ((MARGIN - 0.1) / MARGIN) ** 2
    assert _at(fn, 3.0 - 0.1, 3.0 - 0.1) == pytest.approx(expected, rel=1e-6)


def test_zero_weight_disables_bias():
    fn = _bias_fn(weight=0.0)
    assert _at(fn, 3.0) == pytest.approx(0.0, abs=1e-12)

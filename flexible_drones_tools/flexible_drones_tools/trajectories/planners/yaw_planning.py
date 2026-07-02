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
Concern B yaw output stage: fit the output yaw polynomial after the XYZ solve.

The XYZ trajectory is already fixed when this runs, so the path tangent
``atan2(vy, vx)`` at every sample is a known constant -- not a decision variable.
That makes the whole stage a single convex, equality-constrained least-squares
(one KKT linear solve), with the same degree-7 / C4 structure as the min-snap
machinery in ``trajectories/utilities/kkt.py``:

  minimize   w_align * sum_i gate_i * (yaw(t_i) - tangent_ref_i)^2
           + w_smooth * integral(yaw''(t))^2  dt
           + w_accel_limit * integral((yaw''(t) / yaw_accel_max)^2) dt
           + w_snap   * integral(yaw''''(t))^2 dt
  subject to first/last yaw pinned (value + zero rate/accel/jerk)
             C4 continuity only at interior segment joins.

Everything is in *real* time ``yaw(t) = sum_k c_k t^k`` so the coefficients match
the x/y/z convention used everywhere else in the pipeline. Working from the
precomputed tangent reference means there is no ``atan2`` in the optimization and
forward-facing is automatic (we regress toward the actual travel heading, so the
180-degree-reversed solution never arises).
"""

import math

import numpy as np

# Degree-7 polynomial (8 coefficients), matching x/y/z and utilities/kkt.py.
DEFAULT_COEFFICIENT_COUNT = 8
# Number of continuity orders enforced at each join: position..snap == C4.
CONTINUITY_ORDERS = 5
# Fractional back-off applied to yaw rate/accel caps at the constraint grid so the
# peak *between* grid samples still stays under the true limit (advisory tolerance
# is ~1e-3 absolute). 1% is negligible against conservative policy yaw limits.
CAP_SAFETY_FRACTION = 0.01


def wrap_to_pi(angle):
    """Wrap an angle (radians) to the (-pi, pi] interval."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _falling_factorial(k, m):
    """Return k*(k-1)*...*(k-m+1), the constant from differentiating t^k m times."""
    result = 1.0
    for offset in range(m):
        result *= (k - offset)
    return result


def _derivative_row(n, t, order):
    """
    Monomial-basis row for the ``order``-th time derivative at time ``t``.

    For value (order 0) this is ``[1, t, t^2, ...]``; row . c == d^order/dt^order of
    ``sum_k c_k t^k`` evaluated at ``t``.
    """
    row = np.zeros(n)
    for k in range(order, n):
        row[k] = _falling_factorial(k, order) * (t ** (k - order))
    return row


def _smoothness_hessian(n, duration, order):
    """
    Exact Hessian of ``integral_0^T (d^order/dt^order poly)^2 dt`` in real time.

    Mirrors ``snap_H_block`` from utilities/kkt.py but without normalizing time, so
    it composes directly with the real-time alignment and constraint rows.
    """
    H = np.zeros((n, n))
    for i in range(order, n):
        for j in range(order, n):
            power = i + j - 2 * order + 1
            H[i, j] = (
                _falling_factorial(i, order)
                * _falling_factorial(j, order)
                * duration ** power
                / power
            )
    return H


def _evaluate_axis_derivative(coeffs, t, order):
    """Evaluate the ``order``-th derivative of a real-time polynomial at ``t``."""
    return float(np.dot(_derivative_row(len(coeffs), t, order), coeffs))


def sample_tangent_reference(segments, samples_per_segment, speed_eps):
    """
    Sample the path-tangent heading and a speed gate for every segment.

    Returns ``(times_per_segment, tangent_ref_per_segment, gate_per_segment)`` where
    the tangent reference is globally continuous (NaNs at near-hover samples are
    hold-filled, then the whole sequence is unwrapped). The gate is
    ``speed^2 / (speed^2 + speed_eps^2)`` so near-hover samples, where the tangent
    heading is meaningless, contribute almost nothing to the alignment cost.
    """
    speed_eps_sq = float(speed_eps) ** 2

    times_per_segment = []
    raw_yaw_per_segment = []
    gate_per_segment = []
    for segment in segments:
        duration = float(segment['T'])
        count = max(int(samples_per_segment), DEFAULT_COEFFICIENT_COUNT)
        times = np.linspace(0.0, duration, count)
        c_x = np.asarray(segment['c_x'], dtype=np.float64)
        c_y = np.asarray(segment['c_y'], dtype=np.float64)

        yaw_values = np.full(count, np.nan)
        gates = np.zeros(count)
        for index, t in enumerate(times):
            vx = _evaluate_axis_derivative(c_x, t, 1)
            vy = _evaluate_axis_derivative(c_y, t, 1)
            speed_sq = vx * vx + vy * vy
            gates[index] = speed_sq / (speed_sq + speed_eps_sq)
            if math.hypot(vx, vy) >= 1e-6:
                yaw_values[index] = math.atan2(vy, vx)

        times_per_segment.append(times)
        raw_yaw_per_segment.append(yaw_values)
        gate_per_segment.append(gates)

    _hold_fill_nans(raw_yaw_per_segment)

    # Unwrap globally so the continuous trace knows the true accumulated heading
    # (whether the path turns a little or loops all the way around).
    flat = np.concatenate(raw_yaw_per_segment) if raw_yaw_per_segment else np.array([])
    if flat.size and np.all(np.isnan(flat)):
        flat = np.zeros_like(flat)
    unwrapped = np.unwrap(flat)

    tangent_ref_per_segment = []
    offset = 0
    for times in times_per_segment:
        size = len(times)
        tangent_ref_per_segment.append(unwrapped[offset:offset + size])
        offset += size
    return times_per_segment, tangent_ref_per_segment, gate_per_segment


def _hold_fill_nans(raw_yaw_per_segment):
    """Replace near-hover NaN headings with the nearest finite neighbour (both ways)."""
    finite = [
        value
        for segment in raw_yaw_per_segment
        for value in segment
        if np.isfinite(value)
    ]
    if not finite:
        return  # whole path is near-hover; caller falls back to a flat trace

    previous = finite[0]
    for segment in raw_yaw_per_segment:
        for index in range(len(segment)):
            if np.isfinite(segment[index]):
                previous = segment[index]
            else:
                segment[index] = previous

    following = finite[-1]
    for segment in reversed(raw_yaw_per_segment):
        for index in reversed(range(len(segment))):
            if np.isfinite(segment[index]):
                following = segment[index]
            else:
                segment[index] = following


def resolve_boundary_pin(target_yaw, trace_endpoint):
    """
    Pick the ``+2*pi*n`` representative of ``target_yaw`` nearest the tangent trace.

    A quaternion only encodes yaw mod 2*pi, so a raw pin can force a spurious
    near-full rotation through the interior (e.g. +179 -> -179 sweeping -358 instead
    of +2). Choosing the representative nearest the continuous trace keeps the
    boundary's winding consistent with the actual path. When no explicit yaw is
    given, fall back to the trace endpoint (pure tangent following).
    """
    if target_yaw is None:
        return float(trace_endpoint)
    return float(trace_endpoint) + wrap_to_pi(float(target_yaw) - float(trace_endpoint))


def resolve_target_pin(target_yaw, start_pin, trace_start, trace_end):
    """
    Resolve target yaw relative to the chosen start pin and tangent trace winding.

    Resolving both endpoints independently against a near-hover tangent trace can turn
    ``+170 deg -> -170 deg`` into a long ``-340 deg`` slew. The target reference must
    instead inherit the tangent trace's accumulated change from the already chosen start
    pin; for hover this reduces to the shortest target representative relative to start.
    """
    reference = float(start_pin) + (float(trace_end) - float(trace_start))
    if target_yaw is None:
        return reference
    return reference + wrap_to_pi(float(target_yaw) - reference)


def fit_yaw_coefficients(
    segments,
    start_yaw=None,
    target_yaw=None,
    w_align=1.0,
    w_smooth=1e-2,
    w_snap=0.0,
    w_accel_limit=0.0,
    yaw_accel_max=None,
    yaw_rate_max=None,
    enforce_yaw_limits=False,
    coefficient_count=DEFAULT_COEFFICIENT_COUNT,
    samples_per_segment=16,
    limit_samples_per_segment=40,
    speed_eps=0.05,
    regularization=1e-9,
):
    """
    Solve the Concern B yaw stage and return per-segment real-time yaw coefficients.

    ``start_yaw`` / ``target_yaw`` are the boundary yaw values (radians) from the
    request orientation; pass ``None`` to pin to the path tangent at that end. The
    yaw rate, acceleration, and jerk are pinned to zero only at the first and last
    trajectory points. Interior segment joins only enforce C4 continuity.

    When ``enforce_yaw_limits`` is set and ``yaw_rate_max`` / ``yaw_accel_max`` are
    positive, the fit also enforces the hard caps ``|yaw'| <= yaw_rate_max`` and
    ``|yaw''| <= yaw_accel_max`` at ``limit_samples_per_segment`` points per segment.
    Those inequalities turn the equality-constrained least-squares into a convex QP
    that reuses the same cost and equality rows, so the C4 continuity and
    zero-rate/accel/jerk endpoint pins are preserved exactly. The QP is only invoked
    when the unconstrained fit actually exceeds a cap; otherwise the plain KKT
    solution is returned unchanged. (``yaw_accel_max`` still doubles as the reference
    for the soft ``w_accel_limit`` term regardless of this flag.)
    """
    if not segments:
        return []

    n = int(coefficient_count)
    n_seg = len(segments)
    w_align = max(0.0, float(w_align))
    w_smooth = max(0.0, float(w_smooth))
    w_snap = max(0.0, float(w_snap))
    w_accel_limit = max(0.0, float(w_accel_limit))
    yaw_accel_max = 0.0 if yaw_accel_max is None else float(yaw_accel_max)

    times_per_segment, tangent_ref_per_segment, gate_per_segment = sample_tangent_reference(
        segments, samples_per_segment, speed_eps
    )

    trace_start = tangent_ref_per_segment[0][0]
    trace_end = tangent_ref_per_segment[-1][-1]
    start_pin = resolve_boundary_pin(start_yaw, trace_start)
    target_pin = resolve_target_pin(target_yaw, start_pin, trace_start, trace_end)

    total_vars = n_seg * n

    # --- Cost: H (quadratic) and linear term f, both block-structured by segment. ---
    H = np.zeros((total_vars, total_vars))
    f = np.zeros(total_vars)
    for seg in range(n_seg):
        block = slice(seg * n, (seg + 1) * n)
        duration = float(segments[seg]['T'])

        H_block = np.zeros((n, n))
        if w_smooth > 0.0:
            H_block += w_smooth * _smoothness_hessian(n, duration, 2)
        if w_accel_limit > 0.0 and yaw_accel_max > 0.0:
            # Soft policy-limit cost. This remains quadratic/KKT-compatible, unlike
            # a true above-limit hinge or inequality cap.
            H_block += (
                w_accel_limit
                / (yaw_accel_max ** 2)
                * _smoothness_hessian(n, duration, 2)
            )
        if w_snap > 0.0:
            H_block += w_snap * _smoothness_hessian(n, duration, 4)

        if w_align > 0.0:
            times = times_per_segment[seg]
            tangent_ref = tangent_ref_per_segment[seg]
            gates = gate_per_segment[seg]
            for t, ref, gate in zip(times, tangent_ref, gates):
                weight = w_align * float(gate)
                if weight <= 0.0:
                    continue
                row = _derivative_row(n, float(t), 0)
                H_block += weight * np.outer(row, row)
                f[block] -= weight * float(ref) * row

        H[block, block] = H_block

    # Small Tikhonov term keeps the KKT system well-conditioned when alignment is
    # off or a segment is barely sampled; negligible against the real cost.
    H += regularization * np.eye(total_vars)

    # --- Equality constraints A c = b: boundary pins + C4 join continuity. ---
    rows = []
    rhs = []

    def add_row(builder, value):
        full = np.zeros(total_vars)
        for seg_index, seg_row in builder:
            full[seg_index * n:(seg_index + 1) * n] = seg_row
        rows.append(full)
        rhs.append(value)

    last = n_seg - 1
    end_time = float(segments[last]['T'])
    # Global first/last point: value from request/tangent, derivatives 1..3 zero.
    add_row([(0, _derivative_row(n, 0.0, 0))], start_pin)
    for order in range(1, CONTINUITY_ORDERS - 1):
        add_row([(0, _derivative_row(n, 0.0, order))], 0.0)
    add_row([(last, _derivative_row(n, end_time, 0))], target_pin)
    for order in range(1, CONTINUITY_ORDERS - 1):
        add_row([(last, _derivative_row(n, end_time, order))], 0.0)
    # Interior joins: continuity only (orders 0..4), no zero-derivative pins.
    for join in range(1, n_seg):
        left_time = float(segments[join - 1]['T'])
        for order in range(CONTINUITY_ORDERS):
            left = _derivative_row(n, left_time, order)
            right = _derivative_row(n, 0.0, order)
            add_row([(join - 1, left), (join, -right)], 0.0)

    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=np.float64)

    # --- Equality-constrained solve: one KKT linear solve.
    #     [[H, A^T], [A, 0]] [c; lambda] = [-f; b]. This is the limit-free optimum; it
    #     also warm-starts and falls back for the optional capped QP below.
    m = A.shape[0]
    kkt = np.block([[H, A.T], [A, np.zeros((m, m))]])
    kkt_rhs = np.concatenate([-f, b])
    coeffs = np.linalg.solve(kkt, kkt_rhs)[:total_vars]

    # --- Optional hard yaw rate/accel caps. Reuses H, f and the equality rows (A, b),
    #     so C4 continuity and the zero rate/accel/jerk endpoint pins stay exact.
    if enforce_yaw_limits:
        coeffs = _apply_yaw_rate_accel_caps(
            coeffs, H, f, A, b, segments, n, n_seg, total_vars,
            yaw_rate_max, yaw_accel_max, limit_samples_per_segment,
        )

    return [coeffs[seg * n:(seg + 1) * n].copy() for seg in range(n_seg)]


def _apply_yaw_rate_accel_caps(
    coeffs, H, f, A, b, segments, n, n_seg, total_vars,
    yaw_rate_max, yaw_accel_max, limit_samples_per_segment,
):
    """
    Enforce ``|yaw'| <= yaw_rate_max`` / ``|yaw''| <= yaw_accel_max`` on the fit.

    Builds two-sided inequality rows for the enabled caps at a dense per-segment
    grid and, only if the equality-only ``coeffs`` violates one, solves the convex QP
    ``min 1/2 c'Hc + f'c  s.t.  A c = b,  |cap rows| <= (1 - CAP_SAFETY) * cap``
    (IPOPT, warm-started from ``coeffs``). The small safety back-off keeps the peak
    *between* grid samples under the true limit so the downstream advisory check does
    not re-fire on discretization overshoot. Returns capped coefficients, or the input
    unchanged when no cap is active/violated or the QP does not satisfy the caps.
    """
    rate_max = 0.0 if yaw_rate_max is None else float(yaw_rate_max)
    accel_max = 0.0 if yaw_accel_max is None else float(yaw_accel_max)
    cap_rate = rate_max > 0.0
    cap_accel = accel_max > 0.0
    if not (cap_rate or cap_accel):
        return coeffs

    count = max(int(limit_samples_per_segment), DEFAULT_COEFFICIENT_COUNT)
    # (row, true-limit) pairs; the QP constrains to the backed-off limit, the accept
    # check uses the true limit.
    rows = []
    limit = []
    for seg in range(n_seg):
        duration = float(segments[seg]['T'])
        for t in np.linspace(0.0, duration, count):
            if cap_rate:
                row = np.zeros(total_vars)
                row[seg * n:(seg + 1) * n] = _derivative_row(n, float(t), 1)
                rows.append(row)
                limit.append(rate_max)
            if cap_accel:
                row = np.zeros(total_vars)
                row[seg * n:(seg + 1) * n] = _derivative_row(n, float(t), 2)
                rows.append(row)
                limit.append(accel_max)

    G = np.asarray(rows)
    limit = np.asarray(limit)

    # Fast path: the limit-free fit already respects every cap at the grid -> keep it
    # (no QP, no casadi import).
    values = G @ coeffs
    if np.all(np.abs(values) <= limit + 1e-9):
        return coeffs

    import casadi as ca  # local: only the capped path needs the QP backend

    # Back off the grid bound slightly so between-sample peaks stay under the true
    # limit; ridge H so the (only-PSD) cost is strictly convex for the solver.
    bound = limit * (1.0 - CAP_SAFETY_FRACTION)
    x = ca.SX.sym('c', total_vars)
    obj = 0.5 * ca.bilin(ca.DM(H + 1e-6 * np.eye(total_vars)), x, x) + ca.dot(ca.DM(f), x)
    g_all = np.vstack([A, G])
    lbg = np.concatenate([b, -bound])
    ubg = np.concatenate([b, bound])
    qp = {'x': x, 'f': obj, 'g': ca.mtimes(ca.DM(g_all), x)}
    opts = {'print_time': False, 'error_on_fail': False,
            'ipopt': {'print_level': 0, 'sb': 'yes'}}
    try:
        solver = ca.nlpsol('yaw_rate_accel_cap_qp', 'ipopt', qp, opts)
        sol = solver(x0=coeffs, lbg=lbg, ubg=ubg)
        capped = np.asarray(sol['x'].full()).flatten()
    except Exception as exc:  # noqa: B902 - fall back to the equality-only fit
        print(
            f'yaw_rate_accel_cap_qp failed ({exc!r}); enforce_yaw_limits requested a hard '
            'cap but falling back to the uncapped equality-only yaw fit.',
            flush=True,
        )
        return coeffs

    # Accept only if it actually satisfies the true caps at the grid; else fall back.
    if np.all(np.abs(G @ capped) <= limit + 1e-6):
        return capped
    print(
        'yaw_rate_accel_cap_qp solution still violates the true yaw rate/accel caps at '
        'the sample grid; enforce_yaw_limits requested a hard cap but falling back to '
        'the uncapped equality-only yaw fit.',
        flush=True,
    )
    return coeffs

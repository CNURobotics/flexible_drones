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

# Based on code generated in consultation with ChatGPT
import numpy as np
from numpy.polynomial import Polynomial as Poly
from math import comb  # returns the exact binomial coefficient “n choose k”:
# matplotlib is imported lazily in the __main__ demo so importing the KKT math
# (e.g. from kkt_planner or the action server) does not pull in plotting.


# Potential references from quick Google search
# https://apmonitor.com/me575/index.php/Main/KuhnTucker
# https://core-robotics.gatech.edu/2023/08/30/bootcamp-summer-2020-week-karush-kuhn-tucker-kkt-conditions/

# ============================================================
# KKT-based min-snap for 2 segments, degree-7, 1D
# End velocity v(T) is NOT constrained (it is implied by min-snap)
# Normalized solve: p=1, T=1  →  obtain shape & c_v
# ============================================================

# =========================
# Config
# =========================
DEG = 7
NCOEF = DEG + 1  # 8
EPS = 1e-12
REG = 1e-10  # tiny Tikhonov for numerical stability
# =========================
# Helpers for H (snap) and constraints
# =========================


def gram_monomial(n):
    """Gram matrix for monomials over u∈[0,1]: G_ij = ∫0^1 u^(i+j) du = 1/(i+j+1)."""
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = 1.0 / (i + j + 1)
    return G


def D4_matrix(n):
    """Matrix mapping coefficients to the 4th derivative polynomial in u: size n×n."""
    D4 = np.zeros((n, n))
    for k in range(4, n):
        D4[k - 4, k] = k * (k - 1) * (k - 2) * (k - 3)
    return D4


def snap_H_block(Tseg, n=NCOEF):
    """Exact Hessian block for ∫(x''''(t))^2 dt on one segment with duration Tseg."""
    G = gram_monomial(n)
    D4 = D4_matrix(n)
    Hu = D4.T @ G @ D4
    return Hu / (Tseg ** 7)


def basis_row_u(n, u, order):
    """
    Return monomial-basis row for the ``order``-th derivative at u in [0,1].

    No time scaling here; caller applies 1/T^order.
    """
    row = np.zeros(n)
    # derivative of u^k of order m is k*(k-1)*...*(k-m+1) * u^(k-m) for k>=m
    for k in range(order, n):
        coeff = 1.0
        for m in range(order):
            coeff *= (k - m)
        row[k] = coeff * (u ** (k - order))
    return row


def constraint_row(seg, order, u, Tseg, rhs, n=NCOEF):
    """
    Build a single constraint row for seg1 (seg=1) or seg2 (seg=2).

    Constraint: (d^order/dt^order) x_seg(u) = rhs.
    Time scaling 1/Tseg^order applied here.
    (d^order/dt^order) x_seg(u) = rhs  at local u (0 or 1), with time scaling 1/Tseg^order.
    """
    r = np.zeros(2 * n)
    br = basis_row_u(n, u, order) / (Tseg ** order)
    if seg == 1:
        r[:n] = br
    else:
        r[n:] = br
    return r, rhs


def build_constraints_normalized(rho, a_same_start=0.0, a_same_end=0.0):
    """
    Build A, b for the normalized acceleration problem (p=1, T=1).

    - Start jet on seg1 (u=0): x=0, v=0, a=a_same_start, j=0, s=0  (5 rows)
    - Continuity at the knot (orders 0..4): seg1(u=1) == seg2(u=0)  (5 rows)
    - End jet on seg2 (u=1): x=1, a=a_same_end, j=0, s=0  (NOTE: v(T) NOT constrained) (4 rows)
    Total 14 linear constraints.
    """
    assert 0 < rho < 1
    T1 = rho
    T2 = 1.0 - rho
    rows = []
    rhs = []

    # Start jet (x,v,a,j,s) at seg1, u=0
    start_targets = [0.0, 0.0, a_same_start, 0.0, 0.0]
    for k, target in enumerate(start_targets):
        r, b = constraint_row(seg=1, order=k, u=0.0, Tseg=T1, rhs=target)
        rows.append(r)
        rhs.append(b)

    # Continuity at knot: seg1(u=1) - seg2(u=0) == 0, for orders 0..4
    for k in range(5):
        r1, _ = constraint_row(seg=1, order=k, u=1.0, Tseg=T1, rhs=0.0)
        r2, _ = constraint_row(seg=2, order=k, u=0.0, Tseg=T2, rhs=0.0)
        r = r1 - r2
        rows.append(r)
        rhs.append(0.0)

    # End jet on seg2, u=1 (NO v constraint): x=1, a=a_same_end, j=0, s=0
    end_orders = [0, 2, 3, 4]
    end_targets = [1.0, a_same_end, 0.0, 0.0]
    for k, target in zip(end_orders, end_targets):
        r, b = constraint_row(seg=2, order=k, u=1.0, Tseg=T2, rhs=target)
        rows.append(r)
        rhs.append(b)

    A = np.vstack(rows)
    b = np.array(rhs, float)
    return A, b, T1, T2


def solve_min_snap_shape_KKT_normalized(rho, a_same_start=0.0, a_same_end=0.0, reg=0.0):
    """
    Solve the normalized min-snap shape via KKT.

    Minimizes (1/2) c^T H c subject to A c = b.
    Returns coeffs for seg1, seg2, and shape constants.
    """
    A, b, T1, T2 = build_constraints_normalized(rho, a_same_start, a_same_end)

    H1 = snap_H_block(T1)
    H2 = snap_H_block(T2)
    H = np.block([[H1, np.zeros_like(H1)],
                  [np.zeros_like(H2), H2]])
    if reg > 0:
        H = H + reg * np.eye(H.shape[0])

    KKT = np.block([[H, A.T],
                    [A, np.zeros((A.shape[0], A.shape[0]))]])
    rhs = np.concatenate([np.zeros(H.shape[0]), b])

    sol = np.linalg.solve(KKT, rhs)
    c = sol[:2 * NCOEF]
    c1 = c[:NCOEF]
    c2 = c[NCOEF:]

    # Shape constant: normalized endpoint velocity v_norm(1) = P2'(1)/T2
    P2 = Poly(c2)
    c_v = P2.deriv(1)(1.0) / T2

    # Optional: derivative envelopes in normalized time (helpful for sizing T later)
    # Build Poly for seg1 too:
    P1 = Poly(c1)
    # Evaluate max |a|, |j|, |snap| over t∈[0,1]
    N = 2001
    ts1 = np.linspace(0.0, T1, N // 2)
    ts2 = np.linspace(T1, 1.0, N - ts1.size)
    u1 = ts1 / T1
    u2 = (ts2 - T1) / T2
    a1 = P1.deriv(2)(u1) / (T1 ** 2)
    j1 = P1.deriv(3)(u1) / (T1 ** 3)
    s1 = P1.deriv(4)(u1) / (T1 ** 4)
    a2 = P2.deriv(2)(u2) / (T2 ** 2)
    j2 = P2.deriv(3)(u2) / (T2 ** 3)
    s2 = P2.deriv(4)(u2) / (T2 ** 4)
    kA = float(np.max(np.abs(np.concatenate([a1, a2]))))
    kJ = float(np.max(np.abs(np.concatenate([j1, j2]))))
    kS = float(np.max(np.abs(np.concatenate([s1, s2]))))

    # Exact cost value normalized
    J = float(c.T @ H @ c) * 0.5

    return {'c1': c1, 'c2': c2, 'T1': T1, 'T2': T2, 'rho': rho,
            'c_v': c_v, 'kA': kA, 'kJ': kJ, 'kS': kS, 'J': J}


def check_monotone_velocity(v):
    return float(np.min(v))


def t_poly_coeffs_from_normalized(
    c1_norm, c2_norm, rho,
    T_real,
    v1=None, p1=None, c_v=None,
    x0=0.0
):
    """
    Convert normalized shape (p=1, T=1) into real-time t-polynomial coefficients.

    Inputs
    ------
    c1_norm, c2_norm : arrays length 8
        Normalized position coeffs for seg1/seg2 in u ∈ [0,1] (monomial basis, ascending powers).
    rho : float in (0,1)
        Split T1/T.
    T_real : float > 0
        Total time.
    v1 : float, optional
        Desired terminal velocity in real units. Provide either (v1 and c_v) OR p1.
    p1 : float, optional
        Desired total displacement. If given, it is used directly.
    c_v : float, optional
        Normalized endpoint velocity c_v = v_norm(1) = P2'(1)/(1-rho).
        Required if v1 is given and p1 is None.
    x0 : float, optional
        Starting position offset to add to both segments' polynomials.

    Returns
    -------
    dict with:
      - 'T1', 'T2', 'p1'
      - 'seg1_t'          : np.array shape (8,) for x1(t) on [0, T1], ascending powers
      - 'seg2_t_shifted'  : np.array shape (8,) for x2(t) as sum b_k (t - T1)^k
      - 'seg2_t_global'   : np.array shape (8,) for x2(t) expanded as sum a_m t^m
      #- 'evaluators'      : callables (Poly) for convenience

    """
    assert 0.0 < rho < 1.0, 'rho must be in (0,1)'
    T1 = rho * T_real
    T2 = (1.0 - rho) * T_real

    # Determine p1
    if p1 is None:
        if (v1 is None) or (c_v is None):
            raise ValueError('Provide either p1, or (v1 and c_v).')
        p1 = (v1 * T_real) / c_v

    c1 = np.asarray(c1_norm, dtype=float)
    c2 = np.asarray(c2_norm, dtype=float)

    # Segment 1: x1(t) = x0 + sum_k (p1 * c1_k / T1^k) * t^k
    seg1_t = np.zeros_like(c1)
    pow_T1 = 1.0
    for k in range(len(c1)):
        if k > 0:
            pow_T1 *= T1
        seg1_t[k] = p1 * c1[k] / (pow_T1)

    # Segment 2 (shifted): x2(t) = x0 + sum_k (p1 * c2_k / T2^k) * (t - T1)^k
    seg2_t_shifted = np.zeros_like(c2)
    pow_T2 = 1.0
    for k in range(len(c2)):
        if k > 0:
            pow_T2 *= T2
        seg2_t_shifted[k] = p1 * c2[k] / (pow_T2)

    # Add x0 to the constant terms
    seg1_t = seg1_t.copy()
    seg2_t_shifted = seg2_t_shifted.copy()
    seg1_t[0] += x0
    seg2_t_shifted[0] += x0

    return {
        'T1': T1, 'T2': T2, 'p1': p1,
        'seg1_t': seg1_t,
        'seg2_t_shifted': seg2_t_shifted,
    }


def eval_time_histories_normalized(c1, c2, T1, T2, N=2000, t=None):
    """Evaluate x, v, a, j, snap over global time [0, T_total]."""
    T_total = T1 + T2
    print(f'  times = {T1} + {T2} = {T_total} / {N}')
    print(f'c1={c1}')
    print(f'c2={c2}')

    if t is None:
        t = np.linspace(0, T_total, N)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    a = np.zeros_like(t)
    j = np.zeros_like(t)
    s = np.zeros_like(t)
    P1 = Poly(c1)
    P2 = Poly(c2)
    for i, ti in enumerate(t):
        if ti <= T1:
            u = ti / T1
            x[i] = P1(u)
            v[i] = P1.deriv(1)(u) / T1
            a[i] = P1.deriv(2)(u) / (T1 ** 2)
            j[i] = P1.deriv(3)(u) / (T1 ** 3)
            s[i] = P1.deriv(4)(u) / (T1 ** 4)
        else:
            tau = (ti - T1) / T2
            x[i] = P2(tau)
            v[i] = P2.deriv(1)(tau) / T2
            a[i] = P2.deriv(2)(tau) / (T2 ** 2)
            j[i] = P2.deriv(3)(tau) / (T2 ** 3)
            s[i] = P2.deriv(4)(tau) / (T2 ** 4)
    return t, x, v, a, j, s


def eval_time_histories(c1, c2, T1, T2, N=2000):
    """
    Evaluate piecewise polynomial time histories.

    c1: coeffs a_k for x1(t) = sum a_k t^k on [0, T1].
    c2: coeffs b_k for x2(t) = sum b_k (t - T1)^k on [T1, T1+T2].
    """
    P1_t = Poly(c1)
    T_total = T1

    if c2 is not None and T2 is not None:
        P2_tau = Poly(c2)  # evaluate at tau = t - T1
        T_total += T2

    t = np.linspace(0.0, T_total, N)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    a = np.zeros_like(t)
    j = np.zeros_like(t)
    s = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti <= T1:
            x[i] = P1_t(ti)
            v[i] = P1_t.deriv(1)(ti)
            a[i] = P1_t.deriv(2)(ti)
            j[i] = P1_t.deriv(3)(ti)
            s[i] = P1_t.deriv(4)(ti)
        else:
            tau = ti - T1
            x[i] = P2_tau(tau)
            v[i] = P2_tau.deriv(1)(tau)
            a[i] = P2_tau.deriv(2)(tau)
            j[i] = P2_tau.deriv(3)(tau)
            s[i] = P2_tau.deriv(4)(tau)

    return t, x, v, a, j, s


def reverse_local_poly(a, T, C):
    """
    Return the time-reversed polynomial R(t) = C - P(T - t).

    Given local poly P(t) = sum_k a[k] t^k on t in [0, T],
    returns coefficients r such that R(t) = sum_m r[m] t^m.
    """
    a = np.asarray(a, dtype=float)
    deg = len(a) - 1
    r = np.zeros_like(a)
    # m=0 (constant term) gets the C offset minus P(T)
    PT = sum(a[k] * (T**k) for k in range(deg + 1))
    r[0] = C - PT
    # m>=1:
    for m in range(1, deg + 1):
        s = 0.0
        for k in range(m, deg + 1):
            s += a[k] * comb(k, m) * (T**(k - m)) * ((-1)**m)
        r[m] = -s
    return r


def build_decel_to_stop_from_shape(c1_norm, c2_norm, rho, T_real, p1, v1, x0=0.0):
    """
    Build a deceleration-to-stop trajectory by time-reversing the accelerating shape.

    Uses the accelerating shape (0 to free v) and reverses it to get v0 to 0.
    Steps: choose p1 so accel endpoint speed equals v0, build accel t-polys,
    then time-reverse the piecewise polynomials.
    Returns dict with decel seg1/seg2 coeffs (local-time forms) and T1, T2, p1.
    """
    assert 0.0 < rho < 1.0
    T1 = rho * T_real
    T2 = (1.0 - rho) * T_real

    # Build accelerating t-polys with your existing helper:
    tp_accel = t_poly_coeffs_from_normalized(
        c1_norm=c1_norm, c2_norm=c2_norm, rho=rho,
        T_real=T_real,
        v1=v1,
        p1=p1,
        x0=x0
    )
    # Local polynomials:
    seg1_accel = tp_accel['seg1_t']             # on [0, T1]
    seg2_accel_shift = tp_accel['seg2_t_shifted']  # as (t - T1)^k on [T1, T1+T2]

    # We want local-in-time forms for each segment interval starting at 0.
    # seg2_accel_shift is already local in tau = t - T1 ∈ [0, T2].
    # seg1_accel is local on [0, T1].

    # End position:
    x_end = x0 + p1

    # Decel segment 1 (duration T2) = x_end - seg2_accel(T2 - t)
    seg1_decel = reverse_local_poly(seg2_accel_shift, T2, C=x_end)

    # Decel segment 2 (duration T1) = x_end - seg1_accel(T1 - t)
    seg2_decel = reverse_local_poly(seg1_accel, T1, C=x_end)

    return {
        'T1': T2,                 # note: decel first segment has duration T2
        'T2': T1,                 # and second has duration T1 (swapped order)
        'p1': p1,
        'seg1_t': seg1_decel,     # decel segment 1: local poly on [0, T1]
        'seg2_t_shifted': seg2_decel,  # decel segment 2: local poly on [T2, T1+T2] (to be used after T2)
        'x0': x0,
        'x_end': x_end
    }


# ================================================================
# Corner blending
def build_inside_corner_constraints_axis(P0c, P2c, v0c, v2c, T1, T2, n=NCOEF):
    """
    Build constraints for an inside-the-corner blend on one axis.

    - Start jet at seg1 u=0: x=P0c, v=v0c, a=j=s=0.
    - Knot: C^4 continuity seg1(u=1) == seg2(u=0) for orders 0..4.
    - End jet at seg2 u=1: x=P2c, v=v2c, a=j=s=0.
    NOTE: No row pins the knot position to P1c -- the optimizer cuts inside.
    """
    rows = []
    rhs = []

    # Start jet (seg1, u=0)
    start_targets = [P0c, v0c, 0.0, 0.0, 0.0]
    for k, target in enumerate(start_targets):
        r, b = constraint_row(seg=1, order=k, u=0.0, Tseg=T1, rhs=target, n=n)
        rows.append(r)
        rhs.append(b)

    # Knot continuity (orders 0..4)
    for k in range(5):
        r1, _ = constraint_row(seg=1, order=k, u=1.0, Tseg=T1, rhs=0.0, n=n)
        r2, _ = constraint_row(seg=2, order=k, u=0.0, Tseg=T2, rhs=0.0, n=n)
        rows.append(r1 - r2)
        rhs.append(0.0)

    # End jet (seg2, u=1)
    end_targets = [P2c, v2c, 0.0, 0.0, 0.0]
    for k, target in enumerate(end_targets):
        r, b = constraint_row(seg=2, order=k, u=1.0, Tseg=T2, rhs=target, n=n)
        rows.append(r)
        rhs.append(b)

    A = np.vstack(rows)
    b = np.array(rhs, float)
    return A, b


def solve_axis_inside_corner_KKT(P0c, P2c, v0c, v2c, T1, T2, reg=REG):
    """
    Solve min-snap KKT for one axis with an inside-corner blend.

    Minimizes (1/2) c^T H c subject to A c = b for two segments.
    Returns coeffs c1 (seg1) and c2 (seg2), and cost.
    """
    n = NCOEF
    H1 = snap_H_block(T1, n)
    H2 = snap_H_block(T2, n)
    H = np.block([[H1, np.zeros_like(H1)],
                  [np.zeros_like(H2), H2]])

    if reg > 0.0:
        H = H + reg * np.eye(2 * n)

    A, b = build_inside_corner_constraints_axis(P0c, P2c, v0c, v2c, T1, T2, n=n)

    # KKT solve
    KKT = np.block([[H, A.T],
                    [A, np.zeros((A.shape[0], A.shape[0]))]])
    rhs = np.concatenate([np.zeros(2 * n), b])

    sol = np.linalg.solve(KKT, rhs)
    c = sol[:2 * n]
    c1 = c[:n]
    c2 = c[n:]
    cost = 0.5 * c @ (H @ c)

    # Segment 1: x1(t) = x0 + sum_k (p1 * c1_k / T1^k) * t^k
    seg1_t = np.zeros_like(c1)
    pow_T1 = 1.0
    for k in range(len(c1)):
        if k > 0:
            pow_T1 *= T1
        seg1_t[k] = c1[k] / (pow_T1)

    # Segment 2 (shifted): x2(t) = x0 + sum_k (p1 * c2_k / T2^k) * (t - T1)^k
    seg2_t = np.zeros_like(c2)
    pow_T2 = 1.0
    for k in range(len(c2)):
        if k > 0:
            pow_T2 *= T2
        seg2_t[k] = c2[k] / (pow_T2)

    return seg1_t, seg2_t, float(cost)


def unit(v):
    n = np.linalg.norm(v)
    return v / (n if n > EPS else 1.0)


def points_from_corner(P1, u_in, u_out, s_in, s_out):
    """
    Build P01 and P12 from corner P1 by stepping along inbound/outbound rays.

    u_in and u_out must be unit-length direction vectors.
    s_in and s_out are positive distances in meters.
    """
    P01 = P1 - s_in * u_in   # back along inbound ray toward where we came from
    P12 = P1 + s_out * u_out  # forward along outbound ray
    return P01, P12


def endpoint_velocities_from_rays(u_in, u_out, speed_in, speed_out):
    """
    Make endpoint velocities aligned with the trajectory rays.

    v0 is aligned with +u_in toward the knot (increasing position along seg1).
    v2 is aligned with +u_out away from the knot.
    """
    v0 = speed_in * u_in
    v2 = speed_out * u_out
    return v0, v2


def solve_corner_with_alpha(P0, P1, P2,
                            T1, T2,
                            speed_in, speed_out,
                            alpha=0.5, reg=REG):
    """
    Wrap the corner-solver when durations (T1, T2) and aggressiveness alpha are fixed.

    Uses: s_in = v_in*T1/alpha,  s_out = v_out*T2/alpha

    Returns: dictionary of
      P01, P12,  - points in and out of corner
      v0, v2,    - velocity vectors
      cx1, cy1,  - x, y coeffecients for first segment
      cx2, cy2,  - x, y coeffecients for second segment
      Jx, Jy     - snap cost
    """
    if alpha <= 0:
        raise ValueError('alpha must be positive')
    speed_in = float(speed_in)
    speed_out = float(speed_out)

    # Compute distances along the corner rays
    s_in = (speed_in * T1) / alpha
    s_out = (speed_out * T2) / alpha

    # Normalize directions and build endpoints/endpoint velocities
    u_in = unit(P1 - P0)
    u_out = unit(P2 - P1)
    P01, P12 = points_from_corner(P1, u_in, u_out, s_in, s_out)
    v0, v2 = endpoint_velocities_from_rays(u_in, u_out, speed_in, speed_out)

    # Solve per axis
    cx1, cx2, Jx = solve_axis_inside_corner_KKT(P01[0], P12[0], v0[0], v2[0], T1, T2, reg=reg)
    cy1, cy2, Jy = solve_axis_inside_corner_KKT(P01[1], P12[1], v0[1], v2[1], T1, T2, reg=reg)

    return {'p_in': P01,
            'p_out': P12,
            'v_in': v0,
            'v_out': v2,
            'seg1_t': (cx1, cy1),
            'seg2_t': (cx2, cy2),
            'J': Jx + Jy}


# ============================================================
# Demo / Usage
# ============================================================
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    show_normalized = False
    show_accel = False
    show_decel = False
    show_corner = True
    rho = 0.4
    res = solve_min_snap_shape_KKT_normalized(rho, a_same_start=0.0, a_same_end=0.0, reg=0.0)

    c1, c2 = res['c1'], res['c2']
    T1, T2 = res['T1'], res['T2']
    c_v, kA, kJ, kS, J = res['c_v'], res['kA'], res['kJ'], res['kS'], res['J']
    print(f'[Normalized shape] rho={rho:.3f}, T1={T1:.3f}, T2={T2:.3f}')
    print(f'  c_v = v_norm(1) = {c_v:.6g}')
    print(f'  kA={kA:.6g}, kJ={kJ:.6g}, kS={kS:.6g}, J={J:.6g}')

    # --- 1) Plot normalized time histories (T=1) and check monotonicity ---
    t, x, v, a, j, s = eval_time_histories_normalized(c1, c2, rho, 1 - rho, N=2000)
    vmin = check_monotone_velocity(v)
    print(f'min v over [0,1] = {vmin:.6g}')

    mask1 = t <= T1
    mask2 = ~mask1

    if show_normalized:
        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], x[mask1], label='x seg1')
        plt.plot(t[mask2], x[mask2], label='x seg2')
        plt.xlabel('t (normalized)')
        plt.ylabel('position')
        plt.legend()
        plt.title('Position (normalized)')

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], v[mask1], label='v seg1')
        plt.plot(t[mask2], v[mask2], label='v seg2')
        plt.xlabel('t (normalized)')
        plt.ylabel('velocity')
        plt.legend()
        plt.title('Velocity (normalized)')

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], a[mask1], label='a seg1')
        plt.plot(t[mask2], a[mask2], label='a seg2')
        plt.xlabel('t (normalized)')
        plt.ylabel('acceleration')
        plt.legend()
        plt.title('Acceleration (normalized)')

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], j[mask1], label='j seg1')
        plt.plot(t[mask2], j[mask2], label='j seg2')
        plt.xlabel('t (normalized)')
        plt.ylabel('jerk')
        plt.legend()
        plt.title('Jerk (normalized)')

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], s[mask1], label='snap seg1')
        plt.plot(t[mask2], s[mask2], label='snap seg2')
        plt.xlabel('t (normalized)')
        plt.ylabel('snap')
        plt.legend()
        plt.title('Snap (normalized)')

        plt.tight_layout()

    # --- 2) How to scale: given (v1, T_real) compute p1; or given (p1, T_real) compute v1 ---
    v1_desired = 0.2   # example terminal velocity (real units)
    T_real = 0.8   # example total time (real units)
    p1_needed = v1_desired * T_real / c_v
    print(f'Given v1={v1_desired}, T={T_real} → p1 should be {p1_needed:.6g}')

    tp = t_poly_coeffs_from_normalized(
        c1_norm=c1, c2_norm=c2,
        rho=rho,
        T_real=T_real,
        p1=p1_needed,   # or use v1=v1_desired, c_v=c_v
        x0=0.0
    )

    print('T1,T2:', tp['T1'], tp['T2'])
    print('seg1_t coeffs (ascending powers):', tp['seg1_t'])
    print('seg2_t (t-T1) coeffs:', tp['seg2_t_shifted'])
    # print("seg2_t global coeffs:", tp['seg2_t_global'])

    T1 = tp['T1']
    T2 = tp['T2']
    t, x, v, a, j, s = eval_time_histories(
        c1=tp['seg1_t'],
        c2=tp['seg2_t_shifted'],
        T1=T1,
        T2=T2,
        N=2000
    )

    vmin = check_monotone_velocity(v)
    print(f'min v over [0,1] = {vmin:.6g}')

    mask1 = t <= T1
    mask2 = ~mask1

    if show_accel:
        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], x[mask1], label='x seg1')
        plt.plot(t[mask2], x[mask2], label='x seg2')
        plt.xlabel('t')
        plt.ylabel('position')
        plt.legend()
        plt.title('Position')

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], v[mask1], label='v seg1')
        plt.plot(t[mask2], v[mask2], label='v seg2')
        plt.xlabel('t')
        plt.ylabel('velocity')
        plt.legend()
        plt.title('Velocity')

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], a[mask1], label='a seg1')
        plt.plot(t[mask2], a[mask2], label='a seg2')
        plt.xlabel('t')
        plt.ylabel('acceleration')
        plt.legend()
        plt.title('Acceleration')

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], j[mask1], label='j seg1')
        plt.plot(t[mask2], j[mask2], label='j seg2')
        plt.xlabel('t')
        plt.ylabel('jerk')
        plt.legend()
        plt.title('Jerk')

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], s[mask1], label='snap seg1')
        plt.plot(t[mask2], s[mask2], label='snap seg2')
        plt.xlabel('t')
        plt.ylabel('snap')
        plt.legend()
        plt.title('Snap')

        plt.tight_layout()

    # Now do a deceleration
    T_real = 0.8
    v0 = 0.2  # starting speed; end speed will be 0
    p1_decel = v0 * T_real / c_v  # displacement implied by the accel shape

    decel = build_decel_to_stop_from_shape(
        c1_norm=c1, c2_norm=c2, rho=rho,
        T_real=T_real, p1=p1_decel, v1=v0, x0=0.0
    )

    # Evaluate and plot:
    t, x, v, a, j, s = eval_time_histories(
        c1=decel['seg1_t'],
        c2=decel['seg2_t_shifted'],
        T1=decel['T1'],
        T2=decel['T2'],
        N=2000
    )

    mask1 = t <= T1
    mask2 = ~mask1

    if show_decel:
        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], x[mask1], label='x seg1')
        plt.plot(t[mask2], x[mask2], label='x seg2')
        plt.xlabel('t')
        plt.ylabel('position')
        plt.legend()
        plt.title('Position Decel')

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], v[mask1], label='v seg1')
        plt.plot(t[mask2], v[mask2], label='v seg2')
        plt.xlabel('t')
        plt.ylabel('velocity')
        plt.legend()
        plt.title('Velocity  Decel')

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], a[mask1], label='a seg1')
        plt.plot(t[mask2], a[mask2], label='a seg2')
        plt.xlabel('t')
        plt.ylabel('acceleration')
        plt.legend()
        plt.title('Deceleration')

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], j[mask1], label='j seg1')
        plt.plot(t[mask2], j[mask2], label='j seg2')
        plt.xlabel('t')
        plt.ylabel('jerk')
        plt.legend()
        plt.title('Jerk Decel')

        plt.figure(figsize=(10, 6))
        plt.plot(t[mask1], s[mask1], label='snap seg1')
        plt.plot(t[mask2], s[mask2], label='snap seg2')
        plt.xlabel('t')
        plt.ylabel('snap')
        plt.legend()
        plt.title('Snap Decel')

        plt.tight_layout()

    # Now do a corner demo
    P0 = np.array((-0.05, 0.75))
    P1 = np.array((0.00, 0.75))  # used only to size T1/T2
    P2 = np.array((0.026919432271293273, 0.7921348331691399))
    v01 = unit(P1 - P0)
    v12 = unit(P2 - P1)

    alpha = 1.2
    T1 = 0.3
    T2 = 0.3
    speed = 0.22
    P0 = P1 - 2 * T1 * v01 * speed
    P2 = P1 + 2 * T2 * v12 * speed

    res = solve_corner_with_alpha(P0, P1, P2, T1, T2,
                                  speed, speed,
                                  alpha=alpha, reg=REG)
    P01 = res['p_in']
    P12 = res['p_out']

    print(f'corner={res}')
    tx, x, vx, ax, jx, sx = eval_time_histories(
        c1=res['seg1_t'][0],
        c2=res['seg2_t'][0],
        T1=T1,
        T2=T2,
        N=2000
    )
    ty, y, vy, ay, jy, sy = eval_time_histories(
        c1=res['seg1_t'][1],
        c2=res['seg2_t'][1],
        T1=T1,
        T2=T2,
        N=2000
    )

    if show_corner:
        mask1 = tx <= T1
        mask2 = ~mask1
        vx1 = vx[mask1]
        vx2 = vx[mask2]
        ax1 = ax[mask1]
        ax2 = ax[mask2]
        jx1 = jx[mask1]
        jx2 = jx[mask2]
        sx1 = sx[mask1]
        sx2 = sx[mask2]
        vy1 = vy[mask1]
        vy2 = vy[mask2]
        ay1 = ay[mask1]
        ay2 = ay[mask2]
        jy1 = jy[mask1]
        jy2 = jy[mask2]
        sy1 = sy[mask1]
        sy2 = sy[mask2]

        # =========================
        # Figure 1: geometry + x(t), y(t)
        # =========================
        fig1 = plt.figure(figsize=(9, 8))
        gs1 = fig1.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 1.0])
        ax_geom = fig1.add_subplot(gs1[0])
        ax_xt = fig1.add_subplot(gs1[1])
        ax_yt = fig1.add_subplot(gs1[2])
        plt.subplots_adjust(hspace=0.32)

        ax_geom.plot([P0[0], P1[0], P2[0]], [P0[1], P1[1], P2[1]], 's:', ms=5, label='P0,P1,P2 (P1 not enforced)')
        l1, = ax_geom.plot(x[mask1], y[mask1], '-', lw=2, label='Seg1')
        l2, = ax_geom.plot(x[mask2], y[mask2], '-', lw=2, label='Seg2')
        ax_geom.plot([P01[0], P1[0], P12[0]], [P01[1], P1[1], P12[1]], 'o', ms=5, label='P01,P1,P12')
        ax_geom.axis('equal')
        ax_geom.grid(True, ls=':', lw=0.7)
        ax_geom.legend(loc='best')
        ax_geom.set_title('Two-segment min-snap (C⁴ blend at knot, NO pass-through P1) via H + KKT')

        c1 = l1.get_color()
        c2 = l2.get_color()

        ax_xt.plot(tx[mask1], x[mask1], '-', color=c1, lw=1.8, label='Seg1 x(t)')
        ax_xt.plot(tx[mask2], x[mask2], '-', color=c2, lw=1.8, label='Seg2 x(t)')
        ax_yt.plot(ty[mask1], y[mask1], '-', color=c1, lw=1.8, label='Seg1 y(t)')
        ax_yt.plot(ty[mask2], y[mask2], '-', color=c2, lw=1.8, label='Seg2 y(t)')
        ax_xt.set_title('x(t)')
        ax_yt.set_title('y(t)')
        ax_xt.grid(True, ls=':', lw=0.6)
        ax_yt.grid(True, ls=':', lw=0.6)
        ax_xt.legend(loc='best')
        ax_yt.legend(loc='best')

        # =========================
        # Figure 2: velocity, acceleration, jerk, snap components (by segment color)
        # =========================
        fig2 = plt.figure(figsize=(11, 10))
        gs2 = fig2.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

        axes = np.empty((4, 2), dtype=object)
        titles = [['vx(t)', 'vy(t)'], ['ax(t)', 'ay(t)'], ['jx(t)', 'jy(t)'], ['sx(t)', 'sy(t)']]
        data1 = [[vx1, vy1], [ax1, ay1], [jx1, jy1], [sx1, sy1]]
        data2 = [[vx2, vy2], [ax2, ay2], [jx2, jy2], [sx2, sy2]]

        for i in range(4):
            for j in range(2):
                ax = fig2.add_subplot(gs2[i, j])
                axes[i, j] = ax
                ax.plot(tx[mask1], data1[i][j], '-', color=c1, lw=1.5, label='Seg1')
                ax.plot(tx[mask2], data2[i][j], '-', color=c2, lw=1.5, label='Seg2')
                ax.set_title(titles[i][j])
                ax.grid(True, ls=':', lw=0.6)
                if i == 0 and j == 0:
                    ax.legend(loc='best')

        fig2.suptitle('Time histories (components), color-coded by segment — C⁴ blend, inside corner', y=0.99)

    plt.show()

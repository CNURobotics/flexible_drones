# Trajectory planner design notes

Theory and rationale behind the CasADi/IPOPT planner
([`casadi_obstacle_planner.py`](casadi_obstacle_planner.py)) and the staged anytime
pipeline that runs it ([`optimization.py`](optimization.py),
[`validation.py`](validation.py)). Code comments point back here with
`See README.md (<section>)`.

## Module layout

The action server ([`../plan_trajectory_action_server.py`](../plan_trajectory_action_server.py))
is a thin ROS node; the planning logic lives in this package:

| Module | Responsibility |
|--------|----------------|
| `casadi_obstacle_planner.py` | The NLP planner (objective, constraints, solve). |
| `seeding.py` | Initial-trajectory seeding strategies (growable). |
| `optimization.py` | Planner registry + the staged anytime pipeline run in one spawned subprocess (queue contract, `best_valid` retention). |
| `obstacles.py` | Bounds + obstacle handling: capsule geometry and the `ObstacleManager`. |
| `validation.py` | Dense post-solve gates: boundary, kinematic-limit, position-bound, obstacle-clearance, continuity (shippable vs. looser seed tolerance profiles). |
| `geometry.py` | Shared numpy transform / capsule-distance helpers. |
| `errors.py` | Shared planning exceptions. |

The planner solves a single non-linear program (NLP) for a start→goal trajectory
made of `n_segments` 7th-order polynomial pieces (8 coefficients per axis per
segment), parameterized on a normalized `s ∈ [0, 1]` per segment with physical
time `t = T·s` and a free segment duration `T`. Each derivative w.r.t. real time
picks up a factor of `1/T`: velocity `= (1/T)·dx/ds`, acceleration `= (1/T²)·d²x/ds²`,
and so on. That `1/Tᵏ` factor drives most of the design choices below.

## The objective

```
f = w_time·(ΣT / T_ref)                      # minimum time
  + 0.25·((T - T_prev)/T_ref)²               # segment-time smoothing
  + Σ w_d·J_d   for d in {acc, jerk, snap}   # normalized control effort
  + w_yaw_acc·J_yaw                          # path-tangent yaw-acceleration cost
  + boundary_bias                            # soft keep-off-the-walls
  + obstacle_penalty                         # soft avoidance (margin-band hinge)
  + progress_costs                           # optional backtrack / behind-start (off by default)
```

with kinematic limits, continuity, and the flight-volume box imposed as **hard
constraints** — obstacle avoidance is the exception, a soft penalty (see *Obstacles
and boundaries*).

The **yaw term** (`casadi_yaw_acceleration_terms`, `w_yaw_acc`) penalizes the
acceleration (2nd derivative) of the **path-tangent** heading `yaw = atan2(vy, vx)`
implied by the x/y motion. Yaw is not a decision variable in this planner — output yaw
is fit to the tangent afterward (or zeroed) — so this is the weaker statement "minimize
the 2nd derivative of the heading the path already implies," not an independent yaw flat
output as in full Mellinger handling. The cost works directly from x/y derivatives, with
a smooth speed gate so near-hover samples (where tangent yaw is ill-defined) don't
dominate. It discourages snappy heading reversals the same way the effort terms
discourage translational ones.

The **output yaw fit** is a separate solve after x/y/z planning
([`yaw_planning.py`](yaw_planning.py)). The base fit is a single linear
equality-constrained least-squares (KKT) solve that pins the first/last yaw value with
zero rate/accel/jerk at the endpoints and C4 continuity at interior joins, with a soft
`w_yaw_accel_limit` policy cost `integral((yaw'' / yaw_accel_max)^2) dt`. When the
caller sets `enforce_yaw_limits`, the fit *also* imposes **hard** `|yaw'| ≤ yaw_rate_max`
and `|yaw''| ≤ yaw_accel_max` at a dense per-segment grid. Those inequalities turn the
stage into a small convex QP (IPOPT, warm-started from the least-squares solution) that
reuses the same cost and equality rows — so the C4 continuity and zero-endpoint-derivative
pins are preserved exactly. The QP runs only when the unconstrained fit actually exceeds a
cap; otherwise the plain KKT solution is returned. A small back-off on the grid bound keeps
the peak *between* samples under the true limit so the dense advisory check
(`validation.find_yaw_limit_warnings`) does not re-fire on discretization overshoot.

This is Tier 1 of yaw handling: it makes the *output* yaw respect its own rate/accel
limits without touching x/y/z. It cannot fix a path whose *tangent* heading is itself
infeasible (too sharp a turn for the yaw limits in the time available) — that coupling
lives in the x/y solve via `w_yaw_acc` and the tangent-yaw-rate cap.

The optional **progress costs** (`backtrack_objective_weight`,
`behind_start_objective_weight`) penalize moving backward along, or behind, the
start→goal direction. Both default to `0` — see the brake-and-reverse note for why a
positional anti-backtrack cost is off by default.

## Differential flatness: why these derivatives

Why penalize *these* position derivatives at all? Because (assuming the usual quadrotor
flatness model) position `(x, y, z)` and yaw are **flat outputs**: every state and motor
input can be written as a function of those outputs and their derivatives. Each higher
position derivative maps to a deeper level of the control stack:

| Derivative | Position term | Physical / control meaning |
|---|---|---|
| 2nd | acceleration | thrust magnitude + desired roll/pitch attitude |
| 3rd | jerk | body angular rates |
| 4th | snap | angular acceleration → body torques → differential motor commands |

So **snap is not "attitude acceleration"** — it sits one level deeper: minimizing snap
minimizes the *rate of change of the torque/thrust inputs the motors must produce*,
keeping commands smooth, bounded, and unsaturated, which is what makes a trajectory
physically trackable. Snap is the canonical Mellinger & Kumar cost because it is the
**lowest** derivative that directly penalizes the motor inputs (torque ~ angular
acceleration ~ snap) — low enough to shape the inputs, not so high it over-constrains.
Yaw enters the input map at a lower derivative order than the lateral outputs, which is
why it is handled separately at its **2nd** derivative (above).

This is the physical justification for the `acc/jerk/snap` ladder; the next two
sections cover *how much* to weight them — the brake-and-reverse symptom that motivates
a nonzero `w_acc`, and the normalization that makes the weights comparable.

## Why control-effort costs exist (the brake-and-reverse problem)

A pure minimum-time objective is **bang-bang in acceleration**: the optimum
saturates the acceleration limits. With nothing rewarding *lower* acceleration,
the solver is happy to brake hard and reverse around an obstacle — a maneuver
that maximizes acceleration — instead of a gentle sweeping arc, because the arc
covers more distance and the `w_time` term penalizes that.

The remedy is an explicit **control-effort cost** `∫‖dᵏx/dtᵏ‖² dt` that trades a
little time for smoothness. Acceleration (`d²/dt²`) is the *right* derivative to
penalize for this symptom, because reversing is a large-acceleration event; snap
(`d⁴/dt⁴`) barely sees a smooth-but-hard reverse. Jerk is a useful companion for
damping the abruptness of transitions. See `casadi_effort_terms`.

A purely positional "don't move backward" cost (`backtrack_objective_weight`)
was tried and is **off by default**: it also penalizes the legitimate lateral /
backward motion a real detour needs, so it fights the very sweep we want. The
effort cost permits the detour while penalizing the jerky reverse.

## Weight normalization (the most important part)

Each effort term integrates a different time-derivative, so the *raw* terms carry
different units and different powers of `T`:

| Term            | Raw form                | T-scaling | Units   |
|-----------------|-------------------------|-----------|---------|
| time            | `w_time·T`              | `T¹`      | s       |
| acceleration    | `w_acc·‖a‖²·(T·ds)`     | `~1/T³`   | m²/s³   |
| jerk            | `w_jerk·‖j‖²·(T·ds)`    | `~1/T⁵`   | m²/s⁵   |
| snap            | `w_snap·‖snap‖²·(T·ds)` | `~1/T⁷`   | m²/s⁷   |
| obstacle/bound  | position-based          | `T⁰`      | —       |

Two problems with the raw form:

1. **Weights aren't a common currency.** `w_snap` and `w_acc` have different
   units, so their numeric values can't be compared or reasoned about.
2. **Catastrophic conditioning.** Over the feasible `T ∈ [0.4, 60] s`, the snap
   term's `1/T⁷` scaling swings ~10 orders of magnitude. IPOPT's Hessian becomes
   ill-conditioned, and the solver gets a perverse lever: every effort term
   *shrinks* as `T` grows (snap fastest), so a snap-heavy objective inflates `T`
   — you ask for smoothness and get a needlessly slow flight.

We fix both with two normalizations (`casadi_effort_terms`):

```
J_d = (1 / T_ref) · Σ_samples ( ‖dᵏx/dtᵏ‖² / limit_d² ) · (T·ds)
```

- **Divide each derivative by its kinematic limit** (`a_max`, `j_max`, `s_max`).
  This nondimensionalizes the term (it becomes a *fraction of saturation*) and,
  because the limits are *enforced constraints*, bounds it: `‖a‖²/a_max² ≲ 3`
  pointwise regardless of how the coefficients and `T` trade off. That kills the
  `1/Tᵏ` blow-up — the coefficients and `T` are tied together by the constraint.
  It also erases the `a_max=4` vs `j_max=10` vs `s_max=50` magnitude disparity.
- **Divide by a reference time `T_ref`** (and normalize the time and
  segment-smoothing terms by it too). This removes the residual `T_total` scaling
  so each `J_d` is a time-*average* of a bounded quantity, ~`O(1)`.

The result: every weight is a **dimensionless `O(1)` trade-off ratio** that
transfers across problem scales (a 1 m hover and a 6 m dash use the same
weights), and the dynamic range that wrecked conditioning is compressed to `O(1)`.

### Choosing `T_ref` (`_reference_time`)

`T_ref` = time to traverse the **flight-volume box diagonal** at **half** of
`v_max`. The diagonal is the longest straight path through the workspace; half
`v_max` is an average-speed proxy (an accel-limited point-to-point move never
cruises at `v_max`). It is a **constant** per `(bounds, limits)` configuration —
deliberately *not* per-problem.

Why a constant is the right call: because `T_ref` divides *both* the time term and
the effort terms, it factors out of the time-vs-effort balance entirely (scaling
co-scaled terms doesn't move their trade-off). Its real job is (a) making the
weights dimensionless and (b) setting the overall scale of the kinematic block
relative to the **un-normalized** obstacle and boundary penalties. A stable,
workspace-derived constant gives a predictable kinematic-vs-penalty balance you
tune once, rather than one that drifts with each start/goal pair.

### Default weights

The planner *class* defaults are `w_time=10, w_acc=1, w_jerk=0, w_snap=1, w_yaw_acc=0`.
The values that actually run come from the **ROS node** parameters, which override
them: `w_time=13`, `w_acc=1`, `w_jerk=0`, `w_snap=2`, `w_yaw_acc=5` (with both progress
weights `0`). So at runtime time dominates ~13:1, snap effort is weighted twice
acceleration, jerk is available but off, and the path-tangent yaw-acceleration cost is
active. Because the terms are normalized, these numbers mean what they look like. Tune
via the node parameters, not the class defaults.

## Obstacles and boundaries

- **Capsules are the inflated boundary.** Capsules are the unifying primitive
  (`casadi_capsule_distance_sq`); cylinders are a special case, and URDF
  box/sphere/cylinder collisions are converted to bounding capsules upstream. The
  capsule radius already includes the safety inflation (`obstacle_safety_margin`), so
  "clearance" below is signed distance to that *inflated* surface — the real vehicle
  keeps the inflation as a buffer underneath.
- **Soft avoidance, not a hard constraint.** Obstacle avoidance is a soft one-sided
  quadratic hinge on clearance (`casadi_obstacle_clearance_penalty`): zero value *and*
  gradient once the path is `obstacle_penalty_margin` metres clear of the inflated
  surface, rising quadratically as it approaches or crosses — the same margin-band shape
  as `casadi_boundary_bias`, scaled by `obstacle_penalty_weight`. This **replaced** an
  earlier hard clearance inequality (`distance² − r² ≥ 0`) plus a stiff `tanh` wall.
  Three reasons the soft form is better here:
  - *No barrier / feasibility-restoration cost.* Hundreds of nonconvex keep-out
    inequalities made IPOPT's interior-point solve slow and timeout-prone even from a
    collision-free seed; the hinge is just an objective term.
  - *Non-binding obstacles are free.* Outside the margin band the penalty contributes
    nothing (value and gradient), so distant obstacles don't distort the path and the
    solve degrades gracefully to the obstacle-free problem.
  - *Buy-off is safe.* Because the capsule is already inflated, a shallow incursion the
    min-time objective might "buy" still leaves real clearance — the inflation, not the
    constraint, is the safety margin.
  Dense post-solve validation (below) is the actual guarantee: a candidate that breaches
  clearance is rejected, not shipped.
- **Margin sizing at tight gates.** Because `obstacle_penalty_margin` is measured against
  the inflated surface, it must fit inside the passage. A gate whose inflated opening is
  ~0.24 m leaves its centre only ~0.12 m clear, so a margin above that would put the
  required gate waypoint inside the penalty band and fight the pinned boundary condition.
  Keep the margin below the tightest gate's half-clearance (per-obstacle if needed) so the
  waypoint stays penalty-free.
- **Boundary bias** (`casadi_boundary_bias`) is the same margin-band hinge for the
  flight-volume walls, layered *under* the still-**hard** box constraints: zero (with
  zero slope, so C1) beyond `boundary_bias_margin` inside each wall, rising to
  `boundary_bias_weight` at the wall. It keeps free flight off the walls while yielding
  near hard-constrained targets.
- **Adaptive obstacle refinement.** The penalty is sampled at a finite set of `s` per
  segment, so a path can still graze *between* samples. After the first constrained solve,
  `_find_obstacle_refinements` scans each segment/obstacle pair densely and, where the path
  runs close, adds extra penalty samples (spaced by `obstacle_refine_*`) and re-solves from
  the previous solution, concentrating the penalty where the path grazes. The dense
  validation gate is the backstop for any residual between-sample incursion.
- **Not yet distance-gated.** Every (segment, obstacle, sample) currently contributes a
  term even for far-apart pairs; most are structurally zero (outside the margin) but still
  built. Gating obviously-distant pairs at build time is a pending optimization.

## Kinematic limits

Velocity, acceleration, jerk, and snap are all bounded in the main solve by a
direct two-sided constraint per `(segment, axis)` at each sample:
`−limit ≤ val(sᵢ) ≤ limit`. (An earlier version used a per-segment slack
reformulation, `slack ∈ [0, limit]` with `slack² − val² ≥ 0`; it enforced the
same feasible set but added nonconvex auxiliary variables and converged ~3×
slower on the gate scenarios for identical trajectories.) Two caveats:

- **Per-axis box, not Euclidean ball.** Each of x/y/z is capped independently, so
  the true vector magnitude can reach `√3·limit` on a diagonal. This is also why
  the limit-normalized effort terms are bounded by ~3 rather than 1.
- **Sampled in the solve, dense-checked after.** The in-solve limits hold only at the
  `n_eval` sample points, so a high-order polynomial can overshoot between them. More
  in-solve samples tighten this, but the real guard is the **dense post-solve gate**
  (`validation.validate_kinematic_limits`, 200 samples) — a candidate that overshoots
  v/a/j/s between samples is rejected, not shipped. Obstacle clearance and position
  bounds are dense-checked the same way (`validate_obstacle_clearance`,
  `validate_position_bounds`). See *Validation and the success contract*.
- **Optimizer headroom.** The `kinematic_constraint_scale` parameter defaults to
  `0.975`, so the sampled NLP enforces `0.975 * limit` for v/a/j/s while dense
  validation still checks the real configured limits. Set it to `1.0` to disable
  this headroom.

## Segments and initialization

- **Minimum 2 segments.** With 8 coefficients/segment and 5 endpoint derivatives
  (pos…snap) constrained at each end (10 conditions/axis), one segment (8 DOF)
  can't satisfy them; two segments (16 DOF − 5 continuity = 11) can.
- **3 is the sweet spot for a single obstacle** (approach arc / bypass / exit arc),
  with a junction near closest approach so curvature concentrates where needed.
- **More segments help** (more DOF, less per-piece excursion, less between-sample
  overshoot, freedom to spread curvature) **at a cost** (larger NLP, slower solve,
  more local minima, harder time allocation). Heuristic:
  `n_segments ≈ (turns to make) + 1`, clamped to ~`[2, 6]`.
- **Initialization is decisive for obstacle problems.** The ROS parameter
  `seed_strategy` selects a strategy from `seeding.SEED_STRATEGIES`; the default launch
  configuration uses `dubins`, a Dubins-inspired geometric initializer that chooses the
  shortest feasible left/right circle-tangent-circle candidate from the endpoint frames
  and travel directions. `obstacle_boundary_subdivision`, `velocity_biased_chord`, and
  `straight_line` remain available as simpler baselines. More segments only help once
  the seed and time allocation already route *around* the obstacle; otherwise they give
  the solver more rope to tie a worse knot.
- **The cheap seed is built outside the subprocess.** The action server calls the
  planner's deterministic `build_base_seed(...)` first — a fast geometric/closed-form
  guess, never an optimizer-backed one — and passes those `initial_segments` into the
  subprocess. `CasadiObstaclePlanner.plan(...)` requires a supplied seed and focuses on
  constructing/solving the NLP.

## Staged anytime pipeline

The constrained solve is slow and can fail or time out, so planning runs as a staged
pipeline in **one spawned subprocess** (`optimization.run_planner_pipeline_process` /
`planner_pipeline_process_main`), supervised by a per-goal thread in the node. `spawn`
(not `fork`) is used so the child does not inherit lock state from the threaded ROS
parent. The stages:

1. **Geometric seed** — the cheap `build_base_seed` guess (built in the parent).
2. **Skip bounds/kinematics warm-start** — the pipeline keeps the geometric seed
   unchanged for the obstacle stage. This avoids spending time on an optimizer-backed
   initializer that can move the path into the wrong gate topology before real
   obstacles are considered.
3. **Constrained solve** — the full NLP (hard bounds/kinematic constraints plus the soft
   obstacle penalty) at tight tolerances, seeded directly by the geometric guess and
   followed by adaptive obstacle refinement. The obstacle set is the real planner
   obstacle set.
4. **Dense validation** — see below; only a candidate that passes becomes shippable.

Each stage streams a status dictionary back over a queue (`stage`, `msg`, `progress`,
per-stage `timing`, an optional serialized `trajectory`, and an optional `obstacles`
list). `valid` is a **promotion signal**: candidates stream `valid=False` (for
visualization only), and a trajectory is re-streamed `valid=True` only after it passes
dense validation. The supervisor retains the latest `valid=True` payload as
`best_valid`, because the child's in-memory state is lost when it is terminated on
cancel/timeout. `obstacles`, when present, is the exact set that stage solved
against (only set when it differs from what's already on screen, e.g. the
gate-adjacent subset at the warm-start stage above); the action server's
`pipeline_message` callback republishes it via `ObstacleManager.publish_markers_if_subscribed`
so the marker display tracks what's actually influencing the solve, not just the
full request-time obstacle set.

### Validation and the success contract

After the constrained/refined solve, the candidate is gated by dense
(`~200`-sample) checks in `validation.py`: endpoint **boundary** match, **kinematic
limits** (v/a/j/s), **position bounds**, **obstacle clearance**, and junction
**continuity**.

**Success means a dense-validated plan, nothing else.** The action succeeds only when a
`best_valid` trajectory exists; otherwise it returns timeout/cancel. An unvalidated
candidate is never shipped — failing to find a plan within the deadline is an accepted
outcome.

**Solver/validation tolerance coupling.** Each dense tolerance is set *looser* than the
`constr_viol_tol` of the stage that produced the candidate, so a converged (or
accepted-at-iteration-limit) solution is not rejected on solver-noise-level violations.
The non-shippable warm-start seed is gated with a separate, much looser `SEED_*`
tolerance profile (it only needs gross sanity), while the shippable `CONTINUITY_TOLERANCES`
sit above the constrained stage's acceptable tolerance. Tight, *informational*
continuity warnings use the distinct `DIAGNOSTIC_CONTINUITY_TOLERANCES`.

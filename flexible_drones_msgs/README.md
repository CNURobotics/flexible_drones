# flexible_drones_msgs

ROS 2 interface package for the Flexible Drones stack. This package is the
shared API boundary between drone managers, trajectory tools, swarm commands,
FlexBE states, and deployment workflows.

Related in-repository packages:

- [`flexible_drones_core`](../flexible_drones_core/README.md) implements the
  backend-neutral manager/orchestrator logic that consumes these interfaces.
- [`flexible_drones_tools`](../flexible_drones_tools/README.md) provides CLI
  clients, demos, plotting, logging, and trajectory utilities for these
  interfaces.
- [`flexible_drones_description`](../flexible_drones_description/README.md)
  provides shared models and visualization assets used by managers.

## Messages

- `DroneStatus`: platform-neutral manager status snapshot with compact lifecycle
  `state`, common `status_flags`, raw platform bits in `vendor_status_flags`,
  battery voltage/percent, link/odom/time-sync hints, and operator-facing status
  and warning text. Unknown percentages use `PERCENT_UNKNOWN`; unknown age or
  latency fields use `-1.0`.
- `FullState`: stamped pose and body-frame twist command/state payload.
  `header.frame_id` names the pose/world frame; `child_frame_id` names the body
  frame for twist. `valid_mask` marks which grouped ROS-frame fields are
  meaningful: position, orientation, linear velocity, and yaw rate.
- `Trajectory`: polynomial trajectory identified by `trajectory_id`.
- `TrajectoryPolynomialPiece`: polynomial coefficients for x, y, z, yaw, plus
  segment duration.
- `TrajectoryPoint`: sampled x, y, z, yaw point with `time_from_start`.
- `SampledTrajectory`: named sampled trajectory with ordered
  `TrajectoryPoint` entries.
- `TrajectoryBoundary`: framed pose/twist constraint used as the start or end
  boundary condition in trajectory-planning requests.
- `TrajectoryBounds`: axis-aligned flight-space bounds (`[min, max]` per axis)
  in the planner's preferred frame.
- `TrajectoryObstacle`: obstacle primitive for trajectory planning. Only
  `type: "cylinder"` is supported: a vertical cylinder centered at (x, y);
  a non-positive or NaN `height` means full flight-volume height (`z_max`).
- `OnboardStatus`: companion-computer health snapshot published by the PiHawk
  onboard node. Reports autopilot type and lifecycle `state`, the
  `ready_for_actions` verdict, positive health bits (masked by
  `applicable_health_flags`), blocker bits explaining why actions are
  rejected, warning bits, and time-sync, mocap, heartbeat, odometry-age, and
  FCU-load diagnostics.

## Services

- `Arm`: arm or disarm a drone with a timeout.
- `EmergencyStop`: request an emergency stop and return a success/message pair.
- `GetStatus`: fetch `DroneStatus` for a named drone.
- `Reboot`: safely reboot vehicle flight-controller/firmware; implementations
  must reject unless the vehicle is disarmed and confirmed on the ground.
  `timeout_sec` is how long to wait for reboot/drop confirmation when supported.
- `RealignLocalPosition`: realign the vehicle local-position estimate. The
  `pose` field provides an optional reference; implementations that support it
  use the supplied pose as the target state, while implementations that do not
  (e.g. PiHawk companion bridge, which recalculates a local-frame offset from
  current sensor readings) may derive the target state from live data.
  Response includes `message` explaining any rejection.
- `SetLEDColor`: request a named LED color.
- `SpawnDrone`: spawn a named drone in simulation.
- `SwarmArm`: arm the swarm and report per-request counters:
  `targeted`, `unavailable`, `sent`, `succeeded`, `failed`, and
  `response_timeout`.
- `SetTrajectoryBounds`: update the trajectory planner's axis-aligned
  flight-space bounds; takes effect on subsequent planning requests.
- `SetTrajectoryObstacles`: replace the trajectory planner's obstacle set
  (an empty list clears all obstacles); takes effect on subsequent planning
  requests.

## Actions

### Per-Drone Motion

- `Takeoff`: take off to `height` over `duration`; feedback reports
  `current_height`.
- `Land`: land to `height` over `duration`; feedback reports
  `current_height`.
- `GoTo`: move to an absolute or relative target. Supported frames are
  `FRAME_ABSOLUTE`, `FRAME_RELATIVE_MAP`, and `FRAME_RELATIVE_BODY`; feedback
  reports `current_position` and `current_yaw`.

Each per-drone motion action returns an integer `return_code`; `0` is the
success convention used by managers in this stack.

### Trajectories

- `UploadTrajectory`: upload a `Trajectory` and return the number of uploaded
  pieces.
- `ExecuteTrajectory`: execute an uploaded trajectory by `trajectory_id`.
  Feedback reports `progress`, `elapsed_sec`, `remaining_sec`, and
  `duration_sec`.
- `GetTrajectory`: request a planned polynomial trajectory between start and
  end `TrajectoryBoundary` pose/twist constraints. Feedback reports planner
  `remaining_sec` and operator-facing `status` text.
- `GetSampledTrajectory`: request a sampled trajectory using the same planning
  inputs plus `sample_dt`; feedback uses the same planner feedback fields as
  `GetTrajectory`.

### Controller Sessions

- `EnableController`: generic session-style controller gate; the goal names a
  controller (managers use it for `teleop_control`) with optional
  `params_yaml`. The action stays active while the controller runs; feedback
  reports `controller_active`, command `input_age_sec`, and status text, and
  the result carries the manager `return_code` convention plus the
  `active_controller` name.

### Swarm Commands

The swarm interfaces are kept for orchestrator development. The orchestrator
provider for these interfaces is experimental and has not been release-tested;
basic workflows should use the per-drone services and actions above.

- `SwarmTakeoff`: dispatch takeoff goals to all commandable drones.
- `SwarmLand`: dispatch land goals to all commandable drones; this action is
  safety critical and proceeds despite unavailable drones.
- `SwarmGoTo`: dispatch relative goto offsets to the swarm. Absolute swarm goto
  is intentionally unsupported because a single absolute target cannot be
  shared meaningfully by multiple drones.

Swarm action feedback and results include the same command counters:
`targeted`, `unavailable`, `sent`, `accepted`, `rejected`,
`goal_response_timeout`, `succeeded`, `failed`, `canceled`, and
`result_timeout`. Results also include `return_code`, where `0` means all
accepted goals succeeded, `1` means partial success, `2` means no goals
succeeded, and `5` means the top-level swarm action was canceled. `SwarmGoTo`
may also return `3` for an invalid swarm request.

## Contract Notes

- `group_mask` goals are accepted by a drone when the goal mask is `0` or when
  it overlaps the drone manager's configured group mask.
- Action durations use `builtin_interfaces/Duration`. In CLI YAML, pass them as
  mappings such as `{sec: 5, nanosec: 0}`. Direct per-drone motion actions
  require positive durations; swarm actions may use `0` to request the
  orchestrator's provider default before fan-out.
- Relative body-frame motion uses ROS ENU conventions: x is forward along the
  current heading and y is left.

# flexible_drones

ROS 2 multi-drone orchestration framework for heterogeneous fleets of Crazyflies,
PiHawk/ArduPilot drones, and Gazebo simulation. Supports mixed hardware/simulation
deployments, swarm-level action coordination, FlexBE behavior integration, and
trajectory upload/execution.

> **Release note:** The orchestrator-managed mixed-fleet path is experimental and
> has not been release-tested. For basic bringup and flight checks, follow the
> backend-specific README directions for Crazyflie, PiHawk, or Gazebo launches.

This work is described in:

> E. R. Faith, A. B. Kooiker, A. J. Farney, S. Fox, E. Grimes and D. C. Conner,
> "Flexible Drones: ROS 2 Action-Based Coordination of Multiple Heterogeneous Drones,"
> *SoutheastCon 2026*, Huntsville, AL, USA, 2026, pp. 1–6,
> doi: [10.1109/SoutheastCon63549.2026.11476332](https://ieeexplore.ieee.org/document/11476332)

---

## Repository layout

This repository now owns the backend-neutral Flexible Drones packages:

| Package | Purpose | Package README |
|---|---|---|
| `flexible_drones` | Aggregate package for the backend-neutral stack. | [`flexible_drones/package.xml`](flexible_drones/package.xml) |
| `flexible_drones_core` | Shared manager API, registry parsing, orchestrator logic, and process supervision. | [`flexible_drones_core/README.md`](flexible_drones_core/README.md) |
| `flexible_drones_msgs` | ROS 2 messages, services, and actions shared by managers, tools, and behaviors. | [`flexible_drones_msgs/README.md`](flexible_drones_msgs/README.md) |
| `flexible_drones_description` | Shared URDF/xacro models, meshes, RViz configs, and description launch helpers. | [`flexible_drones_description/README.md`](flexible_drones_description/README.md) |
| `flexible_drones_tools` | CLI clients, demos, trajectory utilities, monitoring tools, and lightweight GUIs. | [`flexible_drones_tools/README.md`](flexible_drones_tools/README.md) |

Backend, deployment, and behavior packages are separate workspace sibling
repositories:

| Package | Purpose | README |
|---|---|---|
| `chris_serc_deployment` | CNU SERC lab roster, deployment overlays, startup helpers, and trajectories. | `../chris_serc_deployment/README.md` once cloned |
| `flexible_drones_crazyflie` | Crazyflie hardware backend. | `../flexible_drones_crazyflie/README.md` once cloned |
| `flexible_drones_arducopter` | PiHawk/ArduPilot backend. | `../flexible_drones_arducopter/README.md` once cloned |
| `flexible_drones_gazebo` | Gazebo simulation backend. | `../flexible_drones_gazebo/README.md` once cloned |
| `flexible_drones_optitrack` | OptiTrack/NatNet pose bridge. | `../flexible_drones_optitrack/README.md` once cloned |
| `flexible_drones_flexbe` | FlexBE aggregate package plus Flexible Drones states and behaviors. | `../flexible_drones_flexbe/README.md` once cloned |

Typical workspace layout:

```text
src/
  flexible_drones/
    flexible_drones/
    flexible_drones_core/
    flexible_drones_msgs/
    flexible_drones_description/
    flexible_drones_tools/
  chris_serc_deployment/
  flexible_drones_crazyflie/
  flexible_drones_gazebo/
  flexible_drones_optitrack/
  flexible_drones_arducopter/
  flexible_drones_flexbe/
```

---

## Workspace setup

```bash
export WORKSPACE_ROOT=${WORKSPACE_ROOT:-$HOME/chrislab}
mkdir -p "$WORKSPACE_ROOT/src"
cd "$WORKSPACE_ROOT/src"
git clone https://github.com/CNURobotics/flexible_drones.git
vcs import < flexible_drones/flexible_drones.repos
```

The backend-neutral packages `flexible_drones_core`, `flexible_drones_msgs`,
`flexible_drones_description`, and `flexible_drones_tools` are already included
inside this repository. Import only the remaining sibling repositories needed
for your deployment or backend mix. The release `.repos` file imports the
standard backend and optional FlexBE repositories from the `alpha` branch.

For the SERC lab deployment, copy the lab setup helper and Fast DDS profile
template from `chris_serc_deployment` into the workspace root before sourcing
or building. Those files are project-specific and live in the deployment
repository, not in the backend-neutral `flexible_drones` repository. Run this
from the `src` folder after cloning/importing `chris_serc_deployment`. That
deployment package is shown as a commented example in `flexible_drones.repos`
because most non-SERC users should provide their own deployment package.

```bash
cp chris_serc_deployment/setup.bash ../setup.bash
cp chris_serc_deployment/fastdds_wired_only_profile.template.xml ../
source ../setup.bash
```

See `chris_serc_deployment/README.md` and
`chris_serc_deployment/ros2-kilted-multihomed-discovery-fix.md` for the lab
networking details. Other deployments should provide their own setup helper or
source `install/setup.bash` directly.

After importing the repos, choose the core-only or full-stack setup before
running `rosdep`. Note that `flexible_drones` and `flexible_drones_flexbe`
are backend-neutral aggregates: neither one pulls in a hardware/sim backend
or deployment data by itself. For a runnable SERC lab install, build up to
`chris_serc_deployment` instead (see "Aggregate Packages" below); other
projects should build up to their own deployment package or specific
backend package(s).

Core stack without FlexBE:

```bash
touch "$WORKSPACE_ROOT/src/flexible_drones_flexbe/COLCON_IGNORE"
cd "$WORKSPACE_ROOT"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-up-to flexible_drones
source install/setup.bash
```

Full stack including FlexBE states, behaviors, and runtime packages:

```bash
rm -f "$WORKSPACE_ROOT/src/flexible_drones_flexbe/COLCON_IGNORE"
cd "$WORKSPACE_ROOT"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-up-to flexible_drones_flexbe
source install/setup.bash
```

Dependencies not covered by rosdep are documented in the package that needs them:
Crazyflie `cflib` (`pip3 install cflib`), PiHawk `pymavlink`
(`pip3 install pymavlink`), and the pip-installed planner/plotting/GUI
packages for [`flexible_drones_tools`](flexible_drones_tools/README.md)
(see [`flexible_drones_tools/requirements.txt`](flexible_drones_tools/requirements.txt)).

---

## Package overview

Packages are grouped from most general (shared by everything) to most specific
(single-backend or single-scenario).

### 1. Generic

* **[`flexible_drones_msgs`](flexible_drones_msgs/README.md)** — ROS 2 message, service, and action definitions shared
across the entire stack.

Key interfaces:
  * `DroneStatus`,
  * Services for `Arm` and `SwarmArm`
  * Per-drone actions (`Takeoff`, `Land`, `GoTo`, `UploadTrajectory`,
`ExecuteTrajectory`)
  * Swarm actions (`SwarmTakeoff`, `SwarmLand`, `SwarmGoTo`)

* **[`flexible_drones_core`](flexible_drones_core/README.md)** — Everything
that is backend-neutral. Organised into two sub-modules:

  * `flexible_drones_core/manager/` — per-drone manager contract:
    - `DroneManager` — abstract base class all backends subclass; owns ROS interfaces
        (services, action servers, publishers, TF broadcaster), URDF joint parsing, prop
        animation, and group-mask goal filtering
    - `ManagerConfig` / `base_manager_config_from_spec` — resolved manager construction
        contract shared by subprocess, embedded, direct-launch, and SimpleSwarm paths

  * `flexible_drones_core/orchestrator/` — experimental fleet-level supervision:
    - `DroneOrchestratorNode` — desired-state reconciliation timer; swarm-wide fan-out
    for arm, takeoff, land, go_to; admin services (reload registry, spawn/stop drone)
    - `ManagerSupervisor` — routes each `DroneSpec` to either a subprocess or embedded
    manager; stop-before-start ordering prevents ownership overlap on manager-type changes
    - `ProcessManager` — subprocess lifecycle: spawn via `ros2 run`, exponential restart
    backoff, SIGINT→SIGTERM→SIGKILL shutdown sequence
    - `Registry` / `DroneSpec` — thread-safe desired-state store loaded from YAML


### 2. Platform-independent tools

These packages work with any backend and have no hardware-specific dependencies.

* **[`flexible_drones_tools`](flexible_drones_tools/README.md)** — CLI clients for per-drone and swarm commands,
trajectory generation utilities (polynomial, spiral, circle, star), monitoring and
logging nodes, and standalone demo scripts. These utilities wrap the core ROS
interfaces and are used across all backends and deployment scenarios.

* **`flexible_drones_optitrack`** — NatNet driver that publishes `PoseStamped` on
`/<name>/pose` for each tracked rigid body. Works with any backend that accepts
external pose input (Crazyflie ext-pose mode, PiHawk). Also useful standalone for
pose replay and RViz visualization without any drone managers running.

  See `flexible_drones_optitrack/README.md`.

### 3. Deployment and shared project data

* **`chris_serc_deployment`** — Canonical fleet roster and named deployment
setups (`mixed`, `crazyflies`, `pihawks`). Owns the authoritative list of which drones
exist in the project, how each test session enables and positions them, and
trajectory CSV files for named flight demonstrations. Its orchestrator-based launch
files are experimental for this release; prefer the backend package launch files for
basic sessions.

  Users may edit this to update their fleet information.

* **[`flexible_drones_description`](flexible_drones_description/README.md)** — Shared robot URDF/xacro models, gate models,
RViz configurations, and launch helpers for robot descriptions and visualization.
All backends import drone models from here.

  New drone models can be added to this repo.

### 4. Drone backends

Each backend implements `DroneManager` and owns its own defaults YAML,
launch files, and hardware-specific configuration.

* **`flexible_drones_crazyflie`** — Crazyflie hardware backend via `cflib`.
Supports Lighthouse pose estimation and external pose (OptiTrack/Vicon).
The experimental embedded orchestrator path keeps cflib in-process; the SimpleSwarm node
hosts multiple CF managers in one process per radio.

  See `flexible_drones_crazyflie/README.md`.

* **`flexible_drones_arducopter`** — PiHawk/ArduPilot backend via direct MAVLink
(`pymavlink`). Uses OptiTrack for external pose. Intended for Pi-hosted companion
computers connected to ArduPilot flight controllers.

  See `flexible_drones_arducopter/README.md`.

* **`flexible_drones_gazebo`** — Gazebo simulation backend for all drone types via
the `KinematicDrone` C++ plugin. All simulated drone types share one manager
implementation; the URDF/xacro loaded at spawn time is the only type-specific part.

  See `flexible_drones_gazebo/README.md`.


### 5. Autonomous behaviors

The above packages are independent of FlexBE, but the
action-based interfaces support high-level control
using [FlexBE](https://github.com/flexbe/flexbe_behavior_engine) - the Flexible Behavior Engine -
and [FlexBE WebUI](https://github.com/flexbe/flexbe_webui) versions 4.1.4+.

The FlexBE repo provides:

* **`flexible_drones_flexbe`** — aggregate package that depends on the core stack,
the reusable FlexBE states, the behavior package, and the FlexBE runtime packages.
* **`flexible_drones_flexbe_states`** — wraps each swarm/per-drone action and service
as a `FlexBE EventState`.
* **`flexible_drones_flexbe_behaviors`** — composes those states into
`OperatableStateMachine` behaviors for autonomous mission execution.

  See `flexible_drones_flexbe/README.md`.

  Provided behaviors:

  | Behavior | Description | Compatible scenarios |
  |---|---|---|
  | `drone_takeoff` | Arm and take off a single drone | Any single-drone backend |
  | `drone_land` | Land and disarm a single drone | Any single-drone backend |
  | `gazebo_single_drone_cycle` | Arm → takeoff → hover → land on one drone | Gazebo simulation |
  | `cnu_sails` | Interactive single-drone trajectory demo: load, upload, execute, LED color feedback | Single Crazyflie or Gazebo drone |
  | `cnu_3sails` | Full 3-drone concurrent trajectory flight with operator checkpoints | 3 Crazyflies or 3 Gazebo drones |
  | `CapstoneDemo` | Interactive single-drone public demonstration: takeoff, go-to, trajectory, land | Any single-drone backend (validated on Gazebo) |


  If you do not wish to use FlexBE, create
`src/flexible_drones_flexbe/COLCON_IGNORE` before running `rosdep install` and
build up to the core `flexible_drones` aggregate package.

### 6. Aggregate Packages

The core aggregate package is `flexible_drones` (`flexible_drones/package.xml`).
It declares `<exec_depend>` entries for the backend-neutral stack only:
interfaces, core orchestration, shared descriptions, and tools. It
intentionally does **not** pull in any hardware/sim backend (Crazyflie,
Gazebo, PiHawk/ArduCopter, OptiTrack) or deployment roster/overlay data,
since those are project-specific choices, and it does not pull in FlexBE.

```bash
colcon build --symlink-install --packages-up-to flexible_drones
```

For a full install of everything used at the CNU SERC lab, build up to
`chris_serc_deployment` (`chris_serc_deployment/package.xml`) instead.
It depends on `flexible_drones` plus the specific backends deployed in that
lab (Crazyflie, ArduCopter/PiHawk, Gazebo, OptiTrack) and the roster/overlay
data describing the fleet:

```bash
colcon build --symlink-install --packages-up-to chris_serc_deployment
```

Other projects that use a different backend mix should build up to their own
deployment package, or up to the specific backend package(s) they need,
instead of `chris_serc_deployment`.

The FlexBE aggregate package is `flexible_drones_flexbe`
(`flexible_drones_flexbe/package.xml`). It depends on `flexible_drones` plus the
FlexBE runtime, states, and behaviors.

Use this target when you want everything in the core stack plus FlexBE, but
still without any hardware/sim backend:

```bash
colcon build --symlink-install --packages-up-to flexible_drones_flexbe
```

## License

New development is licensed under Apache-2.0.

Some packages and model assets include or derive from third-party work under
compatible licenses. See [NOTICE](NOTICE) and the relevant package/source files
for retained attribution.

### Contributors

* Emma Faith
* Andrew Farney
* Sebastian Fox
* Evangelina Grimes
* Aubrie Kooiker
* David Conner

For more information contact [robotics@cnu.edu](mailto:robotics@cnu.edu)

The authors acknowledge the assistance of ChatGPT, Codex, and Claude to expedite development.

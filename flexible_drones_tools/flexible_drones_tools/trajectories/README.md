# `trajectories` Package Layout

This package owns everything trajectory-related: the action servers, the RViz
visualizer, the pluggable planners, the offline shape generators, and the shared
helper code. It is organized into three subpackages plus the top-level servers.

```
trajectories/
  plan_trajectory_action_server.py   # pluggable-planner action server (+ bounds/obstacle services)
  load_trajectory_action_server.py   # serves named polynomial/sampled trajectories from a YAML config
  visualize_trajectory.py            # RViz nav_msgs/Path visualizer (console_scripts: visualize_trajectory)
  plot_trajectory.py                 # matplotlib 3D / projection / kinematics plots (console_scripts: plot_trajectory)
  planners/                          # planner implementations (TrajectoryPlanner interface)
  generators/                        # offline shape generators + experimental scripts
  utilities/                         # shared, ROS-light helpers (I/O, conversions, math, models)
```

## Top-level servers

- **`plan_trajectory_action_server.py`** — serves `plan_trajectory` (polynomial) and
  `plan_sampled_trajectory` (sampled). It selects a planner via the `planner`
  parameter, transforms the start/goal into `preferred_frame`, and plans in a
  forked process with cache, timeout, and cancel support.
  - Flight bounds and obstacles are runtime state, set via services:
    `set_trajectory_bounds` (`SetTrajectoryBounds`) and
    `set_trajectory_obstacles` (`SetTrajectoryObstacles`). Obstacles start
    unconfigured; planning goals are rejected until the operator calls
    `set_trajectory_obstacles` with either a valid cylinder list or an empty list
    to explicitly clear obstacles. Invalid obstacle updates are rejected and do
    not replace the current obstacle state.
  - Magic numbers are parameters (defaults preserve the historic values):
    `planner` (`casadi_obstacle` | `kkt`), `n_segments`, `n_eval`,
    `limits` (YAML string -> `{v_max,a_max,j_max,s_max}`), and
    `bounds.{x,y,z}_{min,max}` (default `x,y in [-3, 3]`, `z in [0.25, 2.5]`).
- **`load_trajectory_action_server.py`** — serves `load_trajectory` /
  `load_sampled_trajectory` from a YAML config of named CSV trajectories.

## `planners/` — `TrajectoryPlanner` interface

Every planner derives `base_planner.TrajectoryPlanner` and implements
`plan(start, goal, obstacles, bounds, limits, n_segments)`, returning segments
`[{'T', 'c_x', 'c_y', 'c_z'}]` with **real-time, ascending-power** coefficients
(`x(t) = sum_k c_k t^k`). Both planners are start->goal; multi-waypoint paths are
stitched by the caller.

| Module | Description |
| --- | --- |
| [planners/base_planner.py](planners/base_planner.py) | `TrajectoryPlanner` ABC; `DEFAULT_BOUNDS`/`DEFAULT_LIMITS`; bounds/limits merge helpers |
| [planners/casadi_obstacle_planner.py](planners/casadi_obstacle_planner.py) | `CasadiObstaclePlanner`: CasADi/IPOPT time-optimal planner; box bounds + v/a/j/s limits + soft cylinder-obstacle penalty (empty list => clean min-time path) |
| [planners/kkt_planner.py](planners/kkt_planner.py) | `KktPlanner`: closed-form two-segment minimum-snap, obstacle-free (bounds not enforced) |

Add a planner by subclassing `TrajectoryPlanner` and registering it in the
`PLANNERS` dict in `plan_trajectory_action_server.py`.

## `generators/` — offline shape generators + experiments

Shape generators derive `generate_trajectory.GenerateTrajectory` and only
implement `generate_trajectory(...)`; the base `generate_to_file(path, ...)`
runs the shared sample -> polyfit -> drone-order CSV pipeline. These are **run
from source** (not `ros2 run` entry points): each script has a shebang and a
`main()`, and writes to the git-ignored `generators/output/` folder by default.
See [generators/README.md](generators/README.md) for usage.

| Module | Description |
| --- | --- |
| [generators/generate_trajectory.py](generators/generate_trajectory.py) | Base class + `generate_to_file` + static plot helpers |
| [generators/generate_circle_trajectory.py](generators/generate_circle_trajectory.py) | Circle shape |
| [generators/generate_figure8_trajectory.py](generators/generate_figure8_trajectory.py) | Figure-eight shape with configurable radius/height and tangent-following yaw |
| [generators/generate_star_trajectory.py](generators/generate_star_trajectory.py) | Star shape |
| [generators/generate_spiral_trajectory.py](generators/generate_spiral_trajectory.py) | Spiral shape |
| [generators/cnu_sails.py](generators/cnu_sails.py) | Sail-shape generator (uses `utilities/kkt`); writes `sails/*.csv` via shared I/O |
| [generators/vander.py](generators/vander.py) | Vandermonde min-snap from generated/segment-fit data |
| [generators/main_test.py](generators/main_test.py) | Spiral + segment-fit example script |
| [generators/shortest_path_finder.py](generators/shortest_path_finder.py) | Experimental casadi shortest-path script |
| [generators/output/](generators/output/) | Default (git-ignored) output folder for generated CSVs |
| [generators/sails/](generators/sails/) | Committed reference sail CSVs |

## `utilities/` — shared helpers (ROS-light)

| Module | Description |
| --- | --- |
| [utilities/io.py](utilities/io.py) | CSV load/save (writes **drone** coefficient order) + path resolution |
| [utilities/conversion.py](utilities/conversion.py) | Coefficient <-> ROS `PolynomialPiece`/sampled conversions; drone/numpy ordering |
| [utilities/trajectory_path.py](utilities/trajectory_path.py) | `resolve_trajectory_path` / `resolve_trajectory_write_path` |
| [utilities/generate_drone_format.py](utilities/generate_drone_format.py) | Coefficients -> drone-format pieces / CSV |
| [utilities/read_from_csv.py](utilities/read_from_csv.py) | `load_trajectory_data` around `io.load_trajectory_csv` |
| [utilities/model.py](utilities/model.py) | Re-export of `PolyOrderTrajectory` |
| [utilities/poly_order_trajectory.py](utilities/poly_order_trajectory.py) | `PolyOrderTrajectory`: evaluable polynomial trajectory (+ plotting) |
| [utilities/split_trajectory.py](utilities/split_trajectory.py) | `SegmentFit`: sample -> polyfit -> segments |
| [utilities/plotting.py](utilities/plotting.py) | `TrajectoryPlotter`: shared 3D / projection / kinematics matplotlib helpers |
| [utilities/kkt.py](utilities/kkt.py) | Pure KKT minimum-snap math (no ROS deps); used by `kkt_planner` and `cnu_sails` |

## Conventions & notes

- **Coefficient order**: files are written in **drone** order via
  `utilities/io.save_trajectory_csv`; planners/generators may work in numpy
  ascending-power order internally and convert at the I/O boundary with
  `conversion.coefficients_to_drone_order`.
- **`casadi` dependency**: required at runtime by `plan_trajectory_action_server`
  through `planners/casadi_obstacle_planner.py`; it is in `setup.py`
  `install_requires`. `generators/shortest_path_finder.py` also uses casadi.
- **Dependency direction**: `utilities <- planners` and `utilities <- generators`;
  there is no generator->planner coupling.

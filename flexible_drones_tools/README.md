# Flexible Drones Tools

This package bundles operator tools, demo clients, trajectory helpers, loggers,
and lightweight GUIs for the Flexible Drones stack.

The code is distributed under the Apache 2 license.

Related in-repository packages:

- [`flexible_drones_msgs`](../flexible_drones_msgs/README.md) defines the
  actions, services, and messages used by these tools.
- [`flexible_drones_core`](../flexible_drones_core/README.md) provides the
  shared manager/orchestrator contracts that these clients exercise.
- [`flexible_drones_description`](../flexible_drones_description/README.md)
  provides shared robot/course visualization assets used by demos and RViz
  workflows.

## Demos

The demos are the fastest way to exercise the high-level drone actions after
bringup. They use the same ROS namespace remapping pattern as the standalone
clients, so replace `drone1` with the active drone name.

Basic smoke test: arm, take off, move 1 m in relative map `+x`, land, and disarm.

```
clear; ros2 run flexible_drones_tools basic_demo --ros-args -r __ns:=/drone1
```

Useful options:

- `--distance`: relative map `+x` movement in metres
- `--takeoff-height`: target takeoff altitude in metres
- `--land-height`: landing target height in metres
- `--move-duration`: seconds for the relative `go_to` leg

Square flight demo: arm, take off, fly a map-aligned square, land, and disarm.

```
clear; ros2 run flexible_drones_tools square_demo --ros-args -r __ns:=/drone1
clear; ros2 run flexible_drones_tools square_demo --cruise-height 1.2 --leg-duration 4.0 --ros-args -r __ns:=/drone2
```

Trajectory demo: upload, arm, take off, move to the trajectory start, execute,
land, and disarm.

Bare trajectory filenames are looked up in `<deployment_package>/trajectories/`,
where `<deployment_package>` comes from the `FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE`
environment variable. Set it once for the session to the package that owns your
hardware/site trajectories. The SERC demo deployment uses
`chris_serc_deployment`, but other hardware setups should provide their own
deployment package.

```
export FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE=<deployment_package>
clear; ros2 run flexible_drones_tools trajectory_demo figure8.csv --ros-args -r __ns:=/drone1
clear; ros2 run flexible_drones_tools trajectory_demo figure8.csv --execute-in-reverse --ros-args -r __ns:=/drone1
```

For the SERC demo deployment:

```
export FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE=chris_serc_deployment
clear; ros2 run flexible_drones_tools trajectory_demo figure8.csv --ros-args -r __ns:=/drone1
```

Without that variable set, pass a relative or absolute path, which is used
directly:

```
clear; ros2 run flexible_drones_tools trajectory_demo $WORKSPACE_ROOT/src/<deployment_package>/trajectories/figure8.csv --ros-args -r __ns:=/drone1
```

Useful options for `trajectory_demo`:

- `--go-to-duration`: seconds for the leg to the trajectory start
- `--timescale`: trajectory playback timescale
- `--relative` / `--no-relative`: choose whether `ExecuteTrajectory` runs relative to the current pose
- `--execute-in-reverse`: after the forward pass, replay the same uploaded trajectory in reverse
- `--no-visualize-trajectory`: skip publishing the planned path to RViz
- `--visualization-samples`: number of poses in the published planned path
- `--visualization-period-sec`: seconds between planned path publishes
- `--visualization-topic`: topic for the planned `nav_msgs/Path`, default `planned_trajectory`

**Simulation only** (Gazebo, where `command/twist` is not time-stamped and uses
the internal clock):

```
clear; ros2 run flexible_drones_tools basic_demo --ros-args -r __ns:=/drone1 -p use_sim_time:=true
clear; ros2 run flexible_drones_tools trajectory_demo figure8.csv --ros-args -r __ns:=/drone1 -p use_sim_time:=true
```

## Simplified Monitoring Tools

For the monitoring package map, log format, and RViz flight-path visualizer, see
[`flexible_drones_tools/monitoring/README.md`](flexible_drones_tools/monitoring/README.md).

For each drone name, e.g. `drone1`:

`clear; ros2 run flexible_drones_tools drone_logger --ros-args -r __ns:=/drone1`
  * Simplified data logging using numpy
     * These are saved to the workspace `log` folder under the drone's name by default.
  * **Simulation only** (Gazebo — `command/twist` is not time-stamped and uses the internal clock):
    `clear; ros2 run flexible_drones_tools drone_logger --ros-args -r __ns:=/drone1 -p use_sim_time:=true`

For one or more drone names:

```
clear; ros2 launch flexible_drones_tools drone_loggers.launch.py fliers:="[drone1, drone2]"
```

  * Starts one `drone_logger` node per flier namespace.
  * Logs are saved under `$WORKSPACE_ROOT/log/<drone_name>` by default.
  * Optional shared arguments:
    * `logging_buffer_length:=6000`
    * `use_sim_time:=true`

or

`clear; ros2 run flexible_drones_tools run_drone_plot --ros-args -r __ns:=/drone1`
  * Basic "live" plotting tool using Bokeh and browser window
    * On some machines these plots will develop a significant lag that is shown in upper right.
  * Includes simplified data logging by default
     * Use `-p logging:=false` to disable
     * e.g. `clear; ros2 run flexible_drones_tools run_drone_plot --ros-args -r __ns:=/drone1 -p logging:=false`
     * These are saved to the workspace `log` folder under the drone's name by default.
  * **Simulation only** (Gazebo — `command/twist` is not time-stamped and uses the internal clock):
    `clear; ros2 run flexible_drones_tools run_drone_plot --ros-args -r __ns:=/drone1 -p use_sim_time:=true`

`clear; ros2 run flexible_drones_tools plot_logs <ARGS>`
  * Provides tools for plotting data logged by `run_drone_plot`
  * Where `<ARGS>` are basic command line arguments
  ```
  --drone_name  <name>   # (str) name of drone (default='red1')
  --start_time        # (str) optional (latest time is auto-detected)
  --start_idx         # (int) default=0
  --end_idx           # (int) default=None (latest is auto-detected)
  --include-separate  # (flag) optional - each axis on separate plot
  --include-projections  # (flag) optional - xy, xz, and yz plots
  --log_dir           # (Path) optional, defaults to $WORKSPACE_ROOT/log
  ```

  * For example,

    * `clear; ros2 run flexible_drones_tools plot_logs --drone_name drone1`
      * plots 3D trajectory by itself

    * `clear; ros2 run flexible_drones_tools plot_logs --start_time 1716732080 --start_idx 2 --include-separate`
      * finds plots starting at 1716732080 nanoseconds, beginning at index 2 to end and show each axis in separate figure

Visualize a recorded flight in RViz as `nav_msgs/Path`:

`clear; ros2 run flexible_drones_tools visualize_flight --ros-args -p drone_name:=drone1`


## Simplified Trajectory Handling Tools

This module provides tools for basic development, generation, analysis, and visualization of trajectories that can be executed by the drones.

For a map of which modules are production core versus offline generators versus
experimental research code, see
[`flexible_drones_tools/trajectories/README.md`](flexible_drones_tools/trajectories/README.md).

Visualize a drone-order trajectory CSV in RViz as `nav_msgs/Path`.
Bare filenames are looked up in `<deployment_package>/trajectories/` (via
`FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE`), matching `trajectory_demo`:

`export FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE=<deployment_package>`

`clear; ros2 run flexible_drones_tools visualize_trajectory figure8.csv`

`clear; ros2 run flexible_drones_tools visualize_trajectory $WORKSPACE_ROOT/src/<deployment_package>/trajectories/figure8.csv --samples 300`

Plot a drone-order trajectory CSV with matplotlib (3D path, xy/xz/yz projections,
and x/y/z/yaw position/velocity/acceleration/jerk/snap vs. time). Same filename lookup as above:

`clear; ros2 run flexible_drones_tools plot_trajectory figure8.csv`

`clear; ros2 run flexible_drones_tools plot_trajectory cnu_sail0.csv --samples 800`

Upload a trajectory without executing it:

`clear; ros2 run flexible_drones_tools upload_trajectory --ros-args -r __ns:=/drone1 -p trajectory_package:=<deployment_package> -p trajectory_file:=figure8.csv`

## Standalone ROS Clients

Most one-shot clients support `--help` and ROS parameter overrides after `--ros-args`.

Drone control:

- `ros2 run flexible_drones_tools arm --ros-args -r __ns:=/drone1`
- `ros2 run flexible_drones_tools arm --ros-args -r __ns:=/drone1 -p arm:=false`
- `ros2 run flexible_drones_tools emergency_stop --ros-args -r __ns:=/drone1`
- `ros2 run flexible_drones_tools reboot --ros-args -r __ns:=/drone1`
- `ros2 run flexible_drones_tools realign_local_position --ros-args -r __ns:=/drone1 -p frame_id:=map -p x:=0.0 -p y:=0.0 -p z:=0.0`
- `ros2 run flexible_drones_tools set_led_color --ros-args -r __ns:=/drone1 -p color:=green`
- `ros2 run flexible_drones_tools takeoff --ros-args -r __ns:=/drone1 -p height:=1.0`
- `ros2 run flexible_drones_tools land --ros-args -r __ns:=/drone1`
- `ros2 run flexible_drones_tools go_to --ros-args -r __ns:=/drone1 -p frame:=0 -p goal_x:=1.0 -p goal_y:=0.0 -p goal_z:=1.0`
- `ros2 run flexible_drones_tools go_to --ros-args -r __ns:=/drone1 -p frame:=1 -p goal_x:=0.5 -p goal_y:=0.0 -p goal_z:=0.0`
- `go_to` frame values: `0`=absolute ENU, `1`=relative map/ENU, `2`=relative body

Trajectory actions:

- `ros2 run flexible_drones_tools upload_trajectory --ros-args -r __ns:=/drone1 -p trajectory_package:=<deployment_package> -p trajectory_file:=figure8.csv`
- `ros2 run flexible_drones_tools start_execute_trajectory --ros-args -r __ns:=/drone1 -p trajectory_id:=0`
- `ros2 run flexible_drones_tools get_trajectory_client --ros-args -p start_frame_id:=gate_C_target_2 -p end_frame_id:=gate_A_target_1`
- `ros2 run flexible_drones_tools get_sampled_trajectory_client --ros-args -p start_frame_id:=gate_C_target_2 -p end_frame_id:=gate_A_target_1`

Trajectory planner note:

- `plan_trajectory_action_server` writes generated CSVs into `output_dir`
- `plan_trajectory_action_server` serves `plan_trajectory` for polynomial pieces and `plan_sampled_trajectory` for sampled points
- `output_dir` defaults to `~/.ros/traj_planner_gen` so generated trajectories never pollute a package source tree; override it to redirect them (e.g. `-p output_dir:=$WORKSPACE_ROOT/src/<deployment_package>/trajectories/traj_planner_gen`)

Trajectory loader note:

- `load_trajectory_action_server` reads named polynomial CSV trajectories from a YAML config at startup
- it serves `load_trajectory` for polynomial pieces and `load_sampled_trajectory` for sampled points
- requests are looked up by `name`
- for requests with an empty name, the lookup name defaults to `Trajectory <trajectory_id>`
- set `plot_trajectory:=true` to plot returned trajectories

Example server and lookup:

- `ros2 run flexible_drones_tools load_trajectory_action_server --ros-args -p config:=$WORKSPACE_ROOT/src/<deployment_package>/trajectories/trajectories.yaml`
- `ros2 run flexible_drones_tools get_trajectory_client --ros-args -p action_name:=load_trajectory -p name:=figure8`
- `ros2 run flexible_drones_tools get_sampled_trajectory_client --ros-args -p action_name:=load_sampled_trajectory -p name:=figure8 -p period_sec:=0.05`

Example config:

```yaml
trajectories:
  - name: figure8
    package: <deployment_package>
    folder: trajectories
    trajectory_file: figure8.csv
```

Trajectory path note:

- clients that accept `trajectory_path` use that direct CSV path before package/folder/file lookup
- write-side helpers should use an explicit writable output directory for generated trajectories
- package-share writes are a development convenience for symlink-install workspaces, not the preferred installed-deployment path


## Pip-Installed Dependencies

Most of this package is covered by ROS dependencies in `package.xml` and can be
installed through the normal rosdep/colcon flow. A few Python dependencies are
intentionally not declared in `package.xml` for rosdep because lab machines
install them into a workspace virtual environment instead; they are listed in
`requirements.txt`:

- `casadi`: required by the trajectory planner
  (`plan_trajectory_action_server` and the planner demos)
- `matplotlib`: required by the plotting tools (`plot_logs`, `plot_trajectory`,
  `visualize_flight`, `visualize_trajectory`, and trajectory previews)
- `PySide6`: required by `status_gui` and `teleop_gui`
- `bokeh`: required by `run_drone_plot`
- `numpy` / `scipy`: also declared via rosdep, but listed in `requirements.txt`
  so virtual environments that do not use system site-packages get them too

Install these with pip (into the workspace venv when using one) before running
those tools:

```
sudo apt update
sudo apt install libxcb-cursor0 libxcb-xinerama0 libxcb-xinput0 libxkbcommon-x11-0
python3 -m pip install -r "$WORKSPACE_ROOT/src/flexible_drones/flexible_drones_tools/requirements.txt"
```

After installation, these are the main GUI-style tools to demo.

Status GUI: condensed view of swarm `Status` messages.

```
clear; ros2 run flexible_drones_tools status_gui --ros-args \
  -p fliers:="['drone1', 'drone2']" -p window_x:=100 -p window_y:=100
```

Teleop GUI: manual velocity control for one drone through `cmd_vel`.

```
clear; ros2 run flexible_drones_tools teleop_gui --ros-args \
  -r __ns:=/drone1 -p window_x:=100 -p window_y:=400
clear; ros2 run joy joy_node --ros-args -r __ns:=/drone1
```

The joystick node is optional. `teleop_gui` listens to `/<drone_name>/joy` when a
physical joystick/gamepad is available. The default GUI takeoff height is
`0.5 m`, with backend teleop vertical command limits normally clamped between
`0.3 m` and `2.5 m`.

Teleop action gate: keep an existing `cmd_vel` publisher alive, then enable
manual control until Enter cancels the action.

```
clear; ros2 run flexible_drones_tools teleop_control --ros-args -r __ns:=/drone1
```

Teleop flight demo: arm, take off, hold a neutral `cmd_vel` heartbeat, enable
teleop control, wait for Enter, then land and disarm.

```
clear; ros2 run flexible_drones_tools teleop_flight_demo --ros-args -r __ns:=/drone1
```

Live browser plot: Bokeh plot with logging enabled by default.

```
clear; ros2 run flexible_drones_tools run_drone_plot --ros-args -r __ns:=/drone1
clear; ros2 run flexible_drones_tools run_drone_plot --ros-args -r __ns:=/drone1 -p logging:=false
```

**Simulation only**:

```
clear; ros2 run flexible_drones_tools teleop_gui --ros-args -r __ns:=/drone1 -p use_sim_time:=true
clear; ros2 run flexible_drones_tools run_drone_plot --ros-args -r __ns:=/drone1 -p use_sim_time:=true
```


## License

This package is licensed under Apache-2.0. Some tools and model workflows were
informed by MIT-licensed upstream work from Bitcraze, Kimberly McGuire, and the
CrazySwarm2 team; retained third-party notices are documented in
[`../NOTICE`](../NOTICE) and relevant source assets.

## Project status

Actively maintained by `robotics@cnu.edu`

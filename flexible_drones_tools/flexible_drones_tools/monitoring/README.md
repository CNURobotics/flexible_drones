# `monitoring` Package Layout

This folder holds the tools for observing drones during and after a run. The
common thread is telemetry: live status, GUI control surfaces, browser plots,
and `.npy` logs written under the workspace `log` directory.

## Runtime Tools

| Module | Console script | Purpose |
| --- | --- | --- |
| [drone_logger.py](drone_logger.py) | `drone_logger` | Records per-drone telemetry topics to chunked NumPy files. |
| [run_drone_plot.py](run_drone_plot.py) | `run_drone_plot` | Starts the live Bokeh plotter from `drone_plot.py`. Logging is enabled by default. |
| [plot_logs.py](plot_logs.py) | `plot_logs` | Offline matplotlib plots for recorded `.npy` logs. |
| [visualize_flight.py](visualize_flight.py) | `visualize_flight` | Republishes recorded pose/odom logs as `nav_msgs/Path` on `flight_trajectory` for RViz. |
| [status_gui.py](status_gui.py) | `status_gui` | Qt status dashboard for one or more drones. |
| [teleop_gui.py](teleop_gui.py) | `teleop_gui` | Qt manual teleoperation GUI for `cmd_vel` and joystick input. |

## Log Format

`drone_logger` can be left running in the background. It waits for the drone's
`status` topic to report `STATUS_ARMED`, starts a fresh log session at that arm
timestamp, flushes the session when the status becomes disarmed, and then waits
for the next arm event.

It writes one directory per drone, normally:

```text
$WORKSPACE_ROOT/log/<drone_name>/
```

Files are named:

```text
<dataset>_<start_time>_<index>.npy
```

The core datasets used by the flight visualizer are:

- `pose`: `time, x, y, z, qx, qy, qz, qw`
- `odom`: `time, x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz`

`visualize_flight` loads `pose` and `odom` chunks for a selected log session,
uses whichever has more rows, and publishes the first eight columns as a
`nav_msgs/Path`.

## Common Commands

Record one drone:

```bash
ros2 run flexible_drones_tools drone_logger --ros-args -r __ns:=/drone1
```

Record several drones:

```bash
ros2 launch flexible_drones_tools drone_loggers.launch.py fliers:="[drone1, drone2]"
```

Visualize the latest recorded flight in RViz:

```bash
ros2 run flexible_drones_tools visualize_flight --ros-args -p drone_name:=drone1
```

Plot recorded logs offline:

```bash
ros2 run flexible_drones_tools plot_logs --drone_name drone1
```

Run live browser plots:

```bash
ros2 run flexible_drones_tools run_drone_plot --ros-args -r __ns:=/drone1
```

## Boundaries

- Planned trajectory CSV/message utilities live in
  [`../trajectories/README.md`](../trajectories/README.md).
- `visualize_flight` is intentionally here, not in `trajectories`, because it
  visualizes recorded flight logs rather than planned polynomial trajectories.
- `plot_logs.py` has the broadest offline log reader; `visualize_flight.py`
  keeps a narrower reader for RViz path publishing.

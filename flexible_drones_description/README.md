# flexible_drones_description

Shared robot and gate description package for Flexible Drones.

This package owns:

- drone xacros and meshes
- gate xacros
- shared RViz base configs
- launch helpers for publishing `robot_description` topics
- a standalone RViz launch for one fleet or a heterogeneous set of names

Related in-repository packages:

- [`flexible_drones_msgs`](../flexible_drones_msgs/README.md) defines the
  shared message/service/action API used by managers that publish these models.
- [`flexible_drones_core`](../flexible_drones_core/README.md) provides the
  `DroneManager` base class that uses these descriptions for robot description
  publication and prop animation.
- [`flexible_drones_tools`](../flexible_drones_tools/README.md) provides RViz,
  logging, plotting, and demo tools that consume these visualization assets.

## Asset layout

Shared geometry lives under `models/`:

- `models/crazyflie/`
- `models/pihawk/`
- `models/gates/`

Each model folder owns its own `urdf/` and `meshes/` content.
Drone model xacros are discovered automatically from:

```text
models/<type>/urdf/<type>_model.urdf.xacro
```

The `models/gates/` folder is reserved for gate assets and is not advertised as
a drone type.

## Reusable launch helpers

`flexible_drones_description.launch_assets` is written to be imported by other packages.

By default, its helpers resolve assets from this package and from the `models/` tree above. A caller can override that layout when needed:

- `package_name`: look up installed assets from a different package
- `package_root`: resolve source-tree assets from a custom package before install
- `model_root`: use a different models directory inside that package

Example:

```python
from flexible_drones_description.launch_assets import resolve_drone_xacro_path

xacro_path = resolve_drone_xacro_path(
    "crazyflie",
    package_name="my_robot_descriptions",
    package_root="/path/to/my_robot_descriptions",
    model_root="robot_models",
)
```

The module-level defaults remain available as `DESCRIPTION_PACKAGE`, `MODEL_ROOT`, and `GATE_MODEL_NAME`, and Flexible Drones launches continue to rely on those defaults.

## Launches

### `bringup_robot_descriptions.launch.py`

Publishes per-drone `robot_description` topics and starts the shared prop spinner for the selected fleet.

**When you need this:** bag replay and other manager-less workflows. All backend managers (`GazeboDroneManager`, `CrazyflieManager`, `PihawkManager`) inherit from `DroneManager` and self-publish `/<drone>/robot_description` with TRANSIENT_LOCAL QoS, so this launch is redundant whenever any manager is running — Gazebo, hardware, or otherwise.

Set `deployment_package:=<deployment_package>` or
`FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE=<deployment_package>` to use the roster and
deployment setup from that package (this workspace ships
`chris_serc_deployment`).

Example using the default `mixed` deployment:

```bash
ros2 launch flexible_drones_description bringup_robot_descriptions.launch.py \
  deployment_package:=<deployment_package>
```

Example using a different deployment:

```bash
ros2 launch flexible_drones_description bringup_robot_descriptions.launch.py \
  deployment_package:=<deployment_package> \
  deployment:=crazyflies \
  fliers:="[cf1, cf2, cf3]"
```

Arguments:

- `fliers`: YAML list string such as `[cf1, px1]`; empty means all enabled drones listed in the deployment setup, while a non-empty list means exactly that subset
- `roster`: roster YAML path, default `config/roster.yaml`
- `roster_package`: optional package used to resolve the roster path; empty uses `deployment_package`
- `deployment`: deployment folder name, default `mixed`
- `deployment_setup`: optional explicit setup path; empty means `deployments/<deployment>/setup.yaml`
- `deployment_package`: package used to resolve the roster and deployment setup paths, default `$FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE` or empty

### `bringup_rviz.launch.py`

Starts RViz with one `RobotModel` display per selected drone name.

For all enabled drones listed in the default deployment setup:

```bash
ros2 launch flexible_drones_description bringup_rviz.launch.py \
  deployment_package:=<deployment_package>
```

For the staged Crazyflie names in a different named deployment:

```bash
ros2 launch flexible_drones_description bringup_rviz.launch.py \
  deployment_package:=<deployment_package> \
  deployment:=crazyflies \
  fliers:="[cf1, cf2, cf3]"
```

For a heterogeneous combination across multiple backends, replay sources, or logs:

```bash
ros2 launch flexible_drones_description bringup_rviz.launch.py \
  fliers:="[cf1, cf2, px1]"
```

Arguments:

- `fliers`: YAML list string of names to visualize; empty means all enabled drones listed in the deployment setup, while a non-empty list directly visualizes exactly those names and ignores roster/deployment arguments
- `roster`: roster YAML path, default `config/roster.yaml`
- `roster_package`: optional package used to resolve the roster path; empty uses `deployment_package`
- `deployment`: deployment folder name, default `mixed`
- `deployment_setup`: optional explicit setup path; empty means `deployments/<deployment>/setup.yaml`
- `deployment_package`: package used to resolve the roster and deployment setup paths, default `$FLEXIBLE_DRONES_DEPLOYMENT_PACKAGE` or empty
- `rviz_config`: RViz base config in `rviz/` such as `empty.rviz` or `gates.rviz`
- `window_x`: optional RViz window X position in screen pixels
- `window_y`: optional RViz window Y position in screen pixels
- `window_width`: optional RViz window width in screen pixels
- `window_height`: optional RViz window height in screen pixels

### `publish_drone_course.launch.py`

Publishes a configured gate course for hardware, replay, or already-running
systems. This starts the gate static transforms, robot descriptions, and label
markers only; it does not spawn anything in Gazebo and does not open RViz.

The default course is `config/serc_bag.yaml`:

```bash
ros2 launch flexible_drones_description publish_drone_course.launch.py
```

For the smaller 0.5 m gate layout:

```bash
ros2 launch flexible_drones_description publish_drone_course.launch.py \
  course_config:=small_gates.yaml
```

### `rviz_drone_course.launch.py`

Publishes a configured gate course and optionally launches RViz with
`gates.rviz`.

For the SERC BAG course with 1.0 m pipe segments and a 0.5 m first vertical segment:

```bash
ros2 launch flexible_drones_description rviz_drone_course.launch.py
```

To add the course to an RViz session you already launched:

```bash
ros2 launch flexible_drones_description rviz_drone_course.launch.py \
  launch_rviz:=false
```

Arguments:

- `course_config`: course YAML filename in `config/`, or an absolute path
- `frame_id`: parent frame for gate static transforms, default `map`
- `rviz_config`: RViz config to open when `launch_rviz:=true`, default `gates.rviz`
- `launch_rviz`: whether to start RViz as part of this launch
- `label_z`: optional label height override in each gate frame
- `label_rate`: gate label `MarkerArray` publish rate in Hz

## Shared `FLIERS` environment variable

When working across several terminals, it is convenient to define the selected fleet once:

```bash
export FLIERS="['cf1', 'cf2', 'px1']"
```

Then reuse it in every launch command:

```bash
ros2 launch flexible_drones_description bringup_rviz.launch.py fliers:="$FLIERS"
```

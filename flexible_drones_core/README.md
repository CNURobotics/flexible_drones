# flexible_drones_core

Core orchestration package and abstract per-drone manager API for Flexible Drones.

Backend implementations live in separate packages selected by manager profiles. This package provides the orchestrator, registry parsing, process supervision, and the shared `DroneManager` base class.

Related in-repository packages:

- [`flexible_drones_msgs`](../flexible_drones_msgs/README.md) defines the ROS
  interfaces exposed by `DroneManager` and used by the orchestrator.
- [`flexible_drones_description`](../flexible_drones_description/README.md)
  provides shared robot models and launch assets used by managers.
- [`flexible_drones_tools`](../flexible_drones_tools/README.md) provides
  operator clients, demos, logging, plotting, and trajectory tools that call
  the core manager interfaces.

> **Release note:** The orchestrator and `/swarm/...` fan-out path are
> experimental and have not been release-tested. Users doing basic bringup should
> follow the Crazyflie, PiHawk, or Gazebo package README launch directions
> instead of starting `drone_orchestrator_node`.

## Components

- `DroneOrchestratorNode`: Experimental node that loads the registry, applies `selected_drones`, supervises the desired set of managers, and exposes the `/swarm/...` admin and fan-out services.
- `ManagerSupervisor`: Routes each `DroneSpec` to either an embedded manager factory or a subprocess manager.
- `ProcessManager`: Starts, stops, and restarts subprocess-backed managers with backoff.
- `Registry` / `DroneSpec`: Parse the normalized orchestrator registry and preserve per-drone ROS params, namespace, and extra args. In the standard stack, `chris_serc_deployment/launch/swarm.launch.py` materializes that registry from roster + deployment data at launch time.
- `ManagerConfig`: Resolved manager construction contract shared by direct launch, subprocess launch, embedded factories, and multi-manager processes.
- `DroneManager`: Abstract base class for per-drone managers that expose the common services, topics, and actions from `flexible_drones_msgs`.

## Manager Profiles

The orchestrator does not own backend-specific manager profiles. Launch/configuration must provide a `manager_profiles_yaml` mapping from each registry `manager_type` to the backend package, executable, and optional embedded factory.

Those backend packages are selected by deployment profile/configuration rather than declared as unconditional dependencies of `flexible_drones_core`.

### Namespace Model

Manager profiles also define how per-drone ROS names are formed. Managers based on the shared `DroneManager` class create interfaces with the drone name in the topic/service/action name, such as `cf1/arm`, `cf1/takeoff`, and `cf1/odom`. For those managers, set `manager_owns_drone_prefix: true` so subprocess launch keeps the node namespace at `/`; otherwise ROS remapping can double the prefix into names like `/cf1/cf1/arm`.

Leave `manager_owns_drone_prefix` false only for manager implementations that publish namespace-relative names such as `arm`, `takeoff`, and `odom` and are intended to be launched inside the drone namespace. The standard Crazyflie, PiHawk, and Gazebo profiles all use the shared `DroneManager` naming model.

## Embedded Managers

### Why embedded mode exists

Most backends run as subprocesses (`launch_mode: subprocess`): the orchestrator spawns `ros2 run <package> <executable>` per drone and monitors the PID. This keeps backend code fully isolated.

Crazyflie managers cannot use the subprocess model. `cflib` initialises its radio link drivers into a global driver table (`cflib.crtp.CLASSES`) once at process start. Spawning a fresh subprocess per drone — or splitting the radio state across processes — would require each subprocess to independently claim the same USB/radio hardware, which the drivers do not support. All `CrazyflieManager` instances must therefore live **in the same process** as the radio driver initialisation.

### How the plugin mechanism works

Profiles with `launch_mode: embedded` must also declare:

- `embedded_factory_module` — fully-qualified Python module to import at startup
- `embedded_factory_func` — name of the callable within that module

At orchestrator startup `_build_embedded_factories()` loads each factory via `importlib.import_module` and wraps it in a closure that binds the orchestrator node. When the supervisor reconciles a drone with that `manager_type` it resolves the `DroneSpec` into a `ManagerConfig`, calls `factory(config)`, and the wrapper resolves that to `factory_func(config, node)`.

The factory function is responsible for applying backend-specific enrichment to the common `ManagerConfig` (type defaults, `robot_description`, group mask, etc.), constructing the manager object, and returning an `EmbeddedHandle(start, stop, is_alive)` that the supervisor uses for lifecycle control. Because the factory lives in the backend package, core has zero knowledge of backend-specific imports or config files.

### Adding a new embedded backend

1. Create a factory module in your backend package:

   ```python
   # my_package/manager/my_embedded_factory.py
   from flexible_drones_core.manager.config import ManagerConfig
   from flexible_drones_core.orchestrator.manager_supervisor import EmbeddedHandle

   def create_embedded_handle(config: ManagerConfig, node) -> EmbeddedHandle:
       from my_package.manager.my_manager import MyManager
       ros_params = config.to_ros_parameters()
       manager = MyManager(
           drone_name=config.drone_name,
           node=node,
           ros_params=ros_params,
       )
       return EmbeddedHandle(
           start=manager.start_up,
           stop=manager.shut_down,
           is_alive=lambda: manager.connected,
       )
   ```

2. In `manager_profiles_yaml`, add `launch_mode: embedded` plus the two factory keys:

   ```yaml
   my_manager:
     package: my_package
     executable: my_manager_node
     launch_mode: embedded
     embedded_factory_module: my_package.manager.my_embedded_factory
     embedded_factory_func: create_embedded_handle
   ```

3. No changes to `flexible_drones_core` are required.

## Registry Schema

The orchestrator consumes a normalized registry file using the `drones:` list schema. That registry is an orchestrator-facing runtime artifact; `chris_serc_deployment/launch/swarm.launch.py` materializes it from the deployment package at launch time instead of treating a hand-maintained registry file as the primary project configuration.

Example:

```yaml
drones:
  - name: cf1
    uri: radio://0/80/2M/E7E7E7E7E6
    type: crazyflie
    manager_type: example_manager
    enabled: true
    ros_namespace: /cf1
    ros_params: {}
    extra_args: []
```

## Launch Notes

Launch files are in `chris_serc_deployment/launch/`, not here.

- These orchestrator launch files are experimental and not recommended for the
  basic release workflow. Use the backend-specific launch files for normal
  Crazyflie, PiHawk, or Gazebo bringup.
- `swarm.launch.py` resolves the shared deployment roster + setup data, materializes a temporary orchestrator registry, and starts the orchestrator.
- The default selection is `chris_serc_deployment/config/roster.yaml` plus `chris_serc_deployment/deployments/mixed/setup.yaml`.
- `swarm.launch.py` expects the backend package for every selected `manager_type` to be present in the workspace and fails fast if one is missing.
- `hybrid_swarm.launch.py` assumes Gazebo is already running. When Gazebo drones are selected, it renders model descriptions, spawns models, starts the ROS/Gazebo bridge, and then launches the orchestrator.
- Hybrid Gazebo launches require `flexible_drones_gazebo`, `xacro`, `ros_gz_sim`, and `ros_gz_bridge` in the workspace.

## Testing

Run the lightweight unit test suite with:

```bash
colcon test --packages-select flexible_drones_core
colcon test-result --verbose
```

The current suite covers registry parsing, process-manager command construction, manager-supervisor routing, and selected pure-logic helpers in the orchestrator.

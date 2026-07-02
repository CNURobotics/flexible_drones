# Trajectory Generators

These scripts generate trajectory CSVs (drone coefficient order) offline. They are
**run from source**, not installed as `ros2 run` entry points. Generated files are
written to the git-ignored [`output/`](output/) folder by default.

## Prerequisites

Source the workspace so `flexible_drones_tools` is importable:

```
source "$WORKSPACE_ROOT/install/setup.bash"
```

## Running

Each script has a shebang and the executable bit, so run it directly from this
folder:

```
cd "$WORKSPACE_ROOT/src/flexible_drones/flexible_drones_tools/flexible_drones_tools/trajectories/generators"
./generate_circle_trajectory.py
./generate_spiral_trajectory.py
./generate_star_trajectory.py
./generate_figure8_trajectory.py
./generate_x_frequency_sweep_trajectory.py
./generate_z_frequency_sweep_trajectory.py
./cnu_sails.py
```

Equivalently, run as a module from anywhere (workspace sourced):

```
python3 -m flexible_drones_tools.trajectories.generators.generate_circle_trajectory
```

## Output

By default each shape generator writes `output/<shape>_trajectory.csv`; `cnu_sails`
writes `output/cnu_sail0.csv` and `output/cnu_sail1.csv`. Pass an explicit path to
override, and use `--help` for per-script options:

```
./generate_circle_trajectory.py --num-points 800 --num-segments 30
./generate_circle_trajectory.py /tmp/my_circle.csv
./generate_circle_trajectory.py --ramp-duration 0.0
./generate_figure8_trajectory.py --radius 1.5 --z-floor 1.0 --z-ceil 1.8
./generate_spiral_trajectory.py --z-floor 1.0 --z-ceil 1.0 --ramp-duration 0.0
./generate_spiral_trajectory.py --no-reverse --ramp-duration 0.0
./generate_star_trajectory.py --vx 0.75 --alpha 0.05 --corner-sample-multiplier 2.0
./generate_x_frequency_sweep_trajectory.py --start-hz 0.1 --end-hz 1.0
./generate_z_frequency_sweep_trajectory.py --min-z 1.0 --max-z 2.0
```

The `output/` folder is git-ignored (a `.gitignore` keeps the empty folder in the
tree); generated CSVs are local build artifacts and are not committed.

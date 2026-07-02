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

import subprocess
import shutil
import sys
from importlib.resources import files

from flexible_drones_tools.monitoring.cli_help import print_usage_if_requested


USAGE = """\
Usage:
  ros2 run flexible_drones_tools run_drone_plot --ros-args -r __ns:=/drone1

Common examples:
  ros2 run flexible_drones_tools run_drone_plot --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools run_drone_plot --duration 20 --ros-args -r __ns:=/drone1
  ros2 run flexible_drones_tools run_drone_plot --ros-args -r __ns:=/drone1 -p use_sim_time:=true
  ros2 run flexible_drones_tools run_drone_plot --ros-args -r __ns:=/drone1 -p logging:=false

ROS 2 argument notes:
  ROS-specific arguments come after '--ros-args'.
  Set parameters with '-p name:=value'.
  Set the node namespace with '-r __ns:=/drone1'.
  Set the node name with '-r __node:=custom_name'.

Forwarded plot arguments:
  --duration SECONDS
    History window to plot in seconds. Default: 10.0.
"""


def main(args=None):
    if print_usage_if_requested(USAGE, args):
        return

    cli_args = list(sys.argv[1:] if args is None else args)
    script_path = str(files('flexible_drones_tools.monitoring').joinpath('drone_plot.py'))
    if shutil.which('bokeh') is None:
        raise SystemExit(
            'run_drone_plot requires bokeh: python3 -m pip install bokeh\n'
            'See flexible_drones_tools/README.md for optional GUI dependencies.'
        )

    print('Launching drone_plot tool ...')
    cmd = ['bokeh', 'serve', '--show', script_path, '--args'] + cli_args
    print(f'     cmd: <{cmd}>')
    try:
        completed = subprocess.run(cmd)
    except KeyboardInterrupt:
        print('Subprocess interrupted by Ctrl-C. Exiting cleanly.')
        return 130

    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())

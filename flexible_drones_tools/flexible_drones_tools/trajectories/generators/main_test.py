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

# ----Import your trajectory generation files below----
from flexible_drones_tools.trajectories.generators import generate_spiral_trajectory
from flexible_drones_tools.trajectories.utilities import split_trajectory
# import generate_circle_trajectory
# import generate_star_trajectory
# -----------------------------------------------------


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # replace right side of the equals below to match the file you imported, and the class in the file.
    my_traj = generate_spiral_trajectory.GenerateSpiralTrajectory()

    write = True  # Do you want to save this file? Yes=True, No=False
    path = '../my_spiral_trajectory.csv'  # Where do you want to save this file? Provide path as a string.

    # ------You are not advised to change anything below this line-------------

    t, x, y, z, yaw = my_traj.generate_trajectory()
    my_traj.run_all_plots(t, x, y, z, yaw)
    my_traj = split_trajectory.SegmentFit(t, x, y, z, yaw)
    my_traj = my_traj.split_fit_and_plot(write, path)

    plt.show()

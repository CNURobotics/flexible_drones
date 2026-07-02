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

import math

import numpy as np

from flexible_drones_tools.trajectories.generators.generate_spiral_trajectory import GenerateSpiralTrajectory
from flexible_drones_tools.trajectories.utilities.split_trajectory import SegmentFit

# Finds corresponding polynomial coefficients for the provided trajectory segments


def build_min_snap_trajectory_from_segments(raw_segments):
    import cvxpy as cp

    DEGREE = 7  # polynomial degree
    num_coeffs = DEGREE + 1  # x^0 - x^7
    n = len(raw_segments)  # number of segments
    axes = ['x', 'y', 'z']  # the axes being solved for

    # Creates a dictionary of CVXPY optimization variables for each axis.
    # Each entry is an n×8n×8 matrix, where each row holds the 8 coefficients for a segment.
    coeffs = {axis: cp.Variable((n, num_coeffs)) for axis in axes}
    constraints = []
    objective = 0

    # Creates a basis vector to help find the derivative of the polynomial at a specific time
    def get_basis(order, t):
        return np.array([
            math.prod(range(i - order + 1, i + 1)) * t**(i - order) if i >= order else 0
            for i in range(num_coeffs)
        ])

    for axis in axes:
        for i in range(n):
            # Separate the segment at i
            t_seg, x_seg, y_seg, z_seg, _ = raw_segments[i]
            dt = t_seg[-1] - t_seg[0]  # Length of the segment

            # Get the start and end positions of the this segment for the current axis.
            x0 = {'x': x_seg[0], 'y': y_seg[0], 'z': z_seg[0]}[axis]
            x1 = {'x': x_seg[-1], 'y': y_seg[-1], 'z': z_seg[-1]}[axis]

            # Get the basis vector at the start and end of the segment.
            p0 = get_basis(0, 0)
            p1 = get_basis(0, dt)

            # Force this polynomial to match the start and end pos exactly.
            constraints += [
                coeffs[axis][i] @ p0 == x0,
                coeffs[axis][i] @ p1 == x1
            ]

            # Snap cost
            Q = np.zeros((num_coeffs, num_coeffs))
            for i1 in range(4, num_coeffs):
                for i2 in range(4, num_coeffs):
                    Q[i1, i2] = (
                        math.prod(range(i1 - 3, i1 + 1)) *
                        math.prod(range(i2 - 3, i2 + 1)) *
                        dt**(i1 + i2 - 7) / (i1 + i2 - 7 + 1)
                    )
            # Adds the snap cost of this segment to the total optimization objective.
            objective += cp.quad_form(coeffs[axis][i], Q)

            # Position limits
            # Sample 10 points along its duration and ensure the position
            # stays within the specified physical bounds.
            bounds = {'x': (-1.5, 1.5), 'y': (-1.5, 1.5), 'z': (0.0, 2.0)}[axis]
            for t_sample in np.linspace(0, dt, 10):
                p = get_basis(0, t_sample)
                constraints.append(coeffs[axis][i] @ p <= bounds[1])
                constraints.append(coeffs[axis][i] @ p >= bounds[0])

    # Continuity constraints
    # Ensure that the value at the end of the previous segment matches that at the beginning of the next
    for axis in axes:
        for i in range(n - 1):
            t1 = raw_segments[i][0][-1] - raw_segments[i][0][0]
            for order in range(1, 5):  # velocity to snap
                d1 = get_basis(order, t1)
                d2 = get_basis(order, 0)
                constraints.append(coeffs[axis][i] @ d1 == coeffs[axis][i + 1] @ d2)

    # Solve the problem; minimize total snap subject to the constraints
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    # Throw an error if a solution wasnt found
    if problem.status != cp.OPTIMAL:
        raise RuntimeError('Solver failed!')

    # Return as four lists: x, y, z, yaw (not optimized, just interpolated)
    x_coeffs = [coeffs['x'].value[i].tolist() for i in range(n)]
    y_coeffs = [coeffs['y'].value[i].tolist() for i in range(n)]
    z_coeffs = [coeffs['z'].value[i].tolist() for i in range(n)]
    yaw_coeffs = [
        np.polyfit(raw_segments[i][0] - raw_segments[i][0][0], raw_segments[i][4], DEGREE)[::-1].tolist()
        for i in range(n)
    ]

    return x_coeffs, y_coeffs, z_coeffs, yaw_coeffs
    # return coeffs


# def eval_derivative_poly(coeff_array, t, order):
#     if hasattr(coeff_array, 'value') and coeff_array.value is not None:
#         coeff_array = coeff_array.value
#     coeff_array = np.asarray(coeff_array).flatten()

#     result = 0.0
#     for i in range(order, len(coeff_array)):
#         coeff = coeff_array[i]
#         scale = math.prod(range(i - order + 1, i + 1))  # i! / (i - order)!
#         result += coeff * scale * (t ** (i - order))
#     return result


# def plot_trajectory_derivatives_from_segments(raw_segments, coeffs):
#     axes = ['x', 'y', 'z']
#     derivatives = ['position', 'velocity', 'acceleration', 'jerk', 'snap']
#     num_derivatives = len(derivatives)

#     data = {axis: [[] for _ in range(num_derivatives)] for axis in axes}
#     t_all, x_all, y_all, z_all = [], [], [], []

#     for i, (t_seg, x_seg, y_seg, z_seg, _) in enumerate(raw_segments):
#         dt = t_seg[-1] - t_seg[0]
#         t_local = np.linspace(0, dt, 50)
#         t_global = np.linspace(t_seg[0], t_seg[-1], 50)
#         t_all.extend(t_global)
#         x_all.extend([eval_derivative_poly(coeffs['x'][i], t, 0) for t in t_local])
#         y_all.extend([eval_derivative_poly(coeffs['y'][i], t, 0) for t in t_local])
#         z_all.extend([eval_derivative_poly(coeffs['z'][i], t, 0) for t in t_local])

#         for axis in axes:
#             coeff_array = coeffs[axis][i]
#             for order in range(num_derivatives):
#                 data[axis][order].extend([
#                     eval_derivative_poly(coeff_array, t, order) for t in t_local
#                 ])

#     # Plotting each axis in its own figure
#     for axis in axes:
#         fig, axs = plt.subplots(num_derivatives, 1, figsize=(10, 12), sharex=True)
#         fig.suptitle(f'{axis.upper()} Derivatives vs Time', fontsize=16)
#         for i in range(num_derivatives):
#             axs[i].plot(t_all, data[axis][i], label=f'{derivatives[i]}')
#             axs[i].set_ylabel(derivatives[i].capitalize())
#             axs[i].grid(True)
#             axs[i].legend()
#         axs[-1].set_xlabel("Time [s]")
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#         # 3D trajectory
#     fig = plt.figure(figsize=(8, 6))
#     ax3d = fig.add_subplot(111, projection='3d')
#     ax3d.plot(x_all, y_all, z_all, label='Trajectory')
#     ax3d.set_xlabel('X')
#     ax3d.set_ylabel('Y')
#     ax3d.set_zlabel('Z')
#     ax3d.set_title('3D Trajectory')
#     ax3d.set_xlim(-2, 2)
#     ax3d.set_ylim(-2, 2)
#     ax3d.set_zlim(0, 2.5)
#     ax3d.legend()
#     plt.tight_layout()


def eval_derivative_poly(coeffs, t, order):
    n = len(coeffs)
    return sum(
        coeffs[i] * math.prod(range(i - order + 1, i + 1)) * t**(i - order)
        if i >= order else 0
        for i in range(n)
    )


def plot_trajectory_derivatives_from_coeffs(raw_segments, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs):
    import matplotlib.pyplot as plt

    axes = ['x', 'y', 'z', 'yaw']
    labels = ['Position', 'Velocity', 'Acceleration', 'Jerk', 'Snap']
    num_derivatives = 5
    colors = plt.cm.get_cmap('tab20', len(raw_segments))

    data = {axis: [[] for _ in range(num_derivatives)] for axis in axes}
    t_all = []

    x_all, y_all, z_all = [], [], []

    for i, (t_seg, _, _, _, _) in enumerate(raw_segments):
        dt = t_seg[-1] - t_seg[0]
        t_local = np.linspace(0, dt, 50)
        t_global = np.linspace(t_seg[0], t_seg[-1], 50)
        t_all.extend(t_global)

        coeff_dict = {
            'x': x_coeffs[i],
            'y': y_coeffs[i],
            'z': z_coeffs[i],
            'yaw': yaw_coeffs[i]
        }

        x_all.extend([eval_derivative_poly(coeff_dict['x'], t, 0) for t in t_local])
        y_all.extend([eval_derivative_poly(coeff_dict['y'], t, 0) for t in t_local])
        z_all.extend([eval_derivative_poly(coeff_dict['z'], t, 0) for t in t_local])

        for axis in axes:
            for order in range(num_derivatives):
                values = [
                    eval_derivative_poly(coeff_dict[axis], t, order) for t in t_local
                ]
                data[axis][order].append((t_global, values, colors(i)))

    # Plot axis derivatives
    for axis in axes:
        fig, axs = plt.subplots(num_derivatives, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f'{axis.upper()} Derivatives vs Time', fontsize=16)
        for j in range(num_derivatives):
            for t_global, values, color in data[axis][j]:
                axs[j].plot(t_global, values, color=color)
            axs[j].set_ylabel(labels[j])
            axs[j].grid(True)
        axs[-1].set_xlabel('Time [s]')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 3D trajectory plot
    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot(x_all, y_all, z_all, label='Trajectory', color='purple')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('3D Trajectory')
    ax3d.set_xlim(-2, 2)
    ax3d.set_ylim(-2, 2)
    ax3d.set_zlim(0, 2.5)
    ax3d.legend()
    plt.tight_layout()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t, x, y, z, yaw = GenerateSpiralTrajectory().generate_trajectory()
    my_traj = SegmentFit(t, x, y, z, yaw)
    segments, durations = my_traj.split_trajectory()
    x_c, y_c, z_c, yaw_c = build_min_snap_trajectory_from_segments(segments)
    plot_trajectory_derivatives_from_coeffs(segments, x_c, y_c, z_c, yaw_c)
    plt.show()

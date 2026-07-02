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


try:
    from casadi import DM, Opti, norm_2
except ImportError as exc:
    raise ImportError('casadi not found; run: pip3 install casadi') from exc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Parameters
N = 20  # number of discretization points
dim = 3  # 2D path (x, y)
start = np.array([0, 0, 0])
goal = np.array([5, 4, 6])
obstacle = {
    'center': DM([2.0, 2.0, 1.0]),
    'size': DM([1.0, 1.0, 2.0])  # width, depth, height
}

# Optimization
opti = Opti()

# Decision variables: waypoints
X = opti.variable(dim, N)

# Set start and end constraints
opti.subject_to(X[:, 0] == start)
opti.subject_to(X[:, -1] == goal)

obs_min = obstacle['center'] - obstacle['size'] / 2
obs_max = obstacle['center'] + obstacle['size'] / 2
# Penalty weight
penalty_weight = 1000.0

# Objective: minimize total path length plus obstacle avoidance penalty
path_length = 0
for i in range(N - 1):
    dx = X[:, i + 1] - X[:, i]
    path_length += norm_2(dx)

for i in range(1, N - 1):
    inside_x = (X[0, i] >= obs_min[0]) & (X[0, i] <= obs_max[0])
    inside_y = (X[1, i] >= obs_min[1]) & (X[1, i] <= obs_max[1])
    inside_z = (X[2, i] >= obs_min[2]) & (X[2, i] <= obs_max[2])
    inside = inside_x & inside_y & inside_z
    path_length += penalty_weight * inside

opti.minimize(path_length)

# Optional: set initial guess to straight line
for i in range(N):
    alpha = i / (N - 1)
    opti.set_initial(X[:, i], (1 - alpha) * start + alpha * goal)

# Solve
opti.solver('ipopt')
sol = opti.solve()

# Extract solution
X_opt = sol.value(X)


def draw_box(ax, center, size, color='red', alpha=0.3):
    """Draw a 3D box (cuboid) on the given axis."""
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2

    # Define the 8 corners of the box
    corners = np.array([
        [cx - sx, cy - sy, cz - sz],
        [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz],
        [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz],
        [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz],
        [cx - sx, cy + sy, cz + sz]
    ])

    # Define the 6 faces using corner indices
    faces = [
        [corners[j] for j in [0, 1, 2, 3]],  # bottom
        [corners[j] for j in [4, 5, 6, 7]],  # top
        [corners[j] for j in [0, 1, 5, 4]],  # front
        [corners[j] for j in [2, 3, 7, 6]],  # back
        [corners[j] for j in [1, 2, 6, 5]],  # right
        [corners[j] for j in [4, 7, 3, 0]]   # left
    ]

    box = Poly3DCollection(faces, facecolors=color, edgecolors='k', linewidths=0.5, alpha=alpha)
    ax.add_collection3d(box)


# Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.plot(X_opt[0, :], X_opt[1, :], X_opt[2, :], 'b.-', label='Shortest Path')
ax.scatter(start[0], start[1], start[2], c='green', s=20, label='Start')
draw_box(ax, center=(1, 1, 1), size=(1, 0.5, 2.5), color='purple', alpha=0.4)
ax.scatter(goal[0], goal[1], goal[2], c='red', s=20, label='Goal')
plt.title('Shortest Path Using CasADi')
ax.grid(True)
ax.legend()
plt.axis('equal')
plt.show()


# from casadi import *
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from math import comb
# import numpy as np

# # Problem settings
# n_segments = 4
# poly_order = 7  # Minimum-snap requires at least 7th order
# dim = 3
# segment_time = 1.0  # seconds per segment

# opti = Opti()

# # Polynomial coefficients
# C = opti.variable(dim, (poly_order + 1) * n_segments)

# def get_Ck(C, k):
#     return C[:, k*(poly_order+1):(k+1)*(poly_order+1)]

# def poly_t(t, order):
#     return vertcat(*[t**i for i in range(order + 1)])

# def dpoly_t(t, order, d):
#     return vertcat(*[
#         comb(i, d) * np.math.factorial(d) * t**(i - d) if i >= d else DM(0)
#         for i in range(order + 1)
#     ])

# def traj(Ck, t):
#     return mtimes(Ck, poly_t(t, poly_order))

# def traj_derivative(Ck, t, d):
#     return mtimes(Ck, dpoly_t(t, poly_order, d))

# # Start and end points
# x0 = DM([0, 0, 0])
# xf = DM([5, 5, 1])

# # Initial guess: linear initialization
# for k in range(n_segments):
#     alpha = k / (n_segments - 1)
#     opti.set_initial(get_Ck(C, k), np.outer(x0 + alpha * (xf - x0), np.ones(poly_order + 1)))

# # Start constraints
# C0 = get_Ck(C, 0)
# opti.subject_to(traj(C0, 0) == x0)
# v0 = traj_derivative(C0, 0, 1)
# opti.subject_to(dot(v0, v0) <= 4)
# # End constraints
# Cf = get_Ck(C, n_segments - 1)
# opti.subject_to(traj(Cf, segment_time) == xf)
# v1 = traj_derivative(Cf, segment_time, 1)
# opti.subject_to(dot(v1, v1) <= 4)
# opti.subject_to(dot(traj_derivative(Cf, segment_time, 2),
#                     traj_derivative(Cf, segment_time, 2)) <= 1e-4)
# opti.subject_to(dot(traj_derivative(Cf, segment_time, 3),
#                     traj_derivative(Cf, segment_time, 3)) <= 1e-4)

# # Continuity constraints
# for k in range(n_segments - 1):
#     Ck = get_Ck(C, k)
#     Ck1 = get_Ck(C, k + 1)
#     for d in [0, 1, 2, 4]:
#         opti.subject_to(traj_derivative(Ck, segment_time, d) == traj_derivative(Ck1, 0, d))

# # Gate constraints
# gates = [
#     {"center": DM([1.5, 1.0, 1.0]), "size": DM([0.5, 0.5, 1.5]), "dir": "+"},
#     {"center": DM([2.5, 2.0, 1.0]), "size": DM([0.5, 0.5, 1.5]), "dir": "-"},
#     {"center": DM([3.5, 3.0, 1.0]), "size": DM([0.5, 0.5, 1.5]), "dir": "+"},
#     {"center": DM([4.5, 4.0, 1.0]), "size": DM([0.5, 0.5, 1.5]), "dir": "-"},
# ]

# for i, gate in enumerate(gates):
#     Ck = get_Ck(C, i)
#     pk = traj(Ck, segment_time / 2)
#     for d in range(3):
#         opti.subject_to(pk[d] >= gate["center"][d] - gate["size"][d] / 2)
#         opti.subject_to(pk[d] <= gate["center"][d] + gate["size"][d] / 2)
#     x_gate = float(gate["center"][0])
#     if gate["dir"] == "+":
#         opti.subject_to(pk[0] <= x_gate)
#     else:
#         opti.subject_to(pk[0] >= x_gate)

# # Coefficient bounds to prevent numerical instability
# opti.subject_to(opti.bounded(-100, C, 100))

# # Objective: minimize snap
# snap_cost = 0
# for k in range(n_segments):
#     Ck = get_Ck(C, k)
#     for t in np.linspace(0, segment_time, 5):
#         snap = traj_derivative(Ck, t, 4)
#         snap_cost += dot(snap, snap)
# opti.minimize(snap_cost)

# # IPOPT settings
# opti.solver("ipopt", {
#     "print_time": True,
#     "ipopt.tol": 1e-3,
#     "ipopt.linear_solver": "mumps",
#     "ipopt.print_level": 5
# })


# print("Initial C0:\n", opti.debug.value(C))
# # Solve
# sol = opti.solve()

# # Extract trajectory
# traj_points = []
# for k in range(n_segments):
#     Ck = sol.value(get_Ck(C, k))
#     for t in np.linspace(0, segment_time, 10):
#         pt = np.dot(Ck, np.array([t**i for i in range(poly_order + 1)]))
#         traj_points.append(pt)
# traj_points = np.array(traj_points)

# print("Trajectory Points:\n", traj_points)

# # Plotting
# def draw_box(ax, center, size, color='cyan', alpha=0.3):
#     c = center
#     s = size / 2
#     corners = [
#         [c[0]-s[0], c[1]-s[1], c[2]-s[2]],
#         [c[0]+s[0], c[1]-s[1], c[2]-s[2]],
#         [c[0]+s[0], c[1]+s[1], c[2]-s[2]],
#         [c[0]-s[0], c[1]+s[1], c[2]-s[2]],
#         [c[0]-s[0], c[1]-s[1], c[2]+s[2]],
#         [c[0]+s[0], c[1]-s[1], c[2]+s[2]],
#         [c[0]+s[0], c[1]+s[1], c[2]+s[2]],
#         [c[0]-s[0], c[1]+s[1], c[2]+s[2]],
#     ]
#     faces = [
#         [corners[i] for i in [0, 1, 2, 3]],
#         [corners[i] for i in [4, 5, 6, 7]],
#         [corners[i] for i in [0, 1, 5, 4]],
#         [corners[i] for i in [2, 3, 7, 6]],
#         [corners[i] for i in [1, 2, 6, 5]],
#         [corners[i] for i in [4, 7, 3, 0]],
#     ]
#     box = Poly3DCollection(faces, facecolors=color, edgecolors='k', linewidths=0.5, alpha=alpha)
#     ax.add_collection3d(box)

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2], label='Trajectory', color='blue')

# for gate in gates:
#     draw_box(ax, np.array(gate["center"]), np.array(gate["size"]), color='orange', alpha=0.4)
#     direction = 1 if gate["dir"] == "+" else -1
#     center = gate["center"]
#     ax.quiver(center[0] - direction * 0.3, center[1], center[2],
#               direction * 0.5, 0, 0, color='red', arrow_length_ratio=0.4)

# ax.scatter(*x0, color='green', s=100, label='Start')
# ax.scatter(*xf, color='red', s=100, label='Goal')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Minimum Snap Trajectory through Gates')
# ax.legend()
# ax.grid(True)
# ax.view_init(elev=30, azim=135)
# plt.tight_layout()
# plt.show()


# from casadi import *
# import matplotlib.pyplot as plt
# from math import comb  # Python 3.8+
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import numpy as np

# # Problem settings
# n_segments = 4
# poly_order = 7  # Minimum-snap needs at least 7th order
# dim = 3 # num spacial dimentions
# segment_time = 1.0  # seconds per segment

# opti = Opti() # create an optimization problem container

# # Polynomial coefficients per segment
# # decision variable tensor (shape: 3 × 8 × 4) containing polynomial coefficients
# # for each axis (x, y, z), degree (0-7), and segment (0-3).

# # Define as 2D
# C = opti.variable(dim, (poly_order + 1) * n_segments)

# # Helper to access coefficients for segment k
# def get_Ck(C, k):
#     return C[:, k*(poly_order+1):(k+1)*(poly_order+1)]

# # Constructs a vector of monomials: [1,t,t2,...,tn][1,t,t2,...,tn] to evaluate the polynomial at time t.
# def poly_t(t, order):
#     return vertcat(*[t**i for i in range(order + 1)])

# # Derivative basis: returns the coefficients for the d-th derivative of the polynomial at time t.
# def dpoly_t(t, order, d):
#     return vertcat(*[
#         comb(i, d) * np.math.factorial(d) * t**(i - d) if i >= d else 0
#         for i in range(order + 1)
#     ])

# # Multiplies coefficient matrix Ck by the time basis vector to evaluate position at time t.
# def traj(Ck, t):
#     return mtimes(Ck, poly_t(t, poly_order))

# # Computes the d-th derivative (velocity, acceleration, jerk, snap) at time t.
# def traj_derivative(Ck, t, d):
#     return mtimes(Ck, dpoly_t(t, poly_order, d))

# # Start and end points
# x0 = DM([0, 0, 0])
# xf = DM([5, 5, 1])

# # Start constraints
# # Ensure trajectory starts at x0. Constrain initial speed (velocity magnitude) ≤ 2 m/s.
# C0 = get_Ck(C, 0)
# opti.subject_to(traj(C0, 0) == x0)
# v0 = traj_derivative(C0, 0, 1)
# opti.subject_to(sqrt(dot(v0, v0)) <= 2)

# # End constraints
# # Ensure trajectory ends at xf. Enforce velocity ≤ 2 m/s at final point. Enforce zero acceleration and jerk at the endpoint.
# Cf = get_Ck(C, n_segments-1)
# opti.subject_to(traj(Cf, segment_time) == xf)
# v1 = traj_derivative(Cf, segment_time, 1)
# opti.subject_to(sqrt(dot(v1, v1)) <= 2)
# # opti.subject_to(traj_derivative(Cf, segment_time, 2) == 0)
# # opti.subject_to(traj_derivative(Cf, segment_time, 3) == 0)
# opti.subject_to(sqrt(dot(traj_derivative(Cf, segment_time, 2),
#                          traj_derivative(Cf, segment_time, 2))) <= 1e-2)
# opti.subject_to(sqrt(dot(traj_derivative(Cf, segment_time, 3),
#                          traj_derivative(Cf, segment_time, 3))) <= 1e-2)

# opti.subject_to(opti.bounded(-100, C, 100))

# for k in range(n_segments):
#     alpha = k / (n_segments - 1)
#     opti.set_initial(get_Ck(C, k), np.outer(x0 + alpha * (xf - x0), np.ones(poly_order + 1)))


# # Continuity across segments
# # For each pair of consecutive segments, enforce continuity of: pos, vel, acc, snap
# for k in range(n_segments - 1):
#     Ck = get_Ck(C, k)
#     Ck1 = get_Ck(C, k + 1)
#     for d in [0, 1, 2, 4]:  # position, velocity, acceleration, snap
#         opti.subject_to(traj_derivative(Ck, segment_time, d) ==
#                         traj_derivative(Ck1, 0, d))

# # Gate constraints
# # Define 4 gates as boxes in space (center + size). "dir" is used to enforce entry direction constraint.
# gates = [
#     {"center": DM([1.5, 1.0, 1.0]), "size": DM([0.5, 0.5, 1.5]), "dir": "+"},
#     {"center": DM([2.5, 2.0, 1.0]), "size": DM([0.5, 0.5, 1.5]), "dir": "-"},
#     {"center": DM([3.5, 3.0, 1.0]), "size": DM([0.5, 0.5, 1.5]), "dir": "+"},
#     {"center": DM([4.5, 4.0, 1.0]), "size": DM([0.5, 0.5, 1.5]), "dir": "-"},
# ]

# # Forces the midpoint of each segment (t=0.5s) to pass within the gate box.
# # Adds directional constraint: e.g., if "dir" == "+", the drone must pass from negative to positive x.
# for i, gate in enumerate(gates):
#     Ck = get_Ck(C, i)
#     pk = traj(Ck, segment_time / 2)
#     for d in range(3):
#         opti.subject_to(pk[d] >= gate["center"][d] - gate["size"][d] / 2)
#         opti.subject_to(pk[d] <= gate["center"][d] + gate["size"][d] / 2)
#     x_gate = float(gate["center"][0])  # ensure it's a float, not DM
#     if gate["dir"] == "+":
#         opti.subject_to(pk[0] <= x_gate)
#     else:
#         opti.subject_to(pk[0] >= x_gate)


# # Objective: Minimize snap
# snap_cost = 0
# for k in range(n_segments):
#     Ck = get_Ck(C, k)
#     for t in np.linspace(0, segment_time, 5):
#         snap = traj_derivative(Ck, t, 4)
#         snap_cost += dot(snap, snap)

# opti.minimize(snap_cost)

# opti.solver("ipopt", {"print_time": True})

# # print(opti.debug.value(C))  # See if any NaNs

# sol = opti.solve()
# # Get trajectory points
# traj_points = []
# for k in range(n_segments):
#     Ck = sol.value(get_Ck(C, k))
#     for t in np.linspace(0, segment_time, 10):
#         pt = np.dot(Ck, np.array([t**i for i in range(poly_order + 1)]))
#         traj_points.append(pt)

# traj_points = np.array(traj_points)
# print("Trajectory Points:\n", traj_points)


# # Function to draw a box (gate)
# def draw_box(ax, center, size, color='cyan', alpha=0.3):
#     c = center
#     s = size / 2
#     # 8 corners of the box
#     corners = [
#         [c[0]-s[0], c[1]-s[1], c[2]-s[2]],
#         [c[0]+s[0], c[1]-s[1], c[2]-s[2]],
#         [c[0]+s[0], c[1]+s[1], c[2]-s[2]],
#         [c[0]-s[0], c[1]+s[1], c[2]-s[2]],
#         [c[0]-s[0], c[1]-s[1], c[2]+s[2]],
#         [c[0]+s[0], c[1]-s[1], c[2]+s[2]],
#         [c[0]+s[0], c[1]+s[1], c[2]+s[2]],
#         [c[0]-s[0], c[1]+s[1], c[2]+s[2]],
#     ]
#     # Define box faces from the corners
#     faces = [
#         [corners[i] for i in [0, 1, 2, 3]],
#         [corners[i] for i in [4, 5, 6, 7]],
#         [corners[i] for i in [0, 1, 5, 4]],
#         [corners[i] for i in [2, 3, 7, 6]],
#         [corners[i] for i in [1, 2, 6, 5]],
#         [corners[i] for i in [4, 7, 3, 0]],
#     ]
#     box = Poly3DCollection(faces, facecolors=color, edgecolors='k', linewidths=0.5, alpha=alpha)
#     ax.add_collection3d(box)

# # Convert trajectory points to array if needed
# traj_points = np.array(traj_points)

# # Setup 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot trajectory
# ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2], label='Trajectory', color='blue')

# # Plot gates
# for gate in gates:
#     center = np.array(gate["center"])
#     size = np.array(gate["size"])
#     draw_box(ax, center, size, color='orange', alpha=0.4)

#     # Optional: show direction arrow
#     direction = 1 if gate["dir"] == "+" else -1
#     ax.quiver(center[0] - direction * 0.3, center[1], center[2],
#               direction * 0.5, 0, 0, color='red', arrow_length_ratio=0.4)

# # Plot start and end points
# ax.scatter(*x0, color='green', s=100, label='Start')
# ax.scatter(*xf, color='red', s=100, label='Goal')

# # Label and format plot
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Minimum Snap Trajectory through Gates')
# ax.legend()
# ax.grid(True)
# ax.view_init(elev=30, azim=135)
# plt.tight_layout()
# plt.show()

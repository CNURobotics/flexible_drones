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

import numpy as np


class PolyOrderTrajectory():

    def __init__(self, durations, x_coeffs, y_coeffs, z_coeffs, yaw_coeffs):
        self.durations = durations
        # Segment boundaries: one start time plus one end time per segment.
        self.end_times = [0.0] + list(np.cumsum(durations))

        self.x_coeffs = x_coeffs
        self.y_coeffs = y_coeffs
        self.z_coeffs = z_coeffs
        self.yaw_coeffs = yaw_coeffs
        self.last_segment = 0

    def total_duration(self):
        return self.end_times[-1]

    def evaluate_segment(self, i, local_t):
        """Evaluate position, velocity, and acceleration for one segment."""
        pos = np.array([
            np.polyval(self.x_coeffs[i], local_t),
            np.polyval(self.y_coeffs[i], local_t),
            np.polyval(self.z_coeffs[i], local_t),
            np.polyval(self.yaw_coeffs[i], local_t),
        ])

        def eval_deriv(coeffs, order=1):
            deriv_coeffs = np.polyder(coeffs, order)
            return np.polyval(deriv_coeffs, local_t)
        vel = np.array([
            eval_deriv(self.x_coeffs[i]),
            eval_deriv(self.y_coeffs[i]),
            eval_deriv(self.z_coeffs[i]),
            eval_deriv(self.yaw_coeffs[i])
        ])
        acc = np.array([
            eval_deriv(self.x_coeffs[i], 2),
            eval_deriv(self.y_coeffs[i], 2),
            eval_deriv(self.z_coeffs[i], 2),
            eval_deriv(self.yaw_coeffs[i], 2)
        ])
        return pos, vel, acc

    def evaluate(self, t):
        """Evaluate position at global time t."""
        elapsed = 0.0
        end_time = self.total_duration()

        if t > end_time and not np.isclose(t, end_time):
            # No more trajectory to evaluate
            return None, None, None

        if self.durations and np.isclose(t, end_time):
            last_index = len(self.durations) - 1
            self.last_segment = last_index
            return self.evaluate_segment(last_index, self.durations[last_index])

        if self.last_segment < len(self.end_times) - 1:
            if self.end_times[self.last_segment] <= t <= self.end_times[self.last_segment + 1]:
                # Same segment
                return self.evaluate_segment(self.last_segment, t - self.end_times[self.last_segment])
            else:
                self.last_segment = 0
        else:
            self.last_segment = 0

        for i, duration in enumerate(self.durations):
            if t < elapsed + duration:
                local_t = t - elapsed
                self.last_segment = i
                return self.evaluate_segment(i, local_t)
            elapsed += duration
        return None, None, None  # Or hold at last point

    def plot_trajectory(self, samples_per_segment=100):
        """Plot positions, velocities, and accelerations against time for error checking."""
        import matplotlib.pyplot as plt

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Position')

        all_times = []
        all_yaws = []

        global_time = 0.0
        for i, duration in enumerate(self.durations):
            local_ts = np.linspace(0, duration, samples_per_segment)
            global_ts = global_time + local_ts
            pos_xs, pos_ys, pos_zs, pos_yaws = [], [], [], []

            for t in local_ts:
                pos, _, _ = self.evaluate_segment(i, t)
                pos_xs.append(pos[0])
                pos_ys.append(pos[1])
                pos_zs.append(pos[2])
                pos_yaws.append(pos[3])

            ax.plot(pos_xs, pos_ys, pos_zs, label=f'Segment {i + 1}')

            all_times.extend(global_ts)
            all_yaws.extend(pos_yaws)
            global_time += duration

        vel_fig, (v_axs1, v_axs2, v_axs3, v_axs4) = plt.subplots(4)
        global_time = 0.0
        for i, duration in enumerate(self.durations):
            local_ts = np.linspace(0, duration, samples_per_segment)
            global_ts = global_time + local_ts

            # _, vel, _ = self.evaluate_segment(i, local_ts)  # vel is now shape (4, N)

            # v_axs1.plot(global_ts, vel[0], linewidth=2, color='r')
            # v_axs2.plot(global_ts, vel[1], linewidth=2, color='g')
            # v_axs3.plot(global_ts, vel[2], linewidth=2, color='b')
            # v_axs4.plot(global_ts, vel[3], linewidth=2, color='m')
            vxs, vys, vzs, vyaws = [], [], [], []
            for t in local_ts:
                _, vel, _ = self.evaluate_segment(i, t)
                vxs.append(vel[0])
                vys.append(vel[1])
                vzs.append(vel[2])
                vyaws.append(vel[3])

            v_axs1.plot(global_ts, vxs)
            v_axs2.plot(global_ts, vys)
            v_axs3.plot(global_ts, vzs)
            v_axs4.plot(global_ts, vyaws)

            global_time += duration

        vel_fig.supxlabel('Time')
        vel_fig.supylabel('Velocity')
        v_axs4.set_title('Velocity of yaw vs Time')
        v_axs3.set_title('Velocity on the Z vs Time')
        v_axs2.set_title('Velocity on the Y vs Time')
        v_axs1.set_title('Velocity on the X vs Time')

        acc_fig, (a_axs1, a_axs2, a_axs3, a_axs4) = plt.subplots(4)
        global_time = 0.0
        for i, duration in enumerate(self.durations):
            local_ts = np.linspace(0, duration, samples_per_segment)
            global_ts = global_time + local_ts
            # _, _, acc = self.evaluate_segment(i, local_ts)  # acc is now shape (4, N)

            # a_axs1.plot(global_ts, acc[0], linewidth=2, color='r')
            # a_axs2.plot(global_ts, acc[1], linewidth=2, color='g')
            # a_axs3.plot(global_ts, acc[2], linewidth=2, color='b')
            # a_axs4.plot(global_ts, acc[3], linewidth=2, color='m')
            axs, ays, azs, ayaws = [], [], [], []
            for t in local_ts:
                _, _, acc = self.evaluate_segment(i, t)
                axs.append(acc[0])
                ays.append(acc[1])
                azs.append(acc[2])
                ayaws.append(acc[3])

            a_axs1.plot(global_ts, axs)
            a_axs2.plot(global_ts, ays)
            a_axs3.plot(global_ts, azs)
            a_axs4.plot(global_ts, ayaws)

            global_time += duration

        acc_fig.supxlabel('Time')
        acc_fig.supylabel('Acceleration')
        a_axs4.set_title('Yaw rate of change vs Time')
        a_axs3.set_title('Acceleration on the Z vs Time')
        a_axs2.set_title('Acceleration on the Y vs Time')
        a_axs1.set_title('Acceleration on the X vs Time')

        plt.figure()
        plt.plot(all_times, np.unwrap(all_yaws), label='Yaw (unwrapped)')
        plt.xlabel('Time (s)')
        plt.ylabel('Yaw (rad)')
        plt.title('Yaw Angle over Time')
        plt.grid(True)

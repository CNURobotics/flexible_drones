#!/usr/bin/env python3
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

# Run from source (workspace sourced); see generators/README.md. Examples:
#   ./cnu_sails.py                                       # Crazyflie defaults (scale 1.0)
#   ./cnu_sails.py --scale 1.75 --v-nom 1.5 --floor 0.5 --suffix _pihawk
# Writes generators/output/cnu_sail{0,1}_<scale>[suffix].csv (e.g. cnu_sail0_1_75.csv)
# plus a cnu_sails_origins[suffix].py with the relative center of each of the 3 sails.
# Plotting is off by default; add --plot to show the (blocking) matplotlib figures.

from math import ceil

import numpy as np
import os

from flexible_drones_tools.trajectories.utilities.kkt import unit, solve_min_snap_shape_KKT_normalized
from flexible_drones_tools.trajectories.utilities.kkt import t_poly_coeffs_from_normalized
from flexible_drones_tools.trajectories.utilities.kkt import solve_corner_with_alpha
from flexible_drones_tools.trajectories.utilities.kkt import REG, NCOEF
from flexible_drones_tools.trajectories.utilities.kkt import eval_time_histories
from flexible_drones_tools.trajectories.utilities.kkt import build_decel_to_stop_from_shape
from flexible_drones_tools.trajectories.utilities.io import save_trajectory_csv
from flexible_drones_tools.trajectories.utilities.plotting import TrajectoryPlotter


def normalize(v):
    length = np.linalg.norm(v)
    if length == 0:
        return v, 0
    else:
        return v / length, length


def sail(corners, indices):
    """Build the ordered points/vectors/lengths for a sail route through corners."""
    points = [corners[indices[0]]]
    vectors = []
    lengths = []
    total = 0
    for ndx in range(1, len(indices)):
        points.append(corners[indices[ndx]])
        vec = corners[indices[ndx]] - corners[indices[ndx - 1]]
        _, lv = normalize(vec)
        vectors.append(vec)
        lengths.append(lv)
        total += lv
    return points, vectors, lengths, total


def calc_sail_segments(corners, indices, alpha=1.2,
                       speed0=0.2, speed1=0.2,
                       T_in=0.3, T_out=0.3,
                       T_start=0.8):
    # Calculate the normalized acceleration shape
    segments = []  # List of (duration, x_coeff, y_coeff) tuples
    rho = 0.4
    v1_desired = speed0  # terminal speed at start
    points, vectors, lengths, total = sail(corners, indices)
    print(f'Solving sail with spd={speed0} and {speed1}, alpha={alpha}')
    if isinstance(alpha, float):
        alphas = [alpha for _ in vectors]
    else:
        alphas = alpha  # better be list or tuple

    res = solve_min_snap_shape_KKT_normalized(rho, a_same_start=0.0, a_same_end=0.0, reg=0.0)
    c1_norm, c2_norm = res['c1'], res['c2']
    T1, T2 = res['T1'], res['T2']
    c_v = res['c_v']  # kA, kJ, kS, J are unused but available in res

    p1_needed = v1_desired * T_start / c_v
    res_start = t_poly_coeffs_from_normalized(
        c1_norm=c1_norm,
        c2_norm=c2_norm,
        rho=rho,
        T_real=T_start,
        p1=p1_needed,   # or use v1=v1_desired, c_v=c_v
        x0=0.0
    )
    T1 = res_start['T1']
    T2 = res_start['T2']
    c1 = res_start['seg1_t']
    c2 = res_start['seg2_t_shifted']

    # Start from rest
    uv = unit(vectors[0])

    rot = np.array(((uv[0], -uv[1]),
                    (uv[1], uv[0])))

    start1 = rot @ np.array((c1, np.zeros(c1.shape)))
    start2 = rot @ np.array((c2, np.zeros(c2.shape)))
    # print("start1=",start1)
    # print("start2=",start2)
    # print("points[0]=", points[0])

    start1[0][0] += points[0][0]
    start1[1][0] += points[0][1]
    start2[0][0] += points[0][0]
    start2[1][0] += points[0][1]
    # print("start1 =",start1)
    # print("start2 =",start2)

    segments.append((T1, start1[0, :], start1[1, :]))
    segments.append((T2, start2[0, :], start2[1, :]))

    print(f'p1_needed={p1_needed}')
    P00 = rot @ np.array(((p1_needed,), (0.0,)))
    P00[0] += points[0][0]  # Start of linear segment
    P00[1] += points[0][1]  # Start of linear segment
    P00 = P00.T  # others are flat
    if isinstance(T_in, list):
        T1 = T_in[0]
        T2 = T_out[0]
    else:
        T1 = T_in
        T2 = T_out

    speed_in = speed0
    speed_out = speed1

    uv = unit(vectors[0])

    for corn in range(1, len(indices) - 1):
        uv1 = unit(vectors[corn])
        if isinstance(T_out, list):
            T1 = T_in[indices[corn]]
            T2 = T_out[indices[corn]]

        if np.dot(uv, uv1) > 0.99:
            # Continues onto straight section
            print(f'Processing straight #{corn} of {len(indices)} T=({T1}, {T2}) [{indices[corn]}] ...')
            P12 = points[corn]
            linears = np.zeros((2, NCOEF))
            linears[:, 0] = P00  # reshape((2,1))
            linears[:, 1] = uv * speed_in
            linlen = np.linalg.norm(P12 - P00)
            # print(f"linears={linears}\n    P01={P01}\n"
            #       f"    P00={P00}\n    uv={uv}\n"
            #       f"    len={linlen}\n    spd={speed_in}\n"
            #       f"    dur={linlen/speed_in} ", flush=True)

            segments.append((linlen / speed_in, linears[0, :], linears[1, :]))
        else:
            print(f'Processing corner #{corn} of {len(indices)} T=({T1}, {T2}) [{indices[corn]}] ...')
            res = solve_corner_with_alpha(points[corn - 1], points[corn], points[corn + 1],
                                          T1, T2,
                                          speed_in, speed_out,
                                          alpha=alphas[corn],
                                          reg=REG)
            P01 = res['p_in']
            P12 = res['p_out']
            c1 = res['seg1_t']
            c2 = res['seg2_t']

            linvec = P01 - P00
            linlen = np.linalg.norm(linvec)
            if np.dot(uv, linvec.T) < 0:
                print(f'Error: segment {corn} - P00={P00} P01={P01} has negative projection!')
                print(f'   uv={uv}')
                print(f'   linvec={linvec}', flush=True)
                raise Exception('Invalid segment!')

            dur = linlen / speed_in
            if dur < 0.011:
                print(f' Linear segment at segment {corn} is too short to sample dur={dur} seconds!')
            else:
                # print(f"c1={c1}")
                linears = np.zeros((2, NCOEF))
                linears[:, 0] = P00  # reshape((2,1))
                linears[:, 1] = uv * speed_in
                segments.append((dur, linears[0, :], linears[1, :]))
                # print(f"linears={linears}\n    P01={P01}\n"
                #       f"    P00={P00}\n    uv={uv}\n"
                #       f"    len={linlen}\n    spd={speed_in}\n"
                #       f"    dur={linlen/speed_in} ", flush=True)

            segments.append((T1, c1[0], c1[1]))
            segments.append((T2, c2[0], c2[1]))
        # prior departure vector
        uv = uv1

        # Next linear starts at end point of last segment
        P00 = P12

        # Use faster speed next time (loop starts at corn=1, so always the cruise speed)
        speed_in = speed_out
        speed_out = speed1

    duration = 0.0
    seg = 0
    for dur, _, _ in segments:
        duration += dur
        print(f'Duration after segment {seg} = {duration} ({dur})')
        seg += 1
    print(f'Duration after penultimate segment = {duration}')

    # Calculate the final deceleration
    #  p1 = v0 * T_real / c_v
    end_vector = points[0] - P00
    T_real = np.linalg.norm(end_vector) * c_v / speed_in
    print(f'decel zone: end_vector={end_vector} T_real decel = {T_real} speed={speed_in} ')
    # scale = 2.0
    # T_real *= scale
    # c_v *= scale
    # for i in range(len(c1_norm)):
    #     c1_norm[i] *= pow(scale, i)/
    #     c2_norm[i] *= pow(scale, i)

    decel = build_decel_to_stop_from_shape(
        c1_norm=c1_norm, c2_norm=c2_norm, rho=rho,
        T_real=T_real,
        p1=np.linalg.norm(end_vector),
        v1=speed_in,
        x0=0.0
    )
    T1 = decel['T1']
    T2 = decel['T2']
    c1 = decel['seg1_t']
    c2 = decel['seg2_t_shifted']
    p1_needed = decel['p1']
    print(f' decel T1={T1} T2={T2} T={T1 + T2} p1={p1_needed}')
    # Start from rest
    uv = unit(vectors[-1])

    rot = np.array(((uv[0], -uv[1]),
                    (uv[1], uv[0])))

    start1 = rot @ np.array((c1, np.zeros(c1.shape)))
    start2 = rot @ np.array((c2, np.zeros(c2.shape)))
    # print("start1=",start1)
    # print("start2=",start2)
    print('points[-2]=', points[-2])

    start1[0][0] += (points[-1][0] - uv[0] * p1_needed)
    start1[1][0] += (points[-1][1] - uv[1] * p1_needed)
    start2[0][0] += (points[-1][0] - uv[0] * p1_needed)
    start2[1][0] += (points[-1][1] - uv[1] * p1_needed)
    print('start2 =', start2[:, 0])
    print('start1 =', start1[:, 0])

    P20 = points[-1] - uv * p1_needed
    print(f'P20={P20}')
    print('points[-1]=', points[-1])
    approach_vector = P20 - P12
    print(f'vectors[-1]={vectors[-1]}')
    print(f'approach vector = {approach_vector}')
    if approach_vector @ uv > 0:
        # positive distance
        linears = np.zeros((2, NCOEF))
        linears[:, 0] = P12  # reshape((2,1))
        linears[:, 1] = uv * speed1
        linlen = np.linalg.norm(approach_vector)
        print(f'adding linear segment in final deceleration zone |{linlen}| dur={linlen / speed_in}s')
        segments.append((linlen / speed_in, linears[0, :], linears[1, :]))
    segments.append((T1, start1[0, :], start1[1, :]))
    segments.append((T2, start2[0, :], start2[1, :]))

    duration = 0.0
    for dur, _, _ in segments:
        duration += dur

    return segments, duration


def eval_segments(segments, Nsamples=None):
    tprior = 0.0
    time = None
    for ndx, (dur, cx, cy) in enumerate(segments):
        n_samples = Nsamples if Nsamples is not None else ceil(dur * 100)  # 100 hz per segment
        tx, px, vx, ax, jx, sx = eval_time_histories(cx, None, dur, None, N=n_samples)
        ty, py, vy, ay, jy, sy = eval_time_histories(cy, None, dur, None, N=n_samples)
        tx += tprior
        ty += tprior
        tprior = tx[-1]
        if time is None:
            time = tx
            pos = np.vstack((px, py))
            vel = np.vstack((vx, vy))
            acc = np.vstack((ax, ay))
            jrk = np.vstack((jx, jy))
            snp = np.vstack((sx, sy))
        else:
            time = np.concatenate((time, tx))
            pos = np.hstack([pos, np.vstack([px, py])])
            vel = np.hstack([vel, np.vstack([vx, vy])])
            acc = np.hstack([acc, np.vstack([ax, ay])])
            jrk = np.hstack([jrk, np.vstack([jx, jy])])
            snp = np.hstack([snp, np.vstack([sx, sy])])

    return time, pos, vel, acc, jrk, snp


def write_drone_csv(filename, segments, out_dir=None):
    """
    Write drone trajectory segments to a CSV via the shared drone-order I/O.

    Sail segments are 2D: ``segment[1]`` is the x coefficient row and
    ``segment[2]`` is mapped to z (y is held at 0, yaw at 0). Coefficients are
    already in drone (ascending-power) order. Output defaults to the git-ignored
    ``generators/output/`` folder.
    """
    from flexible_drones_tools.trajectories.generators import OUTPUT_DIR

    out_dir = str(OUTPUT_DIR) if out_dir is None else out_dir
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename) if out_dir else filename

    durations = [seg[0] for seg in segments]
    x_coeffs = [np.asarray(seg[1], dtype=float) for seg in segments]
    z_coeffs = [np.asarray(seg[2], dtype=float) for seg in segments]
    zeros = [np.zeros_like(np.asarray(seg[1], dtype=float)) for seg in segments]
    save_trajectory_csv(filepath, durations, x_coeffs, zeros, z_coeffs, zeros,
                        coefficient_order='drone')
    return filepath


def scale_suffix(scale):
    """Return the default filename suffix for a scale ('' at 1.0, else e.g. '_1_75')."""
    if float(scale) == 1.0:
        return ''
    return '_' + str(float(scale)).replace('-', 'm').replace('.', '_')


def sail_filename(index, suffix=''):
    """Return the sail CSV filename (e.g. cnu_sail0_1_75.csv)."""
    return f'cnu_sail{index}{suffix}.csv'


def sails_origins_for(spacing):
    """Return the relative (x, y, z) centers of the three laid-out sails (spaced along x)."""
    spacing = float(spacing)
    return [
        [spacing, 0.0, 0.0],   # Sail 0
        [0.0, 0.0, 0.0],       # Sail 1
        [-spacing, 0.0, 0.0],  # Sail 2
    ]


def write_sails_origins(spacing, scale, suffix=''):
    """Write a sails_origins.py module (next to the CSVs) defining ``sails_origins``."""
    from flexible_drones_tools.trajectories.generators import OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    origins = sails_origins_for(spacing)
    path = OUTPUT_DIR / f'cnu_sails_origins{suffix}.py'
    lines = [
        f'# Generated by cnu_sails.py for scale={scale}.',
        '# Relative centers (x, y, z) for each of the three sails; the spacing is',
        f'# the scaled corner-2 x extent ({float(spacing):.6g} m).',
        '',
        'sails_origins = [',
    ]
    for origin, label in zip(origins, ('Sail 0', 'Sail 1', 'Sail 2')):
        lines.append(f'    [{origin[0]:.6g}, {origin[1]:.6g}, {origin[2]:.6g}],  # {label}')
    lines.append(']')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path, origins


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate the CNU three-sails trajectories (drone coefficient order).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Isotropic scale for sail geometry and timing '
                             '(default 1.0 = Crazyflie; e.g. larger for PiHawk).')
    parser.add_argument('--v-nom', dest='v_nom', type=float, default=0.68,
                        help='Nominal cruise velocity in m/s (default 0.68).')
    parser.add_argument('--alpha', type=float, default=1.2,
                        help='Corner aggressiveness (default 1.2).')
    parser.add_argument('--floor', type=float, default=0.75,
                        help='Height (z) of the lowest point of the sail, in [0.2, 1.5] '
                             '(default 0.75; the scaled sail is shifted to sit on this floor).')
    parser.add_argument('--suffix', default=None,
                        help='Output filename suffix; overrides the default scale suffix '
                             "(e.g. scale 1.75 -> '_1_75'). E.g. '_pihawk' -> cnu_sail0_pihawk.csv.")
    parser.add_argument('--plot', action='store_true',
                        help='Show all plots and the animation (implies the --plot-* flags; blocks).')
    parser.add_argument('--plot-sails', action='store_true',
                        help='Show the combined three-sail layout figure.')
    parser.add_argument('--plot-details', action='store_true',
                        help='Show the per-sail outline / fitted-path detail figures.')
    parser.add_argument('--plot-time', action='store_true',
                        help='Show the per-sail velocity/accel/jerk/snap time histories.')
    parser.add_argument('--animation', action='store_true',
                        help='Show the moving-position animation over the three-sail layout.')
    opts = parser.parse_args(args)

    scale = float(opts.scale)
    v_nom = float(opts.v_nom)
    floor = float(opts.floor)
    if not 0.2 <= floor <= 1.5:
        parser.error('--floor must be between 0.2 and 1.5')
    suffix = opts.suffix if opts.suffix is not None else scale_suffix(scale)

    # Base Crazyflie-tuned sail geometry in the x-z plane (2D; y=0 in 3D).
    # Isotropic scaling is a similarity transform, so it preserves the sail's
    # aspect ratio, proportions, and the spacing between the sails. The scaled
    # sail is then shifted vertically so its lowest point sits at `floor`.
    base_corners = [(-0.28333, 0.75), (0.3833327, 1.75), (0.3833327, 1.35),
                    (0.3833327, 0.75), (0, 0.75)]  # using y for z; y=0 in 3D
    scaled = [np.array(corner) * scale for corner in base_corners]
    floor_shift = floor - min(corner[1] for corner in scaled)
    P0, P1, P2, P3, P4 = [corner + np.array((0.0, floor_shift)) for corner in scaled]

    corners = [P0, P1, P2, P3, P4]
    # The corner blend distance is speed * T / alpha, so to keep it proportional to
    # the geometry (and the shape preserved) at an arbitrary v_nom, the corner and
    # accel timing scales with the geometry AND inversely with the speed change
    # relative to the base speed the timing was tuned for.
    v_nom_base = 0.68
    time_scale = scale * (v_nom_base / v_nom)
    time_in_corners = [t * time_scale for t in (0.245, 0.3, 0.3, 0.3, 0.3)]
    time_out_corners = [t * time_scale for t in (0.30, 0.3, 0.3, 0.3, 0.245)]
    t_start = 0.8 * time_scale  # start/stop accel ramp duration

    sail0 = [0, 1, 3, 0, 1, 3, 4, 0]  # Do 2 trips around, start/stop at 0 vel
    sail1 = [0, 1, 2, 4, 0, 1, 2, 4, 0]

    points0, vectors0, lengths0, total0 = sail(corners, sail0)
    points1, vectors1, lengths1, total1 = sail(corners, sail1)

    print('-------------------')
    v0s = '\n    '.join([str(vec) for vec in vectors0])
    print(f'Sail 0:\n    {v0s}\n    lengths= {lengths0}={total0} = {total0 - lengths0[0]}')
    v1s = '\n    '.join([str(vec) for vec in vectors1])
    print(f'Sail 1:\n    {v1s}\n    lengths= {lengths1}={total1} = {total1 - lengths1[0]}')

    T0, T1 = total0 / v_nom, total1 / v_nom
    Tb0, Tb1 = (total0 - lengths0[0]) / v_nom, (total1 - lengths1[0]) / v_nom
    print(f' Times: 0={T0:.3f}  1={T1:.3f} seconds')
    print(' 0: ', [dv / v_nom for dv in lengths0])
    print('      ', Tb0, Tb0 / Tb1, v_nom * Tb0 / Tb1)
    print(' 1: ', [dv / v_nom for dv in lengths1])
    print('      ', Tb1, Tb1 / Tb1, v_nom * Tb1 / Tb1)
    print('-------------------')

    segments0, duration0 = calc_sail_segments(corners, sail0, alpha=opts.alpha,
                                              speed0=v_nom,
                                              speed1=0.99 * v_nom * Tb0 / Tb1,
                                              T_in=time_in_corners,
                                              T_out=time_out_corners,
                                              T_start=t_start)
    write_drone_csv(sail_filename(0, suffix), segments0)

    segments1, duration1 = calc_sail_segments(corners, sail1, alpha=opts.alpha,
                                              speed0=v_nom,
                                              speed1=1.01 * v_nom * Tb1 / Tb1,
                                              T_in=time_in_corners,
                                              T_out=time_out_corners,
                                              T_start=t_start)
    write_drone_csv(sail_filename(1, suffix), segments1)

    print(30 * '=')
    print(f'Sail 0: {len(segments0)} segments @ {duration0} seconds')
    print(f'Sail 1: {len(segments1)} segments @ {duration1} seconds')

    # Relative centers of the three laid-out sails (spaced along x by the scaled
    # corner-2 x extent). Needed when placing the sails in a behavior.
    spacing = float(P2[0])
    origins_path, origins = write_sails_origins(spacing, scale, suffix=suffix)
    print(f'Relative sail centers (x, y, z), spacing={spacing:.4f} m:')
    for i, origin in enumerate(origins):
        print(f'  Sail {i}: [{origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f}]')
    print(f'Wrote sail origins to {origins_path}')

    # All CSV/origins output is done above; plotting is opt-in because plt.show()
    # blocks. --plot enables everything; the --plot-* flags enable subsets.
    show_sails = opts.plot or opts.plot_sails
    show_details = opts.plot or opts.plot_details
    show_time = opts.plot or opts.plot_time
    show_animation = opts.plot or opts.animation
    if not (show_sails or show_details or show_time or show_animation):
        return

    import matplotlib.pyplot as plt

    # Sampled time histories feed the time grids, the per-sail position overlays,
    # and the animation interpolation.
    time0, pos0, vel0, acc0, jrk0, snp0 = eval_segments(segments0)
    time1, pos1, vel1, acc1, jrk1, snp1 = eval_segments(segments1)

    # The three-sail layout figure also serves as the animation canvas.
    fig_sails = None
    if show_sails or show_animation:
        fig_sails = plt.figure()
        plt.plot(np.array([pnt[0] for pnt in points0]) + P2[0],
                 [pnt[1] for pnt in points0], 'b', linewidth=2, label='Sail 0')
        plt.plot([corners[ndx][0] for ndx in sail1],
                 [corners[ndx][1] for ndx in sail1], 'b', linewidth=2, label='Sail 1')
        plt.plot(np.array([pnt[0] for pnt in points1]) - P2[0],
                 [pnt[1] for pnt in points1], 'b', linewidth=2, label='Sail 2')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('Three-sail layout')
        plt.gca().set_aspect('equal', adjustable='box')

    if show_details:
        # Sail outlines with corner points.
        plt.figure()
        plt.plot([pnt[0] for pnt in points0], [pnt[1] for pnt in points0],
                 'r', linewidth=3, label='Sail 0')
        plt.plot([pnt[0] for pnt in points1], [pnt[1] for pnt in points1],
                 'g:', linewidth=2, label='Sail 1')
        for marker, point, label in zip(
                ('x', 's', 'o', 'd', '*'), (P0, P1, P2, P3, P4),
                ('Point 0', 'Point 1', 'Point 2', 'Point 3', 'Point 4')):
            plt.plot(point[0], point[1], marker, label=label, linewidth=2)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('Sail outlines')
        plt.gca().set_aspect('equal', adjustable='box')

        # Sail 0 fitted path over its outline.
        plt.figure()
        plt.plot([pnt[0] for pnt in points0], [pnt[1] for pnt in points0],
                 'r', linewidth=3, label='Sail 0')
        plt.plot(pos0[0, :], pos0[1, :], '-.', linewidth=1.8)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('Sail 0')
        plt.gca().set_aspect('equal', adjustable='box')

        # Sail 1 fitted path over its outline, with a few segment-start markers.
        plt.figure()
        plt.plot([pnt[0] for pnt in points1], [pnt[1] for pnt in points1],
                 'g:', linewidth=2, label='Sail 1')
        plt.plot(pos1[0, :], pos1[1, :], '-.', linewidth=1.8)
        for ndx, marker in ((10, 'gs'), (11, 'mx'), (12, 'ko')):
            if ndx < len(segments1):
                seg_dur, seg_cx, seg_cy = segments1[ndx]
                _t, seg_px, *_rest = eval_time_histories(seg_cx, None, seg_dur, None, N=2)
                plt.plot(seg_px, [seg_cy[0], seg_cy[0]], marker, linewidth=1.8)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('Sail 1')
        plt.gca().set_aspect('equal', adjustable='box')

    if show_time:
        row_titles = [['vx(t)', 'vy(t)'], ['ax(t)', 'ay(t)'],
                      ['jx(t)', 'jy(t)'], ['sx(t)', 'sy(t)']]
        for label, times, rows in (
                ('Sail 0', time0, [vel0, acc0, jrk0, snp0]),
                ('Sail 1', time1, [vel1, acc1, jrk1, snp1])):
            TrajectoryPlotter.plot_component_grid(
                times, rows, titles=row_titles,
                suptitle=f'{label} time histories (v, a, jerk, snap)')

    if show_animation:
        from matplotlib.animation import FuncAnimation

        t_common = np.arange(max(time1.min(), time0.min()),
                             min(time1.max(), time0.max()), 0.01)
        p0_interp = np.vstack((np.interp(t_common, time0, pos0[0, :]),
                               np.interp(t_common, time0, pos0[1, :])))
        p1_interp = np.vstack((np.interp(t_common, time1, pos1[0, :]),
                               np.interp(t_common, time1, pos1[1, :])))

        plt.figure(fig_sails)
        (point0,) = fig_sails.gca().plot([], [], 'ro', linewidth=4, label='Position 0')
        (point1,) = fig_sails.gca().plot([], [], 'go', linewidth=4, label='Position 1')

        def init():
            point0.set_data([], [])
            point1.set_data([], [])
            return point0, point1

        def update(frame):
            px0, py0 = p0_interp[:, frame]
            point0.set_data([px0 + P2[0]], [py0])  # offset sail 0
            px1, py1 = p1_interp[:, frame]
            point1.set_data([px1], [py1])
            return point0, point1

        # Keep a reference so the animation is not GC'd before plt.show().
        ani = FuncAnimation(fig_sails, update, frames=len(t_common),  # noqa: F841
                            init_func=init, blit=True, interval=50)

    plt.show()


if __name__ == '__main__':
    main()

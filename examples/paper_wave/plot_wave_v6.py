#!/usr/bin/env python3
import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- User parameters --------------------------------------------------------

JSON_FILE            = '../../post/spray_metrics_d25.json'
DROPLET_DIAMETER_MM    = 2.5
DROPLET_RADIUS_MM      = DROPLET_DIAMETER_MM / 2.0
BIN_SIZE_MM            = 0.1

# Treat the plotted "line" as having this physical width in the out-of-plane direction.
# Requested: set equal to the droplet radius.
LINE_WIDTH_MM          = DROPLET_RADIUS_MM

# cluster threshold (effective height per bin, in mm)
MIN_CLUSTER_HEIGHT_MM  = 0.25
# discard bulk clusters larger than this span in y (mm)
MAX_CLUSTER_SIZE_MM    = 30.0

# Toggle: save each figure to OUT_DIR (PNG) instead of plt.show()
SAVE_FIGURES = True
OUT_DIR      = 'figures'

# Colorbar limits (in mm of volume-equivalent height); None => auto
COLORBAR_MIN = 0.0
COLORBAR_MAX = 1.0

# ---------------------------------------------------------------------------

def ensure_out_dir():
    if SAVE_FIGURES and not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

def sphere_slice_volume_between(y0, r, a, b):
    """
    Analytic volume (mm^3) of a sphere (center y0, radius r)
    between planes y=a and y=b (with clipping to the sphere).
      V = π [ r^2 (b' - a') - ( (b'-y0)^3 - (a'-y0)^3 ) / 3 ]
    where a' = max(a, y0 - r), b' = min(b, y0 + r), and V=0 if a' >= b'.
    """
    a_clip = max(a, y0 - r)
    b_clip = min(b, y0 + r)
    if b_clip <= a_clip:
        return 0.0
    term1 = r*r * (b_clip - a_clip)
    term2 = ((b_clip - y0)**3 - (a_clip - y0)**3) / 3.0
    return math.pi * (term1 - term2)

def compute_volume_conserving_height(y_coords_mm, y_edges_mm, radius_mm, line_width_mm):
    """
    For each y-bin [edge_i, edge_{i+1}], compute the *effective height* (mm)
    such that height * BIN_SIZE_MM * line_width_mm equals the true volume
    of all intersecting spheres inside that bin.

    h_eff[i] = (sum_over_droplets V_bin(i)) / (line_width_mm * BIN_SIZE_MM)
    """
    nbins = len(y_edges_mm) - 1
    heights = np.zeros(nbins, dtype=float)

    for y0 in y_coords_mm:
        # vectorized overlap per bin
        a = y_edges_mm[:-1]
        b = y_edges_mm[1:]
        a_clip = np.maximum(a, y0 - radius_mm)
        b_clip = np.minimum(b, y0 + radius_mm)
        valid = b_clip > a_clip
        if not np.any(valid):
            continue
        # volume per valid bin from analytic integral
        term1 = radius_mm**2 * (b_clip[valid] - a_clip[valid])
        term2 = ((b_clip[valid] - y0)**3 - (a_clip[valid] - y0)**3) / 3.0
        V_bins = math.pi * (term1 - term2)  # mm^3
        heights[valid] += V_bins / (line_width_mm * (b[valid] - a[valid]))  # mm

    return heights

def find_clusters(mask):
    clusters = []
    i = 0
    N = len(mask)
    while i < N:
        if mask[i]:
            start = i
            while i < N and mask[i]:
                i += 1
            clusters.append((start, i-1))
        else:
            i += 1
    return clusters

def plot_with_clusters(times, height_mat, y_edges_mm, clusters_dict, z_mm):
    times = np.array(times)
    T = len(times)

    # time cell half-widths for alignment
    dt = np.empty(T)
    for i in range(T):
        if i == 0:
            dt[i] = times[1] - times[0]
        elif i == T-1:
            dt[i] = times[-1] - times[-2]
        else:
            dt[i] = 0.5 * (times[i+1] - times[i-1])

    # extents
    x_min = times[0] - dt[0]/2
    x_max = times[-1] + dt[-1]/2
    y_min = y_edges_mm[0]
    y_max = y_edges_mm[-1]

    # centers for labeling/rectangles
    centers = 0.5 * (y_edges_mm[:-1] + y_edges_mm[1:])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=False, figsize=(8, 6),
        gridspec_kw={'height_ratios':[3,1]},
        constrained_layout=True
    )

    # --- Top: spectrogram (volume-equivalent height) ---
    im_kwargs = dict(
        aspect='auto', origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        interpolation='nearest'
    )
    if COLORBAR_MIN is not None: im_kwargs['vmin'] = COLORBAR_MIN
    if COLORBAR_MAX is not None: im_kwargs['vmax'] = COLORBAR_MAX

    img = ax_top.imshow(height_mat, **im_kwargs)
    ax_top.set_title(f'Volume‑equivalent height vs Time at z = {z_mm:.0f} mm')
    ax_top.set_ylabel('Y [mm]')
    ax_top.set_xlabel('Time [s]')
    ax_top.set_xlim(x_min, x_max)
    cbar = fig.colorbar(img, ax=ax_top, label='Volume‑equivalent height [mm]')

    # overlay red rectangles (exactly aligned to bin/time cells)
    for ti, clusters in clusters_dict.items():
        left = times[ti] - dt[ti]/2
        width = dt[ti]
        for start, end, _vol, _D in clusters:
            bottom = y_edges_mm[start]
            height = y_edges_mm[end+1] - y_edges_mm[start]
            rect = Rectangle((left, bottom), width, height,
                             edgecolor='red', facecolor='none', lw=1.2)
            ax_top.add_patch(rect)

    # --- Bottom: corrected equivalent diameters from cluster volumes ---
    ax_bot.set_xlabel('Time [s]')
    ax_bot.set_ylabel('Corrected dia. [mm]')
    ax_bot.set_xlim(x_min, x_max)
    for ti, clusters in clusters_dict.items():
        t = times[ti]
        for _s, _e, V_cluster, D_corr in clusters:
            ax_bot.scatter(t, D_corr, s=20)
    ax_bot.grid(True, ls='--', alpha=0.5)

    if SAVE_FIGURES:
        fn = os.path.join(OUT_DIR, f'z{int(z_mm)}mm_d{int(DROPLET_DIAMETER_MM*10)}.png')
        fig.savefig(fn, dpi=300)
        print(f"Saved: {fn}")
        plt.close(fig)
    else:
        plt.show()

def main():
    ensure_out_dir()
    with open(JSON_FILE) as f:
        data = json.load(f)

    for key, entry in data.items():
        if not key.startswith('particle_cv_z') or entry.get('system_name') != 'fluid':
            continue

        # parse z
        try:
            z_mm = float(key.split('_')[2].lstrip('z'))
        except Exception:
            continue

        times  = entry['time']
        values = entry['values']

        # skip if empty
        all_ys = np.concatenate(values) if any(values) else np.array([])
        if all_ys.size == 0:
            print(f"Skipping z={z_mm:.0f} mm (no data)")
            continue

        # y-binning
        ys_mm = all_ys * 1000.0
        y_min = ys_mm.min() - DROPLET_RADIUS_MM
        y_max = ys_mm.max() + DROPLET_RADIUS_MM
        y_edges = np.arange(y_min, y_max + BIN_SIZE_MM, BIN_SIZE_MM)
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        # build height matrix (volume‑conserving by construction)
        T = len(times)
        N = len(y_centers)
        height_mat = np.zeros((N, T))
        for ti, ys in enumerate(values):
            if ys:
                ys_mm_t = np.array(ys) * 1000.0
                height_mat[:, ti] = compute_volume_conserving_height(
                    ys_mm_t, y_edges, DROPLET_RADIUS_MM, LINE_WIDTH_MM
                )

        # detect clusters & compute volumes/diameters
        clusters_dict = {}
        for ti in range(T):
            mask = height_mat[:, ti] >= MIN_CLUSTER_HEIGHT_MM
            raw = find_clusters(mask)
            filtered = []
            for start, end in raw:
                span_mm = (end - start + 1) * BIN_SIZE_MM
                if span_mm <= MAX_CLUSTER_SIZE_MM:
                    # cluster *volume* from volume-equivalent heights
                    V_cluster = LINE_WIDTH_MM * BIN_SIZE_MM * height_mat[start:end+1, ti].sum()
                    D_corr = (6.0 * V_cluster / math.pi) ** (1.0/3.0)
                    filtered.append((start, end, V_cluster, D_corr))
            clusters_dict[ti] = filtered

            sizes = [(e-s+1)*BIN_SIZE_MM for s,e,_,_ in filtered]
            Ds    = [D for *_ , D in filtered]
            print(f"t={times[ti]:.2f}s: clusters={len(filtered)}, sizes(y-span)={sizes}, D_corr={['{:.2f}'.format(d) for d in Ds]}")

        # plot
        # imshow expects shape [rows=y, cols=time]; we already have that
        plot_with_clusters(times, height_mat, y_edges, clusters_dict, z_mm)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- User parameters --------------------------------------------------------

JSON_FILE            = '../../post/spray_metrics_d3.json'
DROPLET_DIAMETER_MM    = 3
DROPLET_RADIUS_MM      = DROPLET_DIAMETER_MM / 2.0
BIN_SIZE_MM            = 0.1

# cluster threshold (total x-depth per bin in mm)
MIN_CLUSTER_HEIGHT_MM  = 0.25
# max cluster size to consider (in mm); larger = bulk fluid (drop)
MAX_CLUSTER_SIZE_MM    = 10.0

# Toggle: save each figure to OUT_DIR (PNG) instead of plt.show()
SAVE_FIGURES = True
OUT_DIR      = 'figures'

# Colorbar limits (total x-depth in mm); set to None to auto-scale
COLORBAR_MIN = 0.0
COLORBAR_MAX = 1.0

# ---------------------------------------------------------------------------
def ensure_out_dir():
    if SAVE_FIGURES and not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

def compute_height_profile(y_coords_mm, y_centers_mm, radius_mm):
    h = np.zeros_like(y_centers_mm)
    for y0 in y_coords_mm:
        dy = np.abs(y_centers_mm - y0)
        mask = dy <= radius_mm
        h[mask] += 2 * np.sqrt(radius_mm**2 - dy[mask]**2)
    return h

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

def plot_with_clusters(time_s, height_mat, y_centers_mm, clusters_dict, z_mm):
    times = np.array(time_s)
    T = len(times)
    # compute half‐widths dt for each column
    dt = np.empty(T)
    for i in range(T):
        if i == 0:
            dt[i] = times[1] - times[0]
        elif i == T-1:
            dt[i] = times[-1] - times[-2]
        else:
            dt[i] = 0.5 * (times[i+1] - times[i-1])

    # derive full extent so each cell lines up
    x_min = times[0] - dt[0]/2
    x_max = times[-1] + dt[-1]/2
    y_min = y_centers_mm[0] - BIN_SIZE_MM/2
    y_max = y_centers_mm[-1] + BIN_SIZE_MM/2

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(8, 6),
        gridspec_kw={'height_ratios':[3,1]},
        constrained_layout=True
    )

    # --- Top: spectrogram ---
    im_kwargs = dict(
        aspect='auto',
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        interpolation='nearest'
    )
    if COLORBAR_MIN is not None: im_kwargs['vmin'] = COLORBAR_MIN
    if COLORBAR_MAX is not None: im_kwargs['vmax'] = COLORBAR_MAX

    img = ax_top.imshow(height_mat, **im_kwargs)
    ax_top.set_title(f'X‑depth vs Time at z = {z_mm:.0f} mm')
    ax_top.set_ylabel('Y [mm]')
    ax_top.set_xlabel('Time [s]')           # <— now labeled
    ax_top.set_xlim(x_min, x_max)
    fig.colorbar(img, ax=ax_top, label='Total x‑depth [mm]')

    # overlay red rectangles
    for ti, clusters in clusters_dict.items():
        left = times[ti] - dt[ti]/2
        width = dt[ti]
        for start, end in clusters:
            bottom = y_centers_mm[start] - BIN_SIZE_MM/2
            height = (end - start + 1) * BIN_SIZE_MM
            rect = Rectangle((left, bottom), width, height,
                             edgecolor='red', facecolor='none', lw=1.2)
            ax_top.add_patch(rect)

    # --- Bottom: cluster‐size vs time ---
    ax_bot.set_xlabel('Time [s]')
    ax_bot.set_ylabel('Cluster size [mm]')
    ax_bot.set_xlim(x_min, x_max)
    for ti, clusters in clusters_dict.items():
        t = times[ti]
        for start, end in clusters:
            size_mm = (end - start + 1) * BIN_SIZE_MM
            ax_bot.scatter(t, size_mm, s=20, c='blue')
    ax_bot.grid(True, ls='--', alpha=0.5)

    # save or show
    if SAVE_FIGURES:
        fn = os.path.join(OUT_DIR, f'z{int(z_mm)}mm_clusters.png')
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
        if not key.startswith('particle_cv_z') or entry.get('system_name')!='fluid':
            continue

        # parse z
        try:
            z_mm = float(key.split('_')[2].lstrip('z'))
        except:
            continue

        times  = entry['time']
        values = entry['values']

        # skip empty
        all_ys = np.concatenate(values) if any(values) else np.array([])
        if all_ys.size == 0:
            print(f"Skipping z={z_mm:.0f} mm (no data)")
            continue

        # convert to mm and bin
        ys_mm = all_ys * 1000.0
        y_min = ys_mm.min() - DROPLET_RADIUS_MM
        y_max = ys_mm.max() + DROPLET_RADIUS_MM
        edges   = np.arange(y_min, y_max + BIN_SIZE_MM, BIN_SIZE_MM)
        centers = edges[:-1] + BIN_SIZE_MM/2.0

        # build height matrix
        T = len(times)
        N = len(centers)
        height_mat = np.zeros((N, T))
        for ti, ys in enumerate(values):
            if ys:
                height_mat[:, ti] = compute_height_profile(
                    np.array(ys)*1000.0, centers, DROPLET_RADIUS_MM
                )

        # detect & filter clusters
        clusters_dict = {}
        for ti in range(T):
            mask = height_mat[:, ti] >= MIN_CLUSTER_HEIGHT_MM
            raw = find_clusters(mask)
            filt = []
            for start, end in raw:
                size = (end - start + 1) * BIN_SIZE_MM
                if size <= MAX_CLUSTER_SIZE_MM:
                    filt.append((start, end))
            clusters_dict[ti] = filt

            sizes = [(e-s+1)*BIN_SIZE_MM for s,e in filt]
            print(f"t={times[ti]:.2f}s: {len(filt)} clusters, sizes={sizes}")

        # plot
        plot_with_clusters(times, height_mat, centers, clusters_dict, z_mm)

if __name__ == '__main__':
    main()
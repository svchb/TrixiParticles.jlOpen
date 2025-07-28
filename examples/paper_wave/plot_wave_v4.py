#!/usr/bin/env python3
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt

# --- User parameters --------------------------------------------------------

JSON_FILE            = '../../post/spray_metrics.json'
DROPLET_DIAMETER_MM  = 3
DROPLET_RADIUS_MM    = DROPLET_DIAMETER_MM / 2.0
BIN_SIZE_MM          = 0.1

# Toggle: save each figure to OUT_DIR (PNG) instead of plt.show()
SAVE_FIGURES = True
OUT_DIR      = 'figures'

# Colorbar limits (total x‑depth in mm); set to None to auto-scale
COLORBAR_MIN = 0.0   # e.g. force minimum to 0.0 mm
COLORBAR_MAX = 1.0  # e.g. force maximum to 5.0 mm

# ---------------------------------------------------------------------------

def ensure_out_dir():
    if SAVE_FIGURES and not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

def compute_height_profile(y_coords_mm, y_centers_mm, radius_mm):
    """
    For each bin center y_centers_mm, sum over all droplets the chord-length
    in x: 2*sqrt(r^2 - (y - y0)^2) where |y-y0| <= r.
    """
    heights = np.zeros_like(y_centers_mm)
    for y0 in y_coords_mm:
        dy = np.abs(y_centers_mm - y0)
        mask = dy <= radius_mm
        heights[mask] += 2.0 * np.sqrt(radius_mm**2 - dy[mask]**2)
    return heights

def plot_height_map(time_s, height_matrix, y_centers_mm, z_mm):
    """
    Plot or save the height_matrix spectrogram for this z-slice,
    using the global COLORBAR_MIN / MAX if set.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    im_kwargs = dict(
        aspect='auto',
        origin='lower',
        extent=[time_s[0], time_s[-1],
                y_centers_mm[0], y_centers_mm[-1]],
        interpolation='nearest'
    )
    # apply colorbar limits if provided
    if COLORBAR_MIN is not None: im_kwargs['vmin'] = COLORBAR_MIN
    if COLORBAR_MAX is not None: im_kwargs['vmax'] = COLORBAR_MAX

    c = ax.imshow(height_matrix, **im_kwargs)

    ax.set_title(f'X‑depth vs Time at z = {z_mm:.0f} mm')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Y [mm]')
    fig.colorbar(c, ax=ax, label='Total x‑depth [mm]')
    fig.tight_layout()

    if SAVE_FIGURES:
        fn = os.path.join(OUT_DIR, f'z{int(z_mm)}mm_d{int(DROPLET_DIAMETER_MM*10)}_depth.png')
        fig.savefig(fn, dpi=300)
        print(f"Saved: {fn}")
        plt.close(fig)
    else:
        plt.show()

def main():
    ensure_out_dir()

    # Load JSON
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    # Process each z-slice
    for key, entry in data.items():
        if not key.startswith('particle_cv_z') or entry.get('system_name') != 'fluid':
            continue

        # parse z in mm
        try:
            z_mm = float(key.split('_')[2].lstrip('z'))
        except Exception:
            continue

        times  = entry['time']    # [s]
        values = entry['values']  # [[m]]

        # flatten for range
        all_ys = np.concatenate(values) if any(values) else np.array([])
        if all_ys.size == 0:
            print(f"Skipping z={z_mm:.0f} mm (no data)")
            continue

        all_ys_mm = all_ys * 1000.0
        y_min = all_ys_mm.min() - DROPLET_RADIUS_MM
        y_max = all_ys_mm.max() + DROPLET_RADIUS_MM

        # fixed-size bins via arange
        edges   = np.arange(y_min, y_max + BIN_SIZE_MM, BIN_SIZE_MM)
        centers = edges[:-1] + BIN_SIZE_MM/2.0

        # build height matrix
        height_mat = np.zeros((len(centers), len(times)))
        for ti, ys in enumerate(values):
            if ys:
                ys_mm = np.array(ys) * 1000.0
                height_mat[:, ti] = compute_height_profile(
                    ys_mm, centers, DROPLET_RADIUS_MM
                )

        # plot or save
        plot_height_map(times, height_mat, centers, z_mm)

if __name__ == '__main__':
    main()

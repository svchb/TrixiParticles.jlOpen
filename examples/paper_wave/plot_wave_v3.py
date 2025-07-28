#!/usr/bin/env python3
import json
import math
import numpy as np
import matplotlib.pyplot as plt

# --- User parameters --------------------------------------------------------

# Path to your JSON data
json_file = '../../post/spray_metrics.json'

# Physical droplet diameter (mm) and derived radius (mm)
droplet_diameter_mm = 2.5
droplet_radius_mm = droplet_diameter_mm / 2.0

# Bin size along y (mm)
bin_size_mm = 0.1

# ---------------------------------------------------------------------------

def compute_height_profile(y_coords_mm, y_bins_centers_mm, radius_mm):
    """
    Given a list of droplet centers y_coords_mm and an array of
    bin centers y_bins_centers_mm, returns the total chord-length
    (in mm) in x-direction per bin, summing over all droplets.
    """
    heights = np.zeros_like(y_bins_centers_mm)
    for y0 in y_coords_mm:
        dy = np.abs(y_bins_centers_mm - y0)
        # where the slice cuts the sphere
        mask = dy <= radius_mm
        # chord length = 2 * sqrt(r^2 - dy^2)
        heights[mask] += 2.0 * np.sqrt(radius_mm**2 - dy[mask]**2)
    return heights

def plot_height_map(time_s, height_matrix, y_bins_centers_mm, z_mm):
    """
    Plot the height_matrix (shape: [n_bins, n_times]) with
    time on x-axis and y on y-axis, color = x-depth (height).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    # imshow expects [row=y, col=time], origin='lower' so y increases upward
    c = ax.imshow(
        height_matrix,
        aspect='auto',
        origin='lower',
        extent=[time_s[0], time_s[-1],
                y_bins_centers_mm[0], y_bins_centers_mm[-1]],
        interpolation='nearest'
    )
    ax.set_title(f'X‑depth vs Time at z = {z_mm:.0f} mm')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Y [mm]')
    fig.colorbar(c, ax=ax, label='Total x‑depth [mm]')
    fig.tight_layout()
    plt.show()

def main():
    # Load JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Process each z-slice
    for key, entry in data.items():
        if not key.startswith('particle_cv_z') or entry.get('system_name') != 'fluid':
            continue

        # parse z in mm from key like 'particle_cv_z50_fluid_1'
        try:
            z_str = key.split('_')[2]   # e.g. 'z50'
            z_mm = float(z_str.lstrip('z'))
        except Exception:
            continue

        times = entry['time']       # list of floats [s]
        values = entry['values']    # list of lists of floats [m]

        # flatten all y’s to find bin range
        all_ys = np.concatenate(values) if any(values) else np.array([])
        if all_ys.size == 0:
            print(f"Skipping z={z_mm:.0f} mm (no data)")
            continue

        # convert y from meters → mm
        all_ys_mm = all_ys * 1000.0

        # build y bins from min–radius to max+radius
        y_min = all_ys_mm.min() - droplet_radius_mm
        y_max = all_ys_mm.max() + droplet_radius_mm
        nbins = int(np.ceil((y_max - y_min) / bin_size_mm))
        y_bins = np.linspace(y_min, y_max, nbins)
        y_centers = (y_bins[:-1] + y_bins[1:]) / 2.0

        # prepare matrix: rows = y‑bins, cols = time steps
        height_mat = np.zeros((len(y_centers), len(times)))

        # for each time step, compute height profile
        for ti, ys in enumerate(values):
            if ys:
                ys_mm = np.array(ys) * 1000.0
                height_mat[:, ti] = compute_height_profile(
                    ys_mm, y_centers, droplet_radius_mm
                )

        # plot it
        plot_height_map(times, height_mat, y_centers, z_mm)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import json
import math
import matplotlib.pyplot as plt

# --- User parameters --------------------------------------------------------

# Path to your JSON data
json_file = '../../post/spray_metrics.json'

# Physical droplet diameter [mm]
droplet_diameter_mm = 5

# ---------------------------------------------------------------------------

def mm_to_points(mm):
    """
    Convert a length in millimetres to points (1 point = 1/72 inch).
    1 inch = 25.4 mm.
    """
    return mm * 72.0 / 25.4

def compute_marker_area(diameter_mm):
    """
    Given a diameter in mm, return the scatter 's' value (points^2) 
    so that the displayed marker has the correct physical size.
    """
    d_pts = mm_to_points(diameter_mm)
    r_pts = d_pts / 2.0
    return math.pi * (r_pts ** 2)

def main():
    # load JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # compute marker area once
    marker_area = compute_marker_area(droplet_diameter_mm)

    # find all particle series
    for key, entry in data.items():
        if not key.startswith('particle_cv_z') or entry.get('system_name') != 'fluid':
            continue

        # extract z coordinate from key like 'particle_cv_z50_fluid_1'
        try:
            z_str = key.split('_')[2]   # e.g. 'z50'
            z = float(z_str.lstrip('z'))
        except Exception:
            continue

        times = entry['time']       # list of floats
        values = entry['values']    # list of lists of floats

        # skip if there is no data at all
        if not any(values):
            print(f"Skipping z={z:.0f} (no data)")
            continue

        # create a new figure
        fig, ax = plt.subplots(figsize=(6,4))
        # plot each droplet at each time
        for t, ys in zip(times, values):
            if ys:
                ax.scatter([t]*len(ys), ys, s=marker_area, color='k', zorder=3)

        ax.set_title(f'Particle positions at z = {z:.0f}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Y position [m]')
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()

        # either show or save; here we show
        plt.show()

if __name__ == '__main__':
    main()

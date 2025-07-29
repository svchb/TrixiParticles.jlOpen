#!/usr/bin/env python3
import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- User parameters --------------------------------------------------------

JSON_FILE            = '../../post/spray_metrics.json'
DROPLET_DIAMETER_MM    = 1.0
DROPLET_RADIUS_MM      = DROPLET_DIAMETER_MM / 2.0
BIN_SIZE_MM            = 0.1
WINDOW_DEPTH = 100

# Treat the plotted "line" as having this physical width in the out-of-plane direction.
LINE_WIDTH_MM          = DROPLET_RADIUS_MM

# cluster threshold (volume-equivalent height per bin, in mm)
MIN_CLUSTER_HEIGHT_MM  = 0.25
# discard bulk clusters larger than this span in y (mm)
MAX_CLUSTER_SIZE_MM    = 30.0

# Histogram bin width for radius distribution (mm) — 1 μm bins
RADIUS_BIN_WIDTH_MM    = 0.001

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
    a_clip = max(a, y0 - r)
    b_clip = min(b, y0 + r)
    if b_clip <= a_clip:
        return 0.0
    term1 = r*r * (b_clip - a_clip)
    term2 = ((b_clip - y0)**3 - (a_clip - y0)**3) / 3.0
    return math.pi * (term1 - term2)


def compute_volume_conserving_height(y_coords_mm, y_edges_mm, radius_mm, line_width_mm):
    nbins = len(y_edges_mm) - 1
    heights = np.zeros(nbins, dtype=float)
    for y0 in y_coords_mm:
        a = y_edges_mm[:-1]
        b = y_edges_mm[1:]
        a_clip = np.maximum(a, y0 - radius_mm)
        b_clip = np.minimum(b, y0 + radius_mm)
        valid = b_clip > a_clip
        if not np.any(valid):
            continue
        term1 = radius_mm**2 * (b_clip[valid] - a_clip[valid])
        term2 = ((b_clip[valid] - y0)**3 - (a_clip[valid] - y0)**3) / 3.0
        V_bins = math.pi * (term1 - term2)
        heights[valid] += V_bins / (line_width_mm * (b[valid] - a[valid]))
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

    # time cell half-widths
    dt = np.zeros(T)
    for i in range(T):
        if i == 0:
            dt[i] = times[1] - times[0]
        elif i == T-1:
            dt[i] = times[-1] - times[-2]
        else:
            dt[i] = 0.5 * (times[i+1] - times[i-1])

    # figure: three panels
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, sharex=False, figsize=(8, 10),
        gridspec_kw={'height_ratios':[3,1,1]}, constrained_layout=True
    )

    # extents
    x_min = times[0] - dt[0]/2
    x_max = times[-1] + dt[-1]/2
    y_min = y_edges_mm[0]
    y_max = y_edges_mm[-1]

    # centers for overlay
    centers = 0.5 * (y_edges_mm[:-1] + y_edges_mm[1:])

    # Top: spectrogram
    im_kwargs = dict(aspect='auto', origin='lower', extent=[x_min, x_max, y_min, y_max], interpolation='nearest')
    if COLORBAR_MIN is not None: im_kwargs['vmin']=COLORBAR_MIN
    if COLORBAR_MAX is not None: im_kwargs['vmax']=COLORBAR_MAX
    img = ax_top.imshow(height_mat, **im_kwargs)
    ax_top.set_title(f'Volume-equivalent height vs Time at z = {z_mm:.0f} mm')
    ax_top.set_ylabel('Y [mm]')
    ax_top.set_xlim(x_min, x_max)
    fig.colorbar(img, ax=ax_top, label='Height [mm]')

    # overlay clusters
    for ti, clusters in clusters_dict.items():
        left = times[ti] - dt[ti]/2
        width = dt[ti]
        for start, end, _, _ in clusters:
            bottom = y_edges_mm[start]
            height = y_edges_mm[end+1] - y_edges_mm[start]
            ax_top.add_patch(Rectangle((left, bottom), width, height, edgecolor='red', facecolor='none', lw=1.2))

    # Middle: corrected diameter
    ax_mid.set_xlabel('Time [s]')
    ax_mid.set_ylabel('Corrected dia. [mm]')
    ax_mid.set_xlim(x_min, x_max)
    all_radii = []
    for ti, clusters in clusters_dict.items():
        t = times[ti]
        for _, _, _, D in clusters:
            ax_mid.scatter(t, D, s=20, c='blue')
            all_radii.append(D)
    ax_mid.grid(True, ls='--', alpha=0.5)

    # Bottom: number density vs radius (log-log)
    y_range_mm = y_max - y_min
    vol_slab_m3 = y_range_mm * DROPLET_DIAMETER_MM * WINDOW_DEPTH * 1e-9
    bins = np.arange(0, MAX_CLUSTER_SIZE_MM+RADIUS_BIN_WIDTH_MM, RADIUS_BIN_WIDTH_MM)
    counts, edges = np.histogram(all_radii, bins=bins)
    centres_r = 0.5*(edges[:-1]+edges[1:])
    pdf = counts / vol_slab_m3
    mask = (centres_r>=0.1) & (pdf>0)
    ax_bot.scatter(centres_r[mask], pdf[mask], s=10)
    ax_bot.set_xlabel('Droplet radius [mm]')
    ax_bot.set_ylabel('Number density [#/m$^3$]')
    ax_bot.set_xscale('log')
    ax_bot.set_yscale('log')
    ax_bot.set_xlim(0.1, MAX_CLUSTER_SIZE_MM)
    ax_bot.grid(True, ls='--', alpha=0.5)

    # add breakup reference slopes
    if mask.any():
        r0 = centres_r[mask][-1]
        log_r = np.log10(centres_r[mask])
        log_pdf = np.log10(pdf[mask])
        for m, style, lbl in [(-2.5,'r--','slope −5/2'), (-2.0,'b:','slope −2')]:
            b = np.mean(log_pdf - m*log_r)
            ref_r = np.array([0.4, r0])
            pdf_line = 10**(b + m*np.log10(ref_r))
            ax_bot.loglog(ref_r, pdf_line, style, label=lbl)
        ax_bot.legend()

    # save or show
    if SAVE_FIGURES:
        fn = os.path.join(OUT_DIR, f'z{int(z_mm)}mm_d{int(DROPLET_DIAMETER_MM*10)}.png')
        fig.savefig(fn, dpi=300)
        fn2 = os.path.join(OUT_DIR, f'z{int(z_mm)}mm_d{int(DROPLET_DIAMETER_MM*10)}.svg')
        fig.savefig(fn2)
        print(f"Saved: {fn}")
        plt.close(fig)
    else:
        plt.show()


def main():
    ensure_out_dir()
    with open(JSON_FILE) as f:
        data = json.load(f)

    for key, entry in data.items():
        if not key.startswith('particle_cv_z') or entry.get('system_name')!='fluid': continue
        try:
            z_mm=float(key.split('_')[2].lstrip('z'))
        except: continue
        times,values=entry['time'],entry['values']
        all_ys=np.concatenate(values) if any(values) else np.array([])
        if all_ys.size==0:
            print(f"Skipping z={z_mm:.0f} mm (no data)"); continue
        ys_mm=all_ys*1000.0
        y_min,y_max=ys_mm.min()-DROPLET_RADIUS_MM,ys_mm.max()+DROPLET_RADIUS_MM
        y_edges=np.arange(y_min,y_max+BIN_SIZE_MM,BIN_SIZE_MM)
        T,len_centers=len(times),len(y_edges)-1
        height_mat=np.zeros((len_centers,T))
        for ti,ys in enumerate(values):
            if ys:
                height_mat[:,ti]=compute_volume_conserving_height(np.array(ys)*1000.0,y_edges,DROPLET_RADIUS_MM,LINE_WIDTH_MM)
        clusters_dict={}
        for ti in range(T):
            mask_h=height_mat[:,ti]>=MIN_CLUSTER_HEIGHT_MM
            raw=find_clusters(mask_h)
            flt=[]
            for s,e in raw:
                span=(e-s+1)*BIN_SIZE_MM
                if span<=MAX_CLUSTER_SIZE_MM:
                    V=LINE_WIDTH_MM*BIN_SIZE_MM*height_mat[s:e+1,ti].sum()
                    D=(6*V/math.pi)**(1/3)
                    flt.append((s,e,V,D))
            clusters_dict[ti]=flt
            sizes=[(e-s+1)*BIN_SIZE_MM for s,e,_,_ in flt]
            Ds=[f"{D:.2f}" for *_,D in flt]
            print(f"t={times[ti]:.2f}s: clusters={len(flt)}, sizes={sizes}, D_corr={Ds}")
        plot_with_clusters(times,height_mat,y_edges,clusters_dict,z_mm)

if __name__=='__main__':
    main()
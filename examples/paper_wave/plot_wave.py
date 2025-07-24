# spray_postprocess_scatter.py — auto-detect heights & dynamic subplots (W&I Fig. 8b)
"""
Pool droplet radii per sampling height (auto-detected) and generate
subplots for each detected height, plotting number density N_d [m⁻³ mm⁻¹]
vs droplet radius r [mm], replicating W&I Fig. 8(b) style.
"""
import json
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import math

# ----------------------------------------------------------------------------
# 1. User parameters
# ----------------------------------------------------------------------------
POST_PATH         = Path("/home/bergers1/dev/TrixiParticles.jlOpen/post")           # dir OR file; auto‑detected
AGG_FILE_NAME = "spray_metrics.json"
WAT_DIR       = Path("watanabe")
FIG_NAME      = "sph_vs_watanabe_subplots.png"

BIN_WIDTH_MM  = 0.001               # 1 µm bins
TANK_WIDTH_M  = 0.20          # 200 mm, to match the light box
TANK_LENGTH_M  = 1.0
SLAB_THICK_M      = 0.1

# ----------------------------------------------------------------------------
# 2. Load aggregated JSON to auto-detect height keys
# ----------------------------------------------------------------------------
post_dir = POST_PATH
agg_path = post_dir / AGG_FILE_NAME
if not agg_path.is_file():
    raise FileNotFoundError(f"Aggregated JSON {agg_path} not found")
with open(agg_path) as f:
    js = json.load(f)
# Regex to match droplet_radii_z<height> or droplet_radii_z<height>_fluid_1
pattern = re.compile(r'^droplet_radii_z(\d+)(?:_fluid_1)?$')
heights = []
key_map: Dict[int, str] = {}
for key in js.keys():
    m = pattern.match(key)
    if m:
        h = int(m.group(1))
        heights.append(h)
        key_map[h] = key
if not heights:
    raise ValueError("No droplet_radii keys found in aggregated JSON")
# Sort heights ascending
heights = sorted(set(heights))
# Prepare per-level storage
per_level: Dict[int, List[float]] = {h: [] for h in heights}

# ----------------------------------------------------------------------------
# 3. Pool all radii per height
# ----------------------------------------------------------------------------
# Each series under js[key]['values'] is a list of radii lists
for h in heights:
    jkey = key_map[h]
    series = js.get(jkey, {})
    for rlist in series.get("values", []):
        if isinstance(rlist, list):
            per_level[h].extend([r*1e3 for r in rlist if isinstance(r, (int, float))])

# Fallback: also scan snapshot files if aggregated missing any
# snapshot_files = sorted(post_dir.glob("spray_metrics_*.json"))
# for fn in snapshot_files:
#     rec = json.load(open(fn))
#     for h in heights:
#         # skip if already have series for this file
#         if key_map[h] in js:
#             continue
#         # try snapshot
#         key_plain = f"droplet_radii_z{h}"
#         key_fluid = f"{key_plain}_fluid_1"
#         for k in (key_plain, key_fluid):
#             if k in rec and isinstance(rec[k], list):
#                 per_level[h].extend([r*1e3 for r in rec[k] if isinstance(r, (int, float))])

# Report pooled counts
print("Pooled droplet counts per detected height:")
for h in heights:
    arr = per_level[h]
    print(f" z={h} mm: {len(arr):,} drops, r ∈ [{min(arr or [0]):.2f}, {max(arr or [0]):.2f}] mm")

# ----------------------------------------------------------------------------
# 4. Compute common histogram bins & slab volume
# ----------------------------------------------------------------------------
all_r = np.hstack([per_level[h] for h in heights if per_level[h]]) if any(per_level.values()) else np.array([BIN_WIDTH_MM])
r_max = all_r.max()
bins = np.arange(0.1, r_max * 1.05 + BIN_WIDTH_MM, BIN_WIDTH_MM)
centres = 0.5 * (bins[:-1] + bins[1:])
slab_vol = TANK_WIDTH_M * TANK_LENGTH_M * SLAB_THICK_M

# ----------------------------------------------------------------------------
# 5. Setup subplot grid
# ----------------------------------------------------------------------------
n = len(heights)
cols = 2
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
axes_flat = axes.flatten()

# ----------------------------------------------------------------------------
# 6. Plot each height
# ----------------------------------------------------------------------------
for idx, h in enumerate(heights):
    ax = axes_flat[idx]
    data = per_level[h]
    if data:
        hist, _ = np.histogram(data, bins=bins)
        pdf = hist / BIN_WIDTH_MM / slab_vol
        ax.scatter(centres, pdf, s=5, label="SPH")
    else:
        pdf = np.zeros_like(centres)
    # W&I overlay if available
    csv = WAT_DIR / f"z{h}.csv"
    if csv.is_file():
        import pandas as pd
        df = pd.read_csv(csv, header=None, names=["r", "Nd"] )
        ax.scatter(df.r, df.Nd, s=20, marker='x', color='tab:orange', label="W&I")

    # reference slopes if data present
    mask = pdf > 0
    if mask.any():
        idx_last = np.where(mask)[0][-1]
        r0 = centres[idx_last]       # mm
        log_r = np.log10(centres[mask])
        log_pdf = np.log10(pdf[mask])
        # slopes to test
        for m, style, lbl in [(-2.5,'r--','slope −5/2'), (-2.0,'b:','slope −2')]:
            b = np.mean(log_pdf - m*log_r)   # intercept in log10 space
            ref_r = np.array([0.4, r0])      # mm → user‑requested range
            pdf_line = 10**(b + m*np.log10(ref_r))
            ax.loglog(ref_r, pdf_line, style, label=lbl)
        # last = np.where(nonzero)[0][-1]     # index of last non-empty bin
        # r0   = centres[last]
        # print(f"r0 r0 = {r0:.3f} mm for z = {h} mm")
        # pdf0 = pdf[last]

        # ref_r = np.array([0.4, r0])
        # ax.loglog(ref_r, pdf0*(ref_r/r0)**(-2.5), 'r--', label="slope −5/2")
        # ax.loglog(ref_r, pdf0*(ref_r/r0)**(-2.0), 'b:',  label="slope −2")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f"z = {h} mm")
    ax.grid(True, which='both', linewidth=0.3)
    # axis labels
    row = idx // cols; col = idx % cols
    ax.set_xlabel("Droplet radius r [mm]")
    ax.set_ylabel(f"Nₙ [m⁻³ mm⁻¹]  (bin width {BIN_WIDTH_MM*1e3:.0f} µm)")
    ax.legend(fontsize='small', loc='upper left')

# remove extra axes
for j in range(n, rows*cols):
    fig.delaxes(axes_flat[j])

# ----------------------------------------------------------------------------
# 7. Adjust layout to prevent label overlap
# ----------------------------------------------------------------------------
# Increase spacing between subplots
fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,
                    wspace=0.3, hspace=0.4)

# ----------------------------------------------------------------------------
# 8. Save figure
# ----------------------------------------------------------------------------
print(f"Saving {FIG_NAME}…")
fig.savefig(FIG_NAME, dpi=300)
plt.close(fig)
print("Done.")

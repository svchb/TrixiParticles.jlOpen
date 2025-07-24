# spray_postprocess.py — build SPH droplet PDF & overlay Watanabe (robust version)
"""
This script now supports **both** output styles produced by the Julia callback:

1. **Snapshot‑per‑file** (`post/spray_metrics_0123.json`, one time step each) —
   earlier prototype.
2. **One big file** (`spray_metrics.json`) that stores *series* under
   `droplet_radii_fluid_1` (as uploaded by you).

It automatically detects which layout is present and merges radii falling in
`TIME_WINDOW_S` before binning.

Run with

    python spray_postprocess.py

and check the printed summary – you should no longer see the “No droplet radii
found” error.
"""

from __future__ import annotations
import json
import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. User‑adjustable parameters
# ----------------------------------------------------------------------------
POST_PATH         = Path("/home/bergers1/dev/TrixiParticles.jlOpen/post")           # dir OR file; auto‑detected
AGG_FILE_NAME     = Path("spray_metrics.json")
WAT_DIR           = Path("watanabe")       # digitised csv files
FIG_NAME          = "sph_vs_watanabe.png"

TIME_WINDOW_S     = (0.6, 1.2)             # analysis window [s]
#BIN_RANGE_MM      = (0.05, 5.0)            # histogram limits (matches paper)
N_BINS            = 60                     # number of log bins

# Tank dimensions (used for number‑density normalisation)
TANK_WIDTH_M      = 1.5                    # spanwise width in the SPH domain
TANK_DEPTH_M      = 1.0                    # out‑of‑plane depth (unit)
OBS_DURATION_S    = TIME_WINDOW_S[1] - TIME_WINDOW_S[0]


# ----------------------------------------------------------------------------
# 2. Helper—flatten + convert units
# ----------------------------------------------------------------------------

def _extend_mm(acc: List[float], radii_m: list[float]) -> None:
    """Append radii converted from metres → millimetres to *acc*."""
    acc.extend(r * 1e3 for r in radii_m)


def collect_radii(path: Path, t_range: tuple[float, float]) -> List[float]:
    """Aggregate all droplet radii [mm] that fall inside *t_range*."""
    radii_mm: List[float] = []

    # ---------------------------------------------------------------------
    # Case A — aggregated file (series stored under droplet_radii_fluid_1)
    # ---------------------------------------------------------------------
    agg_candidate = path / AGG_FILE_NAME if path.is_dir() else path
    if agg_candidate.is_file():
        with agg_candidate.open() as fp:
            js = json.load(fp)
        series = js.get("droplet_radii_fluid_1")
        if series is None:
            raise RuntimeError("Aggregated JSON missing 'droplet_radii_fluid_1'")
        times  = series["time"]
        values = series["values"]         # List[List[float]]
        for t, rlist in zip(times, values):
            if t_range[0] <= t <= t_range[1]:
                _extend_mm(radii_mm, rlist)
        return radii_mm

    # ---------------------------------------------------------------------
    # Case B — many snapshot files in a directory
    # ---------------------------------------------------------------------
    if not path.is_dir():
        raise RuntimeError(f"{path} is neither a directory nor the expected big JSON file")

    json_files = sorted(path.glob("spray_metrics_*.json"))
    for file in json_files:
        with file.open() as fp:
            rec = json.load(fp)
        t = rec.get("time")
        if t is None:
            continue  # skip malformed snapshot
        if t_range[0] <= t <= t_range[1]:
            _extend_mm(radii_mm, rec.get("droplet_radii", []))
    return radii_mm


# ----------------------------------------------------------------------------
# 3. Gather radii & bail out early if none
# ----------------------------------------------------------------------------
print("Collecting droplet radii …")
sph_radii_mm = collect_radii(POST_PATH, TIME_WINDOW_S)
if not sph_radii_mm:
    raise RuntimeError("No droplet radii found in the chosen time window — adjust TIME_WINDOW_S?")
print(f"  gathered {len(sph_radii_mm):,} radii between {TIME_WINDOW_S} s")
print(f"    with Min radius = {min(sph_radii_mm):.1f} mm,  Max radius = {max(sph_radii_mm):.1f} mm")

# ----------------------------------------------------------------------------
# 4. Build SPH number‑density PDF
# ----------------------------------------------------------------------------
r_min, r_max = min(sph_radii_mm), max(sph_radii_mm)
BIN_RANGE_MM = (r_min*0.9, r_max*1.1)
log_bins = np.logspace(np.log10(r_min*0.9), np.log10(r_max*1.1), N_BINS)
hist, edges = np.histogram(sph_radii_mm, bins=log_bins)
centres_mm  = 0.5 * (edges[1:] + edges[:-1])
vol_m3      = TANK_WIDTH_M * TANK_DEPTH_M * OBS_DURATION_S
pdf = hist / np.diff(edges) / vol_m3

# ----------------------------------------------------------------------------
# 5. Plot: SPH curve + reference slopes + Watanabe data
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

# Plot SPH PDF only where pdf>0 to avoid useless markers
mask = pdf > 0
ax.loglog(centres_mm[mask], pdf[mask], "s", label="SPH Δx=5 mm")

# Visual slope guides
ref_r = np.array([0.1, 1.0])
ax.loglog(ref_r, 1e5 * ref_r**(-2),  "k--", label="slope −2")
ax.loglog(ref_r, 1e5 * ref_r**(-2.5), "k:",  label="slope −5/2")

# ---------------------------------------------------------
# Optional: overlay digitised Watanabe curves (if present)
# ---------------------------------------------------------
if WAT_DIR.exists():
    for csv_file in sorted(WAT_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, header=None, names=["radius_mm", "N_d"])
            ax.loglog(df.radius_mm, df.N_d, label=f"W&I {csv_file.stem.replace('_',' ')}")
        except Exception as exc:
            print(f"Warning: could not read {csv_file}: {exc}")

# Aesthetics – dynamic y‑min so the plot never looks empty
nonzero_pdf = pdf[mask]
y_min = nonzero_pdf.min()*0.5 if nonzero_pdf.size>0 else 1
ax.set_xlabel("Droplet radius r [mm]")
ax.set_ylabel("Number density N_d [m⁻³ mm⁻¹]")
ax.set_xlim(BIN_RANGE_MM)
ax.set_ylim(bottom=y_min)
ax.grid(True, which="both", lw=0.25)
ax.legend(fontsize="small")
fig.tight_layout()

print(f"Saving {FIG_NAME} …")
fig.savefig(FIG_NAME, dpi=300)
print("Done.")

"""Verify the compact study bundle and print its principal recorded results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def rows(name):
    with (ROOT / "evidence" / name).open() as stream:
        return list(csv.DictReader(stream))


def verify():
    manifest = json.loads((ROOT / "evidence/manifest.json").read_text())
    for name, record in manifest.items():
        path = ROOT / name
        assert path.stat().st_size == record["bytes"], f"Size mismatch: {name}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], f"Hash mismatch: {name}"
    native = rows("native_self_validation.csv")
    assert len(native) == 24 and all(r["status"] == "completed" for r in native)
    failures = [r for r in rows("native_coupling_matrix.csv")
                if r["case"] in ("gradient", "combined") and r["regularization"] == "native"]
    assert len(failures) == 12 and all(r["status"] == "quality_cutoff" for r in failures)
    reference = {r["control"]: r for r in rows("reference_convergence.csv")}
    assert reference["spatial_64_96"]["passed"] == "False"
    assert reference["outlet_12_24"]["passed"] == "True"
    repeats = rows("actual_pr_repeat_statistics.csv")
    assert len(repeats) == 2 and all(r["stationary"] == "False" for r in repeats)
    assert all(float(r["mass_budget_residual"]) == 0 for r in repeats)
    print(f"Verified {len(manifest)} evidence/figure checksums and the principal result flags.")


def summarize():
    print("Controlled Taylor-Green, n=32 (historical research policy):")
    for row in rows("regularized_tgv_summary.csv"):
        if row["n"] == "32":
            print(f"  {row['case']}: velocity={float(row['velocity_percent']):.3f}%, "
                  f"pressure={float(row['pressure_percent']):.3f}%, density={float(row['density_percent']):.3f}%")
    print("Actual-library cylinder repeats, n=16:")
    for row in rows("actual_pr_repeat_statistics.csv"):
        print(f"  {row['case']}: Cd={float(row['mean_drag']):.6f}, "
              f"lift RMS={float(row['lift_rms']):.6f}, St={float(row['strouhal']):.6f}, "
              f"stationary={row['stationary']}")
    print("Independent force reference remains unaccepted. See REPORT.md and TODO.md.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify bundled evidence before summarizing")
    arguments = parser.parse_args()
    if arguments.check:
        verify()
    summarize()

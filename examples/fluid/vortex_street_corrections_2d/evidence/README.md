# Evidence index

These compact tables and figures are snapshots of completed runs. Verify their
SHA-256 values with `python3 ../study/summarize.py --check` when this directory is
the working directory, or use the repository-root command in the main README.
The manifest uses paths relative to the comparison directory.
Evidence attributes disable automatic newline conversion so hashes also survive
checkouts with `core.autocrlf` enabled. CSV line endings match the recorded exports.

| Files | Interpretation |
| --- | --- |
| `regularized_tgv_summary.csv` | Historical controlled Taylor–Green refinement: geometric regularization, research pair viscosity, pressure reduction off. Three paired layouts per configuration. |
| `operator_consistency.csv` | Region-wise moment/gradient diagnostics; not the conservative pressure-force error. |
| `pressure_reference_controls.csv` | Frozen affine-pressure and gauge tests of none/local/global reference policies. These precede production integration. |
| `ghost_cache_before.csv`, `ghost_cache_after.csv` | Isolated unrelated-cache poisoning control for query-local reconstruction. Ordinary affine reconstruction was already at roundoff. |
| `native_coupling_matrix.csv` | Native/no-shifting cases with independently varied pressure and viscosity; `quality_cutoff` is not a full-horizon result. |
| `native_self_*.csv`, `native_clock_controls.csv` | Self-term isolation, Fourier rates, shifting-only evolution, validation, and initialization/timestep-clock controls. |
| `prototype_cylinder_statistics.csv` | Historical primary runs with the broader research viscosity wrapper; also includes the focused uncorrected boundary control. |
| `actual_pr_repeat_statistics.csv` | Actual integrated-library repeats; includes original-window drift, mean wake, density, and recirculation comparisons. |
| `actual_pr_window_sensitivity.csv` | Explicitly post-hoc startup-window diagnostics. They do not replace the predeclared gate. |
| `actual_pr_short_checks.csv` | Actual-library periodic native-PST checks with pressure reduction on/off and plain library viscosity. The stored `viscosity_policy=local` selects that plain type, not a local uncorrected research implementation. |
| `boundary_comparison.json` | Matched first-/zeroth-order comparison, cycle-block drag interval, mean wake, and interpolated reattachment estimate. |
| `reference_statistics.csv`, `reference_convergence.csv` | Independent-reference temporal/spatial/outlet gates. Acceptance remains false. |
| `reference_force_audit.csv`, `analytic_force_quadrature.csv` | Short transient tolerance/sampling sensitivities and manufactured stress integration. Not accepted steady reference data. |
| `coverage.json` | Coverage of the original correction/boundary screening, excluding subsequent native and actual-repeat expansions. |
| `provenance.json` | Curated commit, source-file, and study-script hashes. Absolute workstation paths are omitted. |

Primary case label `gradient` means `MixedKernelGradientCorrection`, and `combined`
adds Shepard density correction. The original experiments used the correction-stack
branch; legacy-API PR regressions were adapted when integrating the fixes.

Large raw trajectories, fields, complete environments, and frozen scripts are
retained locally under `out/correction_impact_study/`. `manifest.json` maps each
bundled file back to its relative raw dataset path. The historical source archive
is recorded in the main README; it is necessary to replay superseded prototypes.

# Correction-impact study

This directory contains the completed study report, its remaining work, compact
numerical evidence, and a production-operator cylinder demonstration.

- **[Results and conclusions](REPORT.md)**
- **[Prioritized TODOs and acceptance criteria](TODO.md)**
- [Evidence index and provenance](evidence/README.md)

The combined base contains upstream [PR #1293][1293] and [PR #1359][1359]. The
integration used for the actual-library repeats is committed as `300997906`.
The pressure, viscosity, shifting, and ghost fixes are library implementations;
the research operator wrappers and process-local source overrides have been retired.

## Verify the recorded report

No simulations or third-party Python dependencies are needed:

```sh
python3 examples/fluid/vortex_street_corrections_2d/study/summarize.py --check
```

This verifies every bundled evidence/figure checksum and checks the principal
numerical claims. It also prints the condensed tables. The CSVs retain the
prototype/actual-library distinction and original stationarity flags.

## Run the maintained demonstration

From the repository root, prepare an isolated Julia environment:

```sh
julia --project=examples/fluid/vortex_street_corrections_2d -e \
  'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --threads=20 --project=examples/fluid/vortex_street_corrections_2d \
  examples/fluid/vortex_street_corrections_2d/comparison.jl \
  --case=combined --end-time=0.1 --output=out/correction_demo
```

Use `--case=all` for the four correction roles, `--spacing-factor=0.125` for
eight particles per diameter, and `--plot-only` to regenerate comparison figures
after all four cases have completed. `--end-time` is physical time in seconds;
the default one second corresponds to `t U / D = 10`. `--max-seconds=570`
sets the per-case wall budget, checked during integration and after export.

The demonstration uses EDAC, deterministic `SerialUpdate()`, the actual library
viscosity, pressure reduction disabled, and a common raw-kernel geometric position
regularizer capped at 3% of a spacing per accepted step. Wall/cylinder models inherit
the fluid corrections. The four cases are uncorrected, Shepard density, mixed
gradient, and combined density/gradient correction.

These short, coarse trajectories illustrate configuration; their snapshots are
not reference errors or an accuracy ranking. The longer completed runs and their
statistical limitations are documented in the report.

## Reproducibility and cleanup

`evidence/manifest.json` records the source dataset location, byte count, and SHA-256
of each bundled table and figure. `evidence/provenance.json` retains the relevant
source hashes and commit IDs. The full CSV trajectories, fields, frozen repeat
scripts, and pinned environments remain in the local `out/correction_impact_study/`
artifact tree; they are not required to read or verify this compact report.

The superseded exploration was archived locally at
`out/correction_impact_study/archive/exploratory_tooling_2026-10-17/`, with a
checksum manifest, before cleanup. That archive contains the historical operator
prototypes, mutation/override experiments, temporary launchers, reference solver
environment, and original report generators. Those historical scripts require
their recorded library source, not an arbitrary current `HEAD`.

[1293]: https://github.com/trixi-framework/TrixiParticles.jl/pull/1293
[1359]: https://github.com/trixi-framework/TrixiParticles.jl/pull/1359

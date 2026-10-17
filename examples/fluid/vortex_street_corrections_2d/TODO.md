# Prioritized follow-up

Keep numerical correctness, statistical settling, and reference accuracy as
separate acceptance questions. A new batch should have a written scope, timed
pilot, wall budget, source hashes, and a stop condition.

## P0 — Finish code and integration readiness

- [ ] Resolve upstream PR #1359's failing JuliaFormatter job using its recorded CI environment.
  - **Accept:** formatting and the existing functional CI checks are green.
- [ ] Review the #1293/#1359 integration, including mixed fluid schemes, correction-role
  configuration, and pressure-reference topology.
  - **Accept:** keep the 865 integration checks and the actual-library pilot parity;
    retain the regression proving a nonzero physical self-gradient is compatible with zero self-shift.
- [ ] Add hardware validation for the new paths when a working CUDA environment is available.
  - **Accept:** no device scalar indexing; active-buffer pressure reduction and corrected pair
    operators agree with CPU within precision-appropriate tolerances.

## P1 — Close the pressure-force consistency gap cheaply

- [ ] Build a paired manufactured-field suite on regular, jittered, and truncated-support clouds.
  Test constant/affine pressure and velocity fields before long flow runs.
- [ ] Measure pressure-force acceleration error directly, alongside direct and difference
  gradient error, condition numbers, and fallback counts.
- [ ] Sweep a spatially constant pressure gauge and verify pair momentum together with accuracy.
- [ ] Check interfaces between fluid systems and wall/open-boundary neighborhoods.
  - **Accept:** a reference policy/stencil retains gauge invariance and reciprocal force balance,
    with documented refinement of affine acceleration error under disorder.
  - **Do not infer success** solely from interpolation moment identities or small density RMS.

## P1 — Establish reference time convergence before more spatial runs

- [ ] At one independent-reference resolution, compare natural solver sampling with controlled
  timestep limits over the full statistical horizon. Record actual sample times.
- [ ] Compare at least two timestep limits and a consistently tight pressure solve.
- [ ] Analyze time-weighted or resampled statistics without forcing extra short integration steps.
- [ ] Record pressure and viscous force contributions separately and monitor divergence.
  - **Accept:** at least 20 settled cycles; window changes <2% for drag/lift RMS and <1% for St;
    timestep refinement meets the same targets.
- [ ] Only after that gate passes, repeat 64/96-cell spatial and outlet-distance checks.
  - **Accept:** drag/lift RMS changes <2%, St <1%; otherwise keep reference acceptance false.

## P1 — Resolve the combined-cylinder settling issue

- [ ] Examine cycle-aligned statistics and amplitude modulation in the retained trajectories.
  Distinguish a long startup from persistent modulation or numerical noise.
- [ ] Preserve the original 50–200 gate. Label alternative startup windows as post-hoc diagnostics.
- [ ] Add accepted-step restart checkpoints before another long extension.
  Include ODE state, active buffers, boundary quantities/history, and necessary solver state.
  - **Accept:** a short restart control reproduces uninterrupted diagnostics within a stated tolerance.
- [ ] Choose a longer sampling horizon from the measured modulation time, then extend only the
  representative actual-library cases that need it.
  - **Accept:** at least 20 settled cycles and all original stationarity tolerances, with uncertainty
    estimated from cycle/block statistics rather than independent time samples.

## P2 — Expand native-shifting validation selectively

- [ ] Test the zero-distance exclusion near walls and open boundaries, including activation,
  buffer reuse, unequal particle states, and 3D.
- [ ] Cover relevant callback and stage-updated PST variants; retain physical self-gradient tests.
- [ ] Measure timestep/strength behavior and separation/energy diagnostics, not only solver retcode.
  - **Accept:** no spurious self-shift, finite/bounded particle states, and no regression in the
    corrected physical operators or required cache preparation.

## P2 — Revisit wider flow refinement after the gates pass

- [ ] Repeat only the actual-library resolutions needed to resolve a demonstrated difference.
- [ ] Hold domain, timestep policy, wall correction, regularizer, and startup/statistics windows fixed.
- [ ] Compare mean wakes, interpolated recirculation, pressure profiles, and cost versus achieved error.
- [ ] Reconsider boundary-model changes only when a measured boundary error exceeds sampling/refinement
  uncertainty; the completed uncorrected order comparison did not resolve a bulk force improvement.

## Completed prerequisites

- [x] Four defects isolated and included in PR #1359.
- [x] Native-PST collapse isolated from pressure and viscosity; 24 zero-self validation runs completed.
- [x] Deterministic ordering controls established.
- [x] Independent solver checked against Taylor–Green; reference-force failure explicitly retained.
- [x] Matched first-order boundary control completed within the focused budget.
- [x] Two actual-library corrected-cylinder repeats completed with zero recorded mass-budget residual.
- [x] Experimental policy wrappers and one-off launchers archived and removed from the live study.

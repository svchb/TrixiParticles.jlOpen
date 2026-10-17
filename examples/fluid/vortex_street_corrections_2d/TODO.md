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

## P1/P2 — Close the explicit control-run backlog

- [ ] Review and disposition every row below before preparing another batch. The original
  16-control matrix is not complete: **1 completed, 2 failed, 1 interrupted, 11 pending,
  and 1 redundant**. One of the pending cases is a legacy diagnostic; ten are production controls.
- [ ] Run selected production controls using the actual integrated library and matched baselines.
  Preserve failed/interrupted results as such; never label a short diagnostic as a completed
  `tU/D=200` control.
- [ ] Use a short pilot for each previously unmeasured boundary implementation. Set an explicit
  wall cap, verify source/parameter parity, and update the estimates before starting a long job.
- [ ] Interpret corrected-case controls only after sufficient settled sampling. The actual-library
  mixed-gradient/combined repeats are completed runs but fail the original stationarity gate.

### Timing assumptions and measured calibration

SPH estimates below are **wall hours on the study machine (Intel Core Ultra 9 285),
20 Julia threads, one BLAS thread, deterministic rebuilding**, n=16 particles per
diameter and Re=200. Production controls target `tU/D=200`, use the same fixed timestep,
and sample `50–200` unless a new protocol explicitly changes that window.

Measured anchors are **6.37 h uncorrected**, **6.00 h first-order uncorrected**,
**7.92 h actual-library mixed gradient**, and **13.32 h actual-library combined**.
The last value supersedes the optimistic 8.5 h combined pilot projection.
These are planning estimates, not promised completion or convergence times:

- Doubling total L/D from 12 to 24 approximately doubles particle/neighbor work.
  The cylinder remains at x/D=3; this changes downstream distance from 9D to 21D.
- Halving the fixed timestep approximately doubles step count.
- For these known scaling changes, ranges include roughly 30% planning allowance.
- Unpiloted boundary estimates assume about 1.5–2 times baseline cost for stage-first
  reconstruction and 1–1.3 times for characteristics, with a further upper allowance.
  Wall-off combined cost is estimated at 0.8–1 times baseline before allowance.
- Ranges exclude development time and fresh dependency installation. Shared-machine
  contention, changed particle distributions, or solver failure can change them materially.

### All original controls: status, cost, and intended disposition

The run IDs are retained verbatim from the original campaign. `original_numerics`
means the **pre-fix** local-pressure/original-viscosity policy, not the current PR's
global reference. Those rows are historical defect diagnostics. Their optional
reproduction estimates apply to a **capped `tU/D<=25` probe**, not their abandoned
full-horizon target of 200. Retained evidence requires no additional simulation.
Reproduction must use the recorded pre-fix source from the archive: setting the current
`average_pressure_reduction=true` selects the fixed policy and does not recreate the legacy one.

| Original run ID | Status | Horizon for the next action | Wall-time estimate | Disposition / priority |
| --- | --- | --- | --- | --- |
| `original_numerics_16_uncorrected` | Failed: nonfinite at 18.2 | No rerun required; optional legacy probe <=25 | 0 h to reuse; 0.5–1 h optional probe | Retain failure evidence; no routine full-horizon rerun |
| `original_numerics_16_density` | Failed: nonfinite at 19.2 | No rerun required; optional legacy probe <=25 | 0 h to reuse; 0.6–1.2 h optional probe | Retain failure evidence; no routine full-horizon rerun |
| `original_numerics_16_gradient` | Interrupted by user at 15.7 | Optional legacy probe <=25 | 1–2 h if selected | Partial stability diagnostic, not a completed control |
| `original_numerics_16_combined` | Pending, legacy only | Optional legacy probe <=25 | 1–2.5 h if selected | Run only to answer an unresolved defect question; otherwise close as superseded |
| `boundary_16_uncorrected_first` | Completed to 200; stationary | Already complete | **6.00 h measured** | Reuse focused result; no repetition planned |
| `boundary_16_uncorrected_stage_first` | Pending | 200 | **10–17 h** | P2; only if a stage-consistency defect warrants reviving the experimental treatment |
| `boundary_16_uncorrected_characteristic` | Pending | 200 | **6.5–11 h** | P2; establish parity and quiet-flow initialization first |
| `boundary_16_combined_first` | Pending | 200 | **12.5–16.5 h** | P2; compare with a sufficiently settled combined baseline |
| `boundary_16_combined_stage_first` | Pending | 200 | **20–35 h** | P2; stage treatment needs a measured justification and pilot |
| `boundary_16_combined_characteristic` | Pending | 200 | **13.5–23 h** | P2; pilot first and retain the combined-case stationarity gate |
| `domain_16_uncorrected` | Pending | 200, total L/D=24 | **13–17 h** | P2; isolate SPH outlet-distance effects |
| `timestep_16_uncorrected` | Pending | 200, dt/2 | **13–17 h** | P1 time-convergence control; pilot confirms approximately doubled work |
| `wall_16_uncorrected` | Redundant | Reuse baseline; optional setup/parity check | **0 h; 5–10 min optional check** | Both wall-role settings select no correction, so do not schedule a full run |
| `domain_16_combined` | Pending | 200, total L/D=24 | **27–35 h** | P2; requires a settled matched baseline |
| `timestep_16_combined` | Pending | 200, dt/2 | **27–35 h** | P1 after restart/settling work; compare settled statistics, not snapshots |
| `wall_16_combined` | Pending | 200, wall/cylinder corrections disabled | **11–18 h** | Conditional P1 diagnostic of boundary coupling; pilot and settling checks required |

Scheduling all ten pending production controls would cost roughly **150–225 sequential
wall hours (6–9 days)** at these rates. This is a backlog, not a new automatic queue.
Select controls after the operator/reference/settling gates identify a question they can resolve.
The two completed actual-library repeats are separate from these 16 controls: mixed gradient
took 7.92 h and combined 13.32 h; neither completes the pending domain/timestep/wall matrix.

### Reference and settling runs needed by the P1 tasks

These are **additional proposed runs**, not entries in the original 16-control matrix.
Reference estimates use **8 Julia threads** and the measured 20-unit audit timings:
64-cell tight/exact, 138.64 s; 64-cell default/natural, 79.78 s; 96-cell tight/exact,
642.24 s. Extrapolation to 200 plus margin gives the ranges below. Natural/timestep-
limited and extended-domain variants are unpiloted; confirm their rates before launch.

| Proposed run | Horizon / method | Estimated wall time | Gate |
| --- | --- | --- | --- |
| Reference 64-cell natural-time baseline | 200, tight pressure solve, controlled maximum dt | **0.4–0.8 h** | Define the sampling protocol and record actual sample times |
| Reference 64-cell timestep refinement | 200, same protocol, maximum dt/2 | **0.8–1.6 h** | Compare with the preceding run before more spatial refinement |
| Reference 96-cell spatial check | 200, validated time/sampling protocol | **1.8–3 h** | Only after 64-cell time convergence passes |
| Reference 64-cell outlet-distance check | 200, total L/D=24, validated protocol | **1–2 h** | Match the accepted 12D reference; not just the earlier failed protocol |
| Restart-equivalence control, each corrected SPH case | Short paired uninterrupted/restarted trajectory, about 2 advective units | **10–20 min per case** | Implement checkpoint contents first; verify equivalence before extensions |
| Fresh mixed-gradient settling run | 0–300, checkpointed | **12–16 h** | Conditional: reassess existing late-window data before selecting it |
| Fresh combined settling run | 0–300, checkpointed | **20–26 h** | Predeclare a settled sampling window and retain the original-window result |
| Future mixed-gradient extension | 200–300 from a validated full checkpoint | **4–5.5 h** | Available only for a future checkpointed run, not the retained outputs |
| Future combined extension | 200–300 from a validated full checkpoint | **7–9 h** | Available only for a future checkpointed run, not the retained outputs |

The existing long SPH runs did **not** save complete restart states; interpolated final
fields are not exact restart checkpoints. Consequently, the cheaper extension budgets
cannot currently be used to continue those runs. No runtime is assigned yet to undefined
3D/GPU configurations or to implementation work; specify the mesh/device and time a pilot first.

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

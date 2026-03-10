# Gallery Regeneration Plan

## Status

- Overall status: Phase 5 complete, Phase 6 ready
- Owner: Unassigned
- Last updated: 2026-03-10
- Primary code repo: `TrixiParticles.jl`
- Public media host: `codebase.helmholtz.cloud` GitLab Pages
- Trigger mode: Manual only

## Goal

Build a reproducible, manual gallery regeneration pipeline that:

1. runs selected examples at higher resolution on the HPC cluster,
2. stores raw simulation output only on HPC storage,
3. renders poster images and animations with ParaView in headless mode where applicable, and supports poster-only workflows for non-ParaView examples,
4. converts selected animations to GIF and/or MP4,
5. publishes only final media to a separate public GitLab Pages repo,
6. regenerates `docs/src/examples.md` from structured metadata.

## Non-Negotiable Constraints

- [ ] Gallery regeneration must never run automatically in CI.
- [ ] Raw `VTK`, `VTU`, `PVD`, logs, and frame sequences must never be committed to GitHub.
- [ ] Final media must not be committed to this repo.
- [ ] The main docs build must consume already-generated gallery content only.
- [ ] Public media URLs must come from the separate GitLab Pages repo.
- [ ] `docs/src/examples.md` must become generated output, not a manually maintained file.
- [ ] The cluster-side workflow must run from a separate cluster-local checkout of `TrixiParticles.jl`.
- [ ] This prompt must not assume it can execute cluster commands directly.
- [ ] The workflow is intentionally split into:
  - [ ] cluster-side simulation,
  - [ ] cluster-side ParaView rendering,
  - [ ] local encoding with `ffmpeg`,
  - [ ] manual upload to GitLab Pages.

## Current State

- [x] `docs/src/examples.md` exists and is currently hand-written.
- [x] Example media is currently referenced via external GitHub asset URLs.
- [x] Example simulations already write ParaView-readable output via `SolutionSavingCallback`.
- [x] Many example files already expose resolution-related parameters that can be overridden.
- [x] GitLab Pages is available publicly on `codebase.helmholtz.cloud`.
- [x] `TrixiParticles.jl` can be checked out directly on the cluster.
- [x] Cluster commands cannot be executed from within the current prompt and must be run by a human in a separate cluster session.
- [x] The intended operator workflow is:
  1. run one script on the cluster to execute all gallery simulations,
  2. run one script on the cluster to generate gallery media with ParaView,
  3. download the rendered images locally,
  4. run `ffmpeg` locally,
  5. upload the final media manually to Helmholtz GitLab.
- [x] `docs/gallery/manifest.toml` exists with a minimal schema and the full current inventory.
- [x] `docs/gallery/src/GalleryPipeline.jl` exists with typed manifest loading via `Configurations.jl`.
- [x] `docs/gallery/run_cluster_gallery.jl` provides the shared Phase 3 command surface.
- [x] `docs/gallery/validate_manifest.jl` is a thin wrapper around the shared validation command.
- [x] Phase 4 run layouts and metadata are derived in `docs/gallery/src/GalleryPipeline.jl`.
- [x] A manifest-driven simulation pipeline exists for `run-examples`.
- [ ] No headless ParaView render automation exists yet.
- [ ] No publishing bridge to the media repo exists yet.
- [ ] No generated gallery page template exists yet.

## Execution Model

The workflow is split into cluster-side generation and a manual publish step.

1. A human operator updates a separate cluster checkout of `TrixiParticles.jl` to the intended commit.
2. The operator runs one cluster-side script that executes all gallery simulations and stores raw outputs on cluster storage.
3. The operator runs one cluster-side script that reads the simulation outputs and generates poster images and frame sequences with ParaView.
4. The operator downloads the rendered images to a local machine.
5. The operator runs local `ffmpeg` post-processing to generate GIF and optional MP4 files.
6. The operator uploads only the final media manually to the Helmholtz GitLab Pages repo.
7. The examples page is generated from deterministic public URL paths that correspond to the uploaded media layout.

## Target Architecture

### Main Repo Responsibilities

- Store code, docs, cluster-run scripts, manifest, templates, and operator instructions.
- Validate manifests and local prerequisites.
- Provide cluster-side scripts for simulation and rendering.
- Provide local post-processing steps for encoding.
- Define deterministic media output paths for manual upload.
- Generate `docs/src/examples.md`.

### Media Repo Responsibilities

- Host only final public gallery assets.
- Serve stable public GitLab Pages URLs for `png`, `gif`, and `mp4`.
- Keep a simple and predictable directory layout.

### HPC Responsibilities

- Run expensive high-resolution simulations when started by a human operator.
- Store raw outputs temporarily.
- Run headless ParaView for poster and frame generation.
- Produce poster PNGs and frame sequences in a known download directory.
- Retain or clean up transient data according to the pipeline policy.

### Local Post-Processing Responsibilities

- Download rendered images from the cluster.
- Run `ffmpeg` locally.
- Produce final `gif` and optional `mp4` files.
- Assemble the final upload directory for the media repo.

## Planned File Layout

This is the intended layout in the main repo.

```text
docs/
  gallery/
    PLAN.md
    manifest.toml
    src/
      GalleryPipeline.jl
    run_cluster_gallery.jl
    paraview/
      render_gallery.py
    templates/
      examples.md.tpl
```

## Success Criteria

- [ ] One command can regenerate one gallery entry end to end.
- [ ] A second command mode can regenerate all configured entries.
- [ ] The pipeline can rerun a single example without touching others.
- [ ] Published media URLs are stable and deterministic.
- [ ] `docs/src/examples.md` is rewritten from the manifest and published URLs.
- [ ] The standard docs deployment can publish the generated page without starting HPC work.
- [ ] Operators can clean up transient outputs safely.
- [ ] The cluster checkout can run the full generation flow with two scripts.
- [ ] Local `ffmpeg` post-processing is a defined step after downloading rendered images.
- [ ] Manual upload to Helmholtz GitLab is the only non-scripted publishing step.

## Progress Tracker

## Phase 0: Confirm Inputs and External Resources

### Objective

Lock down the external assumptions so the implementation does not have to be redesigned later.

### Tasks

- [x] Create the separate GitLab media repo on `codebase.helmholtz.cloud`.
- [x] Enable GitLab Pages for that repo.
- [x] Record the final public base URL for published assets.
- [ ] Decide who owns write access to the media repo.
- [x] Define how the repo checkout is created or updated on the cluster, likely via `git clone` and `git pull`.
- [x] Decide the canonical branch for the media repo.
- [x] Decide whether published assets are overwritten in place or versioned by commit/date.
- [x] Decide whether `gif`, `mp4`, or both are required per example.
- [x] Decide how long transient HPC outputs are retained after publish.
- [x] Determine headless ParaView availability on the cluster.
- [x] Determine `ffmpeg` availability on the cluster.
- [x] Confirm the cluster scheduler interface and required Slurm options.
- [x] Decide the encoding strategy if `ffmpeg` remains unavailable on the cluster.
- [x] Confirm local `ffmpeg` availability for post-processing.
- [x] Decide the exact directory layout of final media that will later be uploaded manually.

### Deliverables

- [x] A documented media repo URL.
- [x] A documented authentication strategy.
- [x] A documented retention policy.
- [x] A documented execution environment for each pipeline stage.
- [x] A documented cluster checkout/update procedure.
- [x] A documented upload directory layout that maps directly to public GitLab Pages URLs.
- [x] A documented cluster tool-availability summary.
- [x] A documented local post-processing environment.

### Exit Criteria

- [x] All external dependencies are known well enough to implement Phase 1 without guessing.

## Phase 1: Define the Gallery Inventory

### Objective

Define exactly which examples belong in the gallery and what each entry needs.

### Tasks

- [x] Review all standalone example scripts under `examples/`.
- [x] Decide which files are direct duplicates or support files rather than gallery entries.
- [x] Identify all additional examples that must be added beyond the current hand-written page.
- [x] Assign each example a stable `id`.
- [x] Assign each example a category such as `fluid`, `fsi`, or `structure`.
- [x] Decide the display title for each example.
- [x] Decide the default output policy for each example: animated entries get `poster.png` plus `animation.gif`, while static entries get `poster.png`, with `animation.mp4` optional and deferred.
- [x] Decide the primary asset strategy for each example.
- [x] Decide whether each example needs one or more sources loaded in ParaView or a non-ParaView poster workflow.
- [x] Decide the default caption text per example.

### Deliverables

- [x] Complete non-duplicate gallery list.
- [x] Stable per-example identifiers and display metadata.
- [x] Inventory recorded in `docs/gallery/INVENTORY.md`.

### Exit Criteria

- [x] The gallery scope is explicit and frozen enough to start implementation.

## Phase 2: Design the Manifest

### Objective

Create one structured source of truth that drives simulation, rendering, publishing, and page generation.

Phase 2 also fixes the global gallery render style:

- all gallery assets use a white background
- all scalar coloring uses the `viridis` color scheme

Phase 2 intentionally uses a minimal manifest schema.

Required per-entry fields:

- `id`
- `title`
- `example`
- `asset_strategy`

Derived fields:

- `category` is derived from the example path prefix
- `outputs` are derived from the asset strategy unless explicitly overridden
- `media_repo_path` is derived from `category` and `id`

Optional fields are added only when an example actually needs them:

- `caption`
- `outputs`
- `run`
  - `run.overrides` holds `trixi_include` keyword overrides such as `tspan` and resolution-related parameters
  - `run.save_dt` can override the example save cadence when the gallery needs a different one
- `render`
- `notes`

### Tasks

- [x] Create `docs/gallery/manifest.toml`.
- [x] Define the top-level schema.
- [x] Define global render defaults in the manifest:
  - [x] `background = "white"`
  - [x] `color_scheme = "viridis"`
- [x] Define one manifest entry per gallery item.
- [x] Reduce the per-entry schema to the smallest sensible required set:
  - [x] `id`
  - [x] `title`
  - [x] `example`
  - [x] `asset_strategy`
- [x] Define derived fields instead of storing them redundantly:
  - [x] `category`
  - [x] `outputs`
  - [x] `media_repo_path`
- [x] Keep only the following as optional per-entry extensions when needed later:
  - [x] `caption`
  - [x] `outputs`
  - [x] `run`
    - [x] use `run.overrides` for `tspan` and resolution-like parameter changes
  - [x] `render`
  - [x] `notes`
- [x] Define validation rules for required fields.
- [x] Decide whether schema validation happens in Julia at runtime or through a separate check.
- [x] Document the meaning of each field inside the plan or inline comments.
- [x] Document that white background and `viridis` are fixed gallery-wide rendering rules, not per-entry style choices.

Validation rules fixed in Phase 2:

- top-level keys are limited to `schema_version`, `render_defaults`, `path_conventions`, `asset_strategies`, and `entries`
- `schema_version` must be `1`
- `render_defaults.background` must be `white`
- `render_defaults.color_scheme` must be `viridis`
- every asset strategy must define `render_backend` and `outputs`
- `render_backend` must currently be one of `paraview` or `julia`
- strategy outputs must be non-empty, unique, and drawn from `poster`, `gif`, `mp4`
- every entry must contain non-empty `id`, `title`, `example`, and `asset_strategy`
- entry `id` values must be unique and match `^[a-z0-9_]+$`
- entry `example` values must be unique, relative, end in `.jl`, and resolve to an existing file under `examples/`
- entry `asset_strategy` must reference a defined strategy
- derived `media_repo_path` values must be unique
- entry `outputs`, if present, must be unique, must contain `poster`, and must use only `poster`, `gif`, `mp4`
- `run`, if present, must be a table
- `run.overrides`, if present, must be a non-empty table with TOML values that can be forwarded as `trixi_include` kwargs
- `run.save_dt`, if present, must be positive
- `render`, if present, must be a table
- `caption` and `notes`, if present, must be non-empty strings

Phase 2 decision on validation:

- validation happens in Julia at runtime
- long term, manifest loading should use a typed configuration library instead of a fully hand-written schema validator
- preferred direction: `Configurations.jl` for TOML-to-typed-struct loading
- fallback direction: `JSONSchema.jl` only if we decide a declarative schema file is more valuable than typed Julia structs
- Phase 3 implements `Configurations.jl`-backed typed loading in `docs/gallery/src/GalleryPipeline.jl`
- semantic checks remain custom only for repo-specific rules such as duplicate IDs, example file existence, and unique derived media paths

### Deliverables

- [x] Manifest file covering the current inventory.
- [x] Documented minimal schema expectations.
- [x] Documented global render defaults for white background and `viridis`.
- [x] Temporary Julia validator for the manifest schema.

### Exit Criteria

- [x] The manifest schema is expressive enough to drive one example end to end without adding new schema fields.
- [x] The current manifest passes a concrete validator.

## Phase 3: Create the Entry Point and Pipeline Skeleton

### Objective

Create a stable CLI surface for cluster-side generation and page generation.

Phase 3 should also replace the bootstrap manifest validator with a library-backed typed loader.

### Tasks

- [x] Add `docs/gallery/src/GalleryPipeline.jl`.
- [x] Add `docs/gallery/run_cluster_gallery.jl`.
- [x] Choose the manifest loading approach.
  - [x] Prefer `Configurations.jl` for typed manifest loading from TOML.
  - [x] Use `JSONSchema.jl` only if we decide to maintain an explicit schema document.
  - [x] Keep only semantic checks custom, such as duplicate IDs, existing example paths, and unique derived media paths.
- [x] Decide the command interface.
- [x] Implement subcommands or flags for:
  - [x] `validate`
  - [x] `run-examples`
  - [x] `render-media`
  - [x] `encode-media`
  - [x] `page`
  - [x] `all`
- [x] Implement argument parsing.
- [x] Implement manifest loading with typed structs.
- [x] Implement structured logging.
- [x] Implement dry-run support where possible.
- [x] Implement selection by example `id`.
- [x] Implement selection of all examples.
- [x] Implement early failure when required configuration is missing.

### Deliverables

- [x] A documented cluster-run command surface.
- [x] A documented local post-processing command surface.
- [x] A reusable Julia module for pipeline orchestration.

### Exit Criteria

- [x] An operator can run `validate` and get a meaningful result.

## Phase 4: Standardize Working Directories and Metadata

### Objective

Make each run reproducible and easy to inspect.

### Tasks

- [x] Define the directory layout used on HPC storage.
- [x] Decide whether runs are keyed by:
  - [x] example `id`
  - [x] git commit
  - [x] manifest hash
  - [x] timestamp
- [x] Create deterministic subdirectories for:
  - [x] raw simulation output
  - [x] ParaView frames
  - [x] final encoded media
  - [x] logs
  - [x] metadata
- [x] Define a metadata file format, likely JSON or TOML.
- [x] Include in metadata:
  - [x] example `id`
  - [x] source example path
  - [x] code commit hash
  - [x] manifest values used
  - [x] job ID
  - [x] tool versions if available
  - [x] start and end timestamps
  - [x] output paths
- [x] Define how reruns overwrite or version previous outputs.

### Phase 4 Decisions

- Workspace roots are external to the repo and resolved from `--workspace-root` or `TRIXIPARTICLES_GALLERY_WORKSPACE`.
- Run directories are keyed by `entry.id`, git commit, and manifest hash.
- Timestamps are recorded in metadata, but are intentionally not part of the deterministic directory key.
- Deterministic run layout:
  - `runs/<entry-id>/git-<commit-short>/manifest-<manifest-hash-short>/raw`
  - `runs/<entry-id>/git-<commit-short>/manifest-<manifest-hash-short>/frames`
  - `runs/<entry-id>/git-<commit-short>/manifest-<manifest-hash-short>/encoded`
  - `runs/<entry-id>/git-<commit-short>/manifest-<manifest-hash-short>/logs`
  - `runs/<entry-id>/git-<commit-short>/manifest-<manifest-hash-short>/metadata`
- Metadata format is TOML.
- The metadata file is stage-specific, for example `metadata/run-examples.toml`.
- Rerun policy is overwrite-in-place for the same `entry.id + git commit + manifest hash`.
- Git dirty state is recorded in metadata so non-committed changes are visible in provenance.

### Deliverables

- [x] Run directory specification.
- [x] Metadata schema.

### Exit Criteria

- [x] Every run can be traced back to the exact code and configuration that created it.

## Phase 5: Implement the HPC Simulation Stage

### Objective

Run one gallery example non-interactively on the cluster from the cluster-local checkout.

### Tasks

- [x] Decide whether the cluster run script calls `sbatch`, `srun`, or direct Julia execution.
- [x] Implement the cluster-side simulation entrypoint in `docs/gallery/run_cluster_gallery.jl`.
- [x] Implement a mode that runs all configured gallery examples.
- [x] Implement a mode that runs a single selected gallery example.
- [x] Reuse `trixi_include` override patterns instead of cloning example files.
- [x] Replace the default saving callback at runtime with a gallery-specific one.
- [x] Ensure output goes to the configured transient HPC directory.
- [x] Implement optional manifest-driven save cadence overrides while preserving example defaults otherwise.
- [x] Ensure log output is captured to a file.
- [x] Capture solver failure cleanly.
- [x] Write a metadata file even on failure.
- [x] Verify that raw outputs are not written into the repo.

### Per-Example Considerations

- [x] Confirm that `fluid/dam_break_2d.jl` can be driven with high-resolution overrides only.
- [x] Confirm that `fsi/dam_break_plate_2d.jl` does not require manual source edits for higher resolution.
- [x] Confirm that structure examples like `oscillating_beam_2d.jl` can be driven similarly.
- [ ] Maintain an empirical watchlist of examples that need physics-setting changes during higher-resolution pilot runs.

Verified locally on 2026-03-10 with `fluid_particle_spacing=0.01` and
`tspan=(0.0, 0.05)` passed via `trixi_include` runtime overrides only. The run finished with
solver retcode `Success` and wrote raw output to `/tmp/trixip-gallery-dam-break-check/raw`
without any source edits.

Verified locally on 2026-03-10 that `fsi/dam_break_plate_2d.jl` also runs without source edits
at a finer setting using only runtime overrides: `fluid_particle_spacing=0.008`,
`n_particles_x=7`, and `tspan=(0.0, 0.05)`. The run kept the default source file unchanged,
finished with solver retcode `Success`, and wrote raw output to
`/tmp/trixip-gallery-plate-check/raw`.

This last point cannot be decided reliably from code inspection alone. Only actual
higher-resolution runs can show whether an example needs physics-setting changes, for example a
different boundary model, tighter tolerances, or split integration.

Empirical watchlist:

- None identified yet from completed pilot runs.
- `fluid/dam_break_2d.jl`: pilot run succeeded with runtime resolution overrides only; no physics-setting change observed.
- `fsi/dam_break_plate_2d.jl`: pilot run succeeded with runtime resolution overrides only; no physics-setting change was required for the tested case.
- `fsi/dam_break_plate_2d.jl`: keep under observation at still finer fluid resolutions, because the source file documents that a different structure boundary model may become preferable when the compact support is no longer fully sampled.

### Deliverables

- [x] One successful local simulation run for one manifest entry.
- [ ] One successful cluster-side simulation run for one manifest entry.

### Exit Criteria

- [x] The pipeline can produce raw `.pvd/.vtu` output for one example in a deterministic location.

## Phase 6: Implement Cluster-Side Run Orchestration

### Objective

Make it easy to run all gallery simulations from the separate cluster checkout.

### Tasks

- [ ] Implement a cluster-side `run-examples` command that iterates over all manifest entries.
- [ ] Decide how failures are summarized after a batch run.
- [ ] Record per-example run status in metadata.
- [ ] Report states such as:
  - [ ] pending
  - [ ] running
  - [ ] failed
  - [ ] completed
- [ ] Support running one example or all examples.
- [ ] Decide whether all examples are submitted independently or as a job array.
- [ ] Surface failed example IDs clearly after the cluster run finishes.

### Deliverables

- [ ] A working `run-examples` flow.

### Exit Criteria

- [ ] An operator can run all gallery simulations from the cluster checkout with a single command.

## Phase 7: Implement Headless ParaView Rendering

### Objective

Render reproducible poster images and frame sequences from raw simulation outputs.

### Tasks

- [ ] Add `docs/gallery/paraview/render_gallery.py`.
- [ ] Implement CLI argument parsing in the ParaView script.
- [ ] Load all required `.pvd` sources for each example.
- [ ] Apply the correct representation for particle data.
- [ ] Reproduce the current Point Gaussian styling in an automated form.
- [ ] Support separate 2D and 3D presets if needed.
- [ ] Apply example-specific camera settings from the manifest.
- [ ] Apply coloring by the configured field.
- [ ] Apply fixed or computed color limits.
- [ ] Render a poster PNG for each example.
- [ ] Render a frame sequence for animated examples.
- [ ] Add a cluster-side `render-media` command that processes all configured examples.
- [ ] Fail clearly if expected source files are missing.
- [ ] Record render outputs in the metadata file.

### Deliverables

- [ ] One poster PNG for one example.
- [ ] One frame sequence for one example.

### Exit Criteria

- [ ] ParaView rendering is fully non-interactive for at least one example.

## Phase 8: Implement Local Encoding to GIF and MP4

### Objective

Turn downloaded frame sequences into final web-consumable media on the local machine.

### Tasks

- [ ] Decide the standard output filenames, for example:
  - [ ] `poster.png`
  - [ ] `animation.gif`
  - [ ] `animation.mp4`
- [ ] Implement `ffmpeg` invocation from the pipeline.
- [ ] Define the local input layout for downloaded frame sequences.
- [ ] Generate MP4 for entries that request video.
- [ ] Generate optimized GIF for entries that request GIF.
- [ ] Use `palettegen` and `paletteuse` for better GIF quality.
- [ ] Decide the target resolution and frame rate.
- [ ] Verify that final files are small enough for Pages and practical browser use.
- [ ] Record final media paths in metadata.

### Deliverables

- [ ] One valid GIF or MP4 for one example.

### Exit Criteria

- [ ] The pipeline can produce final media artifacts from rendered frames without manual work.

## Phase 9: Download Rendered Images and Prepare Final Media for Manual Upload

### Objective

Download rendered images from the cluster and assemble a final upload directory for manual upload.

### Tasks

- [ ] Define the transfer path from cluster rendering outputs to the local machine.
- [ ] Download only the rendered images needed for local encoding.
- [ ] Copy only final media files into the local upload directory.
- [ ] Copy or generate a compact metadata summary if useful.
- [ ] Ensure raw `VTK/PVD` data is excluded from the local upload directory.
- [ ] Verify prepared upload outputs match the manifest entry and expected filenames.
- [ ] Fail clearly if required files are missing.

### Deliverables

- [ ] One upload-ready directory containing final media plus optional metadata for one example.

### Exit Criteria

- [ ] A human operator can upload the prepared directory to Helmholtz GitLab without additional processing.

## Phase 10: Document Manual Upload to the GitLab Pages Media Repo

### Objective

Define the manual upload step clearly and keep it outside the automated pipeline.

### Tasks

- [ ] Decide the public directory layout in the media repo.
- [ ] Document exactly which files are uploaded manually.
- [ ] Document the target directory layout in the GitLab Pages repo.
- [ ] Verify that the expected GitLab Pages URLs are derivable deterministically from the manifest.
- [ ] Verify that the expected GitLab Pages URLs resolve publicly after upload.
- [ ] Record the final public URL pattern for each example.

### Deliverables

- [ ] One example with manually uploaded public media hosted via GitLab Pages.

### Exit Criteria

- [ ] The final media URL for one example is public and stable.

## Phase 11: Implement Gallery Page Generation

### Objective

Replace manual `examples.md` editing with generated content.

### Tasks

- [ ] Add `docs/gallery/templates/examples.md.tpl`.
- [ ] Decide the generated page structure.
- [ ] Implement a page generator in `GalleryPipeline.jl` or a dedicated helper module.
- [ ] Read manifest entries and deterministic public URL paths.
- [ ] Group entries by category.
- [ ] Render markdown for each entry.
- [ ] Support optional poster-only and animated variants.
- [ ] Preserve a clean editing note at the top indicating the file is generated.
- [ ] Fail if any required URL is missing.
- [ ] Overwrite `docs/src/examples.md`.

### Deliverables

- [ ] A generated `docs/src/examples.md`.

### Exit Criteria

- [ ] The examples page can be recreated entirely from the manifest and published URLs.

## Phase 12: Integrate with the Existing Docs Build

### Objective

Ensure normal docs deployment remains lightweight and does not try to regenerate media.

### Tasks

- [ ] Confirm that `docs/make.jl` does not attempt to run the gallery pipeline.
- [ ] Decide whether the page generator runs before docs build or as a separate operator step.
- [ ] Ensure the generated `docs/src/examples.md` is committed when changed.
- [ ] Verify that Documenter can render the new page structure without special handling.
- [ ] Verify that external media URLs display correctly in the rendered docs.

### Deliverables

- [ ] A normal docs build using generated page content only.

### Exit Criteria

- [ ] The gallery page is deployable without cluster access.

## Phase 13: Add Cleanup and Retention

### Objective

Prevent the expensive pipeline from filling storage over time.

### Tasks

- [ ] Define what is deleted after a successful run.
- [ ] Define what is retained temporarily for debugging.
- [ ] Implement cleanup for:
  - [ ] raw simulation files
  - [ ] ParaView frame sequences
  - [ ] stale work directories
  - [ ] failed partial outputs if appropriate
- [ ] Ensure cleanup never removes published media from the media repo.
- [ ] Add a `cleanup` mode if helpful.

### Deliverables

- [ ] A documented retention policy.
- [ ] A working cleanup implementation.

### Exit Criteria

- [ ] Operators can recover storage safely without guessing what can be removed.

## Phase 14: Add Validation, Idempotency, and Failure Handling

### Objective

Make the pipeline safe to rerun and easy to diagnose when something breaks.

### Tasks

- [ ] Implement manifest validation.
- [ ] Implement environment validation for required tools.
- [ ] Implement checks for missing ParaView or `ffmpeg`.
- [ ] Implement checks for missing media repo credentials.
- [ ] Implement deterministic output naming.
- [ ] Define overwrite semantics.
- [ ] Abort page generation when media publish failed.
- [ ] Emit useful non-zero exits on failure.
- [ ] Make it possible to rerun a single failed step.

### Deliverables

- [ ] A pipeline that fails loudly and predictably.

### Exit Criteria

- [ ] Operators can rerun failed work without corrupting the gallery state.

## Phase 15: Document the Operator Workflow

### Objective

Make the gallery maintainable by someone other than the original implementer.

### Tasks

- [ ] Add a short operator guide, likely in `docs/src/development.md` or a dedicated internal note.
- [ ] Document the cluster command to run all gallery examples.
- [ ] Document the cluster command to run ParaView rendering for all gallery examples.
- [ ] Document the exact `git clone` or `git pull` step used to prepare the cluster checkout.
- [ ] Document how rendered images are downloaded from the cluster.
- [ ] Document the local `ffmpeg` command or wrapper used for encoding.
- [ ] Document the manual upload step to the GitLab Pages repo.
- [ ] Document where HPC outputs live.
- [ ] Document where the public media repo lives.
- [ ] Document how to inspect logs and run status on the cluster.
- [ ] Document how to clean old outputs.
- [ ] Document how to add a new example to the manifest.

### Deliverables

- [ ] Minimal but sufficient operator documentation.

### Exit Criteria

- [ ] A team member can run the workflow without reconstructing it from code.

## Phase 16: Rollout Plan

### Milestone 1: One Example End to End

- [ ] Pick `fluid/dam_break_2d.jl` as the pilot example.
- [ ] Create one manifest entry.
- [ ] Run one high-resolution simulation from the cluster checkout.
- [ ] Render one poster image.
- [ ] Download one frame sequence locally.
- [ ] Encode one animation locally.
- [ ] Prepare the upload-ready media directory locally.
- [ ] Upload to GitLab Pages manually.
- [ ] Generate `docs/src/examples.md` for that single entry.
- [ ] Verify the public docs page renders the media correctly.

### Milestone 2: Current Gallery Coverage

- [ ] Add all examples currently present in `docs/src/examples.md`.
- [ ] Verify each entry renders acceptably.
- [ ] Verify all published URLs resolve.
- [ ] Replace the old hand-maintained page content.

### Milestone 3: Hardening

- [ ] Add retries where useful.
- [ ] Add cleanup automation.
- [ ] Add better validation and diagnostics.
- [ ] Add optional support for MP4 where GIF is too large.

## Detailed Acceptance Checklist

### Functional Acceptance

- [ ] Running the pipeline for one entry produces final public media.
- [ ] Running the pipeline for all entries produces a complete examples page.
- [ ] The examples page uses only external GitLab Pages media URLs.
- [ ] The docs build succeeds after page generation.

### Storage Acceptance

- [ ] No raw simulation output is committed to this repo.
- [ ] No final media is committed to this repo.
- [ ] The media repo contains only public final assets.

### Operational Acceptance

- [ ] An operator can inspect job status from the cluster checkout.
- [ ] An operator can rerun one failed example.
- [ ] An operator can clean up transient outputs.
- [ ] An operator can complete the workflow with:
  - [ ] one cluster-side simulation command,
  - [ ] one cluster-side rendering command,
  - [ ] one local encoding step,
  - [ ] one manual upload step.

### Reproducibility Acceptance

- [ ] Every output can be traced to a commit and manifest configuration.
- [ ] Re-running the same entry produces the expected stable output paths or a documented versioned alternative.

## Risks and Watch Items

- [ ] Some examples may require parameter changes beyond simple resolution overrides.
- [ ] Some ParaView pipelines may need per-example special handling.
- [ ] GIF files may become too large for practical web use.
- [ ] Downloading large frame sequences from the cluster may be expensive.
- [ ] Manual upload introduces a human step that can drift from the intended directory layout.
- [ ] Media repo authentication may be awkward on shared systems.
- [ ] Pages propagation delay may complicate verification.

## Open Questions

- [ ] Should the media repo store only the newest outputs, or also versioned historical outputs?
- [ ] Should poster images be mandatory for every entry even if no animation is published?
- [ ] Should MP4 be preferred over GIF for larger examples?
- [ ] Should page generation happen before manual upload, or only after confirming the public URLs resolve?
- [ ] Should the local encoding script live in this repo or remain an operator-only command sequence?
- [ ] Should a lightweight metadata file be published beside each media asset?

## Implementation Order Recommendation

1. Implement Phase 3 around the existing manifest and validator.
2. Implement Milestone 1 for `fluid/dam_break_2d.jl`.
3. Prove publish to GitLab Pages.
4. Prove page generation for one entry.
5. Expand the pipeline from the pilot entry to the full manifest.
6. Harden cleanup, retries, and docs.

## Working Notes

Use this section as an execution log while implementing.

- 2026-03-10: Initial plan created.
- 2026-03-10: Updated plan to assume a separate cluster-local checkout of `TrixiParticles.jl`; this prompt does not execute cluster commands directly.
- 2026-03-10: Updated plan to use the intended three-step workflow: cluster-side simulation script, cluster-side ParaView render script, then manual upload to Helmholtz GitLab.
- 2026-03-10: Updated plan to move animation encoding to the local machine after downloading rendered images from the cluster.
- 2026-03-10: Completed Phase 1 by freezing the full non-duplicate gallery scope in `docs/gallery/INVENTORY.md`.
- 2026-03-10: Completed the Phase 2 manifest design in `docs/gallery/manifest.toml` with a minimal schema and global render defaults.
- 2026-03-10: Added `docs/gallery/validate_manifest.jl` and validated the current manifest successfully with 45 entries and 4 asset strategies.
- 2026-03-10: Implemented Phase 3 with `Configurations.jl`-backed typed manifest loading in `docs/gallery/src/GalleryPipeline.jl`, plus shared `validate` and dry-run command entrypoints.
- 2026-03-10: Implemented Phase 4 run layout and TOML metadata derivation in `docs/gallery/src/GalleryPipeline.jl`, verified via dry-run output and a sample metadata file in `/tmp`.
- Pending: choose the first pilot example for Phase 5 simulation execution.

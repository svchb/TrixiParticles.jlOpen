# Phase 0 Setup Record

## Status

- Status: Complete for implementation
- Date: 2026-03-10
- Scope: External setup and fixed architectural decisions for the gallery pipeline

## Confirmed Decisions

### Hosting

- Final public media will be hosted in a separate public GitLab repo on `codebase.helmholtz.cloud`.
- That repo will expose the media through GitLab Pages.
- This repo will not store final gallery media.
- Raw simulation output will remain on cluster storage only.
- Media repo: `https://codebase.helmholtz.cloud/trixip/trixip_gallery`
- Public Pages base URL: `https://trixip-gallery-ebae4c.pages.hzdr.de/`

### Workflow Boundary

- `TrixiParticles.jl` is checked out separately on the cluster.
- This prompt does not run cluster commands directly.
- The intended operator workflow is:
  1. run one cluster-side script to execute the gallery simulations,
  2. run one cluster-side script to generate posters and frame sequences with ParaView,
  3. download the rendered images to the local machine,
  4. run `ffmpeg` locally to create GIF and optional MP4 files,
  5. upload the final media manually to the GitLab Pages repo.

### Media Repo Policy

- Media repo name in use: `trixip_gallery`
- Recommended default branch: `main`
- Initial publishing policy: overwrite in place to keep stable URLs
- Versioned historical media is explicitly deferred unless later needed
- Recommended upload authentication: manual `git push` using the operator's SSH key registered with `codebase.helmholtz.cloud`
- Fallback upload authentication: HTTPS with a personal access token if SSH is not available

### Media Output Policy

- Every gallery entry must publish `poster.png`
- Animated entries should publish `animation.gif` by default
- `animation.mp4` is optional per manifest entry
- The manifest remains the source of truth for which outputs exist per example

### Cluster Checkout Policy

- The cluster keeps its own checkout of `TrixiParticles.jl`
- The cluster checkout should be updated to the exact commit intended for the gallery run
- Recommended update flow:
  1. `git fetch origin`
  2. `git checkout <commit-or-branch>`
  3. `git pull --ff-only` when operating on a branch

### Cluster Tool Availability

- GUI `paraview` is not usable in the current SSH session because no X server is available
- Headless ParaView is available:
  - `pvbatch` at `/usr/bin/pvbatch`
  - `pvpython` at `/usr/bin/pvpython`
  - reported version: `4.4.0`
- `ffmpeg` is currently not available on `strand`
- Scheduler interface is available:
  - `sbatch` at `/usr/bin/sbatch`
  - `srun` at `/usr/bin/srun`
  - Slurm version: `22.05.9`
- Known partition inventory provided by the operator:

  | Partition | Max Processors | Max Nodes | Max Walltime | Priority | Remarks |
  | --- | ---: | ---: | --- | --- | --- |
  | `pGPU` | 240 | 5 | 72h | mid | 16GB GPU |
  | `p2GPU32` | 144 | 3 | 72h | mid | 2x 32GB GPU |
  | `p2GPU40` | 96 | 2 | 72h | mid | 2x 40GB GPU |
  | `pRoutine` | 96 | 2 | 24h | high | reserved |
  | `pTest` | 96 | 2 | 30m | mid | |
  | `pNode` | 48 | 1 | 168h | mid | |
  | `pCluster` | 960 | 20 | 72h | mid | |
  | `pAll` | all | all | 8h | low | |
  | `pBigCluster` | 720 | 15 | 72h | mid | double memory |
  | `pBigNode` | 48 | 1 | 168h | mid | double memory |
  | `pRush` | 480 | 10 | 6h | high | |
- Current `sinfo` snapshot on `strand` additionally shows:
  - `pGPUTest` is present
  - `pAll` is the default partition in the current scheduler view
  - `pGPU`, `p2GPU32`, `p2GPU40`, `pNode`, `pBigNode`, `pCluster`, `pBigCluster`, `pRush`, `pTest`, and `pGPUTest` are visible in the live scheduler output
- Observed walltimes from the live `sinfo` snapshot are:

  | Partition | Observed Timelimit |
  | --- | --- |
  | `pTest` | `30:00` |
  | `p2GPU40` | `3-00:00:00` |
  | `pGPU` | `4-12:00:00` |
  | `p2GPU32` | `3-00:00:00` |
  | `pNode` | `7-00:00:00` |
  | `pBigNode` | `7-00:00:00` |
  | `pCluster` | `3-00:00:00` |
  | `pBigCluster` | `3-00:00:00` |
  | `pAll` | `8:00:00` |
  | `pRush` | `6:00:00` |
  | `pGPUTest` | `3-00:00:00` |
- Preferred scheduler mode:
  - `sbatch` for full gallery runs
  - `srun` for ad hoc interactive checks if needed
- Partition selection is intentionally left open for Phase 2/3 because it should depend on:
  - example size,
  - whether the example uses CPU or GPU,
  - whether the run is a short smoke test or a production-quality gallery run.

### Encoding Strategy

- Final animation encoding will happen locally after downloading rendered images from the cluster
- The cluster is responsible for simulations, poster generation, and frame-sequence generation only
- Local tooling is responsible for:
  - `animation.gif`
  - optional `animation.mp4`
- Local `ffmpeg` is available:
  - command: `ffmpeg`
  - reported version: `6.1.1-3ubuntu5+esm7`

### Retention Policy

- Raw `VTK/PVD/VTU` output and intermediate frame sequences are transient
- Final upload-ready media stays available until manual upload and verification are complete
- Recommended default cleanup window for raw cluster outputs: 7 days maximum
- The final cleanup policy can be tightened later if storage pressure requires it

## Fixed Directory Layouts

### Cluster-Side Working Layout

Recommended top-level working root on the cluster:

```text
<gallery-root>/
  runs/
    <example-id>/
      raw/
      frames/
      final/
      logs/
      metadata/
```

### Final Upload Layout

Only files from the final upload layout are uploaded manually to the media repo.

```text
public/
  <category>/
    <example-id>/
      poster.png
      animation.gif
      animation.mp4
```

Example:

```text
public/
  fluid/
    dam_break_2d/
      poster.png
      animation.gif
```

This layout must map directly to deterministic public URLs used by the generated docs page.

## Recommended Media Repo Bootstrap

The media repo needs a minimal Pages configuration. A conservative starter `.gitlab-ci.yml` is:

```yaml
deploy-pages:
  stage: deploy
  script:
    - test -d public
  artifacts:
    paths:
      - public
  pages: true
```

This keeps the repo simple: the operator manually adds files under `public/`, commits, and pushes; GitLab Pages then publishes the site.

## External Actions Still Required

These tasks cannot be completed from this prompt and remain operator-owned:

1. Decide who owns write access to the media repo.

## Execution Environment Summary

- Main repo:
  - stores the plan, manifest, templates, and cluster-run scripts
  - generates `docs/src/examples.md`
- Cluster checkout:
  - runs simulations
  - runs ParaView rendering
  - produces posters and frame sequences
- Local workstation:
  - downloads rendered images from the cluster
  - runs `ffmpeg`
  - prepares the final upload directory
- Media repo:
  - receives a manual commit and push of the prepared `public/` tree
  - serves the final public assets via GitLab Pages

## Pending Unknowns

- Media repo write-access policy
- Whether some examples should publish MP4 in addition to GIF from day one

## Recommended Next Step

Phase 0 is sufficiently resolved to begin Phase 1. The remaining media-repo write-access policy is governance, not a technical blocker for implementation.

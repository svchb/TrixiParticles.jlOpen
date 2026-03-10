# Gallery Inventory

## Status

- Status: Phase 1 complete
- Date: 2026-03-10
- Scope: Full non-duplicate gallery inventory for the generated examples page
- Standalone example scripts reviewed: 47
- Included gallery entries: 45
- Explicit exclusions: 2

## Phase 1 Decisions

- The generated gallery now targets every standalone, non-duplicate example script under `examples/*/*.jl`.
- Phase 1 no longer mirrors only the current hand-written `docs/src/examples.md`.
- Dimension variants, solver variants, boundary variants, preprocessing/postprocessing workflows, and benchmark/reference scripts are kept unless they are direct technical duplicates.
- Not every entry is a ParaView animation candidate.
- Each gallery entry gets a stable `id` that will be reused in the manifest.
- Global gallery styling for later phases is fixed:
  - white background for all gallery assets
  - `viridis` as the common color scheme whenever scalar coloring is used
- The initial media policy is:
  - time-dependent simulation entries get `poster.png` and `animation.gif`
  - static, plot-driven, or benchmark entries get `poster.png`
  - `animation.mp4` remains optional and deferred

## Duplicate and Inclusion Policy

1. Include every standalone example script below `examples/` that is intended to be run directly.
2. Exclude helper or support files that are only included by other examples and are not gallery entries by themselves.
3. Exclude technical wrappers that intentionally rerun the same physical setup with a different execution backend.
4. Do not collapse 2D and 3D variants, solver variants, boundary-condition variants, or benchmark/reference variants into a single entry unless they are literal backend duplicates.

## Explicit Exclusions

- `fluid/dam_break_2d_gpu.jl`
  - excluded as a backend duplicate of `fluid/dam_break_2d.jl`
  - it literally reuses the same dam-break setup and only changes the execution backend and neighborhood search details
- `n_body/n_body_system.jl`
  - excluded because it is a support file included by other `n_body` examples
  - it is not a standalone gallery entry

## Asset Strategies

The asset strategies below inherit the same gallery-wide visual defaults:

- white background
- `viridis` color scheme wherever scalar coloring is used

### `paraview_animation`

- Use cluster-side simulation output plus cluster-side headless ParaView rendering.
- Generate `poster.png` and `animation.gif`.
- Typical use: fluid, DEM, FSI, structure, and time-dependent `n_body` simulations.

### `paraview_poster`

- Use exported VTK or geometry data with a single ParaView-rendered poster.
- Generate `poster.png` only.
- Typical use: static 3D preprocessing examples.

### `julia_plot_poster`

- Use the example's Julia-side plotting workflow to generate a static poster image.
- Generate `poster.png` only.
- Typical use: postprocessing examples and 2D preprocessing examples that already build plot layouts.

### `benchmark_poster`

- Generate a static summary poster from benchmark output or a simple comparison layout.
- Generate `poster.png` only.
- Typical use: `n_body` benchmark/reference scripts that do not produce ParaView-ready output.

## Inventory Summary

| category | entries | dominant asset strategy | notes |
| --- | --- | --- | --- |
| `dem` | 2 | `paraview_animation` | both are time-dependent particle simulations |
| `fluid` | 24 | `paraview_animation` | includes solver, dimensional, and multiphase variants |
| `fsi` | 7 | `paraview_animation` | includes 2D and 3D coupled examples |
| `n_body` | 4 | mixed | one simulation animation plus three benchmark/reference posters |
| `postprocessing` | 3 | `julia_plot_poster` | plot-driven examples rather than ParaView gallery scenes |
| `preprocessing` | 4 | mixed | 2D examples are plot-driven, 3D examples are static ParaView posters |
| `structure` | 1 | `paraview_animation` | one dynamic beam example |

- Animation-backed entries: 35
- Poster-only entries: 10

## Inventory

### DEM

| id | title | canonical_example | asset_strategy | outputs |
| --- | --- | --- | --- | --- |
| `dem_collapsing_sand_pile_3d` | `Collapsing Sand Pile 3D` | `dem/collapsing_sand_pile_3d.jl` | `paraview_animation` | `poster,gif` |
| `dem_rectangular_tank_2d` | `Rectangular Tank 2D` | `dem/rectangular_tank_2d.jl` | `paraview_animation` | `poster,gif` |

### Fluid

| id | title | canonical_example | asset_strategy | outputs |
| --- | --- | --- | --- | --- |
| `fluid_accelerated_tank_2d` | `Accelerated Tank 2D` | `fluid/accelerated_tank_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_dam_break_2d` | `Dam Break 2D` | `fluid/dam_break_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_dam_break_2d_iisph` | `Dam Break 2D IISPH` | `fluid/dam_break_2d_iisph.jl` | `paraview_animation` | `poster,gif` |
| `fluid_dam_break_2d_iisph_pressure_boundaries` | `Dam Break 2D IISPH Pressure Boundaries` | `fluid/dam_break_2d_iisph_pressure_boundaries.jl` | `paraview_animation` | `poster,gif` |
| `fluid_dam_break_2phase_2d` | `Dam Break Two-Phase 2D` | `fluid/dam_break_2phase_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_dam_break_3d` | `Dam Break 3D` | `fluid/dam_break_3d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_dam_break_oil_film_2d` | `Dam Break Oil Film 2D` | `fluid/dam_break_oil_film_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_falling_water_column_2d` | `Falling Water Column 2D` | `fluid/falling_water_column_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_falling_water_spheres_2d` | `Falling Water Spheres 2D` | `fluid/falling_water_spheres_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_falling_water_spheres_3d` | `Falling Water Spheres 3D` | `fluid/falling_water_spheres_3d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_hydrostatic_water_column_2d` | `Hydrostatic Water Column 2D` | `fluid/hydrostatic_water_column_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_hydrostatic_water_column_3d` | `Hydrostatic Water Column 3D` | `fluid/hydrostatic_water_column_3d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_lid_driven_cavity_2d` | `Lid-Driven Cavity 2D` | `fluid/lid_driven_cavity_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_moving_wall_2d` | `Moving Wall 2D` | `fluid/moving_wall_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_oscillating_drop_2d` | `Oscillating Drop 2D` | `fluid/oscillating_drop_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_periodic_array_of_cylinders_2d` | `Periodic Array of Cylinders 2D` | `fluid/periodic_array_of_cylinders_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_periodic_channel_2d` | `Periodic Channel 2D` | `fluid/periodic_channel_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_pipe_flow_2d` | `Pipe Flow 2D` | `fluid/pipe_flow_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_pipe_flow_3d` | `Pipe Flow 3D` | `fluid/pipe_flow_3d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_poiseuille_flow_2d` | `Poiseuille Flow 2D` | `fluid/poiseuille_flow_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_sphere_surface_tension_2d` | `Sphere Surface Tension 2D` | `fluid/sphere_surface_tension_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_sphere_surface_tension_3d` | `Sphere Surface Tension 3D` | `fluid/sphere_surface_tension_3d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_sphere_surface_tension_wall_2d` | `Sphere Surface Tension Wall 2D` | `fluid/sphere_surface_tension_wall_2d.jl` | `paraview_animation` | `poster,gif` |
| `fluid_taylor_green_vortex_2d` | `Taylor-Green Vortex 2D` | `fluid/taylor_green_vortex_2d.jl` | `paraview_animation` | `poster,gif` |

### Fluid Structure Interaction

| id | title | canonical_example | asset_strategy | outputs |
| --- | --- | --- | --- | --- |
| `fsi_dam_break_gate_2d` | `Dam Break Gate 2D` | `fsi/dam_break_gate_2d.jl` | `paraview_animation` | `poster,gif` |
| `fsi_dam_break_plate_2d` | `Dam Break Plate 2D` | `fsi/dam_break_plate_2d.jl` | `paraview_animation` | `poster,gif` |
| `fsi_falling_sphere_2d` | `Falling Sphere 2D` | `fsi/falling_sphere_2d.jl` | `paraview_animation` | `poster,gif` |
| `fsi_falling_sphere_3d` | `Falling Sphere 3D` | `fsi/falling_sphere_3d.jl` | `paraview_animation` | `poster,gif` |
| `fsi_falling_spheres_2d` | `Falling Spheres 2D` | `fsi/falling_spheres_2d.jl` | `paraview_animation` | `poster,gif` |
| `fsi_falling_water_column_2d` | `Falling Water Column 2D` | `fsi/falling_water_column_2d.jl` | `paraview_animation` | `poster,gif` |
| `fsi_hydrostatic_water_column_2d` | `Hydrostatic Water Column 2D` | `fsi/hydrostatic_water_column_2d.jl` | `paraview_animation` | `poster,gif` |

### N-Body

| id | title | canonical_example | asset_strategy | outputs |
| --- | --- | --- | --- | --- |
| `n_body_benchmark_reference` | `N-Body Benchmark Reference` | `n_body/n_body_benchmark_reference.jl` | `benchmark_poster` | `poster` |
| `n_body_benchmark_reference_faster` | `N-Body Benchmark Reference Faster` | `n_body/n_body_benchmark_reference_faster.jl` | `benchmark_poster` | `poster` |
| `n_body_benchmark_trixi` | `N-Body Benchmark TrixiParticles` | `n_body/n_body_benchmark_trixi.jl` | `benchmark_poster` | `poster` |
| `n_body_solar_system` | `N-Body Solar System` | `n_body/n_body_solar_system.jl` | `paraview_animation` | `poster,gif` |

### Postprocessing

| id | title | canonical_example | asset_strategy | outputs |
| --- | --- | --- | --- | --- |
| `postprocessing_interpolation_plane` | `Interpolation Plane` | `postprocessing/interpolation_plane.jl` | `julia_plot_poster` | `poster` |
| `postprocessing_interpolation_point_line` | `Interpolation Point and Line` | `postprocessing/interpolation_point_line.jl` | `julia_plot_poster` | `poster` |
| `postprocessing_summary` | `Postprocessing Summary` | `postprocessing/postprocessing.jl` | `julia_plot_poster` | `poster` |

### Preprocessing

| id | title | canonical_example | asset_strategy | outputs |
| --- | --- | --- | --- | --- |
| `preprocessing_complex_shape_2d` | `Complex Shape 2D` | `preprocessing/complex_shape_2d.jl` | `julia_plot_poster` | `poster` |
| `preprocessing_complex_shape_3d` | `Complex Shape 3D` | `preprocessing/complex_shape_3d.jl` | `paraview_poster` | `poster` |
| `preprocessing_packing_2d` | `Packing 2D` | `preprocessing/packing_2d.jl` | `julia_plot_poster` | `poster` |
| `preprocessing_packing_3d` | `Packing 3D` | `preprocessing/packing_3d.jl` | `paraview_poster` | `poster` |

### Structure

| id | title | canonical_example | asset_strategy | outputs |
| --- | --- | --- | --- | --- |
| `structure_oscillating_beam_2d` | `Oscillating Beam 2D` | `structure/oscillating_beam_2d.jl` | `paraview_animation` | `poster,gif` |

## Special Handling Notes

- `fluid/dam_break_2d_iisph.jl` and `fluid/dam_break_2d_iisph_pressure_boundaries.jl` stay separate because they demonstrate distinct solver and boundary-handling variants, not backend duplicates.
- `fluid/dam_break_2phase_2d.jl`, `fluid/dam_break_oil_film_2d.jl`, `fluid/pipe_flow_3d.jl`, `fluid/falling_water_spheres_3d.jl`, and similar files remain distinct gallery entries even though some reuse lower-level setups through `trixi_include`.
- `n_body` benchmark/reference scripts are intentionally included even though they are poster-only and do not use the ParaView path.
- `postprocessing` and part of `preprocessing` require a non-ParaView poster workflow in later phases.

## Exit Criteria Review

Phase 1 is considered complete because:

- every standalone example script under `examples/` has been reviewed,
- the duplicate policy is explicit and intentionally narrow,
- the gallery scope is fixed at 45 included entries,
- each included entry has a stable `id`,
- each included entry has a canonical example path,
- each included entry has a title and category,
- each included entry has an initial asset strategy,
- each included entry has a default output policy.

## Recommended Next Step

Use this inventory together with `docs/gallery/manifest.toml` to add only the per-entry `run` and `render` overrides that are actually needed, instead of expanding the schema up front.

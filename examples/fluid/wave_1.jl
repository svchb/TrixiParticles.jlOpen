# coastal_wave_spray_2d.jl – **with splash-height & droplet-spectrum post-processing**
# -----------------------------------------------------------------------------
# A 2-D dam-break surrogate that slams a wall, throws a spray jet, and records
# (i) instantaneous *splash height* and (ii) a *coarse droplet-size spectrum*.
# -----------------------------------------------------------------------------
# Usage (from the repo root)
#   julia --threads auto -O3 examples/fluid/coastal_wave_spray_2d.jl
# -----------------------------------------------------------------------------
# Dependencies:  TrixiParticles.jl  OrdinaryDiffEq.jl  JSON.jl (only if you want
#                to post-process the generated .json files later)
# -----------------------------------------------------------------------------
# The script shows how to build a **PostprocessCallback** – mirroring the style
# of `validation_util.jl` – to capture custom diagnostics while the solver runs.
# -----------------------------------------------------------------------------

using TrixiParticles
using OrdinaryDiffEq
using Printf

# =============================================================================
# ==== 1.  Physical & numerical parameters
const gravity        = 9.81           # [m/s²]
const fluid_density  = 1_025.0        # [kg/m³] (sea water)
const fluid_viscosity= 1e-6           # [m²/s] (viscosity of sea water)
const H_water        = 0.5            # [m]  initial column height
const W_water        = 0.4            # [m]  initial column width
const tank_height    = 2.0            # [m]
const tank_length    = 1.5            # [m]

const resolution_factor      = 100     # particles per H_water
const Δx_f                   = H_water / resolution_factor       # ≈ 0.005 m
const boundary_layers        = 4
const Δx_b                   = Δx_f

const t_end   = 3.0
const tspan   = (0.0, t_end)

# Weakly-compressible equation of state
const sound_speed   = 40 * sqrt(gravity * H_water)  # ≈ 44 m/s
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1, clip_negative_pressure=false)

# =============================================================================
# ==== 2.  Geometry & tank
initial_fluid_size = (W_water, H_water)
tank_size          = (tank_length, tank_height)

@info "Δx = $(Δx_f*1000) mm  |  tank = $(tank_size) m  |  fluid = $(initial_fluid_size) m"

tank = RectangularTank(Δx_f, initial_fluid_size, tank_size, fluid_density;
                       n_layers = boundary_layers, spacing_ratio = 1,
                       acceleration = (0.0, -gravity), state_equation = state_equation)

# =============================================================================
# ==== 3.  Fluid & boundary systems
smoothing_length  = 1.75 * Δx_f
smoothing_kernel  = WendlandC2Kernel{2}()

fluid_density_calc = ContinuityDensity()
# This is bound to the realizable viscosity at the resolution.
viscosity_model    = ViscosityAdami(nu=max(0.1*Δx_f, fluid_viscosity))

density_diffusion = DensityDiffusionAntuono(tank.fluid, delta = 0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calc,
                                           state_equation, smoothing_kernel,
                                           smoothing_length; viscosity=viscosity_model, density_diffusion,
                                           acceleration = (0.0, -gravity))

boundary_density_calc = AdamiPressureExtrapolation()
# This is just set to the physical value since it is not needed for stability.
wall_viscosity_model  = ViscosityAdami(nu=fluid_viscosity)
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation = state_equation,
                                             boundary_density_calc, viscosity= wall_viscosity_model,
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model;
                                    adhesion_coefficient = 0.0) # <- tweak >0 for wetting

# =============================================================================
# ==== 4.  Semidiscretization & ODE problem
semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search = GridNeighborhoodSearch{2}(update_strategy = nothing),
                          parallelization_backend = PolyesterBackend())

ode = semidiscretize(semi, tspan)

# =============================================================================
# ==== 5.  Post-processing: splash height  &  droplet size spectrum

"Return the maximum fluid y-coordinate (splash height) at the current time step."
function splash_height(system, data, t)
    return maximum(j -> data.coordinates[2, j], axes(data.coordinates, 2))
end

"""
Compute a *coarse* droplet-size spectrum in three bins (>1 cm, 1–2 cm, 2–3 cm).
The routine:
  1. Select particles above the initial free-surface (y > H_water).
  2. Group them by proximity (< 1.1 * smoothing_length) using a simple
     breadth-first search (BFS).  Each group is treated as a droplet.
  3. Convert each cluster’s mass to an *equivalent spherical diameter*  d = (6V/π)^{1/3}.
  4. Count how many droplets fall into each bin.
Returns a 3-element vector so PostprocessCallback stores three time-series.
"""
function coarse_droplet_counts(system, data, t)
    coords   = data.coordinates
    masses   = data.mass                  # per-particle mass (constant here)
    np       = size(coords, 2)
    above    = [j for j in 1:np if coords[2, j] > H_water]  # airborne particles
    visited  = falses(np)
    counts   = zeros(Int, 3)              # (1–2 cm, 2–3 cm, >3 cm)
    rad2     = (1.1 * smoothing_length)^2 # neighbour distance²

    for seed in above
        visited[seed] && continue
        # -- BFS to build a cluster ------------------------------------------
        cluster = Int[]
        push!(cluster, seed)
        visited[seed] = true
        q = [seed]
        while !isempty(q)
            i = popfirst!(q)
            for j in above
                if !visited[j]
                    dx = coords[1, i] - coords[1, j]
                    dy = coords[2, i] - coords[2, j]
                    if dx*dx + dy*dy < rad2
                        push!(cluster, j)
                        visited[j] = true
                        push!(q, j)
                    end
                end
            end
        end
        # -------------------------------------------------------------------
        mass_cluster = length(cluster) * masses[1]          # 2-D mass proxy
        area_cluster = mass_cluster / fluid_density         # 2-D proxy area
        # Treat as a *cylinder of unit depth* -> equivalent diameter in 3-D
        vol3d        = area_cluster      # 1-m thickness assumption
        d_eq         = (6 * vol3d / π)^(1/3)                # [m]
        if 0.01 ≤ d_eq < 0.02
            counts[1] += 1
        elseif 0.02 ≤ d_eq < 0.03
            counts[2] += 1
        elseif d_eq ≥ 0.03
            counts[3] += 1
        end
    end
    return counts
end

function max_vertical_velocity(system, data, t)
    coords = data.coordinates
    airborne_idx = findall(y -> y > H_water, view(coords, 2, :))
    if isempty(airborne_idx)
        return 0.0
    else
        vel = data.velocity
        return maximum(abs.(view(vel, 2, airborne_idx)))
    end
end

post_cb = PostprocessCallback(
    ; dt = 0.01,                        # every 10 ms → lightweight
      output_directory = "post",       # JSON files in ./post/
      filename = "spray_metrics",      # → spray_metrics_*.json
      write_csv = true,
      splash_height,
      coarse_droplet_counts,
      max_vertical_velocity
)

# =============================================================================
# ==== 6.  Callbacks for stats & I/O
info_cb     = InfoCallback(interval = 1000)            # every 100 timesteps
save_cb     = SolutionSavingCallback(; dt = 0.025, output_directory = "output/coastal_wave_spray_2d", prefix="wall_visc")
stepsize_cb = StepsizeCallback(cfl = 0.9)
cbset       = CallbackSet(info_cb, save_cb, stepsize_cb, post_cb)

# =============================================================================
# ==== 7.  Solve
@info "Starting simulation …"
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0,       # initial dt – CFL will reduce it immediately
            save_everystep = false,
            callback = cbset)
@info "Finished.  Results:  ./output/coastal_wave_spray_2d (VTK)  &  ./post (JSON)"

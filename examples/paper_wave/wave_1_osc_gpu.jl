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
using PointNeighbors
using CUDA

# =============================================================================
# ==== 1.  Physical & numerical parameters
const gravity        = 9.81           # [m/s²]
const fluid_density  = 1_025.0        # [kg/m³] (sea water)
const fluid_viscosity= 1e-6           # [m²/s] (viscosity of sea water)
const H_water        = 0.5            # [m]  initial column height
const W_water        = 0.4            # [m]  initial column width
const tank_height    = 2.0            # [m]
const tank_length    = 1.5            # [m]

const resolution_factor      = 500     # particles per H_water
const Δx_f                   = H_water / resolution_factor       # ≈ 0.0025 m
const boundary_layers        = 4
const Δx_b                   = Δx_f

const t_end   = 1.4
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
# This is limited to the realizable viscosity at the resolution.
nu_empirical=max(0.001*Δx_f, fluid_viscosity)
viscosity_model    = ViscosityAdami(nu=nu_empirical)

density_diffusion = DensityDiffusionAntuono(tank.fluid, delta = 0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calc,
                                           state_equation, smoothing_kernel,
                                           smoothing_length; viscosity=viscosity_model, density_diffusion,
                                           acceleration = (0.0, -gravity))

boundary_density_calc = AdamiPressureExtrapolation()
# This is just set to the physical value since it is not needed for stability.
wall_viscosity_model  = ViscosityAdami(nu=50*nu_empirical)
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation = state_equation,
                                             boundary_density_calc, viscosity= wall_viscosity_model,
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model;
                                    adhesion_coefficient = 0.0)

# =============================================================================
# ==== 4.  Semidiscretization & ODE problem
min_corner = minimum(tank.boundary.coordinates, dims=2)
max_corner = maximum(tank.boundary.coordinates, dims=2)
cell_list = FullGridCellList(; min_corner, max_corner)
nhs_gpu = GridNeighborhoodSearch{2}(; cell_list)

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search = nhs_gpu,
                          parallelization_backend = CUDABackend())

ode = semidiscretize(semi, tspan)

# =============================================================================
# ==== 5.  Post-processing: splash height  &  droplet size spectrum

"Return the maximum fluid y-coordinate (splash height) at the current time step."
function splash_height(system, data, t)
    return maximum(j -> data.coordinates[2, j], axes(data.coordinates, 2))
end

##############################
# 2.  Coarse droplet counts  #
##############################

"""
    coarse_droplet_counts(system, data, t) -> Vector{Int}

Return a three‑element integer vector `[n₁, n₂, n₃]` where
  • `n₁` counts droplets with 10–20 mm diameter,
  • `n₂` counts droplets with 20–30 mm diameter,
  • `n₃` counts droplets larger than 30 mm.
Each droplet diameter is derived from the cluster area using the
*equivalent‑sphere* relation `d = (6V/π)^{1/3}` with `V = area × 1 m` (unit
out‑of‑plane width).
"""
function coarse_droplet_counts(system, data, t)
    clusters = airborne_clusters(data.coordinates)
    counts   = zeros(Int, 3)
    for c in clusters
        area = length(c) * data.mass[1] / fluid_density       # m² (2‑D proxy)
        d_eq = (6 * area / π)^(1/3)                          # m
        if 0.01 ≤ d_eq < 0.02       # 10–20 mm bin
            counts[1] += 1
        elseif 0.02 ≤ d_eq < 0.03   # 20–30 mm bin
            counts[2] += 1
        elseif d_eq ≥ 0.03          # >30 mm bin
            counts[3] += 1
        end
    end
    return counts
end

##############################
# 3.  Max vertical velocity  #
##############################

"""
    max_vertical_velocity(system, data, t) -> Float64

Return the maximum absolute *vertical* velocity of particles above the free
surface.  Returns `0.0` when no such particles exist (e.g. before impact).
"""
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

##############################
# 4.  Droplet radii list     #
##############################
function particle_cv(system, v_ode, u_ode, semi, t, h;
                     light_sheet_height=0.1,
                     volume_thickness=0.1)

    # 1. unwrap & get coords
    u_sys = TrixiParticles.wrap_u(u_ode, system, semi)
    coords = TrixiParticles.current_coordinates(u_sys, system)
    # N = size(coords, 2)
    Δz = light_sheet_height / 2

    coords_svector = reinterpret(reshape, SVector{ndims(system), eltype(coords)}, coords)
    found_particles = findall(x -> x[1] > tank_length - volume_thickness && abs(x[2] - h) <= Δz, coords_svector)

    return Array(coords[2, found_particles])
end

particle_cv_z50 = (system, v_ode, u_ode, semi, t)->particle_cv(system, v_ode, u_ode, semi, t, 0.05)
particle_cv_z150 = (system, v_ode, u_ode, semi, t)->particle_cv(system, v_ode, u_ode, semi, t, 0.15)
particle_cv_z250 = (system, v_ode, u_ode, semi, t)->particle_cv(system, v_ode, u_ode, semi, t, 0.25)
particle_cv_z350 = (system, v_ode, u_ode, semi, t)->particle_cv(system, v_ode, u_ode, semi, t, 0.35)
particle_cv_z450 = (system, v_ode, u_ode, semi, t)->particle_cv(system, v_ode, u_ode, semi, t, 0.45)
particle_cv_z600 = (system, v_ode, u_ode, semi, t)->particle_cv(system, v_ode, u_ode, semi, t, 0.60)
particle_cv_z800 = (system, v_ode, u_ode, semi, t)->particle_cv(system, v_ode, u_ode, semi, t, 0.80)


post_cb = PostprocessCallback(
    ; dt = 0.02,                        # 500hz
      output_directory = "post",       # JSON files in ./post/
      filename = "spray_metrics",      # → spray_metrics_*.json
      write_csv = true,
    #   splash_height,
    #   coarse_droplet_counts,
    #   max_vertical_velocity,
    #   particle_cv,
      particle_cv_z50,
      particle_cv_z150,
      particle_cv_z250,
      particle_cv_z350,
      particle_cv_z450,
      particle_cv_z600,
      particle_cv_z800
)

# =============================================================================
# ==== 6.  Callbacks for stats & I/O
info_cb     = InfoCallback(interval = 1000)            # every 100 timesteps
save_cb     = SolutionSavingCallback(; dt = 0.025, output_directory = "output/coastal_wave_spray_2d", prefix="visc0001_wallVisc50_{$resolution_factor}")
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

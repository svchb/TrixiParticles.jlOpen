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
const water_density  = 1_025.0        # [kg/m³] (sea water)
const water_viscosity= 8.9e-7         # [m²/s] (viscosity of water)
const air_density    = 1.2            # [kg/m³] (air)
const air_viscosity  = 1.544E-5       # [m²/s] (viscosity of dry air)
const H_water        = 0.5            # [m]  initial column height
const W_water        = 0.4            # [m]  initial column width
const tank_height    = 2.0            # [m]
const tank_length    = 1.5            # [m]

const resolution_factor      = 100     # particles per H_water
const Δx_f                   = H_water / resolution_factor       # ≈ 0.0025 m
const boundary_layers        = 4
const Δx_b                   = Δx_f
const smoothing_length_factor = 1.75

const t_end   = 1.4
const tspan   = (0.0, t_end)

# This is limited to the realizable viscosity at the resolution.
const nu_empirical=max(0.001*Δx_f, water_viscosity)
const nu_ratio = air_viscosity/water_viscosity
water_viscosity_model    = ViscosityAdami(nu=nu_empirical)
air_viscosity_model  = ViscosityAdami(nu=nu_ratio*nu_empirical)
wall_viscosity_model  = ViscosityAdami(nu=50*nu_empirical)

# Weakly-compressible equation of state
# const sound_speed   = 40 * sqrt(gravity * H_water)
# const sound_speed  = 150.0 # survives crash but still penetration (+100% time)
const sound_speed  = 100.0

state_equation = StateEquationCole(; sound_speed, reference_density=water_density,
                                   exponent=7, background_pressure=100000, clip_negative_pressure=false)

air_eos = StateEquationCole(; sound_speed, reference_density=air_density, exponent=1.4,
                            background_pressure=5000, clip_negative_pressure=false)

# =============================================================================
# ==== 2.  Geometry & tank
initial_water_size = (W_water, H_water)
initial_air_size_ontop = (W_water, tank_height - H_water)
initial_air_size_side = (tank_length - W_water, tank_height)
tank_size          = (tank_length, tank_height)

@info "Δx = $(Δx_f*1000) mm  |  tank = $(tank_size) m  |  fluid = $(initial_water_size) m"

tank = RectangularTank(Δx_f, initial_water_size, tank_size, water_density;
                       n_layers = boundary_layers, spacing_ratio = 1,
                       acceleration = (0.0, -gravity), state_equation = state_equation)

air_tank_ontop = RectangularShape(Δx_f,
                              round.(Int, initial_air_size_ontop ./ Δx_f),
                              zeros(length(initial_air_size_ontop)), density=air_density)

# move on top of the water
for i in axes(air_tank_ontop.coordinates, 2)
    air_tank_ontop.coordinates[:, i] .+= [0.0, H_water]
end

air_tank_side = RectangularShape(Δx_f,
                              round.(Int, initial_air_size_side ./ Δx_f),
                               (W_water, 0.0), density=air_density)

air_in_tank = union(air_tank_ontop, air_tank_side)

# =============================================================================
# ==== 3.  Fluid & boundary systems
smoothing_length  = smoothing_length_factor * Δx_f
smoothing_kernel  = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

density_diffusion = DensityDiffusionAntuono(tank.fluid, delta = 0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length; viscosity=water_viscosity_model, density_diffusion,
                                           acceleration = (0.0, -gravity))

#density_diffusion_air = DensityDiffusionAntuono(air_in_tank, delta = 0.1)
air_system = WeaklyCompressibleSPHSystem(air_in_tank, fluid_density_calculator,
                                                air_eos, smoothing_kernel, smoothing_length,
                                                viscosity=air_viscosity_model,
                                                #density_diffusion=density_diffusion_air,
                                                acceleration=(0.0, -gravity))


# boundary_density_calc = AdamiPressureExtrapolation(pressure_offset=1000)
boundary_density_calc = ContinuityDensity()
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

semi = Semidiscretization(fluid_system, air_system, boundary_system,
                          neighborhood_search = nhs_gpu,
                          parallelization_backend = CUDABackend())


# semi = Semidiscretization(fluid_system, air_system, boundary_system,
#                           neighborhood_search = GridNeighborhoodSearch{2}(update_strategy = nothing),
#                           parallelization_backend = PolyesterBackend())

ode = semidiscretize(semi, tspan)

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
info_cb     = InfoCallback(interval = 250)            # every 100 timesteps
save_cb     = SolutionSavingCallback(; dt = 0.025, output_directory = "output/coastal_wave_spray_2d", prefix="visc0001_wallVisc50_$resolution_factor")
stepsize_cb = StepsizeCallback(cfl = 0.9)
cbset       = CallbackSet(info_cb, save_cb, stepsize_cb, post_cb)

# =============================================================================
# ==== 7.  Solve
@info "Starting simulation …"
# sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
#             dt = 1.0,       # initial dt – CFL will reduce it immediately
#             save_everystep = false,
#             callback = cbset)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            maxiters=1e7,
            save_everystep=false, callback=cbset);

@info "Finished.  Results:  ./output/coastal_wave_spray_2d (VTK)  &  ./post (JSON)"

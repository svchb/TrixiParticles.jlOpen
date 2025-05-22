# 2D coastal wave spray simulation (simplified wave impact/dam‑break style)
#
# This script sets up a single breaking wave / dam‑break surrogate in a 2‑D tank
# to generate spray that impacts a vertical wall, suitable for corrosion exposure studies.
#
# It is adapted from the standard `dam_break_2d.jl` example in TrixiParticles.jl,
# with modified geometry and output settings.
#
# Key differences:
#   * Smaller water column (0.4 m × 0.5 m) released into a 1.5 m‑long tank
#     with a 1 m freeboard so spray can loft.
#   * Finer particle spacing (~5 mm) to begin resolving droplets ≥1 cm.
#   * Output of particle data every 1 ms to capture fast spray dynamics.
#   * The right tank wall at x = tank_length acts as the impact wall.
#     No outlet boundary: the wall is modeled with boundary particles.
#
# Usage (from package root):
#   julia --threads auto -O3 examples/fluid/coastal_wave_spray_2d.jl
#
# Dependencies:
#   TrixiParticles.jl, OrdinaryDiffEq.jl
#
# Note: adjust `output_directory` to a suitable location; by default it writes
#       to `output/coastal_wave_spray_2d`.
###############################################################################

using TrixiParticles
using OrdinaryDiffEq
using Printf

# =============================================================================
# ==== Physical & numerical parameters
gravity = 9.81            # [m/s²]
fluid_density = 1_025.0   # [kg/m³]  (sea water)
H_water = 0.5             # initial water column height [m]
W_water = 0.4             # initial water column width  [m]
tank_height = 1.0         # total tank height [m]
tank_length = 1.5         # total tank length [m]

resolution_factor = 100   # particles per water height
fluid_particle_spacing = H_water / resolution_factor   # ~0.005 m
boundary_layers = 4
spacing_ratio = 1
boundary_particle_spacing = fluid_particle_spacing / spacing_ratio

# Time span: simulate 3 seconds (adjust as needed)
t_end = 3.0
tspan = (0.0, t_end)

# Sound speed: 20 × √(g H) is common for weakly‑compressible SPH
sound_speed = 20 * sqrt(gravity * H_water)
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1, clip_negative_pressure=false)

# =============================================================================
# ==== Geometry: rectangular tank with left‑side water column
initial_fluid_size = (W_water, H_water)
tank_size = (tank_length, tank_height)

@info "Fluid particle spacing Δx = $(fluid_particle_spacing*1000) mm"
@info "Tank size = $(tank_size) m, initial fluid size = $(initial_fluid_size) m"

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density;
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       acceleration=(0.0, -gravity), state_equation=state_equation)

# =============================================================================
# ==== Fluid system
smoothing_length = 1.75 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

# Optional: density diffusion for robustness
density_diffusion = DensityDiffusionAntuono(tank.fluid, delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, -gravity),
                                           correction=nothing, surface_tension=nothing,
                                           reference_particle_spacing=0)

# =============================================================================
# ==== Boundary system (solid walls)
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             correction=nothing,
                                             reference_particle_spacing=0)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model; adhesion_coefficient=0.0)

# =============================================================================
# ==== Semidiscretization & ODE setup
semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing),
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

# =============================================================================
# ==== Callbacks & output
output_directory = joinpath(pwd(), "output", "coastal_wave_spray_2d")
mkpath(output_directory)

info_callback    = InfoCallback(interval=100)  # console info every 0.05 s
saving_callback  = SolutionSavingCallback(;output_directory, dt=0.01) # write every 1 ms
stepsize_cb      = StepsizeCallback(cfl=0.9)

callbacks = CallbackSet(info_callback, saving_callback, stepsize_cb)

# =============================================================================
# ==== Solve
@info "Starting simulation…"
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false);
            dt = 1.0,   # initial dt, overwritten by CFL control
            save_everystep = false,
            callback = callbacks)

@info "Simulation finished. Data written to $(output_directory)"

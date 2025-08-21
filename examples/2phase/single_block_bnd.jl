# ============================================================================
# Single SPH water particle falling through air (using RectangularTank + setdiff)
# ============================================================================
using TrixiParticles
using OrdinaryDiffEq
using Statistics

# --- Domain & resolution -----------------------------------------------------
dx = 0.05
boundary_layers = 3
spacing_ratio   = 1
dx_b            = dx / spacing_ratio

# --- Physics -----------------------------------------------------------------
g   = 0
tend = 2.0
tspan = (0.0, tend)

h      = 1.75 * dx
kernel = WendlandC2Kernel{2}()
cs     = 100.0                   # WCSPH "sound speed" (raise to reduce compressibility)

ρ_air   = 1.0
ρ_water = 1000.0

# Viscosities (scale water relative to air)
ν_air_phys   = 1.544e-5
ν_water_phys = 8.9e-7
ν_ratio      = ν_water_phys / ν_air_phys
ν_air_sim    = 0.02 * h * cs
ν_water_sim  = ν_ratio * ν_air_sim

visc_air   = ViscosityMorris(nu=ν_air_sim)
visc_water = ViscosityMorris(nu=ν_water_sim)

# Equations of state
eos_air   = StateEquationCole(; sound_speed=cs, reference_density=ρ_air,   exponent=1,
                              clip_negative_pressure=false)
eos_water = StateEquationCole(; sound_speed=cs, reference_density=ρ_water, exponent=7,
                              clip_negative_pressure=false)

ρ_calc = ContinuityDensity()


# tank_air = RectangularTank(dx, (1.0, 1.0), (1.0, 1.0), ρ_air;
#                            n_layers=boundary_layers, spacing_ratio=spacing_ratio,
#                            acceleration=(0.0, -g), state_equation=eos_air)


tank_air = RectangularTank(dx, (1.0, 1.0), (1.0, 1.0), ρ_air;
                           n_layers=boundary_layers, spacing_ratio=spacing_ratio)

# air lattice from tank
air_full = tank_air.fluid

sphere1 = SphereShape(dx, 2*dx, (0.475, 0.475),
                      ρ_water, sphere_type=VoxelSphere(), velocity=(0.0, 0.0))

# remove the overlapping AIR particle at (cx, cy) in one shot
air_shape = setdiff(air_full, sphere1)

# --- Systems -----------------------------------------------------------------
# boundary_density_calculator = AdamiPressureExtrapolation()
boundary_density_calculator = ContinuityDensity()
boundary_model = BoundaryModelDummyParticles(
    tank_air.boundary.density, tank_air.boundary.mass,
    state_equation=eos_air, boundary_density_calculator, kernel, h
)

boundary_system = BoundarySPHSystem(tank_air.boundary, boundary_model; adhesion_coefficient=0.0)

air_system = WeaklyCompressibleSPHSystem(
    air_shape, ρ_calc, eos_air, kernel, h;
    viscosity=visc_air, density_diffusion=nothing,
    acceleration=(0.0, -g), correction=nothing, reference_particle_spacing=0
)

boundary_model_sphere = BoundaryModelDummyParticles(
    sphere1.density, sphere1.mass,
    state_equation=eos_water, boundary_density_calculator, kernel, h
)

sphere_system = BoundarySPHSystem(sphere1, boundary_model_sphere; adhesion_coefficient=0.0)


# --- Simulation --------------------------------------------------------------
semi = Semidiscretization(sphere_system, air_system, boundary_system;
    neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing),
    parallelization_backend=PolyesterBackend()
)
ode = semidiscretize(semi, tspan)

info_cb     = InfoCallback(interval=100)
save_cb     = SolutionSavingCallback(dt=0.02, prefix="single_bnd_fall")
stepsize_cb = StepsizeCallback(cfl=0.9)
callbacks   = CallbackSet(info_cb, save_cb, stepsize_cb)

sol = solve(ode, RDPK3SpFSAL35();
            abstol=1e-5, reltol=1e-4, dtmax=1e-2,
            save_everystep=false, callback=callbacks)

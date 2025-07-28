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

# =============================================================================
# ==== 1.  Physical & numerical parameters
const gravity        = 9.81           # [m/s²]
const fluid_density  = 1_025.0        # [kg/m³] (sea water)
const fluid_viscosity= 1e-6           # [m²/s] (viscosity of sea water)
const H_water        = 0.5            # [m]  initial column height
const W_water        = 0.4            # [m]  initial column width
const tank_height    = 2.0            # [m]
const tank_length    = 1.5            # [m]

const resolution_factor      = 150     # particles per H_water
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

##############################
# 1.  Clustering             #
##############################

"""
    airborne_clusters(coords; z_cut = H_water) -> Vector{Vector{Int}}

Return a list of particle‑index clusters representing individual **droplets**.
A droplet is any connected set of particles whose *centres* are closer than
`link_radius = 1.1*h` (where `h = smoothing_length`).  Only particles with
`y > z_cut` are considered (i.e. detached from the bulk free‑surface).
"""
function airborne_clusters(coords; z_cut=0.0, max_particles=ceil(Int, π*(0.01/2)^2 / (Δx_f^2)))
    # Precompute all airborne particle indices
    airborne = findall(y -> y > z_cut, view(coords, 2, :))
    visited  = falses(size(coords, 2))
    clusters = Vector{Vector{Int}}()
    link2    = (1.1 * smoothing_length)^2

    for seed in airborne
        visited[seed] && continue
        # start a new cluster BFS
        cluster = Int[seed]
        visited[seed] = true
        queue = [seed]
        too_big = false

        while !isempty(queue)
            i = popfirst!(queue)
            for j in airborne
                if !visited[j]
                    dx = coords[1,i] - coords[1,j]
                    dy = coords[2,i] - coords[2,j]
                    if dx*dx + dy*dy < link2
                        push!(cluster, j)
                        visited[j] = true
                        # early bail if cluster too large
                        if length(cluster) > max_particles
                            too_big = true
                            break
                        end
                        push!(queue, j)
                    end
                end
            end
            too_big && break
        end

        # only keep clusters under the size threshold
        if !too_big
            push!(clusters, cluster)
        end
    end
    return clusters
end

"Equivalent spherical radius [m] of a cluster."
function cluster_radius(c, mass)
    area = length(c)*mass/fluid_density
    return (6*area/π)^(1/3)/2      # radius = d/2
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

"""
    particle_cv(system, data, t) -> Vector{Float64} | Vector{Any}

Return the list of *equivalent spherical radii* (metres) for **all** airborne
clusters in the current frame.  An **empty JSON list `[]`** is returned when no
particles have detached, making downstream parsing trivial.
"""
# function particle_cv(system, data, t)
#     clusters = airborne_clusters(data.coordinates)
#     if isempty(clusters)
#         return []                       # empty *untyped* vector ⇒ `[]` in JSON
#     end
#     radii = Float64[]
#     for c in clusters
#         area = length(c) * data.mass[1] / fluid_density
#         push!(radii, (6 * area / π)^(1/3) / 2)           # radius = d/2 [m]
#     end
#     return radii
# end

"Return droplet radii [m] in clusters whose centroid lies within ±Δz of a level."
# function particle_cv(system, v_ode, u_ode, semi, t, h; light_sheet_height=0.1, volume_thickness=0.1)
#     u_system  = TrixiParticles.wrap_u(u_ode, system, semi)
#     coords    = TrixiParticles.current_coordinates(u_system, system)
#     particle_y_in_volume=Float64[];
#     Δz=light_sheet_height/2

#     # TrixiParticles.@threaded semi
#     for particle in TrixiParticles.eachparticle(system)
#         # Get the particle's position
#         pos = coords[:, particle]
#         # Check if the particle's y-coordinate is within the specified range
#         if abs(pos[2] - h) ≤ Δz && pos[1] < volume_thickness
#             push!(particle_y_in_volume, pos[2])
#         end
#     end

#     return particle_y_in_volume
# end

buf = Vector{Float64}(undef, size(tank.fluid.coordinates[2, :]))


function particle_cv(system, v_ode, u_ode, semi, t, h;
                     light_sheet_height=0.1,
                     volume_thickness=0.1)

    # 1. unwrap & get coords
    u_sys = TrixiParticles.wrap_u(u_ode, system, semi)
    coords = TrixiParticles.current_coordinates(u_sys, system)
    N = size(coords, 2)

    # 2. prep slab half‑thickness
    Δz = light_sheet_height / 2

    # 3. pre‑allocate a Float64 buffer of length N
    ys = coords[2, :]
    xs = coords[1, :]
    buf = Vector{Float64}(undef, N)
    # global buf
    cnt = 0

    # 4. scan once, inbounds + SIMD, fill buffer
    @inbounds @simd for i in 1:N
        y = ys[i]
        # first test cheap x‑cut, then y‑slab
        if xs[i] > tank_length - volume_thickness && abs(y - h) ≤ Δz
            cnt += 1
            buf[cnt] = y
        end
    end

    # 5. return only the filled portion
    return @view buf[1:cnt]
end

# function particle_cv(system, v_ode, u_ode, semi, t, h;
#                      light_sheet_height=0.1,
#                      volume_thickness=0.1)

#     # 1. unwrap & grab coords
#     u_sys = TrixiParticles.wrap_u(u_ode, system, semi)
#     coords = TrixiParticles.current_coordinates(u_sys, system)
#     N = size(coords, 2)
#     xs = coords[1, :]
#     ys = coords[2, :]

#     # 2. slab half‑thickness
#     Δz = light_sheet_height / 2

#     # 3. prepare one Float64 vector per thread
#     nt = Threads.nthreads()
#     local_bufs = [Float64[] for _ in 1:nt]

#     # 4. parallel loop over all particles
#     TrixiParticles.@threaded semi for i in TrixiParticles.eachparticle(system)
#         y = ys[i]; x = xs[i]
#         if x < volume_thickness && abs(y - h) ≤ Δz
#             push!(local_bufs[Threads.threadid()], y)
#         end
#     end

#     # 5. concatenate and return
#     return vcat(local_bufs...)
# end

# Generate one function per height
# particle_cv_z50 = (sys,d,t)->particle_cv(sys,d,t,0.05)
# particle_cv_z150 = (sys,d,t)->particle_cv(sys,d,t,0.15)
# particle_cv_z250 = (sys,d,t)->particle_cv(sys,d,t,0.25)
# particle_cv_z350 = (sys,d,t)->particle_cv(sys,d,t,0.35)
# particle_cv_z450 = (sys,d,t)->particle_cv(sys,d,t,0.45)
# particle_cv_z600 = (sys,d,t)->particle_cv(sys,d,t,0.60)
# particle_cv_z800 = (sys,d,t)->particle_cv(sys,d,t,0.80)

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

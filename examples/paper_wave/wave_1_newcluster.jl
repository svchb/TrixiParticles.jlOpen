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
# function coarse_droplet_counts(system, data, t)
#     clusters = airborne_clusters(data.coordinates)
#     counts   = zeros(Int, 3)
#     for c in clusters
#         area = length(c) * data.mass[1] / fluid_density       # m² (2‑D proxy)
#         d_eq = (6 * area / π)^(1/3)                          # m
#         if 0.01 ≤ d_eq < 0.02       # 10–20 mm bin
#             counts[1] += 1
#         elseif 0.02 ≤ d_eq < 0.03   # 20–30 mm bin
#             counts[2] += 1
#         elseif d_eq ≥ 0.03          # >30 mm bin
#             counts[3] += 1
#         end
#     end
#     return counts
# end

##############################
# 3.  Max vertical velocity  #
##############################

"""
    max_vertical_velocity(system, data, t) -> Float64

Return the maximum absolute *vertical* velocity of particles above the free
surface.  Returns `0.0` when no such particles exist (e.g. before impact).
"""
# function max_vertical_velocity(system, data, t)
#     coords = data.coordinates
#     airborne_idx = findall(y -> y > H_water, view(coords, 2, :))
#     if isempty(airborne_idx)
#         return 0.0
#     else
#         vel = data.velocity
#         return maximum(abs.(view(vel, 2, airborne_idx)))
#     end
# end

##############################
# 4.  Droplet radii list     #
##############################

mutable struct ClusterScratch
    parent::Vector{Int}        # Union‑Find parent pointers
    size::Vector{Int}          # cluster size; -1 => oversize; 0 => not airborne
    sumy::Vector{Float64}      # sum of y’s for centroid calculation
    airborne::Vector{Int}      # list of indices that are airborne
end

function ClusterScratch(N::Int)
    ClusterScratch(collect(1:N), zeros(Int, N), zeros(Float64, N), Int[])
end

# Reset per frame
function _reset!(scratch::ClusterScratch, coords::AbstractMatrix{<:Real}, z_cut::Float64)
    parent, size, sumy, airborne = scratch.parent, scratch.size, scratch.sumy, scratch.airborne
    empty!(airborne)
    @inbounds @simd for i in eachindex(parent)
        parent[i] = i
        yi = coords[2,i]
        if yi > z_cut
            size[i] = 1
            sumy[i] = yi
            push!(airborne, i)
        else
            size[i] = 0
        end
    end
end

# Path‑compressed find
@inline function _find!(parent::Vector{Int}, i::Int)
    @inbounds while parent[i] != i
        parent[i] = parent[parent[i]]
        i = parent[i]
    end
    return i
end

# Union by size, mark oversize
@inline function _union!(scratch::ClusterScratch, a::Int, b::Int, maxp::Int)
    p, size, sumy = scratch.parent, scratch.size, scratch.sumy
    ra = _find!(p, a); rb = _find!(p, b)
    ra == rb && return
    sa, sb = size[ra], size[rb]
    # if either is already oversize, result stays oversize
    if sa == -1 || sb == -1
        p[rb] = ra
        size[ra] = -1
        return
    end
    # union smaller into larger
    if sa < sb
        ra, rb = rb, ra
        sa, sb = sb, sa
    end
    p[rb] = ra
    newsize = sa + sb
    if newsize > maxp
        size[ra] = -1
    else
        size[ra] = newsize
        sumy[ra] += sumy[rb]
    end
end

const _radius_factor = (6 / π)^(1/3) / 2

function droplet_radii_at_level(system, v_ode, u_ode, semi, t, h;
                                light_sheet_height=0.1,
                                z_cut=0.0,
                                max_particles=ceil(Int, π*(0.01/2)^2 / (Δx_f^2)))
    u_system  = TrixiParticles.wrap_u(u_ode, system, semi)
    coords    = TrixiParticles.current_coordinates(u_system, system)
    pmass     = TrixiParticles.hydrodynamic_mass(system, 1)

    global scratch

    # build clusters on‐the‐fly
    _reset!(scratch, coords, z_cut)
    TrixiParticles.foreach_point_neighbor(system, system, coords, coords, semi;
                           points = TrixiParticles.each_moving_particle(system)) do p, q, _, _
        # only link airborne particles, avoid double work with p<q
        if p<q && scratch.size[p]>0 && scratch.size[q]>0
            _union!(scratch, p, q, max_particles)
        end
    end

    # extract radii for clusters whose centroid lies near height h
    Δz = light_sheet_height/2
    radii = Float64[]
    pvec  = scratch.parent
    svec  = scratch.size
    sy    = scratch.sumy

    @inbounds for i in scratch.airborne
        # only consider roots
        if pvec[i]==i && svec[i]>0
            ybar = sy[i] / svec[i]
            if abs(ybar - h) ≤ Δz
                vol = svec[i] * pmass / fluid_density
                push!(radii, _radius_factor * vol^(1/3))
            end
        end
    end

    return radii
end

# Generate one function per height
const scratch = ClusterScratch(size(tank.fluid.coordinates, 2))
droplet_radii_z50 = (system, v_ode, u_ode, semi, t)->droplet_radii_at_level(system, v_ode, u_ode, semi, t, 0.05)
droplet_radii_z150 = (system, v_ode, u_ode, semi, t)->droplet_radii_at_level(system, v_ode, u_ode, semi, t, 0.15)
droplet_radii_z250 = (system, v_ode, u_ode, semi, t)->droplet_radii_at_level(system, v_ode, u_ode, semi, t, 0.25)
droplet_radii_z350 = (system, v_ode, u_ode, semi, t)->droplet_radii_at_level(system, v_ode, u_ode, semi, t, 0.35)
droplet_radii_z450 = (system, v_ode, u_ode, semi, t)->droplet_radii_at_level(system, v_ode, u_ode, semi, t, 0.45)
droplet_radii_z600 = (system, v_ode, u_ode, semi, t)->droplet_radii_at_level(system, v_ode, u_ode, semi, t, 0.60)
droplet_radii_z800 = (system, v_ode, u_ode, semi, t)->droplet_radii_at_level(system, v_ode, u_ode, semi, t, 0.80)


post_cb = PostprocessCallback(
    ; dt = 0.02,                        # 500hz
      output_directory = "post",       # JSON files in ./post/
      filename = "spray_metrics",      # → spray_metrics_*.json
      write_csv = true,
      #splash_height,
      #coarse_droplet_counts,
      #max_vertical_velocity,
      #droplet_radii,
      droplet_radii_z50,
      droplet_radii_z150,
      droplet_radii_z250,
      droplet_radii_z350,
      droplet_radii_z450,
      droplet_radii_z600,
      droplet_radii_z800
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

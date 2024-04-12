"""
    Semidiscretization(systems...; neighborhood_search=GridNeighborhoodSearch,
                       periodic_box_min_corner=nothing, periodic_box_max_corner=nothing,
                       threaded_nhs_update=true)

The semidiscretization couples the passed systems to one simulation.

The type of neighborhood search to be used in the simulation can be specified with
the keyword argument `neighborhood_search`. A value of `nothing` means no neighborhood search.

# Arguments
- `systems`: Systems to be coupled in this semidiscretization

# Keywords
- `neighborhood_search`:    The type of neighborhood search to be used in the simulation.
                            By default, the [`GridNeighborhoodSearch`](@ref) is used.
                            Use [`TrivialNeighborhoodSearch`](@ref) or `nothing` to loop
                            over all particles (no neighborhood search).
- `periodic_box_min_corner`:    In order to use a (rectangular) periodic domain, pass the
                                coordinates of the domain corner in negative coordinate
                                directions.
- `periodic_box_max_corner`:    In order to use a (rectangular) periodic domain, pass the
                                coordinates of the domain corner in positive coordinate
                                directions.
- `threaded_nhs_update=true`:   Can be used to deactivate thread parallelization in the neighborhood search update.
                                This can be one of the largest sources of variations between simulations
                                with different thread numbers due to particle ordering changes.

# Examples
```jldoctest; output = false, setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), sol=nothing); ref_system = fluid_system)
semi = Semidiscretization(fluid_system, boundary_system)

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=TrivialNeighborhoodSearch)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Semidiscretization                                                                               │
│ ══════════════════                                                                               │
│ #spatial dimensions: ………………………… 2                                                                │
│ #systems: ……………………………………………………… 2                                                                │
│ neighborhood search: ………………………… TrivialNeighborhoodSearch                                        │
│ total #particles: ………………………………… 636                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
struct Semidiscretization{S, RU, RV, NS}
    systems               :: S
    ranges_u              :: RU
    ranges_v              :: RV
    neighborhood_searches :: NS
    systems_have_collided :: MutableBool

    # Dispatch at `systems` to distinguish this constructor from the one below when
    # 4 systems are passed.
    # This is an internal constructor only used in `test/count_allocations.jl`.
    function Semidiscretization(systems::Tuple, ranges_u, ranges_v, neighborhood_searches)
        new{typeof(systems), typeof(ranges_u),
            typeof(ranges_v), typeof(neighborhood_searches)}(systems, ranges_u, ranges_v,
                                                             neighborhood_searches, MutableBool(false))
    end
end

function Semidiscretization(systems...; neighborhood_search=GridNeighborhoodSearch,
                            periodic_box_min_corner=nothing,
                            periodic_box_max_corner=nothing, threaded_nhs_update=true)
    systems = filter(system -> !isnothing(system), systems)

    # Check e.g. that the boundary systems are using a state equation if EDAC is not used.
    # Other checks might be added here later.
    check_configuration(systems)

    sizes_u = [u_nvariables(system) * n_moving_particles(system)
               for system in systems]
    ranges_u = Tuple((sum(sizes_u[1:(i - 1)]) + 1):sum(sizes_u[1:i])
                     for i in eachindex(sizes_u))
    sizes_v = [v_nvariables(system) * n_moving_particles(system)
               for system in systems]
    ranges_v = Tuple((sum(sizes_v[1:(i - 1)]) + 1):sum(sizes_v[1:i])
                     for i in eachindex(sizes_v))

    # Create (and initialize) a tuple of n neighborhood searches for each of the n systems
    # We will need one neighborhood search for each pair of systems.
    searches = Tuple(Tuple(create_neighborhood_search(system, neighbor,
                                                      Val(neighborhood_search),
                                                      periodic_box_min_corner,
                                                      periodic_box_max_corner,
                                                      threaded_nhs_update)
                           for neighbor in systems)
                     for system in systems)

    return Semidiscretization(systems, ranges_u, ranges_v, searches)
end

# Inline show function e.g. Semidiscretization(neighborhood_search=...)
function Base.show(io::IO, semi::Semidiscretization)
    @nospecialize semi # reduce precompilation time

    print(io, "Semidiscretization(")
    for system in semi.systems
        print(io, system, ", ")
    end
    print(io, "neighborhood_search=")
    print(io, semi.neighborhood_searches |> eltype |> eltype |> nameof)
    print(io, ")")
end

# Show used during summary printout
function Base.show(io::IO, ::MIME"text/plain", semi::Semidiscretization)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "Semidiscretization")
        summary_line(io, "#spatial dimensions", ndims(semi.systems[1]))
        summary_line(io, "#systems", length(semi.systems))
        summary_line(io, "neighborhood search",
                     semi.neighborhood_searches |> eltype |> eltype |> nameof)
        summary_line(io, "total #particles", sum(nparticles.(semi.systems)))
        summary_footer(io)
    end
end

function create_neighborhood_search(system, neighbor,
                                    ::Union{Val{nothing}, Val{TrivialNeighborhoodSearch}},
                                    periodic_box_min_corner, periodic_box_max_corner,
                                    threaded_nhs_update)
    radius = compact_support(system, neighbor)
    TrivialNeighborhoodSearch{ndims(system)}(radius, eachparticle(neighbor),
                                             periodic_box_min_corner=periodic_box_min_corner,
                                             periodic_box_max_corner=periodic_box_max_corner)
end

function create_neighborhood_search(system, neighbor, ::Val{GridNeighborhoodSearch},
                                    periodic_box_min_corner, periodic_box_max_corner,
                                    threaded_nhs_update)
    radius = compact_support(system, neighbor)
    search = GridNeighborhoodSearch{ndims(system)}(radius, nparticles(neighbor),
                                                   periodic_box_min_corner=periodic_box_min_corner,
                                                   periodic_box_max_corner=periodic_box_max_corner,
                                                   threaded_nhs_update=threaded_nhs_update)
    # Initialize neighborhood search
    initialize!(search, initial_coordinates(neighbor))

    return search
end

@inline function compact_support(system, neighbor)
    (; smoothing_kernel, smoothing_length) = system
    return compact_support(smoothing_kernel, smoothing_length)
end

@inline function compact_support(system::TotalLagrangianSPHSystem,
                                 neighbor::TotalLagrangianSPHSystem)
    (; smoothing_kernel, smoothing_length) = system
    return compact_support(smoothing_kernel, smoothing_length)
end

@inline function compact_support(system::Union{TotalLagrangianSPHSystem, BoundarySPHSystem,
                                               RigidSPHSystem},
                                 neighbor)
    return compact_support(system, system.boundary_model, neighbor)
end

@inline function compact_support(system, model::BoundaryModelMonaghanKajtar, neighbor)
    # Use the compact support of the fluid for solid-fluid interaction
    return compact_support(neighbor, system)
end

@inline function compact_support(system, model::BoundaryModelMonaghanKajtar,
                                 neighbor::BoundarySPHSystem)
    # This NHS is never used
    return 0.0
end

@inline function compact_support(system, model::BoundaryModelDummyParticles, neighbor)
    # TODO: Monaghan-Kajtar BC are using the fluid's compact support for solid-fluid
    # interaction. Dummy particle BC use the model's compact support, which is also used
    # for density summations.
    (; smoothing_kernel, smoothing_length) = model
    return compact_support(smoothing_kernel, smoothing_length)
end

@inline function get_neighborhood_search(system, semi)
    (; neighborhood_searches) = semi

    system_index = system_indices(system, semi)

    return neighborhood_searches[system_index][system_index]
end

@inline function get_neighborhood_search(system, neighbor_system, semi)
    (; neighborhood_searches) = semi

    system_index = system_indices(system, semi)
    neighbor_index = system_indices(neighbor_system, semi)

    return neighborhood_searches[system_index][neighbor_index]
end

@inline function system_indices(system, semi)
    # Note that this takes only about 5 ns, while mapping systems to indices with a `Dict`
    # is ~30x slower because `hash(::System)` is very slow.
    index = findfirst(==(system), semi.systems)

    if isnothing(index)
        throw(ArgumentError("system is not in the semidiscretization"))
    end

    return index
end

# This is just for readability to loop over all systems without allocations
@inline foreach_system(f, semi::Union{NamedTuple, Semidiscretization}) = foreach_noalloc(f,
                                                                                         semi.systems)
@inline foreach_system(f, systems) = foreach_noalloc(f, systems)

"""
    semidiscretize(semi, tspan; reset_threads=true)

Create an `ODEProblem` from the semidiscretization with the specified `tspan`.

# Arguments
- `semi`: A [`Semidiscretization`](@ref) holding the systems involved in the simulation.
- `tspan`: The time span over which the simulation will be run.

# Keywords
- `reset_threads`: A boolean flag to reset Polyester.jl threads before the simulation (default: `true`).
  After an error within a threaded loop, threading might be disabled. Resetting the threads before the simulation
  ensures that threading is enabled again for the simulation.
  See also [trixi-framework/Trixi.jl#1583](https://github.com/trixi-framework/Trixi.jl/issues/1583).

# Returns
A `DynamicalODEProblem` (see [the OrdinaryDiffEq.jl docs](https://docs.sciml.ai/DiffEqDocs/stable/types/dynamical_types/))
to be integrated with [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).
Note that this is not a true `DynamicalODEProblem` where the acceleration does not depend on the velocity.
Therefore, not all integrators designed for `DynamicalODEProblems` will work properly.
However, all integrators designed for `ODEProblems` can be used.

# Examples
```jldoctest; output = false, filter = r"u0: .*", setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), sol=nothing); ref_system = fluid_system)
semi = Semidiscretization(fluid_system, boundary_system)
tspan = (0.0, 1.0)
ode_problem = semidiscretize(semi, tspan)

# output
ODEProblem with uType RecursiveArrayTools.ArrayPartition{Float64, Tuple{Vector{Float64}, Vector{Float64}}} and tType Float64. In-place: true
timespan: (0.0, 1.0)
u0: ([...], [...]) *this line is ignored by filter*
```
"""
function semidiscretize(semi, tspan; reset_threads=true)
    (; systems) = semi

    @assert all(system -> eltype(system) === eltype(systems[1]), systems)
    ELTYPE = eltype(systems[1])

    # Optionally reset Polyester.jl threads. See
    # https://github.com/trixi-framework/Trixi.jl/issues/1583
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    if reset_threads
        Polyester.reset_threads!()
    end

    # Initialize all particle systems
    @trixi_timeit timer() "initialize particle systems" begin
        foreach_system(semi) do system
            # Get the neighborhood search for this system
            neighborhood_search = get_neighborhood_search(system, semi)

            # Initialize this system
            initialize!(system, neighborhood_search)
        end
    end

    sizes_u = (u_nvariables(system) * n_moving_particles(system) for system in systems)
    sizes_v = (v_nvariables(system) * n_moving_particles(system) for system in systems)
    u0_ode = Vector{ELTYPE}(undef, sum(sizes_u))
    v0_ode = Vector{ELTYPE}(undef, sum(sizes_v))

    # Set initial condition
    foreach_system(semi) do system
        u0_system = wrap_u(u0_ode, system, semi)
        v0_system = wrap_v(v0_ode, system, semi)

        write_u0!(u0_system, system)
        write_v0!(v0_system, system)
    end

    return DynamicalODEProblem(kick!, drift!, v0_ode, u0_ode, tspan, semi)
end

"""
    restart_with!(semi, sol)

Set the initial coordinates and velocities of all systems in `semi` to the final values
in the solution `sol`.
[`semidiscretize`](@ref) has to be called again afterwards, or another
[`Semidiscretization`](@ref) can be created with the updated systems.

# Arguments
- `semi`:   The semidiscretization
- `sol`:    The `ODESolution` returned by `solve` of `OrdinaryDiffEq`
"""
function restart_with!(semi, sol; reset_threads=true)
    # Optionally reset Polyester.jl threads. See
    # https://github.com/trixi-framework/Trixi.jl/issues/1583
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    if reset_threads
        Polyester.reset_threads!()
    end

    foreach_system(semi) do system
        v = wrap_v(sol.u[end].x[1], system, semi)
        u = wrap_u(sol.u[end].x[2], system, semi)

        restart_with!(system, v, u)
    end

    return semi
end

# We have to pass `system` here for type stability,
# since the type of `system` determines the return type.
@inline function wrap_u(u_ode, system, semi)
    (; ranges_u) = semi

    range = ranges_u[system_indices(system, semi)]

    @boundscheck @assert length(range) == u_nvariables(system) * n_moving_particles(system)

    # This is a non-allocating version of:
    # return unsafe_wrap(Array{eltype(u_ode), 2}, pointer(view(u_ode, range)),
    #                    (u_nvariables(system), n_moving_particles(system)))
    return PtrArray(pointer(view(u_ode, range)),
                    (StaticInt(u_nvariables(system)), n_moving_particles(system)))
end

@inline function wrap_v(v_ode, system, semi)
    (; ranges_v) = semi

    range = ranges_v[system_indices(system, semi)]

    @boundscheck @assert length(range) == v_nvariables(system) * n_moving_particles(system)

    return PtrArray(pointer(view(v_ode, range)),
                    (StaticInt(v_nvariables(system)), n_moving_particles(system)))
end

function calculate_dt(v_ode, u_ode, cfl_number, semi::Semidiscretization)
    (; systems) = semi

    return minimum(system -> calculate_dt(v_ode, u_ode, cfl_number, system), systems)
end

function drift!(du_ode, v_ode, u_ode, semi, t)
    @trixi_timeit timer() "drift!" begin
        @trixi_timeit timer() "reset ∂u/∂t" set_zero!(du_ode)

        @trixi_timeit timer() "velocity" begin
            # Set velocity and add acceleration for each system
            foreach_system(semi) do system
                du = wrap_u(du_ode, system, semi)
                v = wrap_v(v_ode, system, semi)

                @threaded for particle in each_moving_particle(system)
                    # This can be dispatched per system
                    add_velocity!(du, v, particle, system)
                end
            end
        end
        # @trixi_timeit timer() "collision interaction" collision_interaction!(du_ode, v_ode,
        # u_ode, semi)
    end

    return du_ode
end

@inline function add_velocity!(du, v, particle, system)
    for i in 1:ndims(system)
        du[i, particle] = v[i, particle]
    end

    return du
end

@inline add_velocity!(du, v, particle, system::BoundarySPHSystem) = du

function kick!(dv_ode, v_ode, u_ode, semi, t)
    @trixi_timeit timer() "kick!" begin
        @trixi_timeit timer() "reset ∂v/∂t" set_zero!(dv_ode)

        @trixi_timeit timer() "update systems and nhs" update_systems_and_nhs(v_ode, u_ode,
                                                                              semi, t)

        @trixi_timeit timer() "system interaction" system_interaction!(dv_ode, v_ode, u_ode,
                                                                       semi)

        @trixi_timeit timer() "source terms" add_source_terms!(dv_ode, v_ode, u_ode, semi)

        @trixi_timeit timer() "collision interaction" collision_interaction!(dv_ode, v_ode,
        u_ode, semi)
    end

    return dv_ode
end

# Update the systems and neighborhood searches (NHS) for a simulation before calling `interact!` to compute forces
function update_systems_and_nhs(v_ode, u_ode, semi, t)
    # First update step before updating the NHS
    # (for example for writing the current coordinates in the solid system)
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        update_positions!(system, v, u, v_ode, u_ode, semi, t)
    end

    # Update NHS
    @trixi_timeit timer() "update nhs" update_nhs(u_ode, semi)

    # Second update step.
    # This is used to calculate density and pressure of the fluid systems
    # before updating the boundary systems,
    # since the fluid pressure is needed by the Adami interpolation.
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        update_quantities!(system, v, u, v_ode, u_ode, semi, t)
    end

    # Perform correction and pressure calculation
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        update_pressure!(system, v, u, v_ode, u_ode, semi, t)
    end

    # Final update step for all remaining systems
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        update_final!(system, v, u, v_ode, u_ode, semi, t)
    end
end

function update_nhs(u_ode, semi)
    # Update NHS for each pair of systems
    foreach_system(semi) do system
        foreach_system(semi) do neighbor
            u_neighbor = wrap_u(u_ode, neighbor, semi)
            neighborhood_search = get_neighborhood_search(system, neighbor, semi)

            update!(neighborhood_search, nhs_coords(system, neighbor, u_neighbor))
        end
    end
end

function add_source_terms!(dv_ode, v_ode, u_ode, semi)
    foreach_system(semi) do system
        dv = wrap_v(dv_ode, system, semi)
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        @threaded for particle in each_moving_particle(system)
            # Dispatch by system type to exclude boundary systems
            add_acceleration!(dv, particle, system)
            add_source_terms_inner!(dv, v, u, particle, system, source_terms(system))
        end
    end

    return dv_ode
end

@inline source_terms(system) = nothing
@inline source_terms(system::RigidSPHSystem) = nothing
@inline source_terms(system::Union{FluidSystem, SolidSystem}) = system.source_terms

@inline add_acceleration!(dv, particle, system) = dv

@inline function add_acceleration!(dv, particle, system::Union{FluidSystem, SolidSystem})
    (; acceleration) = system

    for i in 1:ndims(system)
        dv[i, particle] += acceleration[i]
    end

    return dv
end

@inline function add_source_terms_inner!(dv, v, u, particle, system, source_terms_)
    coords = current_coords(u, system, particle)
    velocity = current_velocity(v, system, particle)
    density = particle_density(v, system, particle)
    pressure = particle_pressure(v, system, particle)

    source = source_terms_(coords, velocity, density, pressure)

    # Loop over `eachindex(source)`, so that users could also pass source terms for
    # the density when using `ContinuityDensity`.
    for i in eachindex(source)
        dv[i, particle] += source[i]
    end

    return dv
end

@inline add_source_terms_inner!(dv, v, u, particle, system, source_terms_::Nothing) = dv

@doc raw"""
    SourceTermDamping(; damping_coefficient)

A source term to be used when a damping step is required before running a full simulation.
The term ``-c \cdot v_a`` is added to the acceleration ``\frac{\mathrm{d}v_a}{\mathrm{d}t}``
of particle ``a``, where ``c`` is the damping coefficient and ``v_a`` is the velocity of
particle ``a``.

# Keywords
- `damping_coefficient`:    The coefficient ``d`` above. A higher coefficient means more
                            damping. A coefficient of `1e-4` is a good starting point for
                            damping a fluid at rest.

# Examples
```jldoctest; output = false
source_terms = SourceTermDamping(; damping_coefficient=1e-4)

# output
SourceTermDamping{Float64}(0.0001)
```
"""
struct SourceTermDamping{ELTYPE}
    damping_coefficient::ELTYPE

    function SourceTermDamping(; damping_coefficient)
        return new{typeof(damping_coefficient)}(damping_coefficient)
    end
end

@inline function (source_term::SourceTermDamping)(coords, velocity, density, pressure)
    (; damping_coefficient) = source_term

    return -damping_coefficient * velocity
end

function system_interaction!(dv_ode, v_ode, u_ode, semi)
    # Call `interact!` for each pair of systems
    foreach_system(semi) do system
        foreach_system(semi) do neighbor
            # Construct string for the interactions timer.
            # Avoid allocations from string construction when no timers are used.
            if timeit_debug_enabled()
                system_index = system_indices(system, semi)
                neighbor_index = system_indices(neighbor, semi)
                timer_str = "$(timer_name(system))$system_index-$(timer_name(neighbor))$neighbor_index"
            else
                timer_str = ""
            end

            interact!(dv_ode, v_ode, u_ode, system, neighbor, semi, timer_str=timer_str)
        end
    end

    return dv_ode
end

# Function barrier to make benchmarking interactions easier.
# One can benchmark, e.g. the fluid-fluid interaction, with:
# dv_ode, du_ode = copy(sol.u[end]).x; v_ode, u_ode = copy(sol.u[end]).x;
# @btime TrixiParticles.interact!($dv_ode, $v_ode, $u_ode, $fluid_system, $fluid_system, $semi);
@inline function interact!(dv_ode, v_ode, u_ode, system, neighbor, semi; timer_str="")
    dv = wrap_v(dv_ode, system, semi)
    v_system = wrap_v(v_ode, system, semi)
    u_system = wrap_u(u_ode, system, semi)

    v_neighbor = wrap_v(v_ode, neighbor, semi)
    u_neighbor = wrap_u(u_ode, neighbor, semi)
    nhs = get_neighborhood_search(system, neighbor, semi)

    @trixi_timeit timer() timer_str begin
        interact!(dv, v_system, u_system, v_neighbor, u_neighbor, nhs, system, neighbor)
    end
end

# start of recursion
function collision_interaction!(dv_ode, v_ode, u_ode, semi)

    collision_interaction!(dv_ode, v_ode, u_ode, semi, true)

    return dv_ode
end

@inline function reset_system_collision_status(semi)
    foreach_system(semi) do system
        if system isa RigidSPHSystem || system isa TotalLagrangianSPHSystem
            system.has_collided.value = false
        end
    end
end

function collision_interaction!(dv_ode, v_ode, u_ode, semi, systems_have_collided)
    if systems_have_collided
        semi.systems_have_collided.value = false
        reset_system_collision_status(semi)

        # Call `interact!` for each pair of systems
        foreach_system(semi) do system
            foreach_system(semi) do neighbor
                system_index = system_indices(system, semi)
                neighbor_index = system_indices(neighbor, semi)

                # the same system does not collide with its self
                if system_index != neighbor_index
                    # Construct string for the interactions timer.
                    # Avoid allocations from string construction when no timers are used.
                    if timeit_debug_enabled()
                        timer_str = "$(timer_name(system))$system_index-$(timer_name(neighbor))$neighbor_index"
                    else
                        timer_str = ""
                    end

                    collision_interact!(dv_ode, v_ode, u_ode, system, neighbor, semi,
                                        timer_str=timer_str)
                end
            end
        end

        # if systems have collided we need to call this again since there can be more collisions
        collision_interaction!(dv_ode, v_ode, u_ode, semi, semi.systems_have_collided.value)
    end

    return dv_ode
end



# FluidSystems don't collide with each other
@inline function collision_interact!(dv_ode, v_ode, u_ode, system::FluidSystem,
                                     neighbor::FluidSystem, semi; timer_str="")
    return dv_ode
end

# BoundarySPHSystems don't collide with each other use RigidSPHSystem
@inline function collision_interact!(dv_ode, v_ode, u_ode,
                                     system::BoundarySPHSystem, neighbor::BoundarySPHSystem,
                                     semi; timer_str="")
    return dv_ode
end

@inline function set_collision_status!(semi, system)
    return semi
end

@inline function set_collision_status!(semi, system::Union{RigidSPHSystem, TotalLagrangianSPHSystem})
    semi.systems_have_collided.value = semi.systems_have_collided.value || system.has_collided.value
    return semi
end

# Function barrier to make benchmarking interactions easier.
# One can benchmark, e.g. the fluid-fluid interaction, with:
# dv_ode, du_ode = copy(sol.u[end]).x; v_ode, u_ode = copy(sol.u[end]).x;
# @btime TrixiParticles.interact!($dv_ode, $v_ode, $u_ode, $fluid_system, $fluid_system, $semi);
@inline function collision_interact!(dv_ode, v_ode, u_ode, system, neighbor, semi;
                                     timer_str="")
    dv = wrap_v(dv_ode, system, semi)
    # dv = wrap_u(dv_ode, system, semi)

    v_system = wrap_v(v_ode, system, semi)
    u_system = wrap_u(u_ode, system, semi)

    v_neighbor = wrap_v(v_ode, neighbor, semi)
    u_neighbor = wrap_u(u_ode, neighbor, semi)
    nhs = get_neighborhood_search(system, neighbor, semi)

    @trixi_timeit timer() timer_str begin
        collision_interact!(dv, v_system, u_system, v_neighbor, u_neighbor, nhs, system,
                            neighbor)
    end

    set_collision_status!(semi, system)
end

@inline function collision_interact!(dv, v_system, u_system, v_neighbor, u_neighbor,
                                     nhs, system, neighbor)
    return dv
end

# Systems representing solids colliding with a boundary under the assumption the boundary is not influenced by the collision
@inline function collision_interact!(dv, v_particle_system, u_particle_system,
                                     v_neighbor_system, u_neighbor_system,
                                     neighborhood_search,
                                     particle_system::Union{RigidSPHSystem,
                                                            TotalLagrangianSPHSystem},
                                     neighbor_system::BoundarySPHSystem)
    (; particle_spacing) = neighbor_system
    neighbor_radius = 0.5 * particle_spacing

    (; particle_spacing) = particle_system
    particle_radius = 0.5 * particle_spacing

    collision_distance = neighbor_radius + particle_radius

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    total_dv = zeros(ndims(particle_system))

    max_overlap = 0.0
    max_normal = zeros(ndims(particle_system))

    # Detect the maximum overlap and calculate the normal vector at maximum overlap
    for_particle_neighbor(particle_system, neighbor_system, system_coords, neighbor_system_coords, neighborhood_search) do particle, neighbor, pos_diff, distance
        overlap = collision_distance - distance
        if overlap > max_overlap
            max_overlap = overlap
            max_normal = pos_diff / distance  # Normal vector pointing from particle to neighbor
        end
    end

    # Only proceed if there is an overlap
    if max_overlap > 0
        # current_dv_normal = dot(dv[:, particle], max_normal) * max_normal

        # Calculate the total initial velocity in the direction of the normal
        particle_velocity = extract_svector(v_particle_system, particle_system, 1)
        initial_normal_velocity = dot(particle_velocity, max_normal)

        # Coefficient of restitution (0 < e <= 1)
        # e = 1   -> perfectly elastic collision
        # 0 < e < 1 -> inelastic collision

        e = 1.0
        if norm(particle_velocity) < sqrt(eps())
            e= 0.0
        end

        # Calculate the required change in velocity along the normal (reversing it to simulate a bounce)
        change_in_velocity_normal = -(1 + e) * initial_normal_velocity * max_normal

        #collision_time = max_overlap/norm(particle_velocity)
        #println("collison_time", collision_time)

        # Apply the change uniformly across all particles
        @inbounds for particle in each_moving_particle(particle_system)
            # current_dv_normal = dot(dv[:, particle], max_normal) * max_normal
            # dv[:, particle] = change_in_velocity_normal/collision_time
            dv[:, particle] = change_in_velocity_normal
        end
    end



    # doesn't work
    # for particle in each_moving_particle(particle_system)
    #     @inbounds for i in 1:ndims(particle_system)
    #         u_particle_system[i, particle] -= position_correction[i]
    #     end
    # end



    # normal = pos_diff / distance

    # part_v = extract_svector(v_particle_system, particle_system, particle)
    # nghbr_v = extract_svector(v_neighbor_system, neighbor_system, neighbor)
    # rel_vel = part_v - nghbr_v

    # # nothing to do we are in contact
    # norm(rel_vel) <= sqrt(eps()) && return

    # println("collide")

    # rel_vel_normal = dot(rel_vel, normal)
    # collision_damping_coefficient = 0.1
    # force_magnitude = collision_damping_coefficient * rel_vel_normal
    # force = force_magnitude * normal

    # @inbounds for i in 1:ndims(particle_system)
    #     total_dv[i] += force[i] / mass[particle]
    # end

    # for particle in each_moving_particle(particle_system)
    #     for i in 1:ndims(particle_system)
    #         dv[i, particle] += total_dv[i]
    #     end
    # end

    return dv
end

# collision influences both systems
@inline function collision_interact!(dv, v_particle_system, u_particle_system,
    v_neighbor_system, u_neighbor_system,
    neighborhood_search,
    particle_system::Union{RigidSPHSystem,
                           TotalLagrangianSPHSystem},
    neighbor_system::Union{RigidSPHSystem,
    TotalLagrangianSPHSystem})

    return dv
end

# NHS updates
# All systems that always move update every time
function nhs_coords(system::FluidSystem,
                    neighbor::Union{FluidSystem, TotalLagrangianSPHSystem, RigidSPHSystem},
                    u)
    return current_coordinates(u, neighbor)
end

function nhs_coords(system::TotalLagrangianSPHSystem,
                    neighbor::Union{FluidSystem, RigidSPHSystem}, u)
    return current_coordinates(u, neighbor)
end

function nhs_coords(system::RigidSPHSystem,
                    neighbor::Union{FluidSystem, TotalLagrangianSPHSystem}, u)
    return current_coordinates(u, neighbor)
end

# Only update when moving
function nhs_coords(system::Union{FluidSystem, TotalLagrangianSPHSystem, RigidSPHSystem},
                    neighbor::BoundarySPHSystem, u)
    if neighbor.ismoving[1]
        return current_coordinates(u, neighbor)
    end

    # Don't update
    return nothing
end

function nhs_coords(system::BoundarySPHSystem,
                    neighbor::Union{FluidSystem, TotalLagrangianSPHSystem,
                                    BoundarySPHSystem, RigidSPHSystem}, u)
    # Don't update
    return nothing
end

function nhs_coords(system::BoundarySPHSystem{<:BoundaryModelDummyParticles},
                    neighbor::FluidSystem, u)
    return current_coordinates(u, neighbor)
end

function nhs_coords(system::TotalLagrangianSPHSystem, neighbor::TotalLagrangianSPHSystem, u)
    # Don't update
    return nothing
end

function nhs_coords(system::RigidSPHSystem, neighbor::RigidSPHSystem, u)
    # Don't update
    return nothing
end

function check_configuration(systems)
    foreach_system(systems) do system
        check_configuration(system, systems)
    end
end

check_configuration(system, systems) = nothing

function check_configuration(boundary_system::BoundarySPHSystem, systems)
    (; boundary_model) = boundary_system

    foreach_system(systems) do neighbor
        if neighbor isa WeaklyCompressibleSPHSystem &&
           boundary_model isa BoundaryModelDummyParticles &&
           isnothing(boundary_model.state_equation)
            throw(ArgumentError("`WeaklyCompressibleSPHSystem` cannot be used without " *
                                "setting a `state_equation` for all boundary systems"))
        end
    end
end

function check_configuration(system::Union{TotalLagrangianSPHSystem, RigidSPHSystem},
                             systems)
    (; boundary_model) = system

    foreach_system(systems) do neighbor
        if neighbor isa FluidSystem && boundary_model === nothing
            throw(ArgumentError("a boundary model for `TotalLagrangianSPHSystem` must be " *
                                "specified when simulating a fluid-structure interaction."))
        end
    end

    if boundary_model isa BoundaryModelDummyParticles &&
       boundary_model.density_calculator isa ContinuityDensity
        throw(ArgumentError("`BoundaryModelDummyParticles` with density calculator " *
                            "`ContinuityDensity` is not yet supported for a `TotalLagrangianSPHSystem`"))
    end
end

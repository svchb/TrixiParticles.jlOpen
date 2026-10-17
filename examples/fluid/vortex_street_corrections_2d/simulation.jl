module VortexStreetCorrections

using CSV
using DataFrames
using JSON
using LinearAlgebra
using OrdinaryDiffEqLowStorageRK
using Printf
using Statistics
using TrixiParticles

const TP = TrixiParticles
const CASES = (:uncorrected, :density, :gradient, :combined)
const LABELS = Dict(:uncorrected => "Uncorrected",
                    :density => "Shepard density",
                    :gradient => "Mixed gradient",
                    :combined => "Density + gradient")

# Short boundary loops and plain contiguous array operations are faster serially
# than launching 20 tasks. Particle-neighbor loops over the active fluid indices
# retain all Julia worker threads.
struct ComparisonBackend <: TP.PointNeighbors.AbstractThreadingBackend end

@inline function TP.PointNeighbors.parallel_foreach(f, iterator, ::ComparisonBackend)
    if iterator isa AbstractUnitRange || length(iterator) < 2048
        TP.PointNeighbors.parallel_foreach(f, iterator, SerialBackend())
    else
        TP.PointNeighbors.parallel_foreach(f, iterator, ThreadsDynamicBackend())
    end
end

function configuration(; spacing_factor=0.125, end_time=1.0, max_seconds=570.0,
                       output_directory="out/vortex_street_corrections", save_vtk=false,
                       thread_backend=:adaptive)
    0 < spacing_factor <= 0.2 || throw(ArgumentError("invalid spacing factor"))
    end_time > 0 || throw(ArgumentError("end time must be positive"))
    max_seconds > 0 || throw(ArgumentError("runtime budget must be positive"))
    thread_backend in (:adaptive, :dynamic) ||
        throw(ArgumentError("thread backend must be :adaptive or :dynamic"))
    diameter = 0.1
    spacing = spacing_factor * diameter
    # Round the domain to whole particle cells, keeping geometry identical in all cases.
    domain = Tuple(round.(Int, (12diameter, 6diameter) ./ spacing) .* spacing)
    return (; diameter, spacing, domain, center=(3diameter, domain[2] / 2),
            density=1000.0, velocity=1.0, reynolds=200.0, sound_speed=15.0,
            smoothing_length=2spacing, boundary_layers=4, open_layers=8,
            shift_strength=0.2, shift_limit=0.03,
            end_time, max_seconds, output_directory, save_vtk, thread_backend)
end

function corrections(case)
    case in CASES || throw(ArgumentError("unknown correction case: $case"))
    return (;
            density_correction=case in (:density, :combined) ?
                               ShepardKernelCorrection() : nothing,
            gradient_correction=case in (:gradient, :combined) ?
                                MixedKernelGradientCorrection() : nothing,
            force_correction=nothing)
end

function build_case(case, cfg; shifting_technique=nothing,
                    neighborhood_update=ParallelUpdate(), parallelization_backend=nothing,
                    reverse_static_neighbors=false,
                    boundary_model=BoundaryModelMirroringTafuni(;
                                                                mirror_method=ZerothOrderMirroring()),
                    average_pressure_reduction=false, viscosity_model=nothing,
                    correct_walls=true, directed_boundaries=false)
    (; diameter, spacing, domain, center, density, velocity, sound_speed,
     smoothing_length, boundary_layers, open_layers) = cfg
    direction = SVector(1.0, 0.0)
    # A deterministic, localized transverse disturbance starts the shedding mode.
    # Its physical amplitude and location are identical in every correction case.
    initial_velocity = x -> SVector(velocity,
                                    0.01velocity *
                                    exp(-((x[1] - center[1] - 1.5diameter) /
                                          (2diameter))^2 -
                                        ((x[2] - center[2]) / (1.5diameter))^2))
    pipe = RectangularTank(spacing, domain, domain, density;
                           n_layers=boundary_layers, velocity=initial_velocity,
                           faces=(false, false, true, true))
    zone_size = (open_layers * spacing, domain[2])
    inlet = RectangularTank(spacing, zone_size, zone_size, density;
                            n_layers=boundary_layers, velocity=velocity * direction,
                            min_coordinates=(-zone_size[1], 0.0),
                            faces=(false, false, true, true))
    outlet = RectangularTank(spacing, zone_size, zone_size, density;
                             n_layers=boundary_layers, velocity=velocity * direction,
                             min_coordinates=(domain[1], 0.0),
                             faces=(false, false, true, true))
    cylinder_ic = SphereShape(spacing, diameter / 2, center, density;
                              sphere_type=RoundSphere())
    fluid_ic = setdiff(pipe.fluid, cylinder_ic)
    kernel = WendlandC6Kernel{2}()
    viscosity = isnothing(viscosity_model) ?
                ViscosityAdami(nu=velocity * diameter / cfg.reynolds) : viscosity_model
    # Evolve pressure with the same EDAC scheme in every case. This avoids an
    # EOS startup pressure transient from the different summation-density estimates.
    background_pressure = 0.0
    pressure_formulation = TP.inter_particle_averaged_pressure
    fluid = EntropicallyDampedSPHSystem(fluid_ic;
                                        smoothing_kernel=kernel, smoothing_length,
                                        density_calculator=SummationDensity(),
                                        sound_speed, viscosity,
                                        pressure_acceleration=pressure_formulation,
                                        shifting_technique,
                                        average_pressure_reduction,
                                        buffer_size=nparticles(inlet.fluid),
                                        corrections(case)...)
    inflow = BoundaryZone(;
                          boundary_face=([0.0, 0.0], [0.0, domain[2]]),
                          face_normal=direction, open_boundary_layers=open_layers,
                          density, particle_spacing=spacing,
                          reference_velocity=velocity * direction,
                          reference_density=density,
                          boundary_type=directed_boundaries ? InFlow() :
                                        BidirectionalFlow(),
                          initial_condition=inlet.fluid)
    outflow = BoundaryZone(;
                           boundary_face=([domain[1], 0.0], [domain[1], domain[2]]),
                           face_normal=(-direction), open_boundary_layers=open_layers,
                           density, particle_spacing=spacing, reference_pressure=0.0,
                           boundary_type=directed_boundaries ? OutFlow() :
                                         BidirectionalFlow(),
                           initial_condition=outlet.fluid)
    open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system=fluid,
                                       boundary_model,
                                       pressure_acceleration=pressure_formulation,
                                       buffer_size=nparticles(inlet.fluid))
    wall_ic = union(pipe.boundary, inlet.boundary, outlet.boundary)
    wall_roles = correct_walls ? corrections(case) :
                 (; density_correction=nothing, gradient_correction=nothing,
                  force_correction=nothing)
    wall = WallBoundarySystem(wall_ic,
                              BoundaryModelDummyParticles(wall_ic; fluid_system=fluid,
                                                          wall_roles...))
    cylinder = WallBoundarySystem(cylinder_ic,
                                  BoundaryModelDummyParticles(cylinder_ic;
                                                              fluid_system=fluid,
                                                              viscosity, wall_roles...))
    min_corner = vec(minimum(wall_ic.coordinates; dims=2)) .- spacing
    max_corner = vec(maximum(wall_ic.coordinates; dims=2)) .+ spacing
    nhs = GridNeighborhoodSearch{2}(;
                                    cell_list=FullGridCellList(; min_corner, max_corner),
                                    update_strategy=neighborhood_update)
    default_backend = cfg.thread_backend == :adaptive ? ComparisonBackend() :
                      ThreadsDynamicBackend()
    backend = isnothing(parallelization_backend) ? default_backend :
              parallelization_backend
    semi = Semidiscretization(fluid, open_boundary, wall, cylinder;
                              neighborhood_search=nhs,
                              parallelization_backend=backend)
    ode = semidiscretize(semi, (0.0, cfg.end_time))
    if reverse_static_neighbors
        # An ordering-only diagnostic: geometry and neighbor membership are unchanged.
        for neighbor in ode.p.semi.systems[3:4]
            search = TP.get_neighborhood_search(ode.p.semi.systems[1], neighbor,
                                                ode.p.semi)
            for index in TP.PointNeighbors.each_cell_index(search.cell_list)
                reverse!(search.cell_list[index])
            end
        end
    end
    # semidiscretize can replace systems when allocating buffers/linking boundaries.
    return (; ode, semi=ode.p.semi, fluid=ode.p.semi.systems[1],
            cylinder=ode.p.semi.systems[4], background_pressure)
end

mutable struct Diagnostics{C}
    cfg        :: C
    rows       :: Vector{Vector{Float64}}
    forces     :: Matrix{Float64}
    started_ns :: UInt64
end

struct GeometricShifting{C}
    cfg      :: C
    velocity :: Matrix{Float64}
end

function (shifting::GeometricShifting)(integrator)
    cfg = shifting.cfg
    semi = integrator.p.semi
    fluid = semi.systems[1]
    v_ode, u_ode = integrator.u.x
    u = TP.wrap_u(u_ode, fluid, semi)
    coordinates = TP.current_coordinates(u, fluid)
    h = cfg.smoothing_length
    kernel = fluid.smoothing_kernel
    Wdx = TP.kernel(kernel, cfg.spacing, h)
    strength = cfg.shift_strength * cfg.sound_speed * (2h)^2 / (2cfg.spacing)
    fill!(shifting.velocity, 0.0)
    TP.foreach_system_wrapped(semi, v_ode, u_ode) do neighbor, v_neighbor, u_neighbor
        TP.has_system_interaction(fluid, neighbor, semi) || return
        x_neighbor = TP.current_coordinates(u_neighbor, neighbor)
        TP.foreach_point_neighbor(fluid, neighbor, coordinates, x_neighbor,
                                  semi) do i, j, diff, distance
            distance > eps(h) || return
            W = TP.kernel(kernel, distance, h)
            grad = TP.kernel_grad_unsafe(kernel, diff, distance, h)
            # Reference volumes and the raw kernel make this geometric regularizer
            # independent of the chosen density and gradient corrections.
            weight = TP.hydrodynamic_mass(neighbor, j) / (2cfg.density)
            delta = -strength * (1 + 0.2 * (W / Wdx)^4) * weight * grad
            for dim in 1:2
                shifting.velocity[dim, i] += delta[dim]
            end
        end
    end
    dt = integrator.t - integrator.tprev
    limit = cfg.shift_limit * cfg.spacing
    TP.PointNeighbors.parallel_foreach(TP.each_active_particle(fluid), semi) do i
        dx = dt * shifting.velocity[1, i]
        dy = dt * shifting.velocity[2, i]
        scale = min(1.0, limit / max(hypot(dx, dy), eps(limit)))
        u[1, i] += scale * dx
        u[2, i] += scale * dy
    end
    TP.SciMLBase.derivative_discontinuity!(integrator, true)
    return nothing
end

const COLUMNS = [:time, :advective_time, :density_rms_percent, :density_min,
    :density_max, :active_particles, :fluid_mass, :drag_coefficient,
    :lift_coefficient, :probe_u, :probe_v, :inlet_flux, :outlet_flux]

# Compute the reaction to physical pressure/viscous forces, excluding artificial
# shifting terms. The same pair-force routine is used by the simulation and FSI.
function cylinder_force!(forces, fluid, cylinder, v_ode, u_ode, semi)
    fill!(forces, 0.0)
    v = TP.wrap_v(v_ode, fluid, semi)
    u = TP.wrap_u(u_ode, fluid, semi)
    v_cylinder = TP.wrap_v(v_ode, cylinder, semi)
    u_cylinder = TP.wrap_u(u_ode, cylinder, semi)
    x = TP.current_coordinates(u, fluid)
    x_cylinder = TP.current_coordinates(u_cylinder, cylinder)
    TP.foreach_point_neighbor(fluid, cylinder, x, x_cylinder, semi) do i, j, diff, distance
        W = TP.smoothing_kernel_grad(fluid, diff, distance, i)
        rho_a = TP.current_density(v, fluid, i)
        rho_b = TP.current_density(v_cylinder, cylinder, j)
        p_a = TP.current_pressure(v, fluid, i)
        p_b = TP.neighbor_pressure(v_cylinder, cylinder, j, p_a)
        m_a = TP.hydrodynamic_mass(fluid, i)
        m_b = TP.hydrodynamic_mass(cylinder, j)
        velocity_a = TP.current_velocity(v, fluid, i)
        velocity_b = TP.current_velocity(v_cylinder, cylinder, j)
        acceleration = TP.add_momentum_equation(zero(W), fluid, cylinder, v,
                                                v_cylinder, i, j, diff, distance, W,
                                                TP.system_sound_speed(fluid), m_a, m_b,
                                                p_a, p_b, rho_a, rho_b,
                                                velocity_a, velocity_b)
        for dim in 1:2
            forces[dim, i] -= m_a * acceleration[dim]
        end
    end
    return vec(sum(forces; dims=2))
end

function (diagnostics::Diagnostics)(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    system === semi.systems[1] || return nothing
    cfg = diagnostics.cfg
    v = TP.wrap_v(v_ode, system, semi)
    active = collect(TP.each_active_particle(system))
    density = TP.current_density(v, system)
    all(i -> isfinite(density[i]) && density[i] > 0, active) ||
        error("nonfinite or nonpositive density at t=$t")
    relative_density = [density[i] / cfg.density - 1 for i in active]
    force = cylinder_force!(diagnostics.forces, system, semi.systems[4],
                            v_ode, u_ode, semi)
    force_scale = 0.5cfg.density * cfg.velocity^2 * cfg.diameter
    probe = [cfg.center[1] + 3cfg.diameter; cfg.center[2] + 0.5cfg.diameter;;]
    probe_data = interpolate_points(probe, semi, system, v_ode, u_ode;
                                    cut_off_bnd=false)
    # Midpoint quadrature on two interior cross-sections, avoiding open-boundary
    # kernel truncation. These are sampled fluxes, not exact particle-transfer counts.
    ny = round(Int, cfg.domain[2] / cfg.spacing)
    points = zeros(2, 2ny)
    for j in 1:ny
        points[:, j] .= (2cfg.smoothing_length, (j - 0.5) * cfg.spacing)
        points[:, ny + j] .= (cfg.domain[1] - 2cfg.smoothing_length,
                              (j - 0.5) * cfg.spacing)
    end
    flux_data = interpolate_points(points, semi, system, v_ode, u_ode;
                                   cut_off_bnd=false)
    flux = flux_data.density .* vec(flux_data.velocity[1, :])
    inlet_flux = sum(view(flux, 1:ny)) * cfg.spacing
    outlet_flux = sum(view(flux, (ny + 1):(2ny))) * cfg.spacing
    row = [t, t * cfg.velocity / cfg.diameter,
        100sqrt(mean(abs2, relative_density)),
        minimum(density[i] for i in active), maximum(density[i] for i in active),
        length(active), sum(system.mass[i] for i in active),
        force[1] / force_scale, force[2] / force_scale,
        probe_data.velocity[1, 1] / cfg.velocity,
        probe_data.velocity[2, 1] / cfg.velocity,
        inlet_flux, outlet_flux]
    all(isfinite, row) || error("nonfinite diagnostics at t=$t")
    push!(diagnostics.rows, row)
    if length(diagnostics.rows) == 1 || length(diagnostics.rows) % 25 == 0
        @printf("  tU/D=%5.1f, density RMS=%6.3f%%, Cd=%7.3f, elapsed=%6.1fs\n",
                row[2], row[3], row[8], (time_ns() - diagnostics.started_ns) / 1.0e9)
        flush(stdout)
    end
    return nothing
end

function sample_field(setup, sol, cfg)
    nx = round(Int, cfg.domain[1] / cfg.spacing) + 1
    ny = round(Int, cfg.domain[2] / cfg.spacing) + 1
    xs = range(0.0, cfg.domain[1]; length=nx)
    ys = range(0.0, cfg.domain[2]; length=ny)
    points = zeros(2, nx * ny)
    for j in 1:ny, i in 1:nx
        points[:, (j - 1) * nx + i] .= (xs[i], ys[j])
    end
    v_ode, u_ode = sol.u[end].x
    TP.update_systems_and_nhs(v_ode, u_ode, setup.semi, sol.t[end])
    data = interpolate_points(points, setup.semi, setup.fluid, v_ode, u_ode;
                              cut_off_bnd=true)
    u = reshape(data.velocity[1, :], nx, ny)
    v = reshape(data.velocity[2, :], nx, ny)
    omega = fill(NaN, nx, ny)
    dx, dy = step(xs), step(ys)
    # The diagnostic operator is identical for all cases, independently of the
    # correction chosen for the SPH solve.
    for j in 2:(ny - 1), i in 2:(nx - 1)
        omega[i, j] = ((v[i + 1, j] - v[i - 1, j]) / (2dx) -
                       (u[i, j + 1] - u[i, j - 1]) / (2dy)) *
                      cfg.diameter / cfg.velocity
    end
    for j in 1:ny, i in 1:nx
        if hypot(xs[i] - cfg.center[1], ys[j] - cfg.center[2]) <=
           cfg.diameter / 2 + cfg.spacing
            omega[i, j] = NaN
        end
    end
    return DataFrame(x=vec(points[1, :]), y=vec(points[2, :]),
                     u=vec(u) / cfg.velocity, v=vec(v) / cfg.velocity,
                     density=data.density / cfg.density,
                     pressure=(data.pressure .- setup.background_pressure) /
                              (0.5cfg.density * cfg.velocity^2),
                     vorticity=vec(omega)), (nx, ny)
end

function run_case(case, cfg; started_ns=time_ns(), neighborhood_update=ParallelUpdate(),
                  parallelization_backend=nothing, reverse_static_neighbors=false,
                  fixed_dt=nothing, record_steps=false)
    case in CASES || throw(ArgumentError("unknown correction case: $case"))
    LinearAlgebra.BLAS.set_num_threads(1)
    println("Running ", LABELS[case], " on ", Threads.nthreads(), " Julia threads")
    setup_seconds = @elapsed setup = build_case(case, cfg; neighborhood_update,
                                                parallelization_backend,
                                                reverse_static_neighbors)
    diagnostics = Diagnostics(cfg, Vector{Float64}[],
                              zeros(2, nparticles(setup.fluid)), started_ns)
    pp = PostprocessCallback(; dt=0.01, write_csv=false, write_json=false,
                             diagnostics)
    budget_condition = (u, t, integrator) -> integrator.stats.naccept % 100 == 0
    budget_affect! = integrator -> begin
        elapsed = (time_ns() - started_ns) / 1.0e9
        elapsed < cfg.max_seconds ||
            error("case exceeded the $(cfg.max_seconds)s wall-clock budget at t=$(integrator.t)")
    end
    shifting = GeometricShifting(cfg, zeros(2, nparticles(setup.fluid)))
    shift_callback = DiscreteCallback((u, t, integrator) -> true, shifting;
                                      save_positions=(false, false))
    step_rows = Vector{Float64}[]
    previous_membership = copy(setup.fluid.buffer.active_particle)
    function record_step!(integrator)
        fluid = integrator.p.semi.systems[1]
        active = collect(TP.each_active_particle(fluid))
        activated = count(i -> fluid.buffer.active_particle[i] && !previous_membership[i],
                          eachindex(previous_membership))
        previous_membership .= fluid.buffer.active_particle
        dt_used = integrator.t - integrator.tprev
        capped = count(i -> dt_used * hypot(shifting.velocity[1, i],
                                  shifting.velocity[2, i]) >
                            cfg.shift_limit * cfg.spacing, active)
        push!(step_rows,
              [integrator.t, dt_used, integrator.dt,
                  integrator.stats.naccept, integrator.stats.nreject,
                  length(active), activated, capped / length(active)])
        # This observer changes neither state nor caches. SciML preserves earlier
        # callbacks' modification flags when folding the callback set.
        TP.SciMLBase.derivative_discontinuity!(integrator, false)
        return nothing
    end
    budget_callback = DiscreteCallback(budget_condition, budget_affect!;
                                       save_positions=(false, false))
    if record_steps
        step_callback = DiscreteCallback((u, t, integrator) -> true, record_step!;
                                         save_positions=(false, false))
        callbacks = CallbackSet(UpdateCallback(), shift_callback, step_callback, pp,
                                budget_callback)
    else
        callbacks = CallbackSet(UpdateCallback(), shift_callback, pp, budget_callback)
    end
    if cfg.save_vtk
        vtk_directory = joinpath(cfg.output_directory, string(case), "vtk")
        saving_callback = SolutionSavingCallback(; dt=0.1,
                                                 output_directory=vtk_directory)
        callbacks = CallbackSet(callbacks, saving_callback)
    end
    dtmax = 0.4cfg.smoothing_length / (cfg.sound_speed + cfg.velocity)
    fixed_dt === nothing || 0 < fixed_dt <= dtmax ||
        throw(ArgumentError("fixed timestep must be positive and no larger than dtmax"))
    dt = fixed_dt === nothing ? 0.2cfg.smoothing_length /
                                (cfg.sound_speed + cfg.velocity) : fixed_dt
    solve_seconds = @elapsed sol = solve(setup.ode, RDPK3SpFSAL35();
                                         abstol=1.0e-5, reltol=1.0e-4,
                                         dt, dtmax, adaptive=fixed_dt === nothing,
                                         save_everystep=false, callback=callbacks,
                                         maxiters=1_000_000)
    TP.SciMLBase.successful_retcode(sol) ||
        error("simulation failed: $(sol.retcode), final time $(sol.t[end])")
    isapprox(sol.t[end], cfg.end_time) || error("simulation did not reach its end time")
    output = joinpath(cfg.output_directory, string(case))
    mkpath(output)
    trace = DataFrame(permutedims(reduce(hcat, diagnostics.rows)), COLUMNS)
    unique!(trace, :time)
    CSV.write(joinpath(output, "diagnostics.csv"), trace)
    if record_steps
        step_columns = [:time, :accepted_dt, :proposed_dt,
            :accepted_steps, :rejected_steps, :active_particles,
            :activated_particles, :capped_shift_fraction]
        CSV.write(joinpath(output, "steps.csv"),
                  DataFrame(permutedims(reduce(hcat, step_rows)), step_columns))
    end
    export_seconds = @elapsed begin
        field, grid_size = sample_field(setup, sol, cfg)
        CSV.write(joinpath(output, "field.csv"), field)
    end
    total_seconds = (time_ns() - started_ns) / 1.0e9
    total_seconds <= cfg.max_seconds ||
        error("case took $(total_seconds)s, exceeding the $(cfg.max_seconds)s wall budget")
    metadata = Dict("case" => string(case), "label" => LABELS[case],
                    "threads" => Threads.nthreads(), "julia_version" => string(VERSION),
                    "git_commit" => TP.compute_git_hash(),
                    "configuration" => Dict(string(k) => v for (k, v) in pairs(cfg)),
                    "grid_size" => collect(grid_size),
                    "neighborhood_update" => string(typeof(neighborhood_update)),
                    "parallelization_backend" =>
                        string(typeof(setup.semi.parallelization_backend)),
                    "reverse_static_neighbors" => reverse_static_neighbors,
                    "adaptive_timestep" => fixed_dt === nothing,
                    "fixed_dt" => fixed_dt,
                    "background_pressure" => setup.background_pressure,
                    "setup_seconds" => setup_seconds, "solve_seconds" => solve_seconds,
                    "export_seconds" => export_seconds,
                    "total_seconds" => total_seconds,
                    "accepted_steps" => sol.stats.naccept,
                    "rejected_steps" => sol.stats.nreject,
                    "rhs_evaluations" => sol.stats.nf,
                    "final_time" => sol.t[end], "retcode" => string(sol.retcode))
    open(joinpath(output, "metadata.json"), "w") do io
        JSON.json(io, metadata; pretty=2)
    end
    @printf("Finished %s: setup %.1fs, solve %.1fs, export %.1fs, total %.1fs\n",
            string(case), setup_seconds, solve_seconds, export_seconds, total_seconds)
    flush(stdout)
    return metadata
end

end # module

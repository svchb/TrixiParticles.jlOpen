@inline function requires_update_callback(system::RigidBodySystem)
    return !isnothing(system.contact_model) &&
           requires_update_callback(system.contact_model)
end

update_rigid_contact_eachstep!(system, v_ode, u_ode, semi, t, integrator) = system

function update_rigid_contact_eachstep!(system::RigidBodySystem{<:Any, <:Any, NDIMS},
                                        v_ode, u_ode, semi, t,
                                        integrator) where {NDIMS}
    requires_update_callback(system) || return system

    v_system = wrap_v(v_ode, system, semi)
    u_system = wrap_u(u_ode, system, semi)
    active_contact_keys = Set{RigidContactKey}()

    foreach_system(semi) do neighbor_system
        neighbor_system === system && return
        update_contact_history_pair!(system, neighbor_system, v_system, u_system,
                                     v_ode, u_ode, semi, integrator.dt,
                                     active_contact_keys)
    end

    remove_inactive_contact_pairs!(system.cache.contact_tangential_displacement,
                                   active_contact_keys)

    return system
end

update_contact_history_pair!(system, neighbor_system, v_system, u_system, v_ode, u_ode,
                             semi, dt, active_contact_keys) = active_contact_keys

function update_contact_history_pair!(system::RigidBodySystem{<:Any, <:Any, NDIMS},
                                      neighbor_system::WallBoundarySystem,
                                      v_system, u_system,
                                      v_ode, u_ode,
                                      semi, dt,
                                      active_contact_keys) where {NDIMS}
    contact_model = system.contact_model
    isnothing(contact_model) && return active_contact_keys

    reset_contact_manifold_cache!(system.cache)

    v_neighbor = wrap_v(v_ode, neighbor_system, semi)
    u_neighbor = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u_system, system)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(system),
                           parallelization_backend=SerialBackend()) do particle, neighbor,
                                                                       pos_diff, distance
        accumulate_wall_contact_pair!(system, v_neighbor, neighbor_system,
                                      particle, neighbor, pos_diff, distance, contact_model)
    end

    neighbor_system_index = system_indices(neighbor_system, semi)
    ELTYPE = eltype(system)
    zero_tangential = zero(SVector{NDIMS, ELTYPE})

    for particle in each_integrated_particle(system)
        n_manifolds = system.cache.contact_manifold_count[particle]
        n_manifolds == 0 && continue

        particle_velocity = current_velocity(v_system, system, particle)

        for manifold_index in 1:n_manifolds
            weight_sum = system.cache.contact_manifold_weight_sum[manifold_index, particle]
            weight_sum <= eps(ELTYPE) && continue

            normal = extract_svector(system.cache.contact_manifold_normal_sum, Val(NDIMS),
                                     manifold_index, particle) / weight_sum
            normal_norm = norm(normal)
            normal_norm <= eps(ELTYPE) && continue
            normal /= normal_norm

            wall_velocity = extract_svector(system.cache.contact_manifold_wall_velocity_sum,
                                            Val(NDIMS), manifold_index, particle) /
                            weight_sum
            penetration_effective = system.cache.contact_manifold_penetration_sum[manifold_index,
                                                                                   particle] /
                                    weight_sum
            relative_velocity = particle_velocity - wall_velocity
            normal_velocity = dot(relative_velocity, normal)
            tangential_velocity = relative_velocity - normal_velocity * normal

            contact_key = wall_contact_key(neighbor_system_index, particle, manifold_index)
            push!(active_contact_keys, contact_key)
            update_contact_tangential_history!(system, contact_key, tangential_velocity,
                                               normal, penetration_effective,
                                               normal_velocity, dt, contact_model,
                                               zero_tangential)
        end
    end

    return active_contact_keys
end

function update_contact_history_pair!(system::RigidBodySystem{<:Any, <:Any, NDIMS},
                                      neighbor_system::RigidBodySystem,
                                      v_system, u_system,
                                      v_ode, u_ode,
                                      semi, dt,
                                      active_contact_keys) where {NDIMS}
    contact_model = system.contact_model
    neighbor_contact_model = neighbor_system.contact_model
    if isnothing(contact_model) || isnothing(neighbor_contact_model)
        return active_contact_keys
    end

    system_coords = current_coordinates(u_system, system)
    v_neighbor = wrap_v(v_ode, neighbor_system, semi)
    u_neighbor = wrap_u(u_ode, neighbor_system, semi)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

    neighbor_system_index = system_indices(neighbor_system, semi)
    ELTYPE = eltype(system)
    zero_tangential = zero(SVector{NDIMS, ELTYPE})

    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(system),
                           parallelization_backend=SerialBackend()) do particle, neighbor,
                                                                       pos_diff, distance
        distance <= eps(ELTYPE) && return

        penetration = max(contact_model.contact_distance,
                          neighbor_contact_model.contact_distance) - distance
        penetration_effective = penetration - contact_model.penetration_slop
        penetration_effective <= 0 && return

        normal = pos_diff / distance
        particle_velocity = current_velocity(v_system, system, particle)
        neighbor_velocity = current_velocity(v_neighbor, neighbor_system, neighbor)
        relative_velocity = particle_velocity - neighbor_velocity
        normal_velocity = dot(relative_velocity, normal)
        tangential_velocity = relative_velocity - normal_velocity * normal

        contact_key = rigid_rigid_contact_key(neighbor_system_index, particle, neighbor)
        push!(active_contact_keys, contact_key)
        update_contact_tangential_history!(system, contact_key, tangential_velocity, normal,
                                           penetration_effective, normal_velocity, dt,
                                           contact_model, zero_tangential)
    end

    return active_contact_keys
end

function update_contact_tangential_history!(system::RigidBodySystem, contact_key,
                                            tangential_velocity, normal,
                                            penetration_effective, normal_velocity, dt,
                                            contact_model::RigidContactModel,
                                            zero_tangential)
    dt_ = isfinite(dt) && dt > 0 ? convert(eltype(system), dt) : zero(eltype(system))
    contact_map = system.cache.contact_tangential_displacement
    isnothing(contact_map) && return contact_map

    tangential_displacement = get(contact_map, contact_key, zero_tangential)
    tangential_displacement += dt_ * tangential_velocity
    tangential_displacement -= dot(tangential_displacement, normal) * normal

    if contact_model.tangential_stiffness > eps(eltype(system))
        normal_force = normal_friction_reference_force(contact_model, penetration_effective,
                                                       normal_velocity, eltype(system))
        max_displacement = contact_model.static_friction_coefficient * normal_force /
                           contact_model.tangential_stiffness
        displacement_norm = norm(tangential_displacement)

        if displacement_norm > max_displacement &&
           displacement_norm > eps(eltype(system))
            tangential_displacement *= max_displacement / displacement_norm
        end
    else
        tangential_displacement = zero_tangential
    end

    contact_map[contact_key] = tangential_displacement

    return contact_map
end

function update_contact_tangential_history!(system::RigidBodySystem, contact_key,
                                            tangential_velocity, normal,
                                            penetration_effective, normal_velocity, dt,
                                            contact_model::RigidContactModel)
    zero_tangential = zero(SVector{ndims(system), eltype(system)})

    return update_contact_tangential_history!(system, contact_key,
                                              tangential_velocity, normal,
                                              penetration_effective, normal_velocity, dt,
                                              contact_model, zero_tangential)
end

function remove_inactive_contact_pairs!(contact_tangential_displacement, active_contact_keys)
    isnothing(contact_tangential_displacement) && return contact_tangential_displacement

    for key in collect(keys(contact_tangential_displacement))
        key in active_contact_keys && continue
        delete!(contact_tangential_displacement, key)
    end

    return contact_tangential_displacement
end

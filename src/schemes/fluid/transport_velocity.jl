"""
    TransportVelocityAdami(background_pressure::Real)

Transport Velocity Formulation (TVF) by [Adami et al. (2013)](@cite Adami2013)
to suppress pairing and tensile instability.
See [TVF](@ref transport_velocity_formulation) for more details of the method.

# Arguments
- `background_pressure`: Background pressure. Suggested is a background pressure which is
                         on the order of the reference pressure.
"""
struct TransportVelocityAdami{T <: Real}
    background_pressure::T
end

# No TVF for a system by default
@inline transport_velocity(system) = nothing

# `δv` is the correction to the particle velocity due to the TVF.
# Particles are advected with the velocity `v + δv`.
@propagate_inbounds function delta_v(system, particle)
    return delta_v(system, transport_velocity(system), particle)
end

@propagate_inbounds function delta_v(system, ::TransportVelocityAdami, particle)
    return extract_svector(system.cache.delta_v, system, particle)
end

# Zero when no TVF is used
@inline function delta_v(system, transport_velocity, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@inline function dv_transport_velocity(::Nothing, system, neighbor_system,
                                       particle, neighbor, v_system, v_neighbor_system,
                                       m_a, m_b, rho_a, rho_b, pos_diff, distance,
                                       grad_kernel, correction)
    return zero(grad_kernel)
end

@inline function dv_transport_velocity(::TransportVelocityAdami, system, neighbor_system,
                                       particle, neighbor, v_system, v_neighbor_system,
                                       m_a, m_b, rho_a, rho_b, pos_diff, distance,
                                       grad_kernel, correction)
    v_a = current_velocity(v_system, system, particle)
    delta_v_a = delta_v(system, particle)

    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    delta_v_b = delta_v(neighbor_system, neighbor)

    A_a = rho_a * v_a * delta_v_a'
    A_b = rho_b * v_b * delta_v_b'

    # The following term depends on the pressure acceleration formulation.
    # See the large comment below. In the original paper (Adami et al., 2013), this is
    #   (V_a^2 + V_b^2) / m_a * ((A_a + A_b) / 2) * ∇W_ab.
    # With the most common pressure acceleration formulation, this is
    #   m_b * (A_a + A_b) / (ρ_a * ρ_b) * ∇W_ab.
    # In order to obtain this, we pass `p_a = A_a` and `p_b = A_b` to the
    # `pressure_acceleration` function.
    return pressure_acceleration(system, neighbor_system, particle, neighbor,
                                 m_a, m_b, A_a, A_b, rho_a, rho_b, pos_diff,
                                 distance, grad_kernel, correction)
end

function update_tvf!(system, transport_velocity, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_tvf!(system, transport_velocity::TransportVelocityAdami, v, u, v_ode,
                     u_ode, semi, t)
    (; cache, correction) = system
    (; delta_v, pn) = cache
    (; background_pressure) = transport_velocity

    sound_speed = system_sound_speed(system)

    set_zero!(delta_v)

    # neighbor_system = system

    foreach_system(semi) do neighbor_system

        # if neighbor_system !== system && system isa FluidSystem
        #     return
        # end

        v_neighbor = wrap_v(v_ode, neighbor_system, semi)
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        # foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
        #                    semi;
        #                    points=each_moving_particle(system)) do particle, neighbor,
        #                                                            pos_diff, distance
        #     kernel_val = smoothing_kernel(system, distance, particle)

        #     # Summing kernel weighted by neighbor volume (m/ρ)
        #     @inbounds pn[particle] += kernel_val *
        #         (hydrodynamic_mass(neighbor_system, neighbor) / current_density(v_neighbor, neighbor_system, neighbor))
        # end

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi;
                               points=each_moving_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
            m_a = @inbounds hydrodynamic_mass(system, particle)
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

            rho_a = @inbounds current_density(v, system, particle)
            rho_b = @inbounds current_density(v_neighbor, neighbor_system, neighbor)

            h = smoothing_length(system, particle)

            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            # In the original paper (Adami et al., 2013), the transport velocity is applied
            # as follows:
            #   v_{1/2} = v_0 + Δt/2 * a,
            # where a is the regular SPH acceleration term (pressure, viscosity, etc.).
            #   r_1 = r_0 + Δt * (v_{1/2} + δv_{1/2}),
            # where v_{1/2} + δv_{1/2} is the transport velocity.
            # We will call δv_{1/2} the shifting velocity, which is given by
            #   δv = -Δt * p_0 / m_a * \sum_b (V_a^2 + V_b^2) * ∇W_ab,
            # where p_0 is the background pressure, V_a = m_a / ρ_a, V_b = m_b / ρ_b.
            # This term depends on the pressure acceleration formulation.
            # In Zhang et al. (2017), the pressure acceleration term
            #   m_b * (p_a / ρ_a^2 + p_b / ρ_b^2) * ∇W_ab
            # is used. They consequently changed the shifting velocity to
            #   δv = -Δt * p_0 * m_b * (1 / ρ_a^2 + 1 / ρ_b^2) * ∇W_ab.
            # We therefore use the function `pressure_acceleration` to compute the
            # shifting velocity according to the used pressure acceleration formulation.
            # In most cases, this will be
            #   δv = -Δt * p_0 * m_b * (1 + 1) / (ρ_a * ρ_b) * ∇W_ab.
            #
            # In these papers, the shifting velocity is scaled by the time step Δt.
            # We generally want the spatial discretization to be independent of the time step.
            # Scaling the shifting velocity by the time step would lead to less shifting
            # when very small time steps are used for testing/debugging purposes.
            # This is especially problematic in TrixiParticles.jl, as the time step can vary
            # significantly between different time integration methods (low vs high order).
            # In order to eliminate the time step from the shifting velocity, we apply the
            # CFL condition used in Adami et al. (2013):
            #   Δt <= 0.25 * h / c,
            # where h is the smoothing length and c is the sound speed.
            # Applying this equation as equality yields the shifting velocity
            #   δv = -p_0 / 8 * h / c * m_b * (1 + 1) / (ρ_a * ρ_b^2) * ∇W_ab.
            # The last part is achieved by passing `p_a = 1` and `p_b = 1` to the
            # `pressure_acceleration` function.
            # p0_a = background_pressure * max(0.0, 1.0 - pn[particle])
            # delta_v_ = p0_a / 8 * h / sound_speed *
            # pressure_acceleration(system, neighbor_system, particle, neighbor,
            #                         m_a, m_b, 1, 1, rho_a, rho_b, pos_diff,
            #                         distance, grad_kernel, correction)

            delta_v_ = background_pressure / 8 * h / sound_speed *
                       pressure_acceleration(system, neighbor_system, particle, neighbor,
                                             m_a, m_b, 1, 1, rho_a, rho_b, pos_diff,
                                             distance, grad_kernel, correction)

            # Write into the buffer
            for i in eachindex(delta_v_)
                @inbounds delta_v[i, particle] += delta_v_[i]
            end
        end
    end

    return system
end

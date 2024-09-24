"""
Calculates the infidelity for the given state 'ψ' and target state
'target.'

Returns: Infidelity
"""
function infidelity_real(
        ψ::AbstractVecOrMat{<: Real},
        target::AbstractVecOrMat{<: Real},
        N_ess::Integer
    )

    R = copy(target)
    N_tot = div(size(target,1), 2)
    # Could be more efficient by just taking dot of the real and imaginary parts directly 
    T = vcat(R[1+N_tot:end,:], -R[1:N_tot,:])
    return 1 - (dot(ψ,R)^2 + dot(ψ,T)^2)/(N_ess^2)
end

function infidelity(
        ψ::AbstractVecOrMat{<: Number},
        target::AbstractVecOrMat{<: Number},
        N_ess::Integer
    )

    ψ_real_valued = vcat(real(ψ), imag(ψ))
    target_real_valued = vcat(real(target), imag(target))
    return infidelity_real(ψ_real_valued, target_real_valued, N_ess)
end



function infidelity(
        prob::SchrodingerProb,
        controls,
        pcof::AbstractVector{<: Real},
        target::AbstractVecOrMat{<: Number};
        order::Integer=2,
        forcing=missing
    )

    history = eval_forward(prob, controls, pcof, order=order, forcing=forcing)
    final_state = history[:,end,:]
    N_ess = prob.N_ess_levels

    return infidelity(final_state, target, N_ess)
end

"""
dt is the timestep size.
T is the total time
W projects a state vector onto the guard subspace 
(should decide whether W should have dimensions of real or complex system
if complex, just take I₂×₂ * W.)
"""
function guard_penalty_real(
        history::AbstractArray{Float64, 3},
        dt::Real,
        T::Real,
        W::AbstractMatrix{<: Real}
    )

    penalty_value = 0.0

    N = size(history, 3)
    W_uv_placeholder = zeros(size(history, 1))
    for i in 1:N
        uv_vec = view(history, :, 1, i)
        mul!(W_uv_placeholder, W, uv_vec)
        if (i == 1 || i == N)
            penalty_value += 0.5*dot(uv_vec, W_uv_placeholder)
        else
            penalty_value += dot(uv_vec, W_uv_placeholder)
        end
    end
    penalty_value *= dt/T

    return penalty_value
end

function guard_penalty_real(
        history::AbstractArray{Float64, 4},
        dt::Real,
        T::Real,
        W::AbstractMatrix{<: Real}
    )

    penalty_value = 0.0

    for initial_condition_index in 1:size(history, 4)
        history_local = view(history, :, :, :, initial_condition_index)
        penalty_value += guard_penalty_real(history_local, dt, T, W)
    end

    return penalty_value
end

"""
Should update this. It's weird when W is the real-valued projector but the
history is complex-valued.
"""
function guard_penalty(
        history::AbstractArray{<: Number, 2},
        dt::Real,
        T::Real,
        W::AbstractMatrix{<: Real}
    )

    history_real_valued = vcat(real(history), imag(history))

    real_system_size = size(history_real_valued, 1)
    nsteps = size(history, 2)

    history_w_derivatives = reshape(
        real_system_size,
        N_tot_levels,
        1,
        nsteps
    )

    return guard_penalty_real(history_w_derivatives, dt, T, W)
end

function guard_penalty(
        history::AbstractArray{<: Number, 3},
        dt::Real,
        T::Real,
        W::AbstractMatrix{<: Real}
    )

    history_real_valued = vcat(real(history), imag(history))

    real_system_size = size(history_real_valued, 1)
    nsteps = size(history, 2)
    N_initial_conditions = size(history, 3)

    history_w_derivatives = reshape(
        history_real_valued,
        real_system_size,
        1,
        nsteps,
        N_initial_conditions
    )

    return guard_penalty_real(history_w_derivatives, dt, T, W)
end

function infidelity_plus_guard(
        prob::SchrodingerProb,
        controls,
        pcof::AbstractVector{<: Real},
        target::AbstractVecOrMat{<: Number};
        order::Integer=2
    )

    history = eval_forward(prob, controls, pcof; order=order)
    final_state = history[:,end,:]
    N_ess = prob.N_ess_levels

    T = prob.tf
    dt = T / prob.nsteps
    W = prob.guard_subspace_projector

    return infidelity(final_state, target, N_ess) + guard_penalty(history, dt, T, W)
end

function compute_terminal_condition(
        prob, controls, pcof, target, final_state;
        order=2, cost_type=:Infidelity, forcing=missing
    )

    terminal_condition = zeros(size(target))

    t = prob.tf
    dt = prob.tf/prob.nsteps

    N_derivatives = div(order, 2)

    uv_mat = zeros(prob.real_system_size, 1+N_derivatives)
    uv_vec = zeros(prob.real_system_size)

    R = target[:,:] # Copy target, converting to matrix if vector (will this code work for vectors?)
    T = vcat(R[1+prob.N_tot_levels:end,:], -R[1:prob.N_tot_levels,:])

    # Set up terminal condition RHS
    if cost_type == :Infidelity
        terminal_RHS = (dot(final_state, R)*R + dot(final_state, T)*T)
        terminal_RHS *= (2.0/(prob.N_ess_levels^2))
    elseif cost_type == :Tracking
        terminal_RHS = -(final_state - target)
    elseif cost_type == :Norm
        terminal_RHS = -final_state
    else
        throw("Invalid cost type: $cost_type")
    end

    # Add forcing
    if !ismissing(forcing)
        terminal_RHS .+= forcing
    end

    function LHS_func_wrapper(uv_out::AbstractVector{Float64}, uv_in::AbstractVector{Float64})
        uv_mat[:,1] .= uv_in
        compute_adjoint_derivatives!(uv_mat, prob, controls, t, pcof, N_derivatives)
        # Do I need to make and adjoint version of this? I don't think so, considering before LHS only used adjoint for utvt!, not the quadrature
        # But maybe there is a negative t I need to worry about. Maybe just provide dt as -dt
        build_LHS!(uv_out, uv_mat, dt, N_derivatives)

        return nothing
    end

    # Create linear map out of LHS_func_wrapper, to use in GMRES
    LHS_map = LinearMaps.LinearMap(
        LHS_func_wrapper,
        prob.real_system_size, prob.real_system_size,
        ismutating=true
    )

    for i in 1:size(target, 2)
        IterativeSolvers.gmres!(uv_vec, LHS_map, terminal_RHS[:,i], abstol=1e-15, reltol=1e-15)
        terminal_condition[:,i] .= uv_vec
    end

    return terminal_condition
end

"""
Arbitrary order version, should make one with target being abstract vector as
well, so I can do state transfer problems.
"""
function discrete_adjoint(
        prob::SchrodingerProb{<: AbstractMatrix{Float64}, <: AbstractMatrix{Float64}},
        controls, pcof::AbstractVector{Float64},
        target::AbstractMatrix{Float64}; 
        order=2, cost_type=:Infidelity, return_lambda_history=false
    )

    history = eval_forward(prob, controls, pcof; order=order)

    forcing = compute_guard_forcing(prob, history)

    R = target[:,:] # Copy target, converting to matrix if vector
    T = vcat(R[1+prob.N_tot_levels:end,:], -R[1:prob.N_tot_levels,:])

    final_state = history[:,1,end,:]

    terminal_condition = compute_terminal_condition(
        prob, controls, pcof, target, final_state, order=order, cost_type=cost_type,
        forcing=forcing[:,end,:]
    )

    lambda_history = eval_adjoint(
        prob, controls, pcof, terminal_condition, order=order, forcing=forcing)

    grad = zeros(length(pcof))

    for initial_condition_index = 1:size(prob.u0,2)
        this_history = view(history, :, :, :, initial_condition_index)
        this_lambda_history = view(lambda_history, :, :, :, initial_condition_index)

        accumulate_gradient!(
            grad, prob, controls, pcof, this_history, this_lambda_history, order=order
        )

    end

    if return_lambda_history
        return grad, history, lambda_history
    end

    return grad
end

"""
Change name to 'accumulate gradient' or something
"""
function accumulate_gradient!(gradient::AbstractVector{Float64},
        prob::SchrodingerProb, controls, pcof::AbstractVector{Float64},
        history::AbstractArray{Float64, 3}, lambda_history::AbstractArray{Float64, 3};
        order=2
    )

    dt = prob.tf / prob.nsteps
    N_derivatives = div(order, 2)

    uv_mat = zeros(prob.real_system_size, 1+N_derivatives)
    uv_partial_mat = zeros(prob.real_system_size, 1+N_derivatives)

    RHS = zeros(prob.real_system_size)
    LHS = zeros(prob.real_system_size)


    # If this is really all it takes, that's pretty easy.
    for n in 0:prob.nsteps-1
        # Used for both times
        lambda_np1 = view(lambda_history, :, 1, 1+n+1)

        # Handle RHS / "Explicit" Part

        t = n*dt
        uv_mat .= view(history, :, :, 1+n)

        for pcof_index in 1:length(pcof)
            compute_partial_derivative!(
                uv_partial_mat, uv_mat, prob, controls, t, pcof, N_derivatives, pcof_index
            )

            build_RHS!(RHS, uv_partial_mat, dt, N_derivatives)
            gradient[pcof_index] -= dot(RHS, lambda_np1)
        end

        # Handle LHS / "Explicit" Part
        t = (n+1)*dt
        uv_mat .= view(history, :, :, 1+n+1)

        for pcof_index in 1:length(pcof)
            compute_partial_derivative!(
                uv_partial_mat, uv_mat, prob, controls, t, pcof, N_derivatives, pcof_index
            )

            build_LHS!(LHS, uv_partial_mat, dt, N_derivatives)
            gradient[pcof_index] += dot(LHS, lambda_np1)
        end
    end
end

"""
History should be 4D array
"""
function compute_guard_forcing(prob, history)
    dt = prob.tf / prob.nsteps

    forcing = zeros(size(history, 1), 1+prob.nsteps, size(history, 4))
    for n in 1:prob.nsteps+1
        forcing[:, n, :] .= prob.guard_subspace_projector * history[:, 1, n, :] * -2*dt/prob.tf
    end
    forcing[:, 1,   :] .*= 0.5
    forcing[:, end, :] .*= 0.5

    return forcing
end

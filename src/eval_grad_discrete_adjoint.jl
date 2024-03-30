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
        order=2, cost_type=:Infidelity, return_lambda_history=false,
        kwargs...
    )

    history = eval_forward(prob, controls, pcof; order=order, kwargs...)

    forcing = compute_guard_forcing(prob, history)

    R = target[:,:] # Copy target, converting to matrix if vector
    T = vcat(R[1+prob.N_tot_levels:end,:], -R[1:prob.N_tot_levels,:])

    final_state = history[:,1,end,:]

    terminal_condition = compute_terminal_condition(
        prob, controls, pcof, target, final_state, order=order, cost_type=cost_type,
        forcing=forcing[:,end,:]
    )

    lambda_history = eval_adjoint(prob, controls, pcof, terminal_condition;
                                  order=order, forcing=forcing, kwargs...)

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

    if (order == 2)
        accumulate_gradient_order2!(gradient, prob, controls, pcof, history, lambda_history)
        return gradient
    elseif (order == 4)
        accumulate_gradient_order4!(gradient, prob, controls, pcof, history, lambda_history)
        return gradient
    end


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
Hard-coded version for order 2
"""
function accumulate_gradient_order2!(gradient::AbstractVector{Float64},
        prob::SchrodingerProb, controls, pcof::AbstractVector{Float64},
        history::AbstractArray{Float64, 3}, lambda_history::AbstractArray{Float64, 3}
    )

    dt = prob.tf / prob.nsteps

    u = zeros(prob.N_tot_levels)
    v = zeros(prob.N_tot_levels)
    lambda_u = zeros(prob.N_tot_levels)
    lambda_v = zeros(prob.N_tot_levels)

    asym_op_lambda_u = zeros(prob.N_tot_levels)
    asym_op_lambda_v = zeros(prob.N_tot_levels)
    sym_op_lambda_u = zeros(prob.N_tot_levels)
    sym_op_lambda_v = zeros(prob.N_tot_levels)

    for i in 1:prob.N_operators
        control = controls[i]
        asym_op = prob.asym_operators[i]
        sym_op = prob.sym_operators[i]
        local_pcof = get_control_vector_slice(pcof, controls, i)

        grad_contrib = zeros(control.N_coeff)
        grad_p = zeros(control.N_coeff)
        grad_q = zeros(control.N_coeff)

        for n in 0:prob.nsteps-1
            lambda_u .= @view lambda_history[1:prob.N_tot_levels,     1, 1+n+1]
            lambda_v .= @view lambda_history[1+prob.N_tot_levels:end, 1, 1+n+1]

            t = n*dt
            u .= @view history[1:prob.N_tot_levels,     1, 1+n]
            v .= @view history[1+prob.N_tot_levels:end, 1, 1+n]

            eval_grad_p_derivative!(grad_p, control, t, local_pcof, 0)
            eval_grad_q_derivative!(grad_q, control, t, local_pcof, 0)

            mul!(asym_op_lambda_u, asym_op, lambda_u)
            mul!(asym_op_lambda_v, asym_op, lambda_v)
            mul!(sym_op_lambda_u,  sym_op,  lambda_u)
            mul!(sym_op_lambda_v,  sym_op,  lambda_v)

            grad_contrib .+= grad_q .* -(dot(u, asym_op_lambda_u) + dot(v, asym_op_lambda_v))
            grad_contrib .+= grad_p .* (-dot(u, sym_op_lambda_v) + dot(v, sym_op_lambda_u))

            t = (n+1)*dt
            u .= @view history[1:prob.N_tot_levels,     1, 1+n+1]
            v .= @view history[1+prob.N_tot_levels:end, 1, 1+n+1]

            eval_grad_p_derivative!(grad_p, control, t, local_pcof, 0)
            eval_grad_q_derivative!(grad_q, control, t, local_pcof, 0)

            grad_contrib .+= grad_q .* -(dot(u, asym_op_lambda_u) + dot(v, asym_op_lambda_v))
            grad_contrib .+= grad_p .* (-dot(u, sym_op_lambda_v) + dot(v, sym_op_lambda_u))
        end

        grad_contrib .*= -0.5*dt

        grad_slice = get_control_vector_slice(gradient, controls, i)
        grad_slice .+= grad_contrib
    end

    return nothing
end



"""
New version which is a bit more informative, but less efficient.
"""
function accumulate_gradient_order2_new!(gradient::AbstractVector{Float64},
        prob::SchrodingerProb, controls, pcof::AbstractVector{Float64},
        history::AbstractArray{Float64, 3}, lambda_history::AbstractArray{Float64, 3}
    )

    dt = prob.tf / prob.nsteps
    working_vector = zeros(prob.real_system_size)

    for i in 1:prob.N_operators
        control = controls[i]
        asym_op = prob.asym_operators[i]
        sym_op = prob.sym_operators[i]
        local_pcof = get_control_vector_slice(pcof, controls, i)

        grad_contrib = zeros(control.N_coeff)
        grad_p = zeros(control.N_coeff)
        grad_q = zeros(control.N_coeff)

        for n in 0:prob.nsteps-1
            # Explicit Part
            t = n*dt
            eval_grad_p_derivative!(grad_p, control, t, local_pcof, 0)
            eval_grad_q_derivative!(grad_q, control, t, local_pcof, 0)

            right_inner = @view lambda_history[:,1,1+n+1]
            left_inner  = @view history[:,1,1+n]

            inner_prod_S = compute_inner_prod_S!(
                left_inner, right_inner, asym_op, working_vector, prob.real_system_size
            )
            inner_prod_K = compute_inner_prod_K!(
                left_inner, right_inner, sym_op, working_vector, prob.real_system_size
            )

            grad_contrib .+= grad_q .* inner_prod_S .* dt .* coefficient(1,1,1)
            grad_contrib .+= grad_p .* inner_prod_K .* dt .* coefficient(1,1,1)

            #Implicit Part
            t = (n+1)*dt
            eval_grad_p_derivative!(grad_p, control, t, local_pcof, 0)
            eval_grad_q_derivative!(grad_q, control, t, local_pcof, 0)

            right_inner = @view lambda_history[:,1,1+n+1]
            left_inner  = @view history[:,1,1+n+1]

            inner_prod_S = compute_inner_prod_S!(
                left_inner, right_inner, asym_op, working_vector, prob.real_system_size
            )
            inner_prod_K = compute_inner_prod_K!(
                left_inner, right_inner, sym_op, working_vector, prob.real_system_size
            )

            grad_contrib .+= grad_q .* inner_prod_S .* dt .* coefficient(1,1,1)
            grad_contrib .+= grad_p .* inner_prod_K .* dt .* coefficient(1,1,1)

        end

        grad_slice = get_control_vector_slice(gradient, controls, i)
        grad_slice .-= grad_contrib
    end

    return nothing
end




function accumulate_gradient_order4!(gradient::AbstractVector{Float64},
        prob::SchrodingerProb, controls, pcof::AbstractVector{Float64},
        history::AbstractArray{Float64, 3},
        lambda_history::AbstractArray{Float64, 3}
    )

    dt = prob.tf / prob.nsteps
    working_vector = zeros(prob.real_system_size)
    working_vector_re = view(working_vector, 1:prob.N_tot_levels)
    working_vector_im = view(working_vector, 1+prob.N_tot_levels:prob.real_system_size)

    left_inner = zeros(prob.real_system_size)
    left_inner_re = view(left_inner, 1:prob.N_tot_levels)
    left_inner_im = view(left_inner, 1+prob.N_tot_levels:prob.real_system_size)

    right_inner = zeros(prob.real_system_size)
    right_inner_re = view(right_inner, 1:prob.N_tot_levels)
    right_inner_im = view(right_inner, 1+prob.N_tot_levels:prob.real_system_size)

    for i in 1:prob.N_operators
        control = controls[i]
        asym_op = prob.asym_operators[i]
        sym_op = prob.sym_operators[i]
        local_pcof = get_control_vector_slice(pcof, controls, i)

        grad_contrib = zeros(control.N_coeff)
        grad_p = zeros(control.N_coeff)
        grad_q = zeros(control.N_coeff)
        grad_pt = zeros(control.N_coeff)
        grad_qt = zeros(control.N_coeff)

        for n in 0:prob.nsteps-1
            # Explicit Part
            t = n*dt
            eval_grad_p_derivative!(grad_p,  control, t, local_pcof, 0)
            eval_grad_p_derivative!(grad_pt, control, t, local_pcof, 1)
            eval_grad_q_derivative!(grad_q,  control, t, local_pcof, 0)
            eval_grad_q_derivative!(grad_qt, control, t, local_pcof, 1)

            c1_exp = dt .* coefficient(1,2,2)
            c2_exp = 0.5*(dt^2)*coefficient(2,2,2)

            # First order contribution
            left_inner .= @view history[:,1,1+n]
            right_inner .= @view lambda_history[:,1,1+n+1]

            inner_prod_S = compute_inner_prod_S!(
                left_inner, right_inner, asym_op, working_vector, prob.real_system_size
            )
            inner_prod_K = compute_inner_prod_K!(
                left_inner, right_inner, sym_op, working_vector, prob.real_system_size
            )
            @. grad_contrib += grad_q * inner_prod_S * c1_exp
            @. grad_contrib += grad_p * inner_prod_K * c1_exp


            # Second order contribution 1 (reuses computation from first order)
            @. grad_contrib += grad_qt * inner_prod_S * c2_exp
            @. grad_contrib += grad_pt * inner_prod_K * c2_exp


            # Second order contribution 2
            left_inner .= 0
            right_inner .= @view lambda_history[:,1,1+n+1]
            working_vector .= @view history[:,1,1+n]

            apply_hamiltonian!(
                left_inner_re, left_inner_im, working_vector_re, working_vector_im,
                prob, controls, t, pcof, use_adjoint=false
            )
            inner_prod_S = compute_inner_prod_S!(
                left_inner, right_inner, asym_op, working_vector, prob.real_system_size
            )
            inner_prod_K = compute_inner_prod_K!(
                left_inner, right_inner, sym_op, working_vector, prob.real_system_size
            )

            @. grad_contrib += grad_q * inner_prod_S * c2_exp
            @. grad_contrib += grad_p * inner_prod_K * c2_exp


            # Second order contribution 3
            left_inner .= @view history[:,1,1+n]
            right_inner .= 0
            working_vector .= @view lambda_history[:,1,1+n+1]

            apply_hamiltonian!(
                right_inner_re, right_inner_im, working_vector_re, working_vector_im,
                prob, controls, t, pcof, use_adjoint=true
            )

            inner_prod_S = compute_inner_prod_S!(
                left_inner, right_inner, asym_op, working_vector, prob.real_system_size
            )
            inner_prod_K = compute_inner_prod_K!(
                left_inner, right_inner, sym_op, working_vector, prob.real_system_size
            )

            @. grad_contrib += grad_q * inner_prod_S * c2_exp
            @. grad_contrib += grad_p * inner_prod_K * c2_exp
            


            #Implicit Part
            t = (n+1)*dt
            eval_grad_p_derivative!(grad_p,  control, t, local_pcof, 0)
            eval_grad_p_derivative!(grad_pt, control, t, local_pcof, 1)
            eval_grad_q_derivative!(grad_q,  control, t, local_pcof, 0)
            eval_grad_q_derivative!(grad_qt, control, t, local_pcof, 1)


            c1_imp = dt .* coefficient(1,2,2)
            c2_imp = -0.5*(dt^2)*coefficient(2,2,2)

            # First order contribution
            left_inner .= @view history[:,1,1+n+1]
            right_inner .= @view lambda_history[:,1,1+n+1]

            inner_prod_S = compute_inner_prod_S!(
                left_inner, right_inner, asym_op, working_vector, prob.real_system_size
            )
            inner_prod_K = compute_inner_prod_K!(
                left_inner, right_inner, sym_op, working_vector, prob.real_system_size
            )
            @. grad_contrib += grad_q * inner_prod_S * c1_imp
            @. grad_contrib += grad_p * inner_prod_K * c1_imp


            # Second order contribution 1 (reuses computation from first order)
            @. grad_contrib += grad_qt * inner_prod_S * c2_imp
            @. grad_contrib += grad_pt * inner_prod_K * c2_imp


            # Second order contribution 2
            left_inner .= 0
            right_inner .= @view lambda_history[:,1,1+n+1]
            working_vector .= @view history[:,1,1+n+1]

            apply_hamiltonian!(
                left_inner_re, left_inner_im, working_vector_re, working_vector_im,
                prob, controls, t, pcof, use_adjoint=false
            )
            inner_prod_S = compute_inner_prod_S!(
                left_inner, right_inner, asym_op, working_vector, prob.real_system_size
            )
            inner_prod_K = compute_inner_prod_K!(
                left_inner, right_inner, sym_op, working_vector, prob.real_system_size
            )

            @. grad_contrib += grad_q * inner_prod_S * c2_imp
            @. grad_contrib += grad_p * inner_prod_K * c2_imp


            # Second order contribution 3
            left_inner .= @view history[:,1,1+n+1]
            right_inner .= 0
            working_vector .= @view lambda_history[:,1,1+n+1]

            apply_hamiltonian!(
                right_inner_re, right_inner_im, working_vector_re, working_vector_im,
                prob, controls, t, pcof, use_adjoint=true
            )

            inner_prod_S = compute_inner_prod_S!(
                left_inner, right_inner, asym_op, working_vector, prob.real_system_size
            )
            inner_prod_K = compute_inner_prod_K!(
                left_inner, right_inner, sym_op, working_vector, prob.real_system_size
            )

            @. grad_contrib += grad_q * inner_prod_S * c2_imp
            @. grad_contrib += grad_p * inner_prod_K * c2_imp
            
        end

        grad_slice = get_control_vector_slice(gradient, controls, i)
        grad_slice .-= grad_contrib
    end

    return nothing
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

function compute_inner_prod_S!(left_inner, right_inner, S, working_vector, real_system_size)
    complex_system_size = div(real_system_size, 2)
    left_inner_re = view(left_inner, 1:complex_system_size)
    left_inner_im = view(left_inner, 1+complex_system_size:real_system_size)

    right_inner_re = view(right_inner, 1:complex_system_size)
    right_inner_im = view(right_inner, 1+complex_system_size:real_system_size)

    working_vector_re = view(working_vector, 1:complex_system_size)
    working_vector_im = view(working_vector, 1+complex_system_size:real_system_size)

    mul!(working_vector_re, S, right_inner_re)
    mul!(working_vector_im, S, right_inner_im)

    inner_prod_K = -dot(left_inner, working_vector)

    return inner_prod_K
end

function compute_inner_prod_K!(left_inner, right_inner, K, working_vector, real_system_size)
    complex_system_size = div(real_system_size, 2)
    left_inner_re = view(left_inner, 1:complex_system_size)
    left_inner_im = view(left_inner, 1+complex_system_size:real_system_size)

    right_inner_re = view(right_inner, 1:complex_system_size)
    right_inner_im = view(right_inner, 1+complex_system_size:real_system_size)

    working_vector_re = view(working_vector, 1:complex_system_size)
    working_vector_neg_im = view(working_vector, 1+complex_system_size:real_system_size)

    mul!(working_vector_re, K, right_inner_im)
    mul!(working_vector_neg_im, K, right_inner_re)

    inner_prod_K = -dot(left_inner_re, working_vector_re)
    inner_prod_K += dot(left_inner_im, working_vector_neg_im)
    return inner_prod_K
end

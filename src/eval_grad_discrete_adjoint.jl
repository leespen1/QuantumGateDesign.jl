function compute_terminal_condition(
        prob::SchrodingerProb,
        controls,
        pcof::AbstractVector{<: Real},
        target::AbstractVecOrMat{<: Real}, # This should be the real-valued one
        final_state::AbstractVecOrMat{<: Real};
        order::Integer=2,
        cost_type=:Infidelity,
        forcing=missing
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

    #TODO Add preconditioner to call
    for i in 1:size(target, 2)
        IterativeSolvers.gmres!(uv_vec, LHS_map, terminal_RHS[:,i],
                                abstol=prob.gmres_abstol, reltol=prob.gmres_reltol)
        terminal_condition[:,i] .= uv_vec
    end

    return terminal_condition
end

"""
    discrete_adjoint(prob, controls, pcof, target; [order=2, cost_type=:Infidelity, return_lambda_history=false])

Compute the gradient using the discrete adjoint method. Return the gradient.

# Arguments
- `prob::SchrodingerProb`: Object containing the Hamiltonians, number of timesteps, etc.
- `controls`: An `AstractControl` or vector of controls, where the i-th control corresponds to the i-th control Hamiltonian.
- `pcof::AbstractVector{<: Real}`: The control vector.
- `target::AbstractMatrix{Float64}`: The target gate, in 'stacked' real-valued format.
- `order::Int64=2`: Which order of the method to use.
- `cost_type=:Infidelity`: The cost function to use (ONLY USE INFIDELITY, OTHERS HAVE NOT BEEN TESTED RECENTLY).
- `return_lambda_history=false`: Whether to return the history of the adjoint variable lambda.
"""
function discrete_adjoint(
        prob::SchrodingerProb{<: AbstractMatrix{Float64}, <: AbstractMatrix{Float64}, P},
        controls,
        pcof::AbstractVector{<: Real},
        target::AbstractMatrix{<: Number}; 
        order=2, cost_type=:Infidelity,
    ) where P
    N_derivatives = div(order, 2)

    grad = zeros(length(pcof))

    history = zeros(prob.real_system_size, 1+N_derivatives, 1+prob.nsteps, prob.N_initial_conditions)
    lambda_history = zeros(prob.real_system_size, 1+N_derivatives, 1+prob.nsteps, prob.N_initial_conditions)
    adjoint_forcing = zeros(prob.real_system_size, 1+prob.nsteps, prob.N_initial_conditions)

    discrete_adjoint!(
        grad, history, lambda_history, adjoint_forcing, prob, controls, pcof,
        target, order=order, cost_type=cost_type
    )
end

"""
Mutating version, arrays pre-allocated
"""
function discrete_adjoint!(
        grad::AbstractVector{Float64}, history::AbstractArray{Float64, 4},
        lambda_history::AbstractArray{Float64, 4}, adjoint_forcing::AbstractArray{Float64, 3},
        prob::SchrodingerProb{<: AbstractMatrix{Float64}, <: AbstractMatrix{Float64}, P},
        controls,
        pcof::AbstractVector{<: Real},
        target::AbstractMatrix{<: Number}; 
        order=2, cost_type=:Infidelity, history_precomputed=false 
    ) where P

    # Set pre-allocated arrays equal to zero (may not be necessary, but being safe)
    if !history_precomputed
        history .= 0
    end
    lambda_history .= 0
    adjoint_forcing .= 0


    N_derivatives = div(order, 2)
    target = complex_to_real(target)

    # FORWARD EVOLUTION (if needed)
    if !history_precomputed
        eval_forward!(history, prob, controls, pcof; order=order)
    end

    # COMPUTE FORCING
    compute_guard_forcing!(adjoint_forcing, prob, history)

    # COMPUTE TERMINAL CONDITION
    final_state = history[:,1,end,:]
    terminal_condition = compute_terminal_condition(
        prob, controls, pcof, target, final_state, order=order, cost_type=cost_type,
        forcing=adjoint_forcing[:,end,:]
    )

    # ADJOINT EVOLUTION
    eval_adjoint!(lambda_history, prob, controls, pcof, terminal_condition;
        order=order, forcing=adjoint_forcing
    )

    # GRADIENT ACCUMULATION (Could be multithreaded)
    grad .= 0
    for initial_condition_index = 1:size(prob.u0,2)
        this_history = view(history, :, :, :, initial_condition_index)
        this_lambda_history = view(lambda_history, :, :, :, initial_condition_index)

        accumulate_gradient!(
            grad, prob, controls, pcof, this_history, this_lambda_history, order=order
        )
    end

    return grad
end

    


#=
function eval_adjoint!(uv_history::AbstractArray{Float64, 3},
        prob::SchrodingerProb{M, V, P}, controls, pcof::AbstractVector{<: Real},
        terminal_condition::AbstractVector{Float64};
        forcing::Union{AbstractArray{Float64, 2}, Missing}=missing,
        order::Int=2,
        use_taylor_guess=true, verbose=false,
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}, P}
=#

"""
Change name to 'accumulate gradient' or something

Maybe I shoudl return the contribution added to the gradient instead of the
gradient itself. That might make it easier to analyze things from the REPL.
"""
function accumulate_gradient!(gradient::AbstractVector{Float64},
        prob::SchrodingerProb, controls, pcof::AbstractVector{Float64},
        history::AbstractArray{Float64, 3}, lambda_history::AbstractArray{Float64, 3};
        order=2
    )

    accumulate_gradient_arbitrary_fast!(gradient, prob, controls, pcof, history, lambda_history, order=order)
    return gradient

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

    return gradient
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
            coeff = coefficient(1,1,1)
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

            @. grad_contrib += grad_q * inner_prod_S * dt * coeff
            @. grad_contrib += grad_p * inner_prod_K * dt * coeff

            println("inner_prod_S: $inner_prod_S")
            println("inner_prod_K: $inner_prod_K")
            println("grad_p: $grad_p")
            println("grad_q: $grad_q")
            println("coeff: $(coeff*dt)")
            println("other factor:")
            println()

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

            @. grad_contrib += grad_q * inner_prod_S * dt * coeff
            @. grad_contrib += grad_p * inner_prod_K * dt * coeff

            println("inner_prod_S: $inner_prod_S")
            println("inner_prod_K: $inner_prod_K")
            println("grad_p: $grad_p")
            println("grad_q: $grad_q")
            println("coeff: $(coeff*dt)")
            println("other factor:")
            println()

        end

        grad_slice = get_control_vector_slice(gradient, controls, i)
        grad_slice .-= grad_contrib
    end

    return gradient
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
            left_inner .= @view history[:,2,1+n]
            right_inner .= @view lambda_history[:,1,1+n+1]
            working_vector .= 0
            
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
            left_inner .= @view history[:,2,1+n+1]
            right_inner .= @view lambda_history[:,1,1+n+1]
            working_vector .= 0

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

    return gradient
end

"""
New version which is a bit more informative, but less efficient.

Efficiency tricks I can do:
1. do implicit and explicit alongside each other, like I do in the hard-coded 4th order
   (actually that may not be so useful or easy) 
2. reuse inner_prod_K/S, like I do in the hard-coded 4th
"""
function accumulate_gradient_arbitrary_fast!(gradient::AbstractVector{Float64},
        prob::SchrodingerProb, controls, pcof::AbstractVector{Float64},
        history::AbstractArray{Float64, 3}, lambda_history::AbstractArray{Float64, 3};
        order=2
    )

    N_derivatives = div(order, 2)
    dt = prob.tf / prob.nsteps
    λₙ₊₁ = zeros(prob.real_system_size) 
    wₙ   = zeros(prob.real_system_size, 1+N_derivatives) 
    wₙ₊₁ = zeros(prob.real_system_size, 1+N_derivatives) 

    working_pcof = zeros(length(pcof))
    working_state_vector = zeros(prob.real_system_size)
    working_state_matrix = zeros(prob.real_system_size, N_derivatives)


    for i in 1:prob.N_operators
        control = controls[i]
        grad_contrib = zeros(control.N_coeff)

        for n in 0:prob.nsteps-1
            λₙ₊₁ .= @view lambda_history[:, 1, 1+n+1]
            wₙ   .= @view history[:, :, 1+n]
            wₙ₊₁ .= @view history[:, :, 1+n+1]
            tₙ = n*dt
            tₙ₊₁ = (n+1)*dt

            #println("#"^20, "\nExplicit\n", "#"^20)
            for k in 0:N_derivatives
                #c_implicit = (-0.5*dt)^k * coefficient(k, N_derivatives, N_derivatives)
                #c_explicit = (0.5*dt)^k  * coefficient(k, N_derivatives, N_derivatives)
                c_implicit = -(-dt)^k * coefficient(k, N_derivatives, N_derivatives)
                c_explicit = (dt)^k  * coefficient(k, N_derivatives, N_derivatives)

                #println("#"^20, "\nOrder $k Contribution\n", "#"^20)
                # Handle explicit
                recursive_magic!(
                    grad_contrib, wₙ, λₙ₊₁, k, c_explicit, prob, controls, tₙ,
                    pcof, i, working_pcof, working_state_vector, working_state_matrix
                )
            end
                
            #println("#"^20, "\nImplicit\n", "#"^20)
            for k in 0:N_derivatives
                #c_implicit = (-0.5*dt)^k * coefficient(k, N_derivatives, N_derivatives)
                #c_explicit = (0.5*dt)^k  * coefficient(k, N_derivatives, N_derivatives)
                c_implicit = -(-dt)^k * coefficient(k, N_derivatives, N_derivatives)
                c_explicit = (dt)^k  * coefficient(k, N_derivatives, N_derivatives)

                #println("#"^20, "\nOrder $k Contribution\n", "#"^20)
                # Handle implicit
                recursive_magic!(
                    grad_contrib, wₙ₊₁, λₙ₊₁, k, c_implicit, prob, controls,
                    tₙ₊₁, pcof, i, working_pcof, working_state_vector,
                    working_state_matrix
                )
            end
        end

        grad_slice = get_control_vector_slice(gradient, controls, i)
        grad_slice .-= grad_contrib
    end

    return gradient
end

"""
I will need a matrix of left_inners, since I have y0, y1, y2, etc. 

May as well make a matrix of right inners, since I will have λ, A₀λ, A₁λ, ...

Does the contribution of ⟨coeff*wⱼ₊₁, λ⟩
"""
function recursive_magic!(grad_contrib::AbstractVector{<: Real},
        w_mat::AbstractMatrix{<: Real}, lambda::AbstractVector{<: Real},
        derivative_order::Integer, coeff::Real,
        prob::SchrodingerProb, controls, t::Real,
        pcof::AbstractVector{<: Real}, control_index::Integer,
        working_pcof::AbstractVector{<: Real},
        working_state_vector::AbstractVector{<: Real},
        working_state_matrix::AbstractMatrix{<: Real}
    )
    control = controls[control_index]
    asym_op = prob.asym_operators[control_index]
    sym_op = prob.sym_operators[control_index]
    local_pcof = get_control_vector_slice(pcof, controls, control_index)
    local_working_pcof = get_control_vector_slice(working_pcof, controls, control_index)
    #TODO Instead of using a working vector, precompute all the control
    #function gradiens and pass them in as a big Array{Float64, 3}

    j = derivative_order-1
    real_system_size = size(w_mat, 1)

    for i in 0:j
        # i=0,j=0 and i=0,j=1 will be the same except for the broadcasting.
        # There should be a way to make use of this to avoid redoing computiation.
        # It seems like once I do any i=i',j=j', I should be able to handle all subsequent
        # cases of i=i',j=any at the same time. Investigate this (also only optimize slow things, don't dig
        # into this prematurely).
        #
        # What I originally had (should work once I use a real history)
        inner_prod_S = compute_inner_prod_S!(
            view(w_mat, :, 1+i), lambda, asym_op, working_state_vector, prob.real_system_size
        )
        inner_prod_K = compute_inner_prod_K!(
            view(w_mat, :, 1+i), lambda, sym_op, working_state_vector, prob.real_system_size
        )

        # local_working_pcof = grad_p
        eval_grad_p_derivative!(local_working_pcof, control, t, local_pcof, j-i) 
        @. grad_contrib += local_working_pcof * inner_prod_K * coeff / ((j+1)*factorial(j-i))

        # local_working_pcof = grad_q
        eval_grad_q_derivative!(local_working_pcof, control, t, local_pcof, j-i)
        @. grad_contrib += local_working_pcof * inner_prod_S * coeff / ((j+1)*factorial(j-i))
    end

    # Better to do in two loops. Makes it more clear how I can make the first
    # loop more efficient by reusing computation.
    # Could also make the loop over 1:j, since if i=0 then this doesn't execute
    for i in 0:j
        # Take special care about how factors are handled
        # Move this outside the loop
        right_inner = @view working_state_matrix[:,1+i]
        right_inner .= 0

        # Using views here might lead to type instability in next recursive_magic! call.
        # Should check this with @code_warn
        right_inner = view(working_state_matrix, :, 1+i)
        working_state_matrix_reduced = view(working_state_matrix, :, 1:i)
        right_inner .= 0
        apply_hamiltonian!(right_inner, lambda, prob, controls, t, pcof;
                           derivative_order=(j-i), use_adjoint=true)
        

        recursive_magic!(
            grad_contrib, w_mat, right_inner, i, coeff/(j+1), prob,
            controls, t, pcof, control_index, working_pcof, working_state_vector,
            working_state_matrix_reduced,
        )
    end

    return grad_contrib
end


"""
Should make 3-dim array version for VectorSchrodingerProb case
"""
function compute_guard_forcing!(forcing_out::AbstractArray{Float64, 3}, 
        prob, history::AbstractArray{Float64, 4}
    )
    forcing_out .= 0 # Maybe unnecessary, since each mul! overwrites
    dt = prob.tf / prob.nsteps

    for n in 1:prob.nsteps+1
        for k in 1:prob.N_initial_conditions
            mul!(
                 view(forcing_out, :, n, k),
                 prob.guard_subspace_projector,
                 view(history, :, 1, n, k)
            )
            @. forcing_out[:, n, k] *= -2*dt/prob.tf
        end
    end
    forcing_out[:, 1,   :] .*= 0.5
    forcing_out[:, end, :] .*= 0.5

    return forcing_out
end

function compute_guard_forcing(prob, history::AbstractArray{Float64, 4})
    forcing = zeros(prob.real_system_size, 1+prob.nsteps, prob.N_initial_conditions)
    compute_guard_forcing!(forcing, prob, history)
    return forcing
end

"""
H = [S K; -K S] (real-valued hamiltonian)
⟨H*left_inner, right_inner⟩
"""
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

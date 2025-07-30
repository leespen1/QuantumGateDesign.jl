mutable struct DiscreteAdjointTimes
    forward::Float64
    adjoint::Float64
    grad_accum::Float64
    function DiscreteAdjointTimes()
        new(NaN, NaN, NaN)
    end
end

function total_time(timer::DiscreteAdjointTimes)
    return timer.forward + timer.adjoint + timer.grad_accum
end


function compute_terminal_condition(
        prob::SchrodingerProb{OpType, StateType, P},
        controls,
        pcof::AbstractVector{<: Real},
        target::AbstractVecOrMat{<: Real}, # This should be the real-valued one
        final_state::AbstractVecOrMat{<: Real};
        order::Integer=2,
        cost_type=:Infidelity,
        forcing=missing,
    ) where {OpType, StateType, P}

    terminal_condition = zeros(size(target))

    t = prob.tf
    dt = prob.tf/prob.nsteps

    N_derivatives = div(order, 2)

    uv_mat = zeros(prob.real_system_size, 1+N_derivatives)
    uv_vec = zeros(prob.real_system_size)

    # ORIGINAL 
    R = target[:,:] # Copy target, converting to matrix if vector (will this code work for vectors?)
    T = vcat(R[1+prob.N_tot_levels:end,:], -R[1:prob.N_tot_levels,:])
    
    ## NEW (DOESN'T SEEM TO WORK)
    #R = vcat(target[1:prob.N_tot_levels,:], -target[1+prob.N_tot_levels:end,:])
    #T = vcat(target[1+prob.N_tot_levels:end,:], target[1:prob.N_tot_levels,:])

    # Set up terminal condition RHS
    if cost_type == :Infidelity
        terminal_RHS = (dot(final_state, R)*R + dot(final_state, T)*T)
        terminal_RHS *= (2.0/(prob.N_ess_levels^2))
    elseif cost_type == :GeneralizedInfidelity
        terminal_RHS = (2.0 / prob.N_ess_levels) .* final_state
        terminal_RHS .-= (2.0 / (prob.N_ess_levels^2)) .* (dot(final_state, R) .* R .+ dot(final_state, T) .* T)
        terminal_RHS .*= -1
    elseif cost_type == :Tracking
        terminal_RHS = -(final_state - target)
    elseif cost_type == :Norm
        terminal_RHS = -final_state
    else
        throw("Invalid cost type: $cost_type")
    end

    #@show terminal_RHS

    # Add forcing
    if !ismissing(forcing)
        terminal_RHS .+= forcing
    end

    # Do the adjoint solve

    lhs_holder = LHSHolderAdjoint(prob, N_derivatives, dt)

    # Create linear map out of LHS_func_wrapper, to use in GMRES
    LHS_map = LinearMaps.LinearMap(
        lhs_holder,
        prob.real_system_size, prob.real_system_size,
        ismutating=true
    )

    Pl = P(prob, order, true)

    gmres_iterable = IterativeSolvers.gmres_iterable!(
        zeros(prob.real_system_size), LHS_map, zeros(prob.real_system_size),
        abstol=prob.gmres_abstol, reltol=prob.gmres_reltol, restart=prob.real_system_size,
        initially_zero=false, Pl=Pl
    )

    fill_p_mat!(lhs_holder.control_vals_real, controls, prob.tf, pcof) 
    fill_q_mat!(lhs_holder.control_vals_imag, controls, prob.tf, pcof) 

    initial_guess = zeros(prob.real_system_size) # Use current timestep as initial guess for gmres
    for i in 1:prob.N_initial_conditions
        update_gmres_iterable!(gmres_iterable, initial_guess, terminal_RHS[:,i])

        N_gmres_iterations = 0
        for iter in gmres_iterable
            N_gmres_iterations += 1
        end
        terminal_condition[:,i] .= gmres_iterable.x
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
        prob::SchrodingerProb,
        controls,
        pcof::AbstractVector{<: Real},
        target::AbstractMatrix{<: Number}; 
        order=2, cost_type=:Infidelity,
        timer::Union{Missing, DiscreteAdjointTimes}=missing
    )

    grad = zeros(length(pcof))
    history = allocate_history(prob, order)
    lambda_history = allocate_history(prob, order)
    adjoint_forcing = allocate_forcing(prob, order)

    discrete_adjoint!(
        grad, history, lambda_history, adjoint_forcing, prob, controls, pcof,
        target, order=order, cost_type=cost_type, timer=timer
    )
end

"""
Mutating version, arrays pre-allocated
"""
function discrete_adjoint!(
        grad::AbstractVector{<: Real},
        history::AbstractArray{Float64},
        lambda_history::AbstractArray{Float64}, 
        adjoint_forcing::AbstractArray{Float64},
        prob::SchrodingerProb,
        controls::ControlsType,
        pcof::AbstractVector{<: Real},
        target::AbstractVecOrMat{<: Number}; 
        order::Integer=2, cost_type=:Infidelity, history_precomputed=false,
        timer::Union{Missing, DiscreteAdjointTimes}=missing,
        forward_gmres_tracker::Union{GMRESTracker, Missing}=missing,
        adjoint_gmres_tracker::Union{GMRESTracker, Missing}=missing,
    ) 
    N_derivatives = div(order, 2)
    # Check sizes of pre-allocated arrays
    @assert size(history) == size(lambda_history)
    @assert size(history, 1) == size(adjoint_forcing, 1) == prob.real_system_size
    @assert size(history, 2) ==  1 + N_derivatives
    @assert size(history, 3) == size(adjoint_forcing, 2) == 1+prob.nsteps
    @assert size(history, 4) == size(adjoint_forcing, 3) == prob.N_initial_conditions

    # Set pre-allocated arrays equal to zero (may not be necessary, but being safe)
    if !history_precomputed
        history .= 0
    end
    lambda_history .= 0
    adjoint_forcing .= 0


    N_derivatives = div(order, 2)
    target = complex_to_real(target)

    # FORWARD EVOLUTION (if needed)
    t_start_forward = ismissing(timer) ? NaN : time()
    if !history_precomputed
        eval_forward!(history, prob, controls, pcof; order=order,
                      gmres_tracker=forward_gmres_tracker)
    end
    t_end_forward = ismissing(timer) ? NaN : time()

    t_start_adjoint = ismissing(timer) ? NaN : time()
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
        order=order, forcing=adjoint_forcing, gmres_tracker=adjoint_gmres_tracker
    )
    t_end_adjoint = ismissing(timer) ? NaN : time()

    t_start_grad_accum = ismissing(timer) ? NaN : time()
    # GRADIENT ACCUMULATION (Could be multithreaded)
    grad .= 0
    for initial_condition_index = 1:size(prob.u0,2)
        this_history = selectdim(history, 4, initial_condition_index)
        this_lambda_history = selectdim(lambda_history, 4, initial_condition_index)

        accumulate_gradient!(
            grad, prob, controls, pcof, this_history, this_lambda_history, order=order
        )
    end
    t_end_grad_accum = ismissing(timer) ? NaN : time()

    if !ismissing(timer)
        timer.forward = t_end_forward - t_start_forward
        timer.adjoint = t_end_adjoint - t_start_adjoint
        timer.grad_accum = t_end_grad_accum - t_start_grad_accum
    end

    return grad
end

    


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

    accumulate_gradient_arbitrary_fast!(gradient, prob, controls, pcof, history, lambda_history, order=order)
    return gradient
end

"""
Hard-coded version for order 2
"""
function accumulate_gradient_order2!(gradient::AbstractVector{Float64},
        prob::SchrodingerProb, controls, pcof::AbstractVector{Float64},
        history::AbstractArray{Float64, 3}, lambda_history::AbstractArray{Float64, 3}
    )
    println("This runs!")

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

        t₀ = 0.0
        eval_grad_p_derivative!(grad_p, control, t₀, local_pcof, 0)
        eval_grad_q_derivative!(grad_q, control, t₀, local_pcof, 0)

        for n in 0:prob.nsteps-1
            lambda_u .= @view lambda_history[1:prob.N_tot_levels,     1, 1+n+1]
            lambda_v .= @view lambda_history[1+prob.N_tot_levels:end, 1, 1+n+1]

            u .= @view history[1:prob.N_tot_levels,     1, 1+n]
            v .= @view history[1+prob.N_tot_levels:end, 1, 1+n]

            mul!(asym_op_lambda_u, asym_op, lambda_u)
            mul!(asym_op_lambda_v, asym_op, lambda_v)
            mul!(sym_op_lambda_u,  sym_op,  lambda_u)
            mul!(sym_op_lambda_v,  sym_op,  lambda_v)

            grad_contrib .+= grad_q .* -(dot(u, asym_op_lambda_u) + dot(v, asym_op_lambda_v))
            grad_contrib .+= grad_p .* (-dot(u, sym_op_lambda_v) + dot(v, sym_op_lambda_u))

            tₙ₊₁ = (n+1)*dt
            u .= @view history[1:prob.N_tot_levels,     1, 1+n+1]
            v .= @view history[1+prob.N_tot_levels:end, 1, 1+n+1]

            eval_grad_p_derivative!(grad_p, control, tₙ₊₁, local_pcof, 0)
            eval_grad_q_derivative!(grad_q, control, tₙ₊₁, local_pcof, 0)

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

    working_state_vector = zeros(prob.real_system_size)
    working_state_matrix = zeros(prob.real_system_size, N_derivatives)

    control_vals_real = zeros(1+N_derivatives, prob.N_operators)
    control_vals_imag = zeros(1+N_derivatives, prob.N_operators)

    # Could move this up the loop heierarchy so I don't have to recompute control values
    for i in 1:prob.N_operators
        control = controls[i]
        grad_contrib = zeros(control.N_coeff)

        local_pcof = get_control_vector_slice(pcof, controls, i)
        local_control_grad_real = zeros(control.N_coeff, 1+N_derivatives)
        local_control_grad_imag = zeros(control.N_coeff, 1+N_derivatives)
        
        # Initial control values
        t₀ = 0.0
        fill_p_mat!(control_vals_real, controls, t₀, pcof) 
        fill_q_mat!(control_vals_imag, controls, t₀, pcof) 
        fill_grad_p_mat!(local_control_grad_real, control, t₀, local_pcof)
        fill_grad_q_mat!(local_control_grad_imag, control, t₀, local_pcof)


        for n in 0:prob.nsteps-1
            λₙ₊₁ .= @view lambda_history[:, 1, 1+n+1]
            wₙ   .= @view history[:, :, 1+n]
            wₙ₊₁ .= @view history[:, :, 1+n+1]
            tₙ₊₁ = (n+1)*dt

            #TODO Make this twice as efficient by moving hamiltonians to the
            #λₙ₊₁ side, as suggested in the paper.

            #println("#"^20, "\nExplicit\n", "#"^20)
            for k in 0:N_derivatives
                #c_implicit = (-0.5*dt)^k * coefficient(k, N_derivatives, N_derivatives)
                #c_explicit = (0.5*dt)^k  * coefficient(k, N_derivatives, N_derivatives)
                c_implicit = -(-dt)^k * coefficient(k, N_derivatives, N_derivatives)
                c_explicit = (dt)^k  * coefficient(k, N_derivatives, N_derivatives)

                #println("#"^20, "\nOrder $k Contribution\n", "#"^20)
                # Handle explicit
                recursive_magic!(
                    grad_contrib, wₙ, λₙ₊₁, k, c_explicit, prob, i,
                    working_state_vector,
                    working_state_matrix, control_vals_real, control_vals_imag, 
                    local_control_grad_real, local_control_grad_imag,
                )
            end


            # These values will be reused next iteration!
            fill_p_mat!(control_vals_real, controls, tₙ₊₁, pcof) 
            fill_q_mat!(control_vals_imag, controls, tₙ₊₁, pcof) 
            fill_grad_p_mat!(local_control_grad_real, control, tₙ₊₁, local_pcof)
            fill_grad_q_mat!(local_control_grad_imag, control, tₙ₊₁, local_pcof)
                
            #println("#"^20, "\nImplicit\n", "#"^20)
            for k in 0:N_derivatives
                #c_implicit = (-0.5*dt)^k * coefficient(k, N_derivatives, N_derivatives)
                #c_explicit = (0.5*dt)^k  * coefficient(k, N_derivatives, N_derivatives)
                c_implicit = -(-dt)^k * coefficient(k, N_derivatives, N_derivatives)
                c_explicit = (dt)^k  * coefficient(k, N_derivatives, N_derivatives)


                #println("#"^20, "\nOrder $k Contribution\n", "#"^20)
                # Handle implicit
                recursive_magic!(
                    grad_contrib, wₙ₊₁, λₙ₊₁, k, c_implicit, prob, i, 
                    working_state_vector,
                    working_state_matrix, control_vals_real, control_vals_imag,
                    local_control_grad_real, local_control_grad_imag,
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

**update** New version, uses precomputed control values
"""
function recursive_magic!(grad_contrib::AbstractVector{<: Real},
        w_mat::AbstractMatrix{<: Real}, lambda::AbstractVector{<: Real},
        derivative_order::Integer, coeff::Real, prob::SchrodingerProb,
        control_index::Integer,
        working_state_vector::AbstractVector{<: Real},
        working_state_matrix::AbstractMatrix{<: Real},
        control_vals_real::AbstractMatrix{Float64},
        control_vals_imag::AbstractMatrix{Float64},
        local_control_grad_real::AbstractMatrix{Float64},
        local_control_grad_imag::AbstractMatrix{Float64},
    )
    asym_op = prob.asym_operators[control_index]
    sym_op = prob.sym_operators[control_index]

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

        fact_j_minus_i = factorial(j-i)

        grad_p = view(local_control_grad_real, :, 1+j-i)
        @. grad_contrib += grad_p * inner_prod_K * coeff / ((j+1)*fact_j_minus_i)

        grad_q = view(local_control_grad_imag, :, 1+j-i)
        @. grad_contrib += grad_q * inner_prod_S * coeff / ((j+1)*fact_j_minus_i)
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
        apply_hamiltonian!(right_inner, lambda, prob, control_vals_real, control_vals_imag;
                           derivative_order=(j-i), use_adjoint=true)
        
        recursive_magic!(
            grad_contrib, w_mat, right_inner, i, coeff/(j+1), prob, control_index,
            working_state_vector, working_state_matrix_reduced,
            control_vals_real, control_vals_imag,
            local_control_grad_real, local_control_grad_imag,
        )
    end

    return grad_contrib
end


"""
Should make 3-dim array version for VectorSchrodingerProb case
"""
function compute_guard_forcing!(forcing_out::AbstractArray{<: Real}, 
        prob::SchrodingerProb, history::AbstractArray{<: Real}
    )
    @assert (ndims(forcing_out) == 3 && ndims(history) == 4) || (ndims(forcing_out) == 2 && ndims(history) == 3) 
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

function compute_guard_forcing(prob::SchrodingerProb, history::AbstractArray{<: Real})
    N_derivatives = size(history, 2) - 1
    method_order = 2*N_derivatives
    forcing = allocate_forcing(prob, method_order)
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

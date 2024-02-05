function eval_grad_forced(prob::SchrodingerProb{M, VM}, controls,
        pcof::AbstractVector{Float64}, target::VM; order=2, 
        cost_type=:Infidelity, return_forcing=false
    ) where {M <: AbstractMatrix{Float64}, VM <: AbstractVecOrMat{Float64}}

    # Allocate space for gradient
    gradient = zeros(length(pcof))

    dt = prob.tf / prob.nsteps
    N_derivatives = div(order, 2)


    # Copy of problem with initial conditions set to zero (to correspond to evolution of ∂ψ/∂θₖ)
    diff_prob = copy(prob)
    diff_prob.u0 .= 0
    diff_prob.v0 .= 0

    R = copy(target)
    T = vcat(R[1+prob.N_tot_levels:end,:], -R[1:prob.N_tot_levels,:])

    # Get state vector history
    history = eval_forward(prob, controls, pcof, order=order)
    final_state = history[:, 1, end, :]



    # For storing the global forcing array
    forcing_ary = zeros(prob.real_system_size, N_derivatives, 1+prob.nsteps, size(prob.u0, 2))
    # For storing the per-timestep forcing matrix
    forcing_matrix = zeros(prob.real_system_size, N_derivatives)
    # For storing the per-timestep state vector and derivatives
    uv_matrix = zeros(prob.real_system_size, 1+N_derivatives)


    global_control_param_index = 1

    # Iterate over each control
    for control_index in 1:prob.N_operators
        local_control = controls[control_index]
        local_pcof = get_control_vector_slice(pcof, controls, control_index)
        local_N_coeff = local_control.N_coeff

        asym_op = prob.asym_operators[control_index]
        sym_op = prob.sym_operators[control_index]

        # Compute the gradients of the control at each time point 
        p_vals = zeros(1+N_derivatives, 1+prob.nsteps, local_N_coeff)
        q_vals = zeros(1+N_derivatives, 1+prob.nsteps, local_N_coeff)

        for n in 0:prob.nsteps
            t = n*dt
            for derivative_i in 0:N_derivatives-1
                p_vals[1+derivative_i, 1+n, :] .= eval_grad_p_derivative(local_control, t, local_pcof, derivative_i)
                q_vals[1+derivative_i, 1+n, :] .= eval_grad_q_derivative(local_control, t, local_pcof, derivative_i)
            end
        end


        for local_control_param_index in 1:local_N_coeff
            ## Compute forcing array (using -i∂H/∂θₖ ψ)
            for initial_condition_index = 1:size(prob.u0, 2)
                for n in 0:prob.nsteps
                    t = n*dt

                    uv_matrix .= history[:, :, 1+n, initial_condition_index]

                    forcing_matrix .= 0

                    for j = 0:(N_derivatives-1)
                        # Get views of the current derivative we are trying to compute (the j+1th derivative)
                        u_derivative = view(forcing_matrix, 1:prob.N_tot_levels,                       1+j)
                        v_derivative = view(forcing_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+j)

                        # Perform the summation (the above is part of the i=j term in summation, this loop completes that term and the rest)
                        for i = j:-1:0
                            u_derivative_prev = view(uv_matrix, 1:prob.N_tot_levels,                       1+i)
                            v_derivative_prev = view(uv_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+i)

                            p_val = p_vals[1+j-i, 1+n, local_control_param_index] / factorial(j-i)
                            q_val = q_vals[1+j-i, 1+n, local_control_param_index] / factorial(j-i)

                            mul!(u_derivative, asym_op, u_derivative_prev, q_val, 1)
                            mul!(u_derivative, sym_op,  v_derivative_prev, p_val, 1)

                            mul!(v_derivative, asym_op, v_derivative_prev, q_val,  1)
                            mul!(v_derivative, sym_op,  u_derivative_prev, -p_val, 1)
                        end

                        # We do not need to divide by (j+1), the factorials are already in uv_matrix
                        # TODO : Is that the reason? Or is it something else?
                    end

                    forcing_ary[:,:,1+n, initial_condition_index] .= forcing_matrix
                end
            end


            # Compute the state history of ∂ψ/∂θₖ
            history_partial_derivative = eval_forward(
                diff_prob, controls, pcof, forcing=forcing_ary,
                order=order
            )

            #=
            if global_control_param_index == 1
                println("History")
                display(history)
                println("\nPartial History")
                display(history_partial_derivative)
            end
            =#

            final_state_partial_derivative = history_partial_derivative[:, 1, end, :]
            
            # Compute the partial derivative of the objective function with respect to θₖ
            # Target Contribution (Infidelity, or whatever we are using at the final time)
            if cost_type == :Infidelity
                gradient[global_control_param_index]  = dot(final_state, R)*dot(final_state_partial_derivative, R)
                gradient[global_control_param_index] += dot(final_state, T)*dot(final_state_partial_derivative, T)
                gradient[global_control_param_index] *= -(2/(prob.N_ess_levels^2))
            elseif cost_type == :Tracking
                gradient[global_control_param_index] = dot(final_state_partial_derivative, final_state - target)
            elseif cost_type == :Norm
                gradient[global_control_param_index] = dot(final_state_partial_derivative, final_state)
            else
                throw("Invalid cost type: $cost_type")
            end

            # Guard contribution
            # Currently just hacking in guard projector
            guard_val = 0.0
            W = prob.guard_subspace_projector
            N = size(history, 3)
            for i in 1:N
                val = dot(history_partial_derivative[:,1,i,:], W*history[:,1,i,:])
                val += dot(history[:,1,i,:], W*history_partial_derivative[:,1,i,:])
                if (i == 1 || i == N)
                    guard_val += 0.5*val
                else
                    guard_val += val
                end
            end
            guard_val *= dt/prob.tf

            gradient[global_control_param_index] += guard_val


            global_control_param_index += 1
        end
    end


    if return_forcing
        return gradient, permutedims(forcing_ary, (1,3,2,4))
    end

    return gradient
end

"""
Evaluates gradient of the provided Schrodinger problem with the given target
gate and control parameter(s) pcof using the "forward differentiation" method,
which evolves a differentiated Schrodinger equation, using the state vector
in the evolution of the original Schrodinger equation as a forcing term.

Returns: gradient
"""
function eval_grad_forced(prob::SchrodingerProb{M, VM}, controls,
        pcof::AbstractVector{Float64}, target::VM; order=2, 
        cost_type=:Infidelity, return_forcing=false
    ) where {M <: AbstractMatrix{Float64}, VM <: AbstractVecOrMat{Float64}}

    # Get state vector history
    history = eval_forward(prob, controls, pcof, order=order, return_time_derivatives=true)

    ## Compute forcing array (using -i∂H/dθₖ ψ)
    # Prepare dH/dα

    gradient = zeros(controls.N_coeff)

    forcing_ary = zeros(prob.real_system_size, 1+prob.nsteps, div(order, 2), size(prob.u0, 2))

    grad_prob = differentiated_prob(prob)


    for control_param_index in 1:controls.N_coeff
        grad_controls = [GradControl(control, control_param_index) for control in controls]
        grad_time_derivative_controls = [TimeDerivativeControl(control) for control in grad_controls]


        for initial_condition_index = 1:size(prob.u0, 2)
            # This could be more efficient by having functions which calculate
            # only the intended partial derivative, not the whole gradient, but
            # I am only focusing on optimizaiton for the discrete adjoint.

            # 2nd order version
            if order == 2
                forcing_vec = zeros(prob.real_system_size)

                u  = zeros(prob.N_tot_levels)
                v  = zeros(prob.N_tot_levels)
                ut = zeros(prob.N_tot_levels)
                vt = zeros(prob.N_tot_levels)

                t = 0.0
                dt = prob.tf/prob.nsteps

                # Get forcing (dH/dα * ψ)
                for n in 0:prob.nsteps
                    copyto!(u, history[1:prob.N_tot_levels    , 1+n, 1, initial_condition_index])
                    copyto!(v, history[1+prob.N_tot_levels:end, 1+n, 1, initial_condition_index])

                    utvt!(ut, vt, u, v, grad_prob, grad_controls, t, pcof)

                    copyto!(forcing_vec, 1,                   ut, 1, prob.N_tot_levels)
                    copyto!(forcing_vec, 1+prob.N_tot_levels, vt, 1, prob.N_tot_levels)

                    forcing_ary[:, 1+n, 1, initial_condition_index] .= forcing_vec

                    t += dt
                end

            # 4th order version
            elseif order == 4
                forcing_vec2 = zeros(prob.real_system_size)
                forcing_vec4 = zeros(prob.real_system_size)

                u  = zeros(prob.N_tot_levels)
                v  = zeros(prob.N_tot_levels)
                ut = zeros(prob.N_tot_levels)
                vt = zeros(prob.N_tot_levels)

                t = 0.0
                dt = prob.tf/prob.nsteps

                # Get forcing (dH/dα * ψ)
                for n in 0:prob.nsteps
                    # Second Order Forcing
                    u .= history[1:prob.N_tot_levels,     1+n, 1, initial_condition_index]
                    v .= history[1+prob.N_tot_levels:end, 1+n, 1, initial_condition_index]

                    utvt!(ut, vt, u, v, grad_prob, grad_controls, t, pcof)

                    forcing_vec2[1:prob.N_tot_levels]     .= ut
                    forcing_vec2[1+prob.N_tot_levels:end] .= vt

                    forcing_ary[:, 1+n, 1, initial_condition_index] .= forcing_vec2

                    # Fourth Order Forcing
                    forcing_vec4 = zeros(prob.real_system_size)

                    utvt!(
                        ut, vt, u, v,
                        grad_prob, grad_time_derivative_controls, t, pcof
                    )
                    forcing_vec4[1:prob.N_tot_levels]     .+= ut
                    forcing_vec4[1+prob.N_tot_levels:end] .+= vt

                    ut .= history[1:prob.N_tot_levels,     1+n, 2, initial_condition_index]
                    vt .= history[1+prob.N_tot_levels:end, 1+n, 2, initial_condition_index]

                    A = zeros(prob.N_tot_levels) # Placeholders
                    B = zeros(prob.N_tot_levels)

                    utvt!(
                        A, B, ut, vt,
                        grad_prob, grad_controls, t, pcof
                    )

                    forcing_vec4[1:prob.N_tot_levels]     .+= A
                    forcing_vec4[1+prob.N_tot_levels:end] .+= B

                    forcing_ary[:, 1+n, 2, initial_condition_index] .= forcing_vec4

                    t += dt
                end
            else 
                throw("Invalid Order: $order")
            end
        end

        # Evolve with forcing
        # Get history of dψ/dα

        # Get history of state vector
        derivative_index = 1 # Don't take derivative
        Q = history[:, end, derivative_index, :]
        dQda = zeros(size(Q)...)

        for initial_condition_index = 1:size(prob.u0,2)
            vec_prob = VectorSchrodingerProb(prob, initial_condition_index)
            timediff_vec_prob = time_diff_prob(vec_prob) 

            history_dQi_da = eval_forward_forced(
                timediff_vec_prob, controls, pcof, forcing_ary[:, :, :, initial_condition_index],
                order=order
            )
            dQda[:,initial_condition_index] .= history_dQi_da[:, end]
        end

        R = copy(target)
        T = vcat(R[1+prob.N_tot_levels:end,:], -R[1:prob.N_tot_levels,:])

        if cost_type == :Infidelity
            gradient[control_param_index] = (dot(Q,R)*dot(dQda,R) + dot(Q,T)*dot(dQda,T))
            gradient[control_param_index] *= -(2/(prob.N_ess_levels^2))
        elseif cost_type == :Tracking
            gradient[control_param_index] = dot(dQda, Q - target)
        elseif cost_type == :Norm
            gradient[control_param_index] = dot(dQda, Q)
        else
            throw("Invalid cost type: $cost_type")
        end
    end

    if return_forcing
        return gradient, forcing_ary
    end

    return gradient
end

function eval_grad_forced_arbitrary_order(prob::SchrodingerProb{M, VM}, controls,
        pcof::AbstractVector{Float64}, target::VM; order=2, 
        cost_type=:Infidelity, return_forcing=false
    ) where {M <: AbstractMatrix{Float64}, VM <: AbstractVecOrMat{Float64}}

    # Allocate space for gradient
    gradient = zeros(controls.N_coeff)

    dt = prob.tf / prob.nsteps
    N_derivatives = div(order, 2)



    ## Compute forcing array (using -i∂H/∂θₖ ψ)

    # Get state vector history
    history = eval_forward_arbitrary_order(prob, controls, pcof, order=order)


    # For storing the global forcing array
    forcing_ary = zeros(prob.real_system_size, N_derivatives, 1+prob.nsteps, size(prob.u0, 2))
    # For storing the per-timestep forcing matrix
    forcing_matrix = zeros(prob.real_system_size, N_derivatives)
    # For storing the per-timestep state vector and derivatives
    uv_matrix = zeros(prob.real_system_size, 1+N_derivatives)

    p_vals = zeros(1+N_derivatives, 1+prob.nsteps, controls.N_coeff)
    q_vals = zeros(1+N_derivatives, 1+prob.nsteps, controls.N_coeff)
    for n in 0:prob.nsteps+1
        t = n*dt
        for derivative_i in 0:N_derivatives
            p_vals[1+derivative_i, 1+n, :] .= eval_grad_p_derivative(control, t, pcof, derivative_i)
            q_vals[1+derivative_i, 1+n, :] .= eval_grad_q_derivative(control, t, pcof, derivative_i)
        end
    end

    for control_param_index in 1:controls.N_coeff # Will only work with single control for now
        # This is bad, want a "global" index, not local. That makes our job
        # easier, since all controls but one become zero (although we need to
        # be careful, since we need to pick the correct control hamiltonian
        # from prob. Having an "apply control" function would be useful here)

        # THIS ONLY WORKS FOR A SINGLE CONTROL PROVIDED
        grad_control = GradControl(controls, control_param_index)
        this_pcof = pcof
        asym_op = prob.asym_operators[1]
        sym_op = prob.sym_operators[1]

        for initial_condition_index = 1:size(prob.u0, 2)
            for n in 0:prob.nsteps
                t = n*dt

                uv_matrix .= history[:, :, 1+n, initial_condition_index]

                for j = 0:(N_derivatives-1)
                    # Get views of the current derivative we are trying to compute (the j+1th derivative)
                    u_derivative = view(forcing_matrix, 1:prob.N_tot_levels,                       1+j)
                    v_derivative = view(forcing_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+j)

                    u_derivative .= 0
                    v_derivative .= 0

                    # Perform the summation (the above is part of the i=j term in summation, this loop completes that term and the rest)
                    for i = j:-1:0
                        #println("j=$j, i=$i, j-i=$(j-i)")
                        u_derivative_prev = view(uv_matrix, 1:prob.N_tot_levels,                       1+i)
                        v_derivative_prev = view(uv_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+i)

                        #p_val = eval_p_derivative(grad_control, t, this_pcof, j-i) / factorial(j-i)
                        #q_val = eval_q_derivative(grad_control, t, this_pcof, j-i) / factorial(j-i)
                        p_val = p_vals[1+j-i,1+n, control_param_index] / factorial(j-i)
                        q_val = q_vals[1+j-i,1+n, control_param_index] / factorial(j-i)

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

        final_state = history[:, 1, end, :]

        # Compute the state history of ∂ψ/∂θₖ

        diff_prob = copy(prob)
        diff_prob.u0 .= 0
        diff_prob.v0 .= 0

        history_partial_derivative = eval_forward_arbitrary_order(
            diff_prob, controls, pcof, forcing=forcing_ary,
            order=order
        )

        final_state_partial_derivative = history_partial_derivative[:, 1, end, :]
        
        # Compute the partial derivative of the objective function with respect to θₖ
        R = copy(target)
        T = vcat(R[1+prob.N_tot_levels:end,:], -R[1:prob.N_tot_levels,:])

        if cost_type == :Infidelity
            gradient[control_param_index]  = dot(final_state, R)*dot(final_state_partial_derivative, R)
            gradient[control_param_index] += dot(final_state, T)*dot(final_state_partial_derivative, T)
            gradient[control_param_index] *= -(2/(prob.N_ess_levels^2))
        elseif cost_type == :Tracking
            gradient[control_param_index] = dot(final_state_partial_derivative, final_state - target)
        elseif cost_type == :Norm
            gradient[control_param_index] = dot(final_state_partial_derivative, final_state)
        else
            throw("Invalid cost type: $cost_type")
        end
    end

    if return_forcing
        return gradient, permutedims(forcing_ary, (1,3,2,4))
    end

    return gradient
end

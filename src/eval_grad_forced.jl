"""
Evaluates gradient of the provided Schrodinger problem with the given target
gate and control parameter(s) pcof using the "forward differentiation" method,
which evolves a differentiated Schrodinger equation, using the state vector
in the evolution of the original Schrodinger equation as a forcing term.

Returns: gradient
"""
function eval_grad_forced(prob::SchrodingerProb{M, VM}, controls,
        pcof::AbstractVector{Float64}, target::VM; order=2, 
        cost_type=:Infidelity
    ) where {M <: AbstractMatrix{Float64}, VM <: AbstractVecOrMat{Float64}}

    # Get state vector history
    history = eval_forward(prob, controls, pcof, order=order, return_time_derivatives=true)

    ## Compute forcing array (using -i∂H/dθₖ ψ)
    # Prepare dH/dα

    gradient = zeros(controls.N_coeff)

    forcing_ary = zeros(prob.real_system_size, 1+prob.nsteps, div(order, 2), size(prob.u0, 2))

    diff_prob = differentiated_prob(prob)


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

                    utvt!(ut, vt, u, v, diff_prob, diff_controls, t, pcof)

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

                    utvt!(ut, vt, u, v, diff_prob, diff_controls, t, pcof)

                    forcing_vec2[1:prob.N_tot_levels]     .= ut
                    forcing_vec2[1+prob.N_tot_levels:end] .= vt

                    forcing_ary[:, 1+n, 1, initial_condition_index] .= forcing_vec2

                    # Fourth Order Forcing
                    forcing_vec4 = zeros(prob.real_system_size)

                    utvt!(
                        ut, vt, u, v,
                        diff_prob, grad_time_derivative_controls, t, pcof
                    )
                    forcing_vec4[1:prob.N_tot_levels]     .+= ut
                    forcing_vec4[1+prob.N_tot_levels:end] .+= vt

                    ut .= history[1:prob.N_tot_levels,     1+n, 2, initial_condition_index]
                    vt .= history[1+prob.N_tot_levels:end, 1+n, 2, initial_condition_index]

                    A = zeros(prob.N_tot_levels) # Placeholders
                    B = zeros(prob.N_tot_levels)

                    utvt!(
                        A, B, ut, vt,
                        diff_prob, grad_controls, t, pcof
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
        timediff_prob = time_diff_prob(prob) 
        # Get history of dψ/dα

        Q = history[:,end,1,:]
        dQda = zeros(size(Q)...)

        for initial_condition_index = 1:size(prob.u0,2)
            vec_prob = VectorSchrodingerProb(prob, initial_condition_index)
            # Set initial condition to zero for evolution of dψ/dα
            vec_prob.u0 .= 0.0
            vec_prob.v0 .= 0.0
            history_dQi_da = eval_forward_forced(
                vec_prob, control, pcof, forcing_ary[:, :, :, initial_condition_index],
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
    return gradient
end

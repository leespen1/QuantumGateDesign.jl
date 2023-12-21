"""
Evaluates gradient of the provided Schrodinger problem with the given target
gate and control parameter(s) pcof using the discrete adjoint method. 

Returns: gradient
"""
function discrete_adjoint(
        prob::SchrodingerProb, controls,
        pcof::AbstractVector{Float64}, target::AbstractMatrix{Float64}; 
        order=2, cost_type=:Infidelity, return_lambda_history=false
    )

    history = eval_forward(prob, controls, pcof; order=order)

    R = target[:,:] # Copy target, converting to matrix if vector
    T = vcat(R[1+prob.N_tot_levels:end,:], -R[1:prob.N_tot_levels,:])

    dt = prob.tf/prob.nsteps

    len_pcof = length(pcof)
    grad = zeros(len_pcof)

    # Set up terminal condition RHS
    if cost_type == :Infidelity
        terminal_RHS = (dot(history[:,end,:],R)*R + dot(history[:,end,:],T)*T)
        terminal_RHS *= (2.0/(prob.N_ess_levels^2))
    elseif cost_type == :Tracking
        terminal_RHS = -(history[:,end,:] - target)
    elseif cost_type == :Norm
        terminal_RHS = -history[:,end,:]
    else
        throw("Invalid cost type: $cost_type")
    end

    full_lambda_history = zeros(prob.real_system_size,1+prob.nsteps, prob.N_ess_levels)

    for initial_condition_index = 1:size(prob.u0,2)
        lambda = zeros(prob.real_system_size)
        lambda_history = zeros(prob.real_system_size,1+prob.nsteps)

        lambda_u   = zeros(prob.N_tot_levels)
        lambda_v   = zeros(prob.N_tot_levels)
        lambda_ut  = zeros(prob.N_tot_levels)
        lambda_vt  = zeros(prob.N_tot_levels)
        lambda_utt = zeros(prob.N_tot_levels)
        lambda_vtt = zeros(prob.N_tot_levels)

        RHS::Vector{Float64} = zeros(prob.real_system_size)

        RHS_lambda_u::Vector{Float64} = zeros(prob.N_tot_levels)
        RHS_lambda_v::Vector{Float64} = zeros(prob.N_tot_levels)


        t = prob.tf
        RHS .= terminal_RHS[:,initial_condition_index]

        if order == 2

            function LHS_func_adj_wrapper_order2(lambda_out::AbstractVector{Float64}, lambda_in::AbstractVector{Float64})
                copyto!(lambda_u, 1, lambda_in, 1,                   prob.N_tot_levels)
                copyto!(lambda_v, 1, lambda_in, 1+prob.N_tot_levels, prob.N_tot_levels)
                LHS_func_adj!(
                    lambda_out, lambda_ut, lambda_vt, lambda_u, lambda_v, 
                    prob, controls, t, pcof, dt, prob.N_tot_levels
                )

                return nothing
            end

            LHS_map = LinearMaps.LinearMap(
                LHS_func_adj_wrapper_order2,
                prob.real_system_size, prob.real_system_size,
                ismutating=true
            )

            # Terminal Condition
            IterativeSolvers.gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

            lambda_history[:,1+prob.nsteps] .= lambda
            copyto!(lambda_u, 1, lambda, 1,                   prob.N_tot_levels)
            copyto!(lambda_v, 1, lambda, 1+prob.N_tot_levels, prob.N_tot_levels)
            

            # Discrete Adjoint Scheme
            for n in prob.nsteps-1:-1:1
                t -= dt
                utvt_adj!(
                    lambda_ut, lambda_vt, lambda_u, lambda_v, prob, controls,
                    t, pcof
                )
                copy!(RHS_lambda_u, lambda_u)
                axpy!(0.5*dt, lambda_ut, RHS_lambda_u)

                copy!(RHS_lambda_v, lambda_v)
                axpy!(0.5*dt, lambda_vt, RHS_lambda_v)

                copyto!(RHS,1,                   RHS_lambda_u, 1, prob.N_tot_levels)
                copyto!(RHS,1+prob.N_tot_levels, RHS_lambda_v, 1, prob.N_tot_levels)

                IterativeSolvers.gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

                lambda_history[:,1+n] .= lambda
                copyto!(lambda_u, 1, lambda, 1,                   prob.N_tot_levels)
                copyto!(lambda_v, 1, lambda, 1+prob.N_tot_levels, prob.N_tot_levels)
            end

            disc_adj_calc_grad!(grad, prob, controls, pcof,
                view(history, :, :, initial_condition_index), lambda_history
            )
            #disc_adj_calc_grad_naive!(grad, prob, controls, pcof,
            #    view(history, :, :, initial_condition_index), lambda_history,
            #    order=order
            #)
        


        elseif order == 4

            function LHS_func_adj_wrapper_order4(lambda_out::AbstractVector{Float64}, lambda_in::AbstractVector{Float64})
                copyto!(lambda_u, 1, lambda_in, 1,                   prob.N_tot_levels)
                copyto!(lambda_v, 1, lambda_in, 1+prob.N_tot_levels, prob.N_tot_levels)
                LHS_func_order4_adj!(
                    lambda_out, lambda_utt, lambda_vtt, lambda_ut, lambda_vt, lambda_u, lambda_v,
                    prob, controls, t, pcof, dt, prob.N_tot_levels
                )

                return nothing
            end

            LHS_map = LinearMaps.LinearMap(
                LHS_func_adj_wrapper_order4,
                prob.real_system_size, prob.real_system_size,
                ismutating=true
            )

            # Terminal Condition
            IterativeSolvers.gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

            lambda_history[:,1+prob.nsteps] .= lambda
            copyto!(lambda_u, 1, lambda, 1,                   prob.N_tot_levels)
            copyto!(lambda_v, 1, lambda, 1+prob.N_tot_levels, prob.N_tot_levels)


            # Discrete Adjoint Scheme
            weights = (1, 1/3)
            for n in prob.nsteps-1:-1:1
                t -= dt
                utvt_adj!(lambda_ut, lambda_vt, lambda_u, lambda_v, prob, controls, t, pcof)

                uttvtt_adj!(
                    lambda_utt, lambda_vtt, lambda_ut, lambda_vt, lambda_u, lambda_v,
                    prob, controls, t, pcof
                )

                copy!(RHS_lambda_u, lambda_u)
                axpy!(0.5*dt*weights[1],     lambda_ut,  RHS_lambda_u)
                axpy!(0.25*dt*dt*weights[2], lambda_utt, RHS_lambda_u)

                copy!(RHS_lambda_v, lambda_v)
                axpy!(0.5*dt*weights[1],     lambda_vt,  RHS_lambda_v)
                axpy!(0.25*dt*dt*weights[2], lambda_vtt, RHS_lambda_v)

                copyto!(RHS, 1,                   RHS_lambda_u, 1, prob.N_tot_levels)
                copyto!(RHS, 1+prob.N_tot_levels, RHS_lambda_v, 1, prob.N_tot_levels)

                # NOTE: LHS and RHS Linear transformations use the SAME TIME

                IterativeSolvers.gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

                lambda_history[:, 1+n] .= lambda
                copyto!(lambda_u, 1, lambda, 1,                   prob.N_tot_levels)
                copyto!(lambda_v, 1, lambda, 1+prob.N_tot_levels, prob.N_tot_levels)
            end

            disc_adj_calc_grad_order_4!(grad, prob, controls, pcof,
                view(history, :, :, initial_condition_index), lambda_history
            )
            #disc_adj_calc_grad_naive!(grad, prob, controls, pcof,
            #    view(history, :, :, initial_condition_index), lambda_history,
            #    order=order
            #)


        else
            throw("Invalid order: $order")
        end

        full_lambda_history[:,:,initial_condition_index] .= lambda_history
    end

    if return_lambda_history
        return grad, history, full_lambda_history
    end
    return grad
end

"""
Need better name
"""
function disc_adj_calc_grad!(gradient::AbstractVector{Float64},
        prob::SchrodingerProb, controls, pcof::AbstractVector{Float64},
        history::AbstractMatrix{Float64}, lambda_history::AbstractMatrix{Float64})

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
        this_pcof = get_control_vector_slice(pcof, controls, i)

        grad_contrib = zeros(control.N_coeff)
        grad_p = zeros(control.N_coeff)
        grad_q = zeros(control.N_coeff)

        for n in 0:prob.nsteps-1
            lambda_u .= @view lambda_history[1:prob.N_tot_levels,     1+n+1]
            lambda_v .= @view lambda_history[1+prob.N_tot_levels:end, 1+n+1]

            u .= @view history[1:prob.N_tot_levels,     1+n]
            v .= @view history[1+prob.N_tot_levels:end, 1+n]
            t = n*dt

            grad_p .= eval_grad_p(control, t, this_pcof)
            grad_q .= eval_grad_q(control, t, this_pcof)

            mul!(asym_op_lambda_u, asym_op, lambda_u)
            mul!(asym_op_lambda_v, asym_op, lambda_v)
            mul!(sym_op_lambda_u,  sym_op,  lambda_u)
            mul!(sym_op_lambda_v,  sym_op,  lambda_v)


            grad_contrib .+= grad_q .* -(dot(u, asym_op_lambda_u) + dot(v, asym_op_lambda_v))
            grad_contrib .+= grad_p .* (-dot(u, sym_op_lambda_v) + dot(v, sym_op_lambda_u))

            u .= @view history[1:prob.N_tot_levels,     1+n+1]
            v .= @view history[1+prob.N_tot_levels:end, 1+n+1]
            t = (n+1)*dt

            grad_p .= eval_grad_p(control, t, this_pcof)
            grad_q .= eval_grad_q(control, t, this_pcof)

            grad_contrib .+= grad_q .* -(dot(u, asym_op_lambda_u) + dot(v, asym_op_lambda_v))
            grad_contrib .+= grad_p .* (-dot(u, sym_op_lambda_v) + dot(v, sym_op_lambda_u))
        end

        grad_contrib .*= -0.5*dt

        grad_slice = get_control_vector_slice(gradient, controls, i)
        grad_slice .+= grad_contrib
    end

    return nothing
end


function disc_adj_calc_grad_order_4!(gradient::AbstractVector{Float64}, prob::SchrodingerProb, controls,
        pcof::AbstractVector{Float64},
        history::AbstractMatrix{Float64}, lambda_history::AbstractMatrix{Float64},
        )

    dt = prob.tf / prob.nsteps

    u = zeros(prob.N_tot_levels)
    v = zeros(prob.N_tot_levels)
    ut = zeros(prob.N_tot_levels)
    vt = zeros(prob.N_tot_levels)

    lambda_u = zeros(prob.N_tot_levels)
    lambda_v = zeros(prob.N_tot_levels)

    sym_op_u = zeros(prob.N_tot_levels)
    sym_op_v = zeros(prob.N_tot_levels)
    asym_op_u = zeros(prob.N_tot_levels)
    asym_op_v = zeros(prob.N_tot_levels)

    sym_op_ut = zeros(prob.N_tot_levels)
    sym_op_vt = zeros(prob.N_tot_levels)
    asym_op_ut = zeros(prob.N_tot_levels)
    asym_op_vt = zeros(prob.N_tot_levels)

    sym_op_lambda_u = zeros(prob.N_tot_levels)
    sym_op_lambda_v = zeros(prob.N_tot_levels)
    asym_op_lambda_u = zeros(prob.N_tot_levels)
    asym_op_lambda_v = zeros(prob.N_tot_levels)


    A = zeros(prob.N_tot_levels)
    B = zeros(prob.N_tot_levels)
    C = zeros(prob.N_tot_levels)

    Hq = zeros(prob.N_tot_levels, prob.N_tot_levels)
    Hp = zeros(prob.N_tot_levels, prob.N_tot_levels)

    len_pcof = length(pcof)

    # NOTE: Revising this for multiple controls will require some thought
    for i in 1:prob.N_operators
        control = controls[i]
        asym_op = prob.asym_operators[i]
        sym_op = prob.sym_operators[i]

        this_pcof = get_control_vector_slice(pcof, controls, i)

        grad_contrib = zeros(control.N_coeff)
        grad_p = zeros(control.N_coeff)
        grad_q = zeros(control.N_coeff)

        # Accumulate Gradient
        # Efficient way, possibly incorrect
        weights_n   = (1, dt/6) 
        weights_np1 = (1, -dt/6)
        for n in 0:prob.nsteps-1
            lambda_u .= lambda_history[1:prob.N_tot_levels,     1+n+1]
            lambda_v .= lambda_history[1+prob.N_tot_levels:end, 1+n+1]

            mul!(asym_op_lambda_u, asym_op, lambda_u)
            mul!(asym_op_lambda_v, asym_op, lambda_v)
            mul!(sym_op_lambda_u,  sym_op,  lambda_u)
            mul!(sym_op_lambda_v,  sym_op,  lambda_v)

            # Qₙ "Explicit Part" contribution
            u .= view(history, 1:prob.N_tot_levels,                       1+n)
            v .= view(history, 1+prob.N_tot_levels:prob.real_system_size, 1+n)
            t = n*dt

            #FIXME THIS IS THE PROBLEM!!! I was write to use product rule, but
            #the hamiltonians which do not have partial derivatives taken
            #should use ALL controls.

            grad_p  = eval_grad_p(control,  t, this_pcof)
            grad_q  = eval_grad_q(control,  t, this_pcof)
            grad_pt = eval_grad_pt(control, t, this_pcof)
            grad_qt = eval_grad_qt(control, t, this_pcof)

            # Do contribution of ⟨w,∂H/∂θₖᵀλ⟩
            # Part involving asym op (q)
            grad_contrib .+= grad_q .* weights_n[1]*(
                -dot(u, asym_op_lambda_u) - dot(v, asym_op_lambda_v)
            )
            # Part involving sym op (p)
            grad_contrib .+= grad_p .* weights_n[1]*(
                -dot(u, sym_op_lambda_v) + dot(v, sym_op_lambda_u)
            )

            # Do contribution of ⟨w,∂Hₜ/∂θₖᵀλ⟩
            # Part involving asym op (q)
            grad_contrib .+= grad_qt .* weights_n[2]*(
                -dot(u, asym_op_lambda_u) - dot(v, asym_op_lambda_v)
            )
            # Part involving sym op (p)
            grad_contrib .+= grad_pt .* weights_n[2]*(
                -dot(u, sym_op_lambda_v) + dot(v, sym_op_lambda_u)
            )
            

            # Do contribution of ⟨∂H/∂θₖ Hw,λ⟩
            # Apply Hamiltonian
            utvt!(
                ut, vt, u, v, 
                prob, controls, t, pcof
            )
            # Do matrix part of ∂H/∂θₖ, so I can "factor out" the scalar function
            mul!(sym_op_ut, sym_op, ut)
            mul!(asym_op_ut, asym_op, ut)
            mul!(sym_op_vt, sym_op, vt)
            mul!(asym_op_vt, asym_op, vt)

            grad_contrib .+= grad_q .* weights_n[2]*dot(asym_op_ut, lambda_u)
            grad_contrib .+= grad_p .* weights_n[2]*dot(sym_op_vt, lambda_u)
            grad_contrib .-= grad_p .* weights_n[2]*dot(sym_op_ut, lambda_v)
            grad_contrib .+= grad_q .* weights_n[2]*dot(asym_op_vt, lambda_v)


            # Do contribution of ⟨H ∂H/∂θₖw,λ⟩
            mul!(asym_op_u, asym_op, u)
            mul!(asym_op_v, asym_op, v)

            # ut and vt are not actually holding ut and vt.
            utvt!(
                ut, vt, asym_op_u, asym_op_v, 
                prob, controls, t, pcof
            )
            grad_contrib .+= grad_q .* weights_n[2]*dot(ut, lambda_u)
            grad_contrib .+= grad_q .* weights_n[2]*dot(vt, lambda_v)

            mul!(sym_op_v, sym_op, v)
            mul!(sym_op_u, sym_op, u) 
            sym_op_u .*= -1

            # ut and vt gets confusing here. But they are just the output of
            # applying the hamiltonian
            #
            # Possibly these should be minuses. I think this is correct, but I
            # may have misfactored
            utvt!(
                ut, vt, sym_op_v, sym_op_u, 
                prob, controls, t, pcof
            )
            grad_contrib .+= grad_p .* weights_n[2]*dot(ut, lambda_u)
            grad_contrib .+= grad_p .* weights_n[2]*dot(vt, lambda_v)
            

            #####################
            # Qₙ₊₁ "Implicit Part" contribution
            #####################

            u .= view(history, 1:prob.N_tot_levels,                       1+n+1)
            v .= view(history, 1+prob.N_tot_levels:prob.real_system_size, 1+n+1)
            t = (n+1)*dt

            grad_p  = eval_grad_p(control,  t, this_pcof)
            grad_q  = eval_grad_q(control,  t, this_pcof)
            grad_pt = eval_grad_pt(control, t, this_pcof)
            grad_qt = eval_grad_qt(control, t, this_pcof)

            # Do contribution of ⟨w,∂H/∂θₖᵀλ⟩
            # Part involving asym op (q)
            grad_contrib .+= grad_q .* weights_np1[1]*(
                -dot(u, asym_op_lambda_u) - dot(v, asym_op_lambda_v)
            )
            # Part involving sym op (p)
            grad_contrib .+= grad_p .* weights_np1[1]*(
                -dot(u, sym_op_lambda_v) + dot(v, sym_op_lambda_u)
            )

            # Do contribution of ⟨w,∂Hₜ/∂θₖᵀλ⟩
            # Part involving asym op (q)
            grad_contrib .+= grad_qt .* weights_np1[2]*(
                -dot(u, asym_op_lambda_u) - dot(v, asym_op_lambda_v)
            )
            # Part involving sym op (p)
            grad_contrib .+= grad_pt .* weights_np1[2]*(
                -dot(u, sym_op_lambda_v) + dot(v, sym_op_lambda_u)
            )
            

            # Do contribution of ⟨∂H/∂θₖ Hw,λ⟩
            # Apply Hamiltonian
            utvt!(
                ut, vt, u, v, 
                prob, controls, t, pcof
            )
            # Do matrix part of ∂H/∂θₖ, so I can "factor out" the scalar function
            mul!(sym_op_ut, sym_op, ut)
            mul!(asym_op_ut, asym_op, ut)
            mul!(sym_op_vt, sym_op, vt)
            mul!(asym_op_vt, asym_op, vt)

            grad_contrib .+= grad_q .* weights_np1[2]*dot(asym_op_ut, lambda_u)
            grad_contrib .+= grad_p .* weights_np1[2]*dot(sym_op_vt, lambda_u)
            grad_contrib .-= grad_p .* weights_np1[2]*dot(sym_op_ut, lambda_v)
            grad_contrib .+= grad_q .* weights_np1[2]*dot(asym_op_vt, lambda_v)


            # Do contribution of ⟨H ∂H/∂θₖw,λ⟩
            mul!(asym_op_u, asym_op, u)
            mul!(asym_op_v, asym_op, v)

            # ut and vt are not actuall holding ut and vt.
            utvt!(
                ut, vt, asym_op_u, asym_op_v, 
                prob, controls, t, pcof
            )
            grad_contrib .+= grad_q .* weights_np1[2]*dot(ut, lambda_u)
            grad_contrib .+= grad_q .* weights_np1[2]*dot(vt, lambda_v)

            mul!(sym_op_v, sym_op, v)
            mul!(sym_op_u, sym_op, u) 
            sym_op_u .*= -1

            # ut and vt gets confusing here. But they are just the output of
            # applying the hamiltonian
            #
            # Possibly these should be minuses. I think this is correct, but I
            # may have misfactored
            utvt!(
                ut, vt, sym_op_v, sym_op_u, 
                prob, controls, t, pcof
            )
            grad_contrib .+= grad_p .* weights_np1[2]*dot(ut, lambda_u)
            grad_contrib .+= grad_p .* weights_np1[2]*dot(vt, lambda_v)

        end

        grad_slice = get_control_vector_slice(gradient, controls, i)
        grad_slice .+= (-0.5*dt) .* grad_contrib
    end

    return nothing
end


"""
A simpler version, but less prone to mistakes by the programmer.

WARNING: I DON'T THINK THIS WORKS FOR 4TH ORDER YET
"""
function disc_adj_calc_grad_naive!(gradient::AbstractVector{Float64},
        prob::SchrodingerProb, controls, pcof::AbstractVector{Float64},
        history::AbstractMatrix{Float64}, lambda_history::AbstractMatrix{Float64};
        order=2
    )

    dt = prob.tf / prob.nsteps

    # "Dummy" system operator, since the partial derivative takes them to zero
    zero_system_op = zeros(prob.N_tot_levels, prob.N_tot_levels)

    u_n = zeros(prob.N_tot_levels)
    v_n = zeros(prob.N_tot_levels)
    u_np1 = zeros(prob.N_tot_levels)
    v_np1 = zeros(prob.N_tot_levels)
    ut = zeros(prob.N_tot_levels)
    vt = zeros(prob.N_tot_levels)
    utt = zeros(prob.N_tot_levels)
    vtt = zeros(prob.N_tot_levels)

    left_inner_prod = zeros(prob.real_system_size)
    lambda_np1 = zeros(prob.real_system_size)

    uv = zeros(prob.real_system_size)
    lambda_uv = zeros(prob.real_system_size)

    for i in 1:prob.N_operators
        control = controls[i]
        asym_op = prob.asym_operators[i]
        sym_op = prob.sym_operators[i]
        this_pcof = get_control_vector_slice(pcof, controls, i)

        grad_contrib = zeros(control.N_coeff)

        grad_p_n = zeros(control.N_coeff)
        grad_q_n = zeros(control.N_coeff)
        grad_pt_n = zeros(control.N_coeff)
        grad_qt_n = zeros(control.N_coeff)

        grad_p_np1 = zeros(control.N_coeff)
        grad_q_np1 = zeros(control.N_coeff)
        grad_pt_np1 = zeros(control.N_coeff)
        grad_qt_np1 = zeros(control.N_coeff)

        if (order == 2)
            weights_n = 1
            weights_np1 = 1
        elseif (order == 4)
            weights_n = (1, 1/3)
            weights_np1 = (1, -1/3)
        else
            throw("Invalid Order: $order")
        end


        for n in 0:prob.nsteps-1

            lambda_np1 .= lambda_history[:, 1+n+1]

            u_n .= @view history[1:prob.N_tot_levels,     1+n]
            v_n .= @view history[1+prob.N_tot_levels:end, 1+n]
            t_n = n*dt

            p_val_n = eval_p(control, t_n, this_pcof)
            q_val_n = eval_q(control, t_n, this_pcof)
            grad_p_n .= eval_grad_p(control, t_n, this_pcof)
            grad_q_n .= eval_grad_q(control, t_n, this_pcof)
            grad_pt_n .= eval_grad_pt(control, t_n, this_pcof)
            grad_qt_n .= eval_grad_qt(control, t_n, this_pcof)

            u_np1 .= @view history[1:prob.N_tot_levels,     1+n+1]
            v_np1 .= @view history[1+prob.N_tot_levels:end, 1+n+1]
            t_np1 = (n+1)*dt

            p_val_np1 = eval_p(control, t_np1, this_pcof)
            q_val_np1 = eval_q(control, t_np1, this_pcof)
            grad_p_np1 .= eval_grad_p(control, t_np1, this_pcof)
            grad_q_np1 .= eval_grad_q(control, t_np1, this_pcof)
            grad_pt_np1 .= eval_grad_pt(control, t_np1, this_pcof)
            grad_qt_np1 .= eval_grad_qt(control, t_np1, this_pcof)

            for i in 1:length(grad_contrib)

                # Do the "explicit" part

                # (∂H/∂θₖ)ψ
                utvt!(
                    ut, vt, u_n, v_n, 
                    zero_system_op, zero_system_op, sym_op, asym_op,
                    grad_p_n[i], grad_q_n[i]
                )
                left_inner_prod[1:prob.N_tot_levels]     .= (0.5*dt*weights_n[1]) .* ut
                left_inner_prod[1+prob.N_tot_levels:end] .= (0.5*dt*weights_n[1]) .* vt

                # Fourth order contribution
                if (order == 4)
                    # (∂H/∂θₖ)Hψ
                    utvt!(
                        ut, vt, u_n, v_n, 
                        prob.system_sym, prob.system_asym, sym_op, asym_op,
                        p_val_n, q_val_n
                    )
                    utvt!(
                        utt, vtt, ut, ut, 
                        zero_system_op, zero_system_op, sym_op, asym_op,
                        grad_p_n[i], grad_q_n[i]
                    )

                    left_inner_prod[1:prob.N_tot_levels]     .+= (0.25*dt*dt*weights_n[2]) .* utt
                    left_inner_prod[1+prob.N_tot_levels:end] .+= (0.25*dt*dt*weights_n[2]) .* vtt
                    
                    # H(∂H/∂θₖ)ψ
                    utvt!(
                        ut, vt, u_n, v_n, 
                        zero_system_op, zero_system_op, sym_op, asym_op,
                        grad_p_n[i], grad_q_n[i]
                    )
                    utvt!(
                        utt, vtt, ut, ut, 
                        prob.system_sym, prob.system_asym, sym_op, asym_op,
                        p_val_n, q_val_n,
                    )

                    left_inner_prod[1:prob.N_tot_levels]     .+= (0.25*dt*dt*weights_n[2]) .* utt
                    left_inner_prod[1+prob.N_tot_levels:end] .+= (0.25*dt*dt*weights_n[2]) .* vtt

                    # (∂Hₜ/∂θₖ)ψ
                    utvt!(
                        ut, vt, u_n, v_n, 
                        zero_system_op, zero_system_op, sym_op, asym_op,
                        grad_pt_n[i], grad_qt_n[i]
                    )

                    left_inner_prod[1:prob.N_tot_levels]     .+= (0.25*dt*dt*weights_n[2]) .* ut
                    left_inner_prod[1+prob.N_tot_levels:end] .+= (0.25*dt*dt*weights_n[2]) .* vt
                end

                # Do the "implicit" part

                # (∂H/∂θₖ)ψ
                utvt!(
                    ut, vt, u_np1, v_np1, 
                    zero_system_op, zero_system_op, sym_op, asym_op,
                    grad_p_np1[i], grad_q_np1[i]
                )
                left_inner_prod[1:prob.N_tot_levels]     .+= (0.5*dt*weights_np1[1]) .* ut
                left_inner_prod[1+prob.N_tot_levels:end] .+= (0.5*dt*weights_np1[1]) .* vt

                # Fourth order contribution
                if (order == 4)
                    @warn "I DON'T THINK THIS WORKS FOR 4TH ORDER YET"
                    # (∂H/∂θₖ)Hψ
                    utvt!(
                        ut, vt, u_np1, v_np1, 
                        prob.system_sym, prob.system_asym, sym_op, asym_op,
                        p_val_np1, q_val_np1
                    )
                    utvt!(
                        utt, vtt, ut, ut, 
                        zero_system_op, zero_system_op, sym_op, asym_op,
                        grad_p_np1[i], grad_q_np1[i]
                    )

                    left_inner_prod[1:prob.N_tot_levels]     .+= (0.25*dt*dt*weights_np1[2]) .* utt
                    left_inner_prod[1+prob.N_tot_levels:end] .+= (0.25*dt*dt*weights_np1[2]) .* vtt
                    
                    # H(∂H/∂θₖ)ψ
                    utvt!(
                        ut, vt, u_np1, v_np1, 
                        zero_system_op, zero_system_op, sym_op, asym_op,
                        grad_p_np1[i], grad_q_np1[i]
                    )
                    utvt!(
                        utt, vtt, ut, ut, 
                        prob.system_sym, prob.system_asym, sym_op, asym_op,
                        p_val_np1, q_val_np1,
                    )

                    left_inner_prod[1:prob.N_tot_levels]     .+= (0.25*dt*dt*weights_np1[2]) .* utt
                    left_inner_prod[1+prob.N_tot_levels:end] .+= (0.25*dt*dt*weights_np1[2]) .* vtt

                    # (∂Hₜ/∂θₖ)ψ
                    utvt!(
                        ut, vt, u_np1, v_np1, 
                        zero_system_op, zero_system_op, sym_op, asym_op,
                        grad_pt_np1[i], grad_qt_np1[i]
                    )

                    left_inner_prod[1:prob.N_tot_levels]     .+= (0.25*dt*dt*weights_np1[2]) .* ut
                    left_inner_prod[1+prob.N_tot_levels:end] .+= (0.25*dt*dt*weights_np1[2]) .* vt
                end

                grad_contrib[i] -= dot(left_inner_prod, lambda_np1)
            end
        end

        grad_slice = get_control_vector_slice(gradient, controls, i)
        grad_slice .+= grad_contrib
    end

    return nothing
end


function compute_terminal_condition(
        prob, controls, pcof, target, final_state;
        order=2, cost_type=:Infidelity
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

    function LHS_func_wrapper(uv_out::AbstractVector{Float64}, uv_in::AbstractVector{Float64})
        uv_mat[:,1] .= uv_in
        arbitrary_order_uv_derivative!(uv_mat, prob, controls, t, pcof, N_derivatives, use_adjoint=true)
        # Do I need to make and adjoint version of this? I don't think so, considering before LHS only used adjoint for utvt!, not the quadrature
        # But maybe there is a negative t I need to worry about. Maybe just provide dt as -dt
        arbitrary_LHS!(uv_out, uv_mat, dt, N_derivatives)

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
function discrete_adjoint_arbitrary_order(
        prob::SchrodingerProb{<: AbstractMatrix{Float64}, <: AbstractMatrix{Float64}},
        controls, pcof::AbstractVector{Float64},
        target::AbstractMatrix{Float64}; 
        order=2, cost_type=:Infidelity, return_lambda_history=false
    )

    history = eval_forward_arbitrary_order(prob, controls, pcof; order=order)

    R = target[:,:] # Copy target, converting to matrix if vector
    T = vcat(R[1+prob.N_tot_levels:end,:], -R[1:prob.N_tot_levels,:])

    final_state = history[:,1,end,:]

    terminal_condition = compute_terminal_condition(
        prob, controls, pcof, target, final_state, order=order, cost_type=cost_type
    )

    lambda_history = eval_adjoint_arbitrary_order(prob, controls, pcof, terminal_condition, order=order)

    grad = zeros(length(pcof))

    this_history = zeros(prob.real_system_size, 1+prob.nsteps)
    this_lambda_history = zeros(prob.real_system_size, 1+prob.nsteps)

    for initial_condition_index = 1:size(prob.u0,2)

        # Get state vector and lambda history in format expected by the functions (the old format)
        for component_i in 1:prob.real_system_size
            for step_i in 1:(1+prob.nsteps)
                this_history[component_i, step_i] = history[component_i, 1, step_i, initial_condition_index]
                this_lambda_history[component_i, step_i] = lambda_history[component_i, 1, step_i, initial_condition_index]
            end
        end


        if (order == 2)
            disc_adj_calc_grad!(grad, prob, controls, pcof,
                this_history, this_lambda_history
            )
        
        elseif (order == 4)
            disc_adj_calc_grad_order_4!(grad, prob, controls, pcof,
                this_history, this_lambda_history
            )

        else
            throw("Invalid order: $order (must be ≤ 4 for now)")
        end

    end

    return grad
end


"""
Evaluates gradient of the provided Schrodinger problem with the given target
gate and control parameter(s) pcof using the discrete adjoint method. 

Returns: gradient
"""
function discrete_adjoint(
        prob::SchrodingerProb, control::AbstractControl,
        pcof::AbstractVector{Float64}, target::AbstractMatrix{Float64}; 
        order=2, cost_type=:Infidelity, return_lambda_history=false
    )

    history = eval_forward(prob, control, pcof; order=order)

    R = target[:,:] # Copy target, converting to matrix if vector
    T = vcat(R[1+prob.N_tot_levels:end,:], -R[1:prob.N_tot_levels,:])

    # For adjoint evolution. Take transpose of entire matrix -> antisymmetric blocks change sign
    Ks_adj = Matrix(transpose(-prob.Ks)) # NOTE THAT K changes sign!
    Ss_adj = Matrix(transpose(prob.Ss))
    p_operator_adj = Matrix(transpose(-prob.p_operator)) # NOTE THAT K changes sign!
    q_operator_adj = Matrix(transpose(prob.q_operator))
    
    p_operator_transpose = Matrix(transpose(prob.p_operator)) # NOTE THAT K changes sign!
    q_operator_transpose = Matrix(transpose(prob.q_operator))

    dt = prob.tf/prob.nsteps

    len_pcof = length(pcof)
    len_pcof_half = div(len_pcof, 2)
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

    for initial_condition_index = 1:size(prob.u0,2)
        lambda = zeros(2*prob.N_tot_levels)
        lambda_history = zeros(2*prob.N_tot_levels,1+prob.nsteps)
        lambda_ut  = zeros(prob.N_tot_levels)
        lambda_vt  = zeros(prob.N_tot_levels)
        lambda_utt = zeros(prob.N_tot_levels)
        lambda_vtt = zeros(prob.N_tot_levels)

        RHS_lambda_u::Vector{Float64} = zeros(prob.N_tot_levels)
        RHS_lambda_v::Vector{Float64} = zeros(prob.N_tot_levels)
        RHS::Vector{Float64} = zeros(2*prob.N_tot_levels)

        grad_contrib = zeros(len_pcof)


        if order == 2
            # Terminal Condition
            t = prob.tf

            RHS .= terminal_RHS[:,initial_condition_index]

            function LHS_func_wrapper_order2(x::AbstractVector{Float64})::Vector{Float64}
                return LHS_func(
                    lambda_ut, lambda_vt, x[1:prob.N_tot_levels],
                    x[1+prob.N_tot_levels:end], Ks_adj, Ss_adj, p_operator_adj,
                    q_operator_adj, control, t, pcof, dt,
                    prob.N_tot_levels
                )
            end

            LHS_map = LinearMaps.LinearMap(
                LHS_func_wrapper_order2,
                2*prob.N_tot_levels, 2*prob.N_tot_levels
            )

            IterativeSolvers.gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

            lambda_history[:,1+prob.nsteps] .= lambda
            lambda_u = copy(lambda[1:prob.N_tot_levels])
            lambda_v = copy(lambda[1+prob.N_tot_levels:2*prob.N_tot_levels])
            
            # Discrete Adjoint Scheme
            for n in prob.nsteps-1:-1:1
                t -= dt
                utvt!(
                    lambda_ut, lambda_vt, lambda_u, lambda_v, Ks_adj, Ss_adj,
                    p_operator_adj, q_operator_adj, control,
                    t, pcof
                )
                copy!(RHS_lambda_u,lambda_u)
                axpy!(0.5*dt,lambda_ut,RHS_lambda_u)

                copy!(RHS_lambda_v,lambda_v)
                axpy!(0.5*dt,lambda_vt,RHS_lambda_v)

                copyto!(RHS,1,RHS_lambda_u,1,prob.N_tot_levels)
                copyto!(RHS,1+prob.N_tot_levels,RHS_lambda_v,1,prob.N_tot_levels)

                IterativeSolvers.gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

                lambda_history[:,1+n] .= lambda
                lambda_u = lambda[1:prob.N_tot_levels]
                lambda_v = lambda[1+prob.N_tot_levels:end]
            end
        
            zero_mat = zeros(prob.N_tot_levels,prob.N_tot_levels)
            uv = zeros(2*prob.N_tot_levels)
            u = zeros(prob.N_tot_levels)
            v = zeros(prob.N_tot_levels)
            # Won't actually hold ut and vt, but rather real/imag parts of dH/dα*ψ
            ut = zeros(prob.N_tot_levels)
            vt = zeros(prob.N_tot_levels)
            dummy_u = zeros(prob.N_tot_levels)
            dummy_v = zeros(prob.N_tot_levels)
            dummy = zeros(2*prob.N_tot_levels)

            MT_lambda_11 = zeros(prob.N_tot_levels)
            MT_lambda_12 = zeros(prob.N_tot_levels)
            MT_lambda_21 = zeros(prob.N_tot_levels)
            MT_lambda_22 = zeros(prob.N_tot_levels)

            for n in 0:prob.nsteps-1
                lambda_u = lambda_history[1:prob.N_tot_levels,1+n+1]
                lambda_v = lambda_history[1+prob.N_tot_levels:end,1+n+1]

                u = history[1:prob.N_tot_levels,     1+n, initial_condition_index]
                v = history[1+prob.N_tot_levels:end, 1+n, initial_condition_index]
                t = n*dt

                grad_p = eval_grad_p(control, t, pcof)
                grad_q = eval_grad_q(control, t, pcof)

                mul!(MT_lambda_11, q_operator_transpose, lambda_u)
                mul!(MT_lambda_12, p_operator_transpose, lambda_v)
                mul!(MT_lambda_21, p_operator_transpose, lambda_u)
                mul!(MT_lambda_22, q_operator_transpose, lambda_v)

                grad_contrib .+= grad_p .* (dot(u, MT_lambda_12) - dot(v, MT_lambda_21))
                grad_contrib .+= grad_q .* (dot(u, MT_lambda_11) + dot(v, MT_lambda_22))

                u = history[1:prob.N_tot_levels,     1+n+1, initial_condition_index]
                v = history[1+prob.N_tot_levels:end, 1+n+1, initial_condition_index]
                t = (n+1)*dt

                grad_p = eval_grad_p(control, t, pcof)
                grad_q = eval_grad_q(control, t, pcof)

                grad_contrib .+= grad_p .* (dot(u, MT_lambda_12) - dot(v, MT_lambda_21))
                grad_contrib .+= grad_q .* (dot(u, MT_lambda_11) + dot(v, MT_lambda_22))
            end

            grad .+=  (-0.5*dt) .* grad_contrib

        elseif order == 4
            # Terminal Condition
            t = prob.tf

            RHS .= terminal_RHS[:,initial_condition_index]

            function LHS_func_wrapper_order4(x::AbstractVector{Float64})::Vector{Float64}
                return LHS_func_order4(
                    lambda_utt, lambda_vtt, lambda_ut, lambda_vt,
                    x[1:prob.N_tot_levels], x[1+prob.N_tot_levels:end],
                    Ks_adj, Ss_adj, p_operator_adj, q_operator_adj,
                    control, t,
                    pcof, dt, prob.N_tot_levels
                )
            end

            LHS_map = LinearMaps.LinearMap(
                LHS_func_wrapper_order4,
                2*prob.N_tot_levels, 2*prob.N_tot_levels
            )

            IterativeSolvers.gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

            lambda_history[:,1+prob.nsteps] .= lambda
            lambda_u = copy(lambda[1:prob.N_tot_levels])
            lambda_v = copy(lambda[1+prob.N_tot_levels:end])
            
            weights = [1,1/3]
            # Discrete Adjoint Scheme
            for n in prob.nsteps-1:-1:1
                t -= dt
                utvt!(
                    lambda_ut, lambda_vt, lambda_u, lambda_v, Ks_adj, Ss_adj,
                    p_operator_adj, q_operator_adj, control,
                    t, pcof
                )
                uttvtt!(
                    lambda_utt, lambda_vtt, lambda_ut, lambda_vt, lambda_u,
                    lambda_v, Ks_adj, Ss_adj, p_operator_adj, q_operator_adj,
                    control, t,
                    pcof
                )

                copy!(RHS_lambda_u,lambda_u)
                axpy!(0.5*dt*weights[1],lambda_ut,RHS_lambda_u)
                axpy!(0.25*dt^2*weights[2],lambda_utt,RHS_lambda_u)

                copy!(RHS_lambda_v,lambda_v)
                axpy!(0.5*dt*weights[1],lambda_vt,RHS_lambda_v)
                axpy!(0.25*dt^2*weights[2],lambda_vtt,RHS_lambda_v)

                copyto!(RHS,1,RHS_lambda_u,1,prob.N_tot_levels)
                copyto!(RHS,1+prob.N_tot_levels,RHS_lambda_v,1,prob.N_tot_levels)

                # NOTE: LHS and RHS Linear transformations use the SAME TIME

                IterativeSolvers.gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

                lambda_history[:,1+n] .= lambda
                lambda_u = lambda[1:prob.N_tot_levels]
                lambda_v = lambda[1+prob.N_tot_levels:2*prob.N_tot_levels]
            end

            zero_mat = zeros(prob.N_tot_levels,prob.N_tot_levels)
            uv = zeros(2*prob.N_tot_levels)
            u = zeros(prob.N_tot_levels)
            v = zeros(prob.N_tot_levels)

            MT_lambda_11 = zeros(prob.N_tot_levels)
            MT_lambda_12 = zeros(prob.N_tot_levels)
            MT_lambda_21 = zeros(prob.N_tot_levels)
            MT_lambda_22 = zeros(prob.N_tot_levels)

            A = zeros(prob.N_tot_levels)
            B = zeros(prob.N_tot_levels)
            C = zeros(prob.N_tot_levels)

            Ap = zeros(prob.N_tot_levels, prob.N_tot_levels)
            Bp = zeros(prob.N_tot_levels, prob.N_tot_levels)
            Am = zeros(prob.N_tot_levels, prob.N_tot_levels)
            Bm = zeros(prob.N_tot_levels, prob.N_tot_levels)

            Cu = zeros(prob.N_tot_levels)
            Cv = zeros(prob.N_tot_levels)
            Du = zeros(prob.N_tot_levels)
            Dv = zeros(prob.N_tot_levels)

            K_full =  prob.Ks .+ prob.p_operator
            S_full =  prob.Ss .+ prob.q_operator

            Hq = zeros(prob.N_tot_levels, prob.N_tot_levels)
            Hp = zeros(prob.N_tot_levels, prob.N_tot_levels)

            # Accumulate Gradient
            len_pcof = length(pcof)
            len_pcof_half = div(len_pcof, 2)

            # Efficient way, possibly incorrect
            weights_n   = [1, dt/6]
            weights_np1 = [1, -dt/6]
            for n in 0:prob.nsteps-1
                lambda_u = lambda_history[1:prob.N_tot_levels,     1+n+1]
                lambda_v = lambda_history[1+prob.N_tot_levels:end, 1+n+1]

                mul!(MT_lambda_11, q_operator_transpose, lambda_u)
                mul!(MT_lambda_12, p_operator_transpose, lambda_v)
                mul!(MT_lambda_21, p_operator_transpose, lambda_u)
                mul!(MT_lambda_22, q_operator_transpose, lambda_v)

                # Qn contribution
                u = history[1:prob.N_tot_levels,     1+n, initial_condition_index]
                v = history[1+prob.N_tot_levels:end, 1+n, initial_condition_index]
                t = n*dt

                grad_p = eval_grad_p(control,   t, pcof)
                grad_q = eval_grad_q(control,   t, pcof)
                grad_pt = eval_grad_pt(control, t, pcof)
                grad_qt = eval_grad_qt(control, t, pcof)

                # H_α
                grad_contrib .+= grad_q .* weights_n[1]*(
                    dot(u, MT_lambda_11) + dot(v, MT_lambda_22)
                )
                grad_contrib .+= grad_p .* weights_n[1]*(
                    dot(u, MT_lambda_12) - dot(v, MT_lambda_21)
                )

                # 4th order Correction
                # H_αt
                grad_contrib .+= grad_qt .* weights_n[2]*(
                    dot(u, MT_lambda_11) + dot(v, MT_lambda_22)
                )
                grad_contrib .+= grad_pt .* weights_n[2]*(
                    dot(u, MT_lambda_12) - dot(v, MT_lambda_21)
                )
                
                # H_α*H
                # part 1
                Hq .= prob.Ss .+ eval_q(control, t, pcof) .* prob.q_operator
                Hp .= prob.Ks .+ eval_p(control, t, pcof) .* prob.p_operator

                mul!(A, Hq, u)
                mul!(A, Hp, v, -1, 1)
                mul!(B, prob.q_operator, A)

                grad_contrib .+= grad_q .* weights_n[2]*dot(B, lambda_u)
                mul!(B, prob.p_operator, A)
                grad_contrib .+= grad_p .* weights_n[2]*dot(B, lambda_v)

                # part 2
                mul!(A, Hp, u)
                mul!(A, Hq, v, 1, 1)
                mul!(B, prob.p_operator, A)

                grad_contrib .-= grad_p .* weights_n[2]*dot(B, lambda_u)
                mul!(B, prob.q_operator, A)
                grad_contrib .+= grad_q .* weights_n[2]*dot(B, lambda_v)


                # H*H_α
                # part 1
                mul!(A, prob.q_operator, u)
                mul!(B, prob.q_operator, v)

                mul!(C, Hq, A)
                mul!(C, Hp, B, -1, 1)
                grad_contrib .+= grad_q .* weights_n[2]*dot(C, lambda_u)

                mul!(C, Hp, A)
                mul!(C, Hq, B, 1, 1)
                grad_contrib .+= grad_q .* weights_n[2]*dot(C, lambda_v)


                # part 2
                mul!(A, prob.p_operator, v)
                mul!(B, prob.p_operator, u)

                mul!(C, Hq, A)
                mul!(C, Hp, B, 1, 1)
                grad_contrib .-= grad_p .* weights_n[2]*dot(C, lambda_u)

                mul!(C, Hp, A)
                mul!(C, Hq, B, -1, 1)
                grad_contrib .-= grad_p .* weights_n[2]*dot(C, lambda_v)
                

                # uv n+1 contribution

                u = history[1:prob.N_tot_levels,     1+n+1, initial_condition_index]
                v = history[1+prob.N_tot_levels:end, 1+n+1, initial_condition_index]
                t = (n+1)*dt

                grad_p = eval_grad_p(control, t, pcof)
                grad_q = eval_grad_q(control, t,pcof)
                grad_pt = eval_grad_pt(control, t,pcof)
                grad_qt = eval_grad_qt(control, t,pcof)

                # H_α
                grad_contrib .+= grad_q .* weights_np1[1]*(
                    dot(u, MT_lambda_11) + dot(v, MT_lambda_22)
                )
                grad_contrib .+= grad_p .* weights_np1[1]*(
                    dot(u, MT_lambda_12) - dot(v, MT_lambda_21)
                )

                # 4th order Correction
                # H_αt
                grad_contrib .+= grad_qt .* weights_np1[2]*(
                    dot(u, MT_lambda_11) + dot(v, MT_lambda_22)
                )
                grad_contrib .+= grad_pt .* weights_np1[2]*(
                    dot(u, MT_lambda_12) - dot(v, MT_lambda_21)
                )
                
                # H_α*H
                # part 1
                Hq .= prob.Ss .+ eval_q(control, t, pcof) .* prob.q_operator
                Hp .= prob.Ks .+ eval_p(control, t, pcof) .* prob.p_operator

                mul!(A, Hq, u)
                mul!(A, Hp, v, -1, 1)
                mul!(B, prob.q_operator, A)

                grad_contrib .+= grad_q .* weights_np1[2]*dot(B, lambda_u)
                mul!(B, prob.p_operator, A)
                grad_contrib .+= grad_p .* weights_np1[2]*dot(B, lambda_v)

                # part 2
                mul!(A, Hp, u)
                mul!(A, Hq, v, 1, 1)
                mul!(B, prob.p_operator, A)

                grad_contrib .-= grad_p .* weights_np1[2]*dot(B, lambda_u)
                mul!(B, prob.q_operator, A)
                grad_contrib .+= grad_q .* weights_np1[2]*dot(B, lambda_v)


                # H*H_α
                # part 1
                mul!(A, prob.q_operator, u)
                mul!(B, prob.q_operator, v)

                mul!(C, Hq, A)
                mul!(C, Hp, B, -1, 1)
                grad_contrib .+= grad_q .* weights_np1[2]*dot(C, lambda_u)

                mul!(C, Hp, A)
                mul!(C, Hq, B, 1, 1)
                grad_contrib .+= grad_q .* weights_np1[2]*dot(C, lambda_v)


                # part 2
                mul!(A, prob.p_operator, v)
                mul!(B, prob.p_operator, u)

                mul!(C, Hq, A)
                mul!(C, Hp, B, 1, 1)
                grad_contrib .-= grad_p .* weights_np1[2]*dot(C, lambda_u)

                mul!(C, Hp, A)
                mul!(C, Hq, B, -1, 1)
                grad_contrib .-= grad_p .* weights_np1[2]*dot(C, lambda_v)
            end
            grad .+=  (-0.5*dt) .* grad_contrib
        else
            throw("Invalid order: $order")
        end
    end

    if return_lambda_history
        return grad, history, lambda_history
    end
    return grad
end

"""
Evaluates gradient of the provided Schrodinger problem with the given target
gate and control parameter(s) pcof using the "forward differentiation" method,
which evolves a differentiated Schrodinger equation, using the state vector
in the evolution of the original Schrodinger equation as a forcing term.

Returns: gradient
"""
function eval_grad_forced(prob::SchrodingerProb{M, VM}, control::AbstractControl,
        pcof::AbstractVector{Float64}, target::VM; order=2, 
        cost_type=:Infidelity
    ) where {M <: AbstractMatrix{Float64}, VM <: AbstractVecOrMat{Float64}}
    @assert length(pcof) == control.N_coeff

    # Get state vector history
    history = eval_forward(prob, control, pcof, order=order, return_time_derivatives=true)

    ## Compute forcing array (-idH/dα ψ)
    # Prepare dH/dα
    
    # System hamiltonian is constant, falls out when taking derivative
    dKs_da::Matrix{Float64} = zeros(prob.N_tot_levels, prob.N_tot_levels)
    dSs_da::Matrix{Float64} = zeros(prob.N_tot_levels, prob.N_tot_levels)

    gradient = zeros(control.N_coeff)

    forcing_ary = zeros(2*prob.N_tot_levels, 1+prob.nsteps, div(order, 2), size(prob.u0, 2))

    for control_param_index in 1:control.N_coeff
        for initial_condition_index = 1:size(prob.u0, 2)
            # This could be more efficient by having functions which calculate
            # only the intended partial derivative, not the whole gradient, but
            # I am only focusing on optimizaiton for the discrete adjoint.
            dpda(t, pcof) = eval_grad_p(control, t, pcof)[control_param_index]
            dqda(t, pcof) = eval_grad_q(control, t, pcof)[control_param_index]
            d2p_dta(t, pcof) = eval_grad_pt(control, t, pcof)[control_param_index]
            d2q_dta(t, pcof) = eval_grad_qt(control, t, pcof)[control_param_index]

            # 2nd order version
            if order == 2
                forcing_vec = zeros(2*prob.N_tot_levels)

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

                    utvt!(
                        ut, vt, u, v,
                        dKs_da, dSs_da, prob.p_operator, prob.q_operator,
                        dpda(t, pcof), dqda(t, pcof)
                    )
                    copyto!(forcing_vec, 1,                   ut, 1, prob.N_tot_levels)
                    copyto!(forcing_vec, 1+prob.N_tot_levels, vt, 1, prob.N_tot_levels)

                    forcing_ary[:, 1+n, 1, initial_condition_index] .= forcing_vec

                    t += dt
                end

            # 4th order version
            elseif order == 4
                forcing_vec2 = zeros(2*prob.N_tot_levels)
                forcing_vec4 = zeros(2*prob.N_tot_levels)

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

                    utvt!(
                        ut, vt, u, v, 
                        dKs_da, dSs_da, prob.p_operator, prob.q_operator,
                        dpda(t, pcof), dqda(t, pcof)
                    )
                    forcing_vec2[1:prob.N_tot_levels]     .= ut
                    forcing_vec2[1+prob.N_tot_levels:end] .= vt

                    forcing_ary[:, 1+n, 1, initial_condition_index] .= forcing_vec2

                    # Fourth Order Forcing
                    forcing_vec4 = zeros(2*prob.N_tot_levels)

                    utvt!(
                        ut, vt, u, v,
                        dKs_da, dSs_da, prob.p_operator, prob.q_operator,
                        d2p_dta(t, pcof), d2q_dta(t, pcof)
                    )
                    forcing_vec4[1:prob.N_tot_levels]     .+= ut
                    forcing_vec4[1+prob.N_tot_levels:end] .+= vt

                    ut .= history[1:prob.N_tot_levels,     1+n, 2, initial_condition_index]
                    vt .= history[1+prob.N_tot_levels:end, 1+n, 2, initial_condition_index]

                    A = zeros(prob.N_tot_levels) # Placeholders
                    B = zeros(prob.N_tot_levels)

                    utvt!(
                        A, B, ut, vt,
                        dKs_da, dSs_da, prob.p_operator, prob.q_operator,
                        dpda(t, pcof), dqda(t, pcof)
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
        differentiated_prob = copy(prob) 
        # Get history of dψ/dα
        # Initial conditions for dψ/dα, all entries zero (control can't change
        # initial condition of ψ)
        differentiated_prob.u0 .= 0.0
        differentiated_prob.v0 .= 0.0

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



"""
Vector control version.

Evaluates gradient of the provided Schrodinger problem with the given target
gate and control parameter(s) pcof using a finite difference method, where a step
size of dpcof is used when perturbing the components of the control vector pcof.

Returns: gradient
"""
function eval_grad_finite_difference(
        prob::SchrodingerProb, control::Control,
        pcof::AbstractVector{Float64}, target::AbstractMatrix{Float64}, 
        dpcof=1e-5; order=2, cost_type=:Infidelity,
    )

    grad = zeros(length(pcof))

    for i in 1:length(pcof)
        # Centered Difference Approximation
        pcof_r = copy(pcof)
        pcof_r[i] += dpcof

        pcof_l = copy(pcof)
        pcof_l[i] -= dpcof

        history_r = eval_forward(prob, control, pcof_r, order=order)
        history_l = eval_forward(prob, control, pcof_l, order=order)
        ψf_r = history_r[:,end,:]
        ψf_l = history_l[:,end,:]
        if cost_type == :Infidelity
            cost_r = infidelity(ψf_r, target, prob.N_ess_levels)
            cost_l = infidelity(ψf_l, target, prob.N_ess_levels)
        elseif cost_type == :Tracking
            cost_r = 0.5*norm(ψf_r - target)^2
            cost_l = 0.5*norm(ψf_l - target)^2
        elseif cost_type == :Norm
            cost_r = 0.5*norm(ψf_r)^2
            cost_l = 0.5*norm(ψf_l)^2
        else
            throw("Invalid cost type: $cost_type")
        end
        grad[i] = (cost_r - cost_l)/(2*dpcof)
    end

    return grad
end

function eval_grad_finite_difference(
        prob::SchrodingerProb{M, V}, control::Control, 
        pcof::AbstractVector{Float64}, target::V; dpcof=1e-5, order=2, 
        cost_type=:Infidelity
    ) where {M<: AbstractMatrix{Float64}, V <: AbstractVector{Float64}}

    grad = zeros(control.N_coeff)

    for i in 1:length(pcof)
        # Centered Difference Approximation
        pcof_r = copy(pcof)
        pcof_r[i] += dpcof

        pcof_l = copy(pcof)
        pcof_l[i] -= dpcof

        history_r = eval_forward(prob, control, pcof_r, order=order)
        history_l = eval_forward(prob, control, pcof_l, order=order)
        ψf_r = history_r[:, end]
        ψf_l = history_l[:, end]
        if cost_type == :Infidelity
            cost_r = infidelity(ψf_r, target, prob.N_ess_levels)
            cost_l = infidelity(ψf_l, target, prob.N_ess_levels)
        elseif cost_type == :Tracking
            cost_r = 0.5*norm(ψf_r - target)^2
            cost_l = 0.5*norm(ψf_l - target)^2
        elseif cost_type == :Norm
            cost_r = 0.5*norm(ψf_r)^2
            cost_l = 0.5*norm(ψf_l)^2
        else
            throw("Invalid cost type: $cost_type")
        end
        grad[i] = (cost_r - cost_l)/(2*dpcof)
    end

    return grad
end



"""
Calculates the infidelity for the given state vector 'ψ' and target state
'target.'

Returns: Infidelity
"""
function infidelity(ψ::AbstractVector{Float64}, target::AbstractVector{Float64}, N_ess::Int64) where {V <: AbstractVector{Float64}}
    R = copy(target)
    N_tot = size(target,1)÷2
    T = vcat(R[1+N_tot:end], -R[1:N_tot])
    return 1 - (dot(ψ,R)^2 + dot(ψ,T)^2)/(N_ess^2)
end



"""
Calculates the infidelity for the given matrix of state vectors 'Q' and matrix
of target states 'target.'

Returns: Infidelity
"""
function infidelity(Q::AbstractMatrix{Float64}, target::AbstractMatrix{Float64}, N_ess::Int64)
    R = copy(target)
    N_tot = size(target,1)÷2
    T = vcat(R[1+N_tot:end,:], -R[1:N_tot,:])
    return 1 - (tr(Q'*R)^2 + tr(Q'*T)^2)/(N_ess^2)
end

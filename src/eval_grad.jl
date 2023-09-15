
"""

    gradient = discrete_adjoint(prob, target, pcof; order=order, cost_type=cost_type, return_lambda_history=false)

Evaluates gradient of the provided Schrodinger problem with the given target
gate and control parameter(s) α using the discrete adjoint method. 

Returns: gradient
"""
function discrete_adjoint(prob::SchrodingerProb, target::Vector{Float64},
        α; order=2, cost_type=:Infidelity, return_lambda_history=false)

    # Get state vector history
    history = eval_forward(prob, α, order=order)

    # Unpacking variables is possibly a bad idea
    Ks = prob.Ks
    Ss = prob.Ss
    a_plus_adag = prob.a_plus_adag
    a_minus_adag = prob.a_minus_adag
    p = prob.p
    q = prob.q
    dpdt = prob.dpdt
    dqdt = prob.dqdt
    dpda = prob.dpda
    dqda = prob.dqda
    d2p_dta = prob.d2p_dta
    d2q_dta = prob.d2q_dta
    u0 = prob.u0
    v0 = prob.v0
    tf = prob.tf
    nsteps = prob.nsteps
    N_ess = prob.N_ess_levels
    N_grd = prob.N_guard_levels
    N_tot = prob.N_tot_levels

    R = copy(target)
    T = vcat(R[1+N_tot:end], -R[1:N_tot])

    # For adjoint evolution. Take transpose of entire matrix -> antisymmetric blocks change sign
    Ks_adj = Matrix(transpose(-Ks)) # NOTE THAT K changes sign!
    Ss_adj = Matrix(transpose(Ss))
    a_plus_adag_adj = Matrix(transpose(-a_plus_adag)) # NOTE THAT K changes sign!
    a_minus_adag_adj = Matrix(transpose(a_minus_adag))
    
    a_plus_adag_transpose = Matrix(transpose(a_plus_adag)) # NOTE THAT K changes sign!
    a_minus_adag_transpose = Matrix(transpose(a_minus_adag))

    dt = tf/nsteps

    len_α = length(α)
    len_α_half = div(len_α, 2)
    grad = zeros(len_α)

    for initial_condition_index = 1:size(u0,2)
        lambda = zeros(2*N_tot)
        lambda_history = zeros(2*N_tot,1+nsteps)
        lambda_ut  = zeros(N_tot)
        lambda_vt  = zeros(N_tot)
        lambda_utt = zeros(N_tot)
        lambda_vtt = zeros(N_tot)

        RHS_lambda_u::Vector{Float64} = zeros(N_tot)
        RHS_lambda_v::Vector{Float64} = zeros(N_tot)
        RHS::Vector{Float64} = zeros(2*N_tot)



        if order == 2
            # Terminal Condition
            t = tf

            if cost_type == :Infidelity
                RHS = (2.0/(N_ess^2))*(dot(history[:,end],R)*R + dot(history[:,end],T)*T)
            elseif cost_type == :Tracking
                RHS = -(history[:,end] - target)
            elseif cost_type == :Norm
                RHS = -history[:,end]
            else
                throw("Invalid cost type: $cost_type")
            end

            LHS_map = LinearMap(
                x -> LHS_func(lambda_ut, lambda_vt, x[1:N_tot], x[1+N_tot:end],
                              Ks_adj, Ss_adj, a_plus_adag_adj, a_minus_adag_adj,
                              p, q, t, α, dt, N_tot),
                2*N_tot,2*N_tot
            )
            gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

            lambda_history[:,1+nsteps] .= lambda
            lambda_u = copy(lambda[1:N_tot])
            lambda_v = copy(lambda[1+N_tot:2*N_tot])
            
            # Discrete Adjoint Scheme
            for n in nsteps-1:-1:1
                t -= dt
                utvt!(lambda_ut, lambda_vt, lambda_u, lambda_v,
                      Ks_adj, Ss_adj, a_plus_adag_adj, a_minus_adag_adj,
                      p, q, t, α)
                copy!(RHS_lambda_u,lambda_u)
                axpy!(0.5*dt,lambda_ut,RHS_lambda_u)

                copy!(RHS_lambda_v,lambda_v)
                axpy!(0.5*dt,lambda_vt,RHS_lambda_v)

                copyto!(RHS,1,RHS_lambda_u,1,N_tot)
                copyto!(RHS,1+N_tot,RHS_lambda_v,1,N_tot)

                # NOTE: LHS and RHS Linear transformations use the SAME TIME

                LHS_map = LinearMap(
                    x -> LHS_func(lambda_ut, lambda_vt, x[1:N_tot], x[1+N_tot:end],
                                  Ks_adj, Ss_adj, a_plus_adag_adj, a_minus_adag_adj,
                                  p, q, t, α, dt, N_tot),
                    2*N_tot,2*N_tot
                )

                gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)
                lambda_history[:,1+n] .= lambda
                lambda_u = lambda[1:N_tot]
                lambda_v = lambda[1+N_tot:end]
            end
        
            zero_mat = zeros(N_tot,N_tot)
            uv = zeros(2*N_tot)
            u = zeros(N_tot)
            v = zeros(N_tot)
            # Won't actually hold ut and vt, but rather real/imag parts of dH/dα*ψ
            ut = zeros(N_tot)
            vt = zeros(N_tot)
            dummy_u = zeros(N_tot)
            dummy_v = zeros(N_tot)
            dummy = zeros(2*N_tot)

            MT_lambda_11 = zeros(N_tot)
            MT_lambda_12 = zeros(N_tot)
            MT_lambda_21 = zeros(N_tot)
            MT_lambda_22 = zeros(N_tot)

            #=
            # Note: the commented out grad[1:len_α_half]'s weer from my original
            # assumption that the first half of the control vector only affects
            # p, and the second half only affects q. 
            #
            # This is not the case for the bspline controls, so I had to revise the
            # method such that grad_p and grad_q are the same size as the gradient.
            #
            # If it is the case that the first half only affects p, then half of
            # grad_p will be empty
            #
            =#
            for n in 0:nsteps-1
                lambda_u = lambda_history[1:N_tot,1+n+1]
                lambda_v = lambda_history[1+N_tot:end,1+n+1]

                u = history[1:N_tot,1+n]
                v = history[1+N_tot:end,1+n]
                t = n*dt

                grad_p = dpda(t,α)
                grad_q = dqda(t,α)


                mul!(MT_lambda_11, a_minus_adag_transpose, lambda_u)
                mul!(MT_lambda_12, a_plus_adag_transpose, lambda_v)
                mul!(MT_lambda_21, a_plus_adag_transpose, lambda_u)
                mul!(MT_lambda_22, a_minus_adag_transpose, lambda_v)

                #grad[1+len_α_half:len_α] .+= grad_q .* (dot(u, MT_lambda_11)
                #                                        + dot(v, MT_lambda_22)
                #                                       )
                #grad[1:len_α_half] .+= grad_p .* (dot(u, MT_lambda_12)
                #                                  - dot(v, MT_lambda_21)
                #                                 )
                grad .+= grad_q .* (dot(u, MT_lambda_11)
                                                        + dot(v, MT_lambda_22)
                                                       )
                grad .+= grad_p .* (dot(u, MT_lambda_12)
                                                  - dot(v, MT_lambda_21)
                                                 )


                u = history[1:N_tot,1+n+1]
                v = history[1+N_tot:end,1+n+1]
                t = (n+1)*dt

                grad_p = dpda(t,α)
                grad_q = dqda(t,α)

                #grad[1+len_α_half:len_α] .+= grad_q .* (dot(u, MT_lambda_11)
                #                                        + dot(v, MT_lambda_22)
                #                                       )
                #grad[1:len_α_half] .+= grad_p .* (dot(u, MT_lambda_12)
                #                                  - dot(v, MT_lambda_21)
                #                                 )
                grad .+= grad_q .* (dot(u, MT_lambda_11)
                                                        + dot(v, MT_lambda_22)
                                                       )
                grad .+= grad_p .* (dot(u, MT_lambda_12)
                                                  - dot(v, MT_lambda_21)
                                                 )
            end
            grad *= -0.5*dt

        elseif order == 4
            # Terminal Condition
            t = tf

            if cost_type == :Infidelity
                RHS = (2/(N_ess^2))*(dot(history[:,end],R)*R + dot(history[:,end],T)*T)
            elseif cost_type == :Tracking
                RHS = -(history[:,end] - target)
            elseif cost_type == :Norm
                RHS = -history[:,end]
            else
                throw("Invalid cost type: $cost_type")
            end

            LHS_map = LinearMap(
                x -> LHS_func_order4(lambda_utt, lambda_vtt, lambda_ut, lambda_vt,
                                     x[1:N_tot], x[1+N_tot:end],
                              Ks_adj, Ss_adj, a_plus_adag_adj, a_minus_adag_adj,
                              p, q, dpdt, dqdt, t, α, dt, N_tot),
                2*N_tot,2*N_tot
            )

            gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

            lambda_history[:,1+nsteps] .= lambda
            lambda_u = copy(lambda[1:N_tot])
            lambda_v = copy(lambda[1+N_tot:end])
            
            weights = [1,1/3]
            # Discrete Adjoint Scheme
            for n in nsteps-1:-1:1
                t -= dt
                utvt!(lambda_ut, lambda_vt, lambda_u, lambda_v,
                      Ks_adj, Ss_adj, a_plus_adag_adj, a_minus_adag_adj,
                      p, q, t, α)
                uttvtt!(lambda_utt, lambda_vtt, lambda_ut, lambda_vt, lambda_u, lambda_v,
                    Ks_adj, Ss_adj, a_plus_adag_adj, a_minus_adag_adj,
                    p, q, dpdt, dqdt, t, α)

                copy!(RHS_lambda_u,lambda_u)
                axpy!(0.5*dt*weights[1],lambda_ut,RHS_lambda_u)
                axpy!(0.25*dt^2*weights[2],lambda_utt,RHS_lambda_u)

                copy!(RHS_lambda_v,lambda_v)
                axpy!(0.5*dt*weights[1],lambda_vt,RHS_lambda_v)
                axpy!(0.25*dt^2*weights[2],lambda_vtt,RHS_lambda_v)

                copyto!(RHS,1,RHS_lambda_u,1,N_tot)
                copyto!(RHS,1+N_tot,RHS_lambda_v,1,N_tot)

                # NOTE: LHS and RHS Linear transformations use the SAME TIME

                LHS_map = LinearMap(
                    x -> LHS_func_order4(lambda_utt, lambda_vtt, lambda_ut, lambda_vt,
                                  x[1:N_tot], x[1+N_tot:end],
                                  Ks_adj, Ss_adj, a_plus_adag_adj, a_minus_adag_adj,
                                  p, q, dpdt, dqdt, t, α, dt, N_tot),
                    2*N_tot,2*N_tot
                )

                gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)
                lambda_history[:,1+n] .= lambda
                lambda_u = lambda[1:N_tot]
                lambda_v = lambda[1+N_tot:2*N_tot]
            end

            zero_mat = zeros(N_tot,N_tot)
            uv = zeros(2*N_tot)
            u = zeros(N_tot)
            v = zeros(N_tot)

            MT_lambda_11 = zeros(N_tot)
            MT_lambda_12 = zeros(N_tot)
            MT_lambda_21 = zeros(N_tot)
            MT_lambda_22 = zeros(N_tot)

            A = zeros(N_tot)
            B = zeros(N_tot)
            C = zeros(N_tot)

            Ap = zeros(N_tot, N_tot)
            Bp = zeros(N_tot, N_tot)
            Am = zeros(N_tot, N_tot)
            Bm = zeros(N_tot, N_tot)

            Cu = zeros(N_tot)
            Cv = zeros(N_tot)
            Du = zeros(N_tot)
            Dv = zeros(N_tot)

            K_full =  Ks .+ a_plus_adag
            S_full =  Ss .+ a_minus_adag

            Hq = zeros(N_tot,N_tot)
            Hp = zeros(N_tot,N_tot)

            # Accumulate Gradient
            len_α = length(α)
            len_α_half = div(len_α, 2)

            # Efficient way, possibly incorrect
            weights_n = [1,dt/6]
            weights_np1 = [1,-dt/6]
            for n in 0:nsteps-1
                lambda_u = lambda_history[1:N_tot,1+n+1]
                lambda_v = lambda_history[1+N_tot:end,1+n+1]

                mul!(MT_lambda_11, a_minus_adag_transpose, lambda_u)
                mul!(MT_lambda_12, a_plus_adag_transpose, lambda_v)
                mul!(MT_lambda_21, a_plus_adag_transpose, lambda_u)
                mul!(MT_lambda_22, a_minus_adag_transpose, lambda_v)

                # Qn contribution
                u = history[1:N_tot,1+n]
                v = history[1+N_tot:end,1+n]
                t = n*dt

                grad_p = dpda(t,α)
                grad_q = dqda(t,α)
                grad_pt = d2p_dta(t,α)
                grad_qt = d2q_dta(t,α)

                # H_α
                #grad[1+len_α_half:len_α] .+= grad_q .* weights_n[1]*(
                #    dot(u, MT_lambda_11) + dot(v, MT_lambda_22)
                #)
                #grad[1:len_α_half] .+= grad_p .* weights_n[1]*(
                #    dot(u, MT_lambda_12) - dot(v, MT_lambda_21)
                #)
                grad .+= grad_q .* weights_n[1]*(
                    dot(u, MT_lambda_11) + dot(v, MT_lambda_22)
                )
                grad .+= grad_p .* weights_n[1]*(
                    dot(u, MT_lambda_12) - dot(v, MT_lambda_21)
                )

                # 4th order Correction
                # H_αt
                #grad[1+len_α_half:len_α] .+= grad_qt .* weights_n[2]*(
                #    dot(u, MT_lambda_11) + dot(v, MT_lambda_22)
                #)
                #grad[1:len_α_half] .+= grad_pt .* weights_n[2]*(
                #    dot(u, MT_lambda_12) - dot(v, MT_lambda_21)
                #)
                grad .+= grad_qt .* weights_n[2]*(
                    dot(u, MT_lambda_11) + dot(v, MT_lambda_22)
                )
                grad .+= grad_pt .* weights_n[2]*(
                    dot(u, MT_lambda_12) - dot(v, MT_lambda_21)
                )
                
                # H_α*H
                # part 1
                Hq .= Ss .+ q(t,α) .* a_minus_adag
                Hp .= Ks .+ p(t,α) .* a_plus_adag

                mul!(A, Hq, u)
                mul!(A, Hp, v, -1, 1)
                mul!(B, a_minus_adag, A)

                #grad[1+len_α_half:len_α] .+= grad_q .* weights_n[2]*dot(B, lambda_u)
                grad .+= grad_q .* weights_n[2]*dot(B, lambda_u)
                mul!(B, a_plus_adag, A)
                #grad[1:len_α_half] .+= grad_p .* weights_n[2]*dot(B, lambda_v)
                grad .+= grad_p .* weights_n[2]*dot(B, lambda_v)

                # part 2
                mul!(A, Hp, u)
                mul!(A, Hq, v, 1, 1)
                mul!(B, a_plus_adag, A)

                #grad[1:len_α_half] .-= grad_p .* weights_n[2]*dot(B, lambda_u)
                grad .-= grad_p .* weights_n[2]*dot(B, lambda_u)
                mul!(B, a_minus_adag, A)
                #grad[1+len_α_half:len_α] .+= grad_q .* weights_n[2]*dot(B, lambda_v)
                grad .+= grad_q .* weights_n[2]*dot(B, lambda_v)


                # H*H_α
                # part 1
                mul!(A, a_minus_adag, u)
                mul!(B, a_minus_adag, v)

                mul!(C, Hq, A)
                mul!(C, Hp, B, -1, 1)
                #grad[1+len_α_half:len_α] .+= grad_q .* weights_n[2]*dot(C, lambda_u)
                grad .+= grad_q .* weights_n[2]*dot(C, lambda_u)

                mul!(C, Hp, A)
                mul!(C, Hq, B, 1, 1)
                #grad[1+len_α_half:len_α] .+= grad_q .* weights_n[2]*dot(C, lambda_v)
                grad .+= grad_q .* weights_n[2]*dot(C, lambda_v)


                # part 2
                mul!(A, a_plus_adag, v)
                mul!(B, a_plus_adag, u)

                mul!(C, Hq, A)
                mul!(C, Hp, B, 1, 1)
                #grad[1:len_α_half] .-= grad_p .* weights_n[2]*dot(C, lambda_u)
                grad .-= grad_p .* weights_n[2]*dot(C, lambda_u)

                mul!(C, Hp, A)
                mul!(C, Hq, B, -1, 1)
                #grad[1:len_α_half] .-= grad_p .* weights_n[2]*dot(C, lambda_v)
                grad .-= grad_p .* weights_n[2]*dot(C, lambda_v)
                

                # uv n+1 contribution

                u = history[1:N_tot,1+n+1]
                v = history[1+N_tot:end,1+n+1]
                t = (n+1)*dt

                grad_p = dpda(t,α)
                grad_q = dqda(t,α)
                grad_pt = d2p_dta(t,α)
                grad_qt = d2q_dta(t,α)

                # H_α
                #grad[1+len_α_half:len_α] .+= grad_q .* weights_np1[1]*(
                #    dot(u, MT_lambda_11) + dot(v, MT_lambda_22)
                #)
                #grad[1:len_α_half] .+= grad_p .* weights_np1[1]*(
                #    dot(u, MT_lambda_12) - dot(v, MT_lambda_21)
                #)
                grad .+= grad_q .* weights_np1[1]*(
                    dot(u, MT_lambda_11) + dot(v, MT_lambda_22)
                )
                grad .+= grad_p .* weights_np1[1]*(
                    dot(u, MT_lambda_12) - dot(v, MT_lambda_21)
                )

                # 4th order Correction
                # H_αt
                #grad[1+len_α_half:len_α] .+= grad_qt .* weights_np1[2]*(
                #    dot(u, MT_lambda_11) + dot(v, MT_lambda_22)
                #)
                #grad[1:len_α_half] .+= grad_pt .* weights_np1[2]*(
                #    dot(u, MT_lambda_12) - dot(v, MT_lambda_21)
                #)
                grad .+= grad_qt .* weights_np1[2]*(
                    dot(u, MT_lambda_11) + dot(v, MT_lambda_22)
                )
                grad .+= grad_pt .* weights_np1[2]*(
                    dot(u, MT_lambda_12) - dot(v, MT_lambda_21)
                )
                
                # H_α*H
                # part 1
                Hq .= Ss .+ q(t,α) .* a_minus_adag
                Hp .= Ks .+ p(t,α) .* a_plus_adag

                mul!(A, Hq, u)
                mul!(A, Hp, v, -1, 1)
                mul!(B, a_minus_adag, A)

                #grad[1+len_α_half:len_α] .+= grad_q .* weights_np1[2]*dot(B, lambda_u)
                grad .+= grad_q .* weights_np1[2]*dot(B, lambda_u)
                mul!(B, a_plus_adag, A)
                #grad[1:len_α_half] .+= grad_p .* weights_np1[2]*dot(B, lambda_v)
                grad .+= grad_p .* weights_np1[2]*dot(B, lambda_v)

                # part 2
                mul!(A, Hp, u)
                mul!(A, Hq, v, 1, 1)
                mul!(B, a_plus_adag, A)

                #grad[1:len_α_half] .-= grad_p .* weights_np1[2]*dot(B, lambda_u)
                grad .-= grad_p .* weights_np1[2]*dot(B, lambda_u)
                mul!(B, a_minus_adag, A)
                #grad[1+len_α_half:len_α] .+= grad_q .* weights_np1[2]*dot(B, lambda_v)
                grad .+= grad_q .* weights_np1[2]*dot(B, lambda_v)


                # H*H_α
                # part 1
                mul!(A, a_minus_adag, u)
                mul!(B, a_minus_adag, v)

                mul!(C, Hq, A)
                mul!(C, Hp, B, -1, 1)
                #grad[1+len_α_half:len_α] .+= grad_q .* weights_np1[2]*dot(C, lambda_u)
                grad .+= grad_q .* weights_np1[2]*dot(C, lambda_u)

                mul!(C, Hp, A)
                mul!(C, Hq, B, 1, 1)
                #grad[1+len_α_half:len_α] .+= grad_q .* weights_np1[2]*dot(C, lambda_v)
                grad .+= grad_q .* weights_np1[2]*dot(C, lambda_v)


                # part 2
                mul!(A, a_plus_adag, v)
                mul!(B, a_plus_adag, u)

                mul!(C, Hq, A)
                mul!(C, Hp, B, 1, 1)
                #grad[1:len_α_half] .-= grad_p .* weights_np1[2]*dot(C, lambda_u)
                grad .-= grad_p .* weights_np1[2]*dot(C, lambda_u)

                mul!(C, Hp, A)
                mul!(C, Hq, B, -1, 1)
                #grad[1:len_α_half] .-= grad_p .* weights_np1[2]*dot(C, lambda_v)
                grad .-= grad_p .* weights_np1[2]*dot(C, lambda_v)
            end
            grad *= -0.5*dt
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
gate and control parameter(s) α using the "forward differentiation" method,
which evolves a differentiated Schrodinger equation, using the state vector
in the evolution of the original Schrodinger equation as a forcing term.

Returns: gradient
"""
function eval_grad_forced(prob::SchrodingerProb, target, α=1.0; order=2, cost_type=:Infidelity)
    # Get state vector history
    history = eval_forward(prob, α, order=order, return_time_derivatives=true)

    N_ess = prob.N_ess_levels
    N_grd = prob.N_guard_levels
    N_tot = prob.N_tot_levels

    ## Prepare forcing (-idH/dα ψ)
    # Prepare dH/dα
    
    # System hamiltonian is constant, falls out when taking derivative
    dKs_da::Matrix{Float64} = zeros(N_tot,N_tot)
    dSs_da::Matrix{Float64} = zeros(N_tot,N_tot)
    a_plus_adag = prob.a_plus_adag
    a_minus_adag = prob.a_minus_adag
    dpda = prob.dpda
    dqda = prob.dqda
    d2p_dta = prob.d2p_dta
    d2q_dta = prob.d2q_dta

    gradient = zeros(length(α))

    for i in 1:length(α)
        dpda(t,pcof) = prob.dpda(t,pcof)[i]
        dqda(t,pcof) = prob.dqda(t,pcof)[i]
        d2p_dta(t, pcof) = prob.d2p_dta(t,pcof)[i]
        d2q_dta(t, pcof) = prob.d2q_dta(t,pcof)[i]
        if order == 2
            forcing_ary = zeros(2*N_tot,1+prob.nsteps,1)
            forcing_vec = zeros(2*N_tot)

            u  = zeros(N_tot)
            v  = zeros(N_tot)
            ut = zeros(N_tot)
            vt = zeros(N_tot)

            nsteps = prob.nsteps
            t = 0.0
            dt = prob.tf/nsteps

            # Get forcing (dH/dα * ψ)
            for n in 0:nsteps
                copyto!(u,history[1:N_tot,1+n,1])
                copyto!(v,history[1+N_tot:end,1+n,1])

                utvt!(ut, vt, u, v,
                      dKs_da, dSs_da, a_plus_adag, a_minus_adag,
                      dpda, dqda, t, α)
                copyto!(forcing_vec,1,ut,1,N_tot)
                copyto!(forcing_vec,1+N_tot,vt,1,N_tot)

                forcing_ary[:,1+n,1] .= forcing_vec

                t += dt
            end
        elseif order == 4
            forcing_ary = zeros(2*N_tot,1+prob.nsteps,2)
            forcing_vec2 = zeros(2*N_tot)
            forcing_vec4 = zeros(2*N_tot)

            u  = zeros(N_tot)
            v  = zeros(N_tot)
            ut = zeros(N_tot)
            vt = zeros(N_tot)

            nsteps = prob.nsteps
            t = 0.0
            dt = prob.tf/nsteps

            # Get forcing (dH/dα * ψ)
            for n in 0:nsteps
                # Second Order Forcing
                u .= history[1:N_tot,1+n,1]
                v .= history[1+N_tot:end,1+n,1]

                utvt!(ut, vt, u, v, 
                      dKs_da, dSs_da, a_plus_adag, a_minus_adag,
                      dpda, dqda, t, α)
                forcing_vec2[1:N_tot] .= ut
                forcing_vec2[1+N_tot:end] .= vt

                forcing_ary[:,1+n,1] .= forcing_vec2

                # Fourth Order Forcing
                forcing_vec4 = zeros(2*N_tot)

                utvt!(ut, vt, u, v,
                      dKs_da, dSs_da, a_plus_adag, a_minus_adag,
                      d2p_dta, d2q_dta, t, α)
                forcing_vec4[1:N_tot] += ut
                forcing_vec4[1+N_tot:end] += vt

                ut .= history[1:N_tot,1+n,2]
                vt .= history[1+N_tot:end,1+n,2]

                A = zeros(N_tot) # Placeholders
                B = zeros(N_tot)
                utvt!(A, B, ut, vt,
                      dKs_da, dSs_da, a_plus_adag, a_minus_adag,
                      dpda, dqda, t, α)
                forcing_vec4[1:N_tot] += A
                forcing_vec4[1+N_tot:end] += B

                forcing_ary[:,1+n,2] .= forcing_vec4

                t += dt
            end
        else 
            throw("Invalid Order: $order")
        end

        # Evolve with forcing
        differentiated_prob = copy(prob) 
        # Get history of dψ/dα
        # Initial conditions for dψ/dα, all entries zero (control can't change
        # initial condition of ψ)
        differentiated_prob.u0 .= 0.0
        differentiated_prob.v0 .= 0.0

        history_dψdα = eval_forward_forced(differentiated_prob, forcing_ary, α, 
                                           order=order)

        dQda = history_dψdα[:,end]
        Q = history[:,end,1]
        R = copy(target)
        T = vcat(R[1+N_tot:end], -R[1:N_tot])

        if cost_type == :Infidelity
            gradient[i] = -(2/(N_ess^2))*(dot(Q,R)*dot(dQda,R) + dot(Q,T)*dot(dQda,T))
        elseif cost_type == :Tracking
            gradient[i] = dot(dQda, Q - target)
        elseif cost_type == :Norm
            gradient[i] = dot(dQda,Q)
        else
            throw("Invalid cost type: $cost_type")
        end
    end
    return gradient
end



"""
Vector control version.

Evaluates gradient of the provided Schrodinger problem with the given target
gate and control parameter(s) α using a finite difference method, where a step
size of dα is used when perturbing the components of the control vector α.

Returns: gradient
"""
function eval_grad_finite_difference(prob::SchrodingerProb, target::AbstractVector{Float64},
        α::AbstractVector{Float64}, dα=1e-5; order=2, cost_type=:Infidelity)

    grad = zeros(length(α))
    for i in 1:length(α)
        # Centered Difference Approximation
        α_r = copy(α)
        α_r[i] += dα
        α_l = copy(α)
        α_l[i] -= dα
        history_r = eval_forward(prob, α_r, order=order)
        history_l = eval_forward(prob, α_l, order=order)
        ψf_r = history_r[:,end]
        ψf_l = history_l[:,end]
        if cost_type == :Infidelity
            cost_r = infidelity(ψf_r, target, N_ess)
            cost_l = infidelity(ψf_l, target, N_ess)
        elseif cost_type == :Tracking
            cost_r = 0.5*norm(ψf_r - target)^2
            cost_l = 0.5*norm(ψf_l - target)^2
        elseif cost_type == :Norm
            cost_r = 0.5*norm(ψf_r)^2
            cost_l = 0.5*norm(ψf_l)^2
        else
            throw("Invalid cost type: $cost_type")
        end
        grad[i] = (cost_r - cost_l)/(2*dα)
    end

    return grad
end

"""
Scalar control version.

Evaluates gradient of the provided Schrodinger problem with the given target
gate and control parameter(s) α using a finite difference method, where a step
size of dα is used when perturbing the control scalar α.

Returns: gradient
"""
function eval_grad_finite_difference(prob::SchrodingerProb, target::AbstractVector{Float64},
        α::Float64, dα=1e-5; order=2, cost_type=:Infidelity)

    α_r = α + dα
    α_l = α - dα
    history_r = eval_forward(prob, α_r, order=order)
    history_l = eval_forward(prob, α_l, order=order)
    ψf_r = history_r[:,end]
    ψf_l = history_l[:,end]
    if cost_type == :Infidelity
        cost_r = infidelity(ψf_r, target, prob.N_ess)
        cost_l = infidelity(ψf_l, target, prob.N_ess)
    elseif cost_type == :Tracking
        cost_r = 0.5*norm(ψf_r - target)^2
        cost_l = 0.5*norm(ψf_l - target)^2
    elseif cost_type == :Norm
        cost_r = 0.5*norm(ψf_r)^2
        cost_l = 0.5*norm(ψf_l)^2
    else
        throw("Invalid cost type: $cost_type")
    end
    grad = (cost_r - cost_l)/(2*dα)

    # Gradient should always be a vector for compatibility with discrete adjoint
    return [grad]
end


"""
Calculates the infidelity for the given state vector 'ψ' and target gate
'target.'

Reutrns: Infidelity
"""
function infidelity(ψ::Vector{Float64}, target::Vector{Float64}, N_ess::Int64)
    R = copy(target)
    N_tot = size(target,1)÷2
    T = vcat(R[1+N_tot:end], -R[1:N_tot])
    return 1 - (dot(ψ,R)^2 + dot(ψ,T)^2)/(N_ess^2)
end

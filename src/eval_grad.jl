function discrete_adjoint(prob::SchrodingerProb, target::Vector{Float64},
        α=missing; order=2, cost_type=:Infidelity, return_lambda_history=false)

    # Get state vector history
    history = eval_forward(prob, α, order=order)

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

    # For adjoint evolution, need transposes 
    #Ks_t = Matrix(transpose(Ks))
    #Ss_t = Matrix(transpose(Ss))
    #a_plus_adag_t = Matrix(transpose(a_plus_adag))
    #a_minus_adag_t = Matrix(transpose(a_minus_adag))
    Ks_t = Matrix(transpose(-Ks)) # NOTE THAT K changes sign!
    Ss_t = Matrix(transpose(Ss))
    a_plus_adag_t = Matrix(transpose(-a_plus_adag)) # NOTE THAT K changes sign!
    a_minus_adag_t = Matrix(transpose(a_minus_adag))
    
    dt = tf/nsteps

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
            RHS = 2*(dot(history[:,end],R)*R + dot(history[:,end],T)*T)
        elseif cost_type == :Tracking
            RHS = -(history[:,end] - target)
        elseif cost_type == :Norm
            RHS = -history[:,end]
        end

        LHS_map = LinearMap(
            x -> LHS_func(lambda_ut, lambda_vt, x[1:N_tot], x[1+N_tot:end],
                          Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
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
                  Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
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
                              Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
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

        # Accumulate Gradient
        grad = 0.0
        for n in 0:nsteps-1
            u = history[1:N_tot,1+n]
            v = history[1+N_tot:end,1+n]
            t = n*dt
            # Note that system matrices go to zero
            utvt!(ut, vt, u, v,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
                  dpda, dqda, t, α)

            dummy_u .= 0.0
            dummy_v .= 0.0

            axpy!(0.5*dt,ut,dummy_u)
            axpy!(0.5*dt,vt,dummy_v)

            dummy[1:N_tot] .= dummy_u
            dummy[1+N_tot:2*N_tot] .= dummy_v

            grad += dot(dummy, lambda_history[:,1+n+1])

            u = history[1:N_tot,1+n+1]
            v = history[1+N_tot:end,1+n+1]
            t = (n+1)*dt
            # Note that system matrices go to zero
            utvt!(ut, vt, u, v,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag,
                  dpda, dqda, t, α)

            dummy_u .= 0.0
            dummy_v .= 0.0

            axpy!(0.5*dt,ut,dummy_u)
            axpy!(0.5*dt,vt,dummy_v)

            dummy[1:N_tot] .= dummy_u
            dummy[1+N_tot:end] .= dummy_v

            grad += dot(dummy, lambda_history[:,1+n+1])
        end
        grad *= -1.0

    elseif order == 4
        # Terminal Condition
        t = tf

        if cost_type == :Infidelity
            RHS = 2*(dot(history[:,end],R)*R + dot(history[:,end],T)*T)
        elseif cost_type == :Tracking
            RHS = -(history[:,end] - target)
        elseif cost_type == :Norm
            RHS = -history[:,end]
        end

        LHS_map = LinearMap(
            x -> LHS_func_order4(lambda_utt, lambda_vtt, lambda_ut, lambda_vt,
                                 x[1:N_tot], x[1+N_tot:end],
                          Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
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
                  Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
                  p, q, t, α)
            uttvtt!(lambda_utt, lambda_vtt, lambda_ut, lambda_vt, lambda_u, lambda_v,
                Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
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
                              Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
                              p, q, dpdt, dqdt, t, α, dt, N_tot),
                2*N_tot,2*N_tot
            )

            gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)
            lambda_history[:,1+n] .= lambda
            lambda_u = lambda[1:N_tot]
            lambda_v = lambda[1+N_tot:2*N_tot]
        end

        # Need different gradient calculation for 4th order!!!
        zero_mat = zeros(N_tot,N_tot)
        uv = zeros(2*N_tot)
        u = zeros(N_tot)
        v = zeros(N_tot)
        # Won't actually hold ut and vt, but rather real/imag parts of dH/dα*ψ
        ut = zeros(N_tot)
        vt = zeros(N_tot)
        utt = zeros(N_tot)
        vtt = zeros(N_tot)
        dummy_u = zeros(N_tot)
        dummy_v = zeros(N_tot)
        dummy = zeros(2*N_tot)

        # Accumulate Gradient
        grad = 0.0
        #weights_n = [1,-1/3]
        #weights_np1 = [1,1/3]
        weights_n = [1,1/3]
        weights_np1 = [1,-1/3]
        for n in 0:nsteps-1
            # Qn contribution
            u = history[1:N_tot,1+n]
            v = history[1+N_tot:end,1+n]
            t = n*dt

            dummy_u .= 0.0
            dummy_v .= 0.0

            # Second Order Corrections
            utvt!(ut, vt, u, v,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
                  dpda, dqda, t, α)

            axpy!(0.5*dt*weights_n[1],ut,dummy_u)
            axpy!(0.5*dt*weights_n[1],vt,dummy_v)

            # Fourth Order Corrections
            utvt!(utt, vtt, ut, vt,
                  Ks, Ss, a_plus_adag, a_minus_adag, 
                  p, q, t, α)
            axpy!(0.25*dt^2*weights_n[2],utt,dummy_u)
            axpy!(0.25*dt^2*weights_n[2],vtt,dummy_v)


            utvt!(ut, vt, u, v,
                  Ks, Ss, a_plus_adag, a_minus_adag, 
                  p, q, t, α)
            utvt!(utt, vtt, ut, vt,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
                  dpda, dqda, t, α)
            axpy!(0.25*dt^2*weights_n[2],utt,dummy_u)
            axpy!(0.25*dt^2*weights_n[2],vtt,dummy_v)


            utvt!(utt, vtt, u, v,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
                  d2p_dta, d2q_dta, t, α)

            axpy!(0.25*dt^2*weights_n[2],utt,dummy_u)
            axpy!(0.25*dt^2*weights_n[2],vtt,dummy_v)

            dummy[1:N_tot] .= dummy_u
            dummy[1+N_tot:end] .= dummy_v

            grad += dot(dummy, lambda_history[:,1+n+1])



            u = history[1:N_tot,1+n+1]
            v = history[1+N_tot:end,1+n+1]
            t = (n+1)*dt

            dummy_u .= 0.0
            dummy_v .= 0.0


            # Second Order Corrections
            utvt!(ut, vt, u, v,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
                  dpda, dqda, t, α)

            axpy!(0.5*dt*weights_np1[1],ut,dummy_u)
            axpy!(0.5*dt*weights_np1[1],vt,dummy_v)

            # Fourth Order Corrections
            utvt!(utt, vtt, ut, vt,
                  Ks, Ss, a_plus_adag, a_minus_adag, 
                  p, q, t, α)
            axpy!(0.25*dt^2*weights_np1[2],utt,dummy_u)
            axpy!(0.25*dt^2*weights_np1[2],vtt,dummy_v)


            utvt!(ut, vt, u, v,
                  Ks, Ss, a_plus_adag, a_minus_adag, 
                  p, q, t, α)
            utvt!(utt, vtt, ut, vt,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
                  dpda, dqda, t, α)
            axpy!(0.25*dt^2*weights_np1[2],utt,dummy_u)
            axpy!(0.25*dt^2*weights_np1[2],vtt,dummy_v)


            utvt!(utt, vtt, u, v,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
                  d2p_dta, d2q_dta, t, α)

            axpy!(0.25*dt^2*weights_np1[2],utt,dummy_u)
            axpy!(0.25*dt^2*weights_np1[2],vtt,dummy_v)

            dummy[1:N_tot] .= dummy_u
            dummy[1+N_tot:end] .= dummy_v

            grad += dot(dummy, lambda_history[:,1+n+1])
        end
        grad *= -1.0
    else
        throw("Invalid order: $order")
    end

    if return_lambda_history
        return grad, history, lambda_history
    end
    return grad
end



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

    if order == 2
        forcing_ary = zeros(2*N_tot,1+prob.nsteps,1)
        forcing_vec = zeros(2*N_tot)

        u  = zeros(N_tot)
        v  = zeros(N_tot)
        ut = zeros(N_tot)
        vt = zeros(N_tot)

        nsteps = prob.nsteps
        t = 0
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
        t = 0
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
    # Initial conditions for dψ/dα
    differentiated_prob.u0 .= 0.0
    differentiated_prob.v0 .= 0.0

    history_dψdα = eval_forward_forced(differentiated_prob, forcing_ary, α, 
                                       order=order)

    dQda = history_dψdα[:,end]
    Q = history[:,end,1]
    R = copy(target)
    T = vcat(R[1+N_tot:end], -R[1:N_tot])

    if cost_type == :Infidelity
        gradient = -2*(dot(Q,R)*dot(dQda,R) + dot(Q,T)*dot(dQda,T))
    elseif cost_type == :Tracking
        gradient = dot(dQda, Q - target)
    elseif cost_type == :Norm
        gradient = dot(dQda,Q)
    else
        throw("Invalid cost type: $cost_type")
    end
    return gradient
end



# In the future, will need to updat to allow dα to work for vector-valued α
# (i.e., compute a whole vector gradient)
function eval_grad_finite_difference(prob::SchrodingerProb, target, α, dα=1e-5; order=2, cost_type=:Infidelity)
    # Centered Difference Approximation
    history_r = eval_forward(prob, α+dα, order=order)
    history_l = eval_forward(prob, α-dα, order=order)
    ψf_r = history_r[:,end]
    ψf_l = history_l[:,end]
    if cost_type == :Infidelity
        cost_r = infidelity(ψf_r, target)
        cost_l = infidelity(ψf_l, target)
    elseif cost_type == :Tracking
        cost_r = 0.5*norm(ψf_r - target)^2
        cost_l = 0.5*norm(ψf_l - target)^2
    elseif cost_type == :Norm
        cost_r = 0.5*norm(ψf_r)^2
        cost_l = 0.5*norm(ψf_l)^2
    else
        throw("Invalid cost type: $cost_type")
    end

    gradient = (cost_r - cost_l)/(2*dα)
    return gradient
end


function infidelity(ψ::Vector{Float64}, target::Vector{Float64})
    R = copy(target)
    N_tot = size(target,1)÷2
    T = vcat(R[1+N_tot:end], -R[1:N_tot])
    return 1 - (dot(ψ,R)^2 + dot(ψ,T)^2)
end

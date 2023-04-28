function discrete_adjoint(prob::SchrodingerProb, target::Vector{Float64}, α=missing; order=2, cost_type=:Infidelity)
    R = target[:]
    T = vcat(R[3:4], -R[1:2])
    
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

    lambda = zeros(4)
    lambda_history = zeros(4,1+nsteps)
    lambda_ut = zeros(2)
    lambda_vt = zeros(2)
    lambda_utt = zeros(2)
    lambda_vtt = zeros(2)

    RHS_lambda_u::Vector{Float64} = zeros(2)
    RHS_lambda_v::Vector{Float64} = zeros(2)
    RHS::Vector{Float64} = zeros(4)

    if order == 2
        println("Discrete Adjoint Order 2")
        # Terminal Condition
        t = tf

        if cost_type == :Infidelity
            RHS = 2*(dot(history[:,end],R)*R + dot(history[:,end],T)*T)
        elseif cost_type == :Norm
            RHS = history[:,end]
        end

        LHS_map = LinearMap(
            x -> LHS_func(lambda_ut, lambda_vt, x[1:2], x[3:4],
                          Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
                          p, q, t, α, dt),
            4,4
        )
        gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

        lambda_history[:,1+nsteps] .= lambda
        lambda_u = copy(lambda[1:2])
        lambda_v = copy(lambda[3:4])
        
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

            copyto!(RHS,1,RHS_lambda_u,1,2)
            copyto!(RHS,3,RHS_lambda_v,1,2)

            # NOTE: LHS and RHS Linear transformations use the SAME TIME

            LHS_map = LinearMap(
                x -> LHS_func(lambda_ut, lambda_vt, x[1:2], x[3:4],
                              Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
                              p, q, t, α, dt),
                4,4
            )

            gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)
            lambda_history[:,1+n+1] .= lambda
            lambda_u = lambda[1:2]
            lambda_v = lambda[3:4]
        end
    
        zero_mat = zeros(2,2)
        uv = zeros(4)
        u = zeros(2)
        v = zeros(2)
        # Won't actually hold ut and vt, but rather real/imag parts of dH/dα*ψ
        ut = zeros(2)
        vt = zeros(2)
        dummy_u = zeros(2)
        dummy_v = zeros(2)
        dummy = zeros(4)

        # Accumulate Gradient
        grad = 0.0
        for n in 0:nsteps-1
            u = history[1:2,1+n]
            v = history[3:4,1+n]
            t = n*dt
            # Note that system matrices go to zero
            utvt!(ut, vt, u, v,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
                  dpda, dqda, t, α)

            dummy_u .= 0.0
            dummy_v .= 0.0

            axpy!(0.5*dt,ut,dummy_u)
            axpy!(0.5*dt,vt,dummy_v)

            dummy[1:2] .= dummy_u
            dummy[3:4] .= dummy_v

            grad += dot(dummy, lambda_history[:,1+n+1])

            u = history[1:2,1+n+1]
            v = history[3:4,1+n+1]
            t = (n+1)*dt
            # Note that system matrices go to zero
            utvt!(ut, vt, u, v,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag,
                  dpda, dqda, t, α)

            dummy_u .= 0.0
            dummy_v .= 0.0

            axpy!(0.5*dt,ut,dummy_u)
            axpy!(0.5*dt,vt,dummy_v)

            dummy[1:2] .= dummy_u
            dummy[3:4] .= dummy_v

            grad += dot(dummy, lambda_history[:,1+n+1])
        end
        grad *= -1.0

    elseif order == 4
        println("Discrete Adjoint Order 4")
        # Terminal Condition
        t = tf

        if cost_type == :Infidelity
            RHS = 2*(dot(history[:,end],R)*R + dot(history[:,end],T)*T)
        elseif cost_type == :Norm
            RHS = history[:,end]
        end

        LHS_map = LinearMap(
            x -> LHS_func_order4(lambda_utt, lambda_vtt, lambda_ut, lambda_vt,
                          x[1:2], x[3:4],
                          Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
                          p, q, dpdt, dqdt, t, α, dt),
            4,4
        )
        #LHS_map = LinearMap(
        #    x -> LHS_func(lambda_ut, lambda_vt,
        #                  x[1:2], x[3:4],
        #                  Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
        #                  p, q, t, α, dt),
        #    4,4
        #)
        gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

        lambda_history[:,1+nsteps] .= lambda
        lambda_u = copy(lambda[1:2])
        lambda_v = copy(lambda[3:4])
        
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

            copyto!(RHS,1,RHS_lambda_u,1,2)
            copyto!(RHS,3,RHS_lambda_v,1,2)

            # NOTE: LHS and RHS Linear transformations use the SAME TIME

            LHS_map = LinearMap(
                x -> LHS_func_order4(lambda_utt, lambda_vtt, lambda_ut, lambda_vt,
                              x[1:2], x[3:4],
                              Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
                              p, q, dpdt, dqdt, t, α, dt),
                4,4
            )
            #LHS_map = LinearMap(
            #    x -> LHS_func(lambda_ut, lambda_vt,
            #                  x[1:2], x[3:4],
            #                  Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
            #                  p, q, t, α, dt),
            #    4,4
            #)

            gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)
            lambda_history[:,1+n] .= lambda
            lambda_u = lambda[1:2]
            lambda_v = lambda[3:4]
        end
        # Need different gradient calculation for 4th order!!!
        zero_mat = zeros(2,2)
        uv = zeros(4)
        u = zeros(2)
        v = zeros(2)
        # Won't actually hold ut and vt, but rather real/imag parts of dH/dα*ψ
        ut = zeros(2)
        vt = zeros(2)
        utt = zeros(2)
        vtt = zeros(2)
        dummy_u = zeros(2)
        dummy_v = zeros(2)
        dummy = zeros(4)

        # Accumulate Gradient
        grad = 0.0
        #weights_n = [1,-1/3]
        #weights_np1 = [1,1/3]
        weights_n = [1,1/3]
        weights_np1 = [1,-1/3]
        for n in 0:nsteps-1
            u = history[1:2,1+n]
            v = history[3:4,1+n]
            t = n*dt
            # Note that system matrices go to zero
            utvt!(ut, vt, u, v,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
                  dpda, dqda, t, α)
            #utvt!(utt, vtt, ut, vt,
            #      zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
            #      dpda, dqda, t, α)
            uttvtt!(utt, vtt, ut, vt, u, v,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
                  dpda, dqda, d2p_dta, d2q_dta, t, α)

            dummy_u .= 0.0
            dummy_v .= 0.0

            axpy!(0.5*dt*weights_n[1],ut,dummy_u)
            axpy!(0.25*dt^2*weights_n[2],utt,dummy_u)

            axpy!(0.5*dt*weights_n[1],vt,dummy_v)
            axpy!(0.25*dt^2*weights_n[2],vtt,dummy_v)

            dummy[1:2] .= dummy_u
            dummy[3:4] .= dummy_v

            grad += dot(dummy, lambda_history[:,1+n+1])

            u = history[1:2,1+n+1]
            v = history[3:4,1+n+1]
            t = (n+1)*dt

            utvt!(ut, vt, u, v,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
                  dpda, dqda, t, α)
            #utvt!(utt, vtt, ut, vt,
            #      zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
            #      dpda, dqda, t, α)
            uttvtt!(utt, vtt, ut, vt, u, v,
                  zero_mat, zero_mat, a_plus_adag, a_minus_adag, 
                  dpda, dqda, d2p_dta, d2q_dta, t, α)

            dummy_u .= 0.0
            dummy_v .= 0.0

            axpy!(0.5*dt*weights_np1[1],ut,dummy_u)
            axpy!(0.25*dt^2*weights_np1[2],utt,dummy_u)

            axpy!(0.5*dt*weights_np1[1],vt,dummy_v)
            axpy!(0.25*dt^2*weights_np1[2],vtt,dummy_v)

            dummy[1:2] .= dummy_u
            dummy[3:4] .= dummy_v

            grad += dot(dummy, lambda_history[:,1+n+1])
        end
        grad *= -1.0
    else
        throw("Invalid order: $order")
    end


    #return lambda_history, grad
    return grad
end



function eval_grad_forced(prob::SchrodingerProb, target, α=1.0; order=2, cost_type=:Infidelity)
    # Get state vector history
    history = eval_forward(prob, α, order=order, return_time_derivatives=true)

    ## Prepare forcing (-idH/dα ψ)
    # Prepare dH/dα
    
    # System hamiltonian is constant, falls out when taking derivative
    dKs_da::Matrix{Float64} = zeros(2,2)
    dSs_da::Matrix{Float64} = zeros(2,2)
    a_plus_adag = prob.a_plus_adag
    a_minus_adag = prob.a_minus_adag
    dpda = prob.dpda
    dqda = prob.dqda
    d2p_dta = prob.d2p_dta
    d2q_dta = prob.d2q_dta

    if order == 2
        forcing_ary = zeros(4,1+prob.nsteps,1)
        forcing_vec = zeros(4)

        u = zeros(2)
        v = zeros(2)
        ut = zeros(2)
        vt = zeros(2)

        nsteps = prob.nsteps
        t = 0
        dt = prob.tf/nsteps

        # Get forcing (dH/dα * ψ)
        for n in 0:nsteps
            copyto!(u,history[1:2,1+n,1])
            copyto!(v,history[3:4,1+n,1])

            utvt!(ut, vt, u, v, dKs_da, dSs_da, a_plus_adag, a_minus_adag, dpda, dqda, t, α)
            copyto!(forcing_vec,1,ut,1,2)
            copyto!(forcing_vec,3,vt,1,2)

            forcing_ary[:,1+n,1] .= forcing_vec

            t += dt
        end
    elseif order == 4
        forcing_ary = zeros(4,1+prob.nsteps,2)
        forcing_vec2 = zeros(4)
        forcing_vec4 = zeros(4)

        u = zeros(2)
        v = zeros(2)
        ut = zeros(2)
        vt = zeros(2)

        nsteps = prob.nsteps
        t = 0
        dt = prob.tf/nsteps

        # Get forcing (dH/dα * ψ)
        for n in 0:nsteps
            # Second Order Forcing
            u .= history[1:2,1+n,1]
            v .= history[3:4,1+n,1]

            utvt!(ut, vt, u, v, 
                  dKs_da, dSs_da, a_plus_adag, a_minus_adag,
                  dpda, dqda, t, α)
            forcing_vec2[1:2] .= ut
            forcing_vec2[3:4] .= vt

            forcing_ary[:,1+n,1] .= forcing_vec2

            # Fourth Order Forcing
            forcing_vec4 = zeros(4)

            utvt!(ut, vt, u, v,
                  dKs_da, dSs_da, a_plus_adag, a_minus_adag,
                  d2p_dta, d2q_dta, t, α)
            forcing_vec4[1:2] += ut
            forcing_vec4[3:4] += vt

            ut .= history[1:2,1+n,2]
            vt .= history[3:4,1+n,2]

            A = zeros(2) # Placeholders
            B = zeros(2)
            utvt!(A, B, ut, vt,
                  dKs_da, dSs_da, a_plus_adag, a_minus_adag,
                  dpda, dqda, t, α)
            forcing_vec4[1:2] += A
            forcing_vec4[3:4] += B

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
    copyto!(differentiated_prob.u0, [0.0,0.0])
    copyto!(differentiated_prob.v0, [0.0,0.0])

    history_dψdα = eval_forward_forced(differentiated_prob, forcing_ary, α, order=order)

    dQda = history_dψdα[:,end]
    Q = history[:,end,1]
    R = copy(target)
    T = vcat(R[3:4], -R[1:2])

    if cost_type == :Infidelity
        gradient = -2*(dot(Q,R)*dot(dQda,R) + dot(Q,T)*dot(dQda,T))
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
    T = vcat(R[3:4], -R[1:2])
    return 1 - (dot(ψ,R)^2 + dot(ψ,T)^2)
end

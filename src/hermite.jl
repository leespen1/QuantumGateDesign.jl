using LinearAlgebra
using LinearMaps
using IterativeSolvers
using ForwardDiff

mutable struct SchrodingerProb
    Ks::Matrix{Float64}
    Ss::Matrix{Float64}
    a_plus_adag::Matrix{Float64} # a + a^†
    a_minus_adag::Matrix{Float64} # a - a^†
    p::Function
    q::Function
    dpdt::Function
    dqdt::Function
    dpda::Function
    dqda::Function
    d2p_dta::Function
    d2q_dta::Function
    u0::Vector{Float64}
    v0::Vector{Float64}
    tf::Float64
    nsteps::Int64
    function SchrodingerProb(
            Ks::Matrix{Float64},
            Ss::Matrix{Float64},
            p::Function,
            q::Function,
            dpdt::Function,
            dqdt::Function,
            dpda::Function,
            dqda::Function,
            d2p_dta::Function,
            d2q_dta::Function,
            u0::Vector{Float64},
            v0::Vector{Float64},
            tf::Float64,
            nsteps::Int64
        )

        a_plus_adag::Matrix{Float64} = [0.0 1.0; 1.0 0.0]
        a_minus_adag::Matrix{Float64} = [0.0 1.0; -1.0 0.0]

        new(copy(Ks), copy(Ss), a_plus_adag, a_minus_adag,
            p, q, dpdt, dqdt, dpda, dqda, d2p_dta, d2q_dta,
            copy(u0), copy(v0), tf, nsteps)
    end
end



function SchrodingerProb(Ks,Ss,p,q,u0,v0,tf,nsteps)
    dpdt(t,a) = nothing
    dqdt(t,a) = nothing
    dpda(t,a) = nothing
    dqda(t,a) = nothing
    d2p_dta(t,a) = nothing
    d2q_dta(t,a) = nothing
    return SchrodingerProb(Ks, Ss,
                           p, q, dpdt, dqdt, dpda, dqda, d2p_dta, d2q_dta,
                           u0,v0,
                           tf,nsteps)
end



function Base.copy(prob::SchrodingerProb)
    return SchrodingerProb(prob.Ks, prob.Ss,
                           prob.p, prob.q, prob.dpdt, prob.dqdt,
                           prob.dpda, prob.dqda, prob.d2p_dta, prob.d2q_dta,
                           prob.u0, prob.v0,
                           prob.tf, prob.nsteps)
end



function utvt!(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α)
    #Should I provide the functions p and q? or the values at (t,α)?

    #mul!(Y, A, B) -> Y
    #Calculates the matrix-matrix or matrix-vector product AB and stores the result in Y, overwriting the existing value of Y. Note that Y must not be aliased with either A or B.
    #mul!(C, A, B, α, β) -> C
    #Combined inplace matrix-matrix or matrix-vector multiply-add A B α + C β. The result is stored in C by overwriting it. Note that C must not be aliased with either A or B.
    
    # Non-Memory-Allocating Version (test performance)
    # ut = (Ss + q(t)(a-a†))u - (Ks + p(t)(a+a†))v
    mul!(ut, Ss, u)
    mul!(ut, a_minus_adag, u, q(t,α), 1)
    mul!(ut, Ks, v, -1, 1)
    mul!(ut, a_plus_adag, v, -p(t,α), 1)

    # vt = (Ss + q(t)(a-a†))v + (Ks + p(t)(a+a†))u
    mul!(vt, Ss, v)
    mul!(vt, a_minus_adag, v, q(t,α), 1)
    mul!(vt, Ks, u, 1, 1)
    mul!(vt, a_plus_adag, u, p(t,α), 1)
end

function uttvtt!(utt, vtt, ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, dpdt, dqdt, t, α)
    ## Make use of utvt!
    utvt!(utt, vtt, ut, vt, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α)
    ## Replicate utvt! explicitly 
    #mul!(utt, Ss, ut)
    #mul!(utt, a_minus_adag, ut, q(t,α), 1)
    #mul!(utt, Ks, vt, -1, 1)
    #mul!(utt, a_plus_adag, vt, -p(t,α), 1)

    mul!(utt, a_minus_adag, u, dqdt(t,α), 1)
    mul!(utt, a_plus_adag, v, -dpdt(t,α), 1)

    ### Replicate utvt! explicitly 
    #mul!(vtt, Ss, vt)
    #mul!(vtt, a_minus_adag, vt, q(t,α), 1)
    #mul!(vtt, Ks, ut, 1, 1)
    #mul!(vtt, a_plus_adag, ut, p(t,α), 1)

    mul!(vtt, a_minus_adag, v, dqdt(t,α), 1)
    mul!(vtt, a_plus_adag, u, dpdt(t,α), 1)
end


function LHS_func(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α, dt)
    utvt!(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α)
    
    LHSu = copy(u)
    axpy!(-0.5*dt,ut,LHSu)
    LHSv = copy(v)
    axpy!(-0.5*dt,vt,LHSv)

    LHS_uv = zeros(4)
    copyto!(LHS_uv,1,LHSu,1,2)
    copyto!(LHS_uv,3,LHSv,1,2)

    return LHS_uv
end

function LHS_func_order4(utt, vtt, ut, vt, u, v,
        Ks, Ss, a_plus_adag, a_minus_adag,
        p, q, dpdt, dqdt, t, α, dt)

    utvt!(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α)
    uttvtt!(utt, vtt, ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, dpdt, dqdt, t, α)

    weights = [1,-1/3]
    
    LHSu = copy(u)
    axpy!(-0.5*dt*weights[1],ut,LHSu)
    axpy!(-0.25*dt^2*weights[2],utt,LHSu)
    LHSv = copy(v)
    axpy!(-0.5*dt*weights[1],vt,LHSv)
    axpy!(-0.25*dt^2*weights[2],vtt,LHSv)

    LHS_uv = zeros(4)
    copyto!(LHS_uv,1,LHSu,1,2)
    copyto!(LHS_uv,3,LHSv,1,2)

    return LHS_uv
end



function eval_forward(prob::SchrodingerProb, α=missing; order=2, return_time_derivatives=false)
    if order == 2
        return eval_forward_order2(prob, α, return_time_derivatives=return_time_derivatives)
    elseif order == 4
        return eval_forward_order4(prob, α, return_time_derivatives=return_time_derivatives)
    end
    throw("Invalid order: $order")
end


function eval_forward_order2(prob::SchrodingerProb, α=missing;
        return_time_derivatives=false)
    Ks = prob.Ks
    Ss = prob.Ss
    a_plus_adag = prob.a_plus_adag
    a_minus_adag = prob.a_minus_adag
    p = prob.p
    q = prob.q
    u0 = prob.u0
    v0 = prob.v0
    tf = prob.tf
    nsteps = prob.nsteps

    t = 0
    dt = tf/nsteps


    uv = zeros(4)
    copyto!(uv,1,u0,1,2)
    copyto!(uv,3,v0,1,2)
    uv_history = Matrix{Float64}(undef,4,1+nsteps)
    uv_history[:,1] .= uv
    utvt_history = Matrix{Float64}(undef,4,1+nsteps)

    RHSu::Vector{Float64} = zeros(2)
    RHSv::Vector{Float64} = zeros(2)
    RHS_uv::Vector{Float64} = zeros(4)

    u = copy(u0)
    v = copy(v0)
    ut = zeros(2)
    vt = zeros(2)

    # Order 2
    println("2nd Order")

    for i in 1:nsteps
        utvt!(ut, vt, u, v,
              Ks, Ss, a_plus_adag, a_minus_adag,
              p, q, t, α)

        utvt_history[1:2,1+(i-1)] .= ut
        utvt_history[3:4,1+(i-1)] .= vt

        copy!(RHSu,u)
        axpy!(0.5*dt,ut,RHSu)

        copy!(RHSv,v)
        axpy!(0.5*dt,vt,RHSv)

        copyto!(RHS_uv,1,RHSu,1,2)
        copyto!(RHS_uv,3,RHSv,1,2)

        t += dt

        LHS_map = LinearMap(
            uv -> LHS_func(ut, vt, uv[1:2], uv[3:4],
                           Ks, Ss, a_plus_adag, a_minus_adag,
                           p, q, t, α, dt),
            4,4
        )

        gmres!(uv, LHS_map, RHS_uv, abstol=1e-15, reltol=1e-15)
        uv_history[:,1+i] .= uv
        u = uv[1:2]
        v = uv[3:4]
    end

    # One last time, for utvt history at final time
    utvt!(ut, vt, u, v,
          Ks, Ss, a_plus_adag, a_minus_adag,
          p, q, t, α)

    utvt_history[1:2,1+nsteps] .= ut
    utvt_history[3:4,1+nsteps] .= vt

    if return_time_derivatives
        return cat(uv_history, utvt_history, dims=3)
    end
    return uv_history
end

function eval_forward_order4(prob::SchrodingerProb, α=missing;
        return_time_derivatives=false)
    Ks = prob.Ks
    Ss = prob.Ss
    a_plus_adag = prob.a_plus_adag
    a_minus_adag = prob.a_minus_adag
    p = prob.p
    q = prob.q
    dpdt = prob.dpdt
    dqdt = prob.dqdt
    u0 = prob.u0
    v0 = prob.v0
    tf = prob.tf
    nsteps = prob.nsteps

    t = 0
    dt = tf/nsteps


    uv = zeros(4)
    copyto!(uv,1,u0,1,2)
    copyto!(uv,3,v0,1,2)
    uv_history = Matrix{Float64}(undef,4,1+nsteps)
    uv_history[:,1] .= uv
    utvt_history = Matrix{Float64}(undef,4,1+nsteps)
    uttvtt_history = Matrix{Float64}(undef,4,1+nsteps)

    RHSu::Vector{Float64} = zeros(2)
    RHSv::Vector{Float64} = zeros(2)
    RHS_uv::Vector{Float64} = zeros(4)

    u = copy(u0)
    v = copy(v0)
    ut = zeros(2)
    vt = zeros(2)
    utt = zeros(2)
    vtt = zeros(2)

    # Order 4
    println("4th Order")

    for i in 1:nsteps
        utvt!(ut, vt, u, v,
              Ks, Ss, a_plus_adag, a_minus_adag,
              p, q, t, α)
        uttvtt!(utt, vtt, ut, vt, u, v,
                Ks, Ss, a_plus_adag, a_minus_adag,
                p, q, dpdt, dqdt, t, α)

        utvt_history[1:2,1+(i-1)] .= ut
        utvt_history[3:4,1+(i-1)] .= vt
        uttvtt_history[1:2,1+(i-1)] .= utt
        uttvtt_history[3:4,1+(i-1)] .= vtt

        weights = [1,1/3]
        copy!(RHSu,u)
        axpy!(0.5*dt*weights[1],ut,RHSu)
        axpy!(0.25*dt^2*weights[2],utt,RHSu)

        copy!(RHSv,v)
        axpy!(0.5*dt*weights[1],vt,RHSv)
        axpy!(0.25*dt^2*weights[2],vtt,RHSv)

        copyto!(RHS_uv,1,RHSu,1,2)
        copyto!(RHS_uv,3,RHSv,1,2)

        t += dt

        LHS_map = LinearMap(
            uv -> LHS_func_order4(utt, vtt, ut, vt, uv[1:2], uv[3:4],
                                  Ks, Ss, a_plus_adag, a_minus_adag,
                                  p, q, dpdt, dqdt, t, α, dt),
            4,4
        )

        gmres!(uv, LHS_map, RHS_uv)
        uv_history[:,1+i] .= uv
        u = uv[1:2]
        v = uv[3:4]
    end

    # One last time, for utvt history at final time
    utvt!(ut, vt, u, v,
          Ks, Ss, a_plus_adag, a_minus_adag,
          p, q, t, α)
    uttvtt!(utt, vtt, ut, vt, u, v,
            Ks, Ss, a_plus_adag, a_minus_adag,
            p, q, dpdt, dqdt, t, α)

    utvt_history[1:2,1+nsteps] .= ut
    utvt_history[3:4,1+nsteps] .= vt
    uttvtt_history[1:2,1+nsteps] .= utt
    uttvtt_history[3:4,1+nsteps] .= vtt

    if return_time_derivatives
        return cat(uv_history, utvt_history, uttvtt_history, dims=3)
    end
    return uv_history
end


function eval_forward_forced(prob::SchrodingerProb, forcing_ary::Array{Float64,3}, α=missing; order=2)
    if order == 2
        return eval_forward_forced_order2(prob, forcing_ary, α)
    elseif order == 4
        return eval_forward_forced_order4(prob, forcing_ary, α)
    end

    throw("Invalid Order: $order")
end

"""
Evolve schrodinger problem with forcing applied, and forcing given as an array
of forces at each discretized point in time.

Maybe I should also do a one with forcing functions as well.
"""
function eval_forward_forced_order2(prob::SchrodingerProb, forcing_ary::Array{Float64,3}, α=missing)
    Ks = prob.Ks
    Ss = prob.Ss
    a_plus_adag = prob.a_plus_adag
    a_minus_adag = prob.a_minus_adag
    p = prob.p
    q = prob.q
    u0 = prob.u0
    v0 = prob.v0
    tf = prob.tf
    nsteps = prob.nsteps

    t = 0
    dt = tf/nsteps

    uv = zeros(4)
    copyto!(uv,1,u0,1,2)
    copyto!(uv,3,v0,1,2)
    uv_history = Matrix{Float64}(undef,4,1+nsteps)
    uv_history[:,1] .= uv

    RHSu::Vector{Float64} = zeros(2)
    RHSv::Vector{Float64} = zeros(2)
    RHS_uv::Vector{Float64} = zeros(4)

    u = copy(u0)
    v = copy(v0)
    ut = zeros(2)
    vt = zeros(2)
    utt = zeros(2)
    vtt = zeros(2)

    for i in 1:nsteps
        utvt!(ut, vt, u, v,
              Ks, Ss, a_plus_adag, a_minus_adag,
              p, q, t, α)
        copy!(RHSu,u)
        axpy!(0.5*dt,ut,RHSu)

        copy!(RHSv,v)
        axpy!(0.5*dt,vt,RHSv)

        copyto!(RHS_uv,1,RHSu,1,2)
        copyto!(RHS_uv,3,RHSv,1,2)

        axpy!(0.5*dt,forcing_ary[:,i,1], RHS_uv)
        axpy!(0.5*dt,forcing_ary[:,i+1,1], RHS_uv)

        t += dt

        LHS_map = LinearMap(
            uv -> LHS_func(ut, vt, uv[1:2], uv[3:4],
                           Ks, Ss, a_plus_adag, a_minus_adag,
                           p, q, t, α, dt),
            4,4
        )

        gmres!(uv, LHS_map, RHS_uv, abstol=1e-15, reltol=1e-15)
        uv_history[:,1+i] .= uv
        u = uv[1:2]
        v = uv[3:4]
    end
    
    return uv_history
end



function eval_forward_forced_order4(prob::SchrodingerProb, forcing_ary::Array{Float64,3}, α=missing)
    Ks = prob.Ks
    Ss = prob.Ss
    a_plus_adag = prob.a_plus_adag
    a_minus_adag = prob.a_minus_adag
    p = prob.p
    q = prob.q
    dpdt = prob.dpdt
    dqdt = prob.dqdt
    u0 = prob.u0
    v0 = prob.v0
    tf = prob.tf
    nsteps = prob.nsteps

    t = 0
    dt = tf/nsteps

    uv = zeros(4)
    copyto!(uv,1,u0,1,2)
    copyto!(uv,3,v0,1,2)
    uv_history = Matrix{Float64}(undef,4,1+nsteps)
    uv_history[:,1] .= uv

    RHSu::Vector{Float64} = zeros(2)
    RHSv::Vector{Float64} = zeros(2)
    RHS_uv::Vector{Float64} = zeros(4)

    u = copy(u0)
    v = copy(v0)
    ut = zeros(2)
    vt = zeros(2)
    utt = zeros(2)
    vtt = zeros(2)

    for i in 1:nsteps
        utvt!(ut, vt, u, v,
              Ks, Ss, a_plus_adag, a_minus_adag,
              p, q, t, α)
        uttvtt!(utt, vtt, ut, vt, u, v,
                Ks, Ss, a_plus_adag, a_minus_adag,
                p, q, dpdt, dqdt, t, α)

        weights = [1,1/3]
        copy!(RHSu,u)
        axpy!(0.5*dt,ut,RHSu)
        axpy!(0.25*dt^2*weights[2],utt,RHSu)

        copy!(RHSv,v)
        axpy!(0.5*dt,vt,RHSv)
        axpy!(0.25*dt^2*weights[2],vtt,RHSv)

        copyto!(RHS_uv,1,RHSu,1,2)
        copyto!(RHS_uv,3,RHSv,1,2)

        axpy!(0.5*dt,forcing_ary[:,i,1], RHS_uv)
        axpy!(0.5*dt,forcing_ary[:,i+1,1], RHS_uv)
        axpy!(0.25*dt^2,forcing_ary[:,i,2], RHS_uv)
        axpy!(0.25*dt^2,forcing_ary[:,i+1,2], RHS_uv)

        t += dt

        LHS_map = LinearMap(
            uv -> LHS_func_order4(utt, vtt, ut, vt, uv[1:2], uv[3:4],
                           Ks, Ss, a_plus_adag, a_minus_adag,
                           p, q, dpdt, dqdt, t, α, dt),
            4,4
        )

        gmres!(uv, LHS_map, RHS_uv, abstol=1e-15, reltol=1e-15)
        uv_history[:,1+i] .= uv
        u = uv[1:2]
        v = uv[3:4]
    end
    
    return uv_history
end



function discrete_adjoint(prob::SchrodingerProb, target::Vector{Float64}, α=missing; order=2)
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
        # Terminal Condition
        t = tf
        LHS_map = LinearMap(
            x -> LHS_func(lambda_ut, lambda_vt, x[1:2], x[3:4],
                          Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
                          p, q, t, α, dt),
            4,4
        )
        RHS = 2*(dot(history[:,end],R)*R + dot(history[:,end],T)*T)
        gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

        lambda_history[:,1+nsteps] .= lambda
        lambda_u = copy(lambda[1:2])
        lambda_v = copy(lambda[3:4])
        

        t = tf - dt
        # Discrete Adjoint Scheme
        for i in nsteps-1:-1:1
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
                x -> LHS_func(lambda_ut, lambda_vt, x[1:2], x[3:4], Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t, p, q, t, α, dt),
                4,4
            )

            gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)
            lambda_history[:,1+i] .= lambda
            lambda_u = lambda[1:2]
            lambda_v = lambda[3:4]
            t -= dt
        end

    elseif order == 4
        # Terminal Condition
        t = tf
        LHS_map = LinearMap(
            x -> LHS_func_order4(lambda_utt, lambda_vtt, lambda_ut, lambda_vt,
                          x[1:2], x[3:4],
                          Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
                          p, q, dpdt, dqdt, t, α, dt),
            4,4
        )
        RHS = 2*(dot(history[:,end],R)*R + dot(history[:,end],T)*T)
        gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

        lambda_history[:,1+nsteps] .= lambda
        lambda_u = copy(lambda[1:2])
        lambda_v = copy(lambda[3:4])
        
        t = tf - dt
        # Discrete Adjoint Scheme
        for i in nsteps-1:-1:1
            utvt!(lambda_ut, lambda_vt, lambda_u, lambda_v,
                  Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
                  p, q, t, α)
            uttvtt!(lambda_utt, lambda_vtt, lambda_ut, lambda_vt, lambda_u, lambda_v,
                Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
                p, q, dpdt, dqdt, t, α)

            weights = [1,1/3]
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

            gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)
            lambda_history[:,1+i] .= lambda
            lambda_u = lambda[1:2]
            lambda_v = lambda[3:4]
            t -= dt
        end


    else
        throw("Invalid order: $order")
    end

    zero_mat = zeros(2,2)
    uv = zeros(4)
    u = zeros(2)
    v = zeros(2)
    # Won't actually hold ut and vt, but rather real/imag parts of dH/dα*ψ
    ut = zeros(2)
    vt = zeros(2)

    grad = 0.0
    for n in 0:nsteps-1
        u = history[1:2,1+n]
        v = history[3:4,1+n]
        t = n*dt
        # Note that system matrices go to zero
        utvt!(ut, vt, u, v, zero_mat, zero_mat, a_plus_adag, a_minus_adag, dpda, dqda, t, α)
        grad += dot(vcat(ut,vt), lambda_history[:,1+n+1])

        u = history[1:2,1+n+1]
        v = history[3:4,1+n+1]
        t = (n+1)*dt
        # Note that system matrices go to zero
        utvt!(ut, vt, u, v, zero_mat, zero_mat, a_plus_adag, a_minus_adag, dpda, dqda, t, α)
        grad += dot(vcat(ut,vt), lambda_history[:,1+n+1])
    end
    grad *= -0.5*dt

    #return lambda_history, grad
    return grad
end



function eval_grad_forced(prob, target, α=1.0; order=2)
    # Get state vector history
    history = eval_forward(prob, α, order=order, return_time_derivatives=true)

    ## Prepare forcing (-idH/dα ψ)
    # Prepare dH/dα
    
    # System hamiltonian is constant, falls out when taking derivative
    Ks::Matrix{Float64} = zeros(2,2)
    Ss::Matrix{Float64} = zeros(2,2)
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
        for i in 0:nsteps
            copyto!(u,history[1:2,1+i,1])
            copyto!(v,history[3:4,1+i,1])

            utvt!(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, dpda, dqda, t, α)
            copyto!(forcing_vec,1,ut,1,2)
            copyto!(forcing_vec,3,vt,1,2)

            forcing_ary[:,1+i,1] .= forcing_vec

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
        for i in 0:nsteps
            # Second Order Forcing
            u .= history[1:2,1+i,1]
            v .= history[3:4,1+i,1]

            utvt!(ut, vt, u, v, 
                  Ks, Ss, a_plus_adag, a_minus_adag,
                  dpda, dqda, t, α)
            forcing_vec2[1:2] .= ut
            forcing_vec2[3:4] .= vt

            forcing_ary[:,1+i,1] .= forcing_vec2

            # Fourth Order Forcing
            #= Do I double count the contributions from first derivative?
            # H*∂H/∂α*ψ
            utvt!(ut, vt, forcing_vec2[1:2], forcing_vec2[3:4],
                  prob.Ks, prob.Ss, a_plus_adag, a_minus_adag,
                  dpda, dqda, t, α)
            forcing_vec4[1:2] .= ut
            forcing_vec4[3:4] .= vt
            =#
            forcing_vec4 = zeros(4)

            utvt!(ut, vt, u, v,
                  Ks, Ss, a_plus_adag, a_minus_adag,
                  d2p_dta, d2q_dta, t, α)
            forcing_vec4[1:2] += ut
            forcing_vec4[3:4] += vt

            u .= history[1:2,1+i,2]
            v .= history[3:4,1+i,2]
            utvt!(ut, vt, u, v,
                  Ks, Ss, a_plus_adag, a_minus_adag,
                  dpda, dqda, t, α)
            forcing_vec4[1:2] += ut
            forcing_vec4[3:4] += vt

            forcing_ary[:,1+i,2] .= forcing_vec4

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

    gradient = -2*(dot(Q,R)*dot(dQda,R) + dot(Q,T)*dot(dQda,T))
    return gradient
end


# Deprecated, dpda, and dqda moved to SchrodingerProblem struct. 
# If trying to use automatic differentiation, do it when creating SchrodingerProblem
#=
"""
Evaluate gradient using differentiated/forced method approach.
Use forward diff for calculating gradients.
"""
function eval_grad_auto_forced(prob::SchrodingerProb, target, α=1.0; order=2)
    # dp/dα and dq/dα using auto (forward) differentiation
    p_wrapped(t_a_vec) = prob.p(t_a_vec[1], t_a_vec[2])
    q_wrapped(t_a_vec) = prob.q(t_a_vec[1], t_a_vec[2])
    dpda(t,α) = ForwardDiff.gradient(p_wrapped, [t,α])[2]
    dqda(t,α) = ForwardDiff.gradient(q_wrapped, [t,α])[2]

    return eval_grad_forced(prob, target, dpda, dqda, α, order=order)
end
=#



function infidelity(ψ::Vector{Float64}, target::Vector{Float64})
    R = copy(target)
    T = vcat(R[3:4], -R[1:2])
    return 1 - (dot(ψ,R)^2 + dot(ψ,T)^2)
end

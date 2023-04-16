using LinearAlgebra
using LinearMaps
using IterativeSolvers

mutable struct SchrodingerProb
    Ks::Matrix{Float64}
    Ss::Matrix{Float64}
    a_plus_adag::Matrix{Float64} # a + a^†
    a_minus_adag::Matrix{Float64} # a - a^†
    p::Function
    q::Function
    u0::Vector{Float64}
    v0::Vector{Float64}
    tf::Float64
    nsteps::Int64
end



function utvt!(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α)
    #Should I provide the functions p and q? or the values at (t,α)?

    #mul!(Y, A, B) -> Y
    #Calculates the matrix-matrix or matrix-vector product AB and stores the result in Y, overwriting the existing value of Y. Note that Y must not be aliased with either A or B.
    #mul!(C, A, B, α, β) -> C
    #Combined inplace matrix-matrix or matrix-vector multiply-add A B α + C β. The result is stored in C by overwriting it. Note that C must not be aliased with either A or B.
    
    ## Memory Allocating Version
    #ut = Ss*u
    #ut += q(t,α)*a_minus_adag*u
    #ut -= Ks*a_plus_adag*u
    #ut -= p(t,α)*a_plus_adag*u

    #vt = Ss*v
    #vt += q(t,α)*a_minus_adag*v
    #vt += Ks*a_plus_adag*v
    #vt += p(t,α)*a_plus_adag*v
    
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



function eval_forward(prob::SchrodingerProb, α=missing)
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

    for i in 1:nsteps
        utvt!(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α)
        copy!(RHSu,u)
        axpy!(0.5*dt,ut,RHSu)

        copy!(RHSv,v)
        axpy!(0.5*dt,vt,RHSv)

        copyto!(RHS_uv,1,RHSu,1,2)
        copyto!(RHS_uv,3,RHSv,1,2)

        t += dt

        LHS_map = LinearMap(
            uv -> LHS_func(ut, vt, uv[1:2], uv[3:4], Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α, dt),
            4,4
        )

        gmres!(uv, LHS_map, RHS_uv)
        uv_history[:,1+i] .= uv
        u = uv[1:2]
        v = uv[3:4]
    end

    return uv_history
end






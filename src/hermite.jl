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
            u0::Vector{Float64},
            v0::Vector{Float64},
            tf::Float64,
            nsteps::Int64
        )

        a_plus_adag::Matrix{Float64} = [0.0 1.0; 1.0 0.0]
        a_minus_adag::Matrix{Float64} = [0.0 1.0; -1.0 0.0]

        new(copy(Ks), copy(Ss), a_plus_adag, a_minus_adag,
            p, q, dpdt, dqdt,
            copy(u0), copy(v0), tf, nsteps)
    end
end

function Base.copy(prob::SchrodingerProb)
    return SchrodingerProb(prob.Ks, prob.Ss,
                           prob.p, prob.q, prob.dpdt, prob.dqdt,
                           prob.u0, prob.v0,
                           prob.tf, prob.nsteps)
end
#=
mutable struct SchrodingerProb
    Ks::Matrix{Float64}
    Ss::Matrix{Float64}
    a_plus_adag::Matrix{Float64} # a + a^†
    a_minus_adag::Matrix{Float64} # a - a^†
    p::Function
    q::Function
    dpdt::Function
    dqdt::Function
    u0::Vector{Float64}
    v0::Vector{Float64}
    tf::Float64
    nsteps::Int64
end
=#

function SchrodingerProb(Ks,Ss,p,q,u0,v0,tf,nsteps)
    dpdt(t,a) = nothing
    dqdt(t,a) = nothing
    return SchrodingerProb(Ks,Ss,p,q, dpdt, dqdt, u0,v0,tf,nsteps)
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



function eval_forward(prob::SchrodingerProb, α=missing; order=2)
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

    if order == 2
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
    else
        # Order 4
        println("4th Order")
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


    end
    return uv_history
end





"""
Evolve schrodinger problem with forcing applied, and forcing given as an array
of forces at each discretized point in time.

Maybe I should also do a one with forcing functions as well.
"""
function eval_forward_forced(prob::SchrodingerProb, forcing_mat::Matrix{Float64}, α=missing)
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

        # Apply forcing (trapezoidal rule on forcing, is explicit)
        axpy!(0.5*dt,forcing_mat[:,i], RHS_uv)
        axpy!(0.5*dt,forcing_mat[:,i+1], RHS_uv)

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



function discrete_adjoint(prob::SchrodingerProb, target::Vector{Float64}, dpda, dqda, α=missing)
    R = target[:]
    T = vcat(R[3:4], -R[1:2])
    
    # Get state vector history
    history = eval_forward(prob, α)

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
    ut = zeros(2)
    vt = zeros(2)

    # Terminal Condition
    t = tf
    LHS_map = LinearMap(
        x -> LHS_func(ut, vt, x[1:2], x[3:4],
                      Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t,
                      p, q, t, α, dt),
        4,4
    )
    RHS = 2*(dot(history[:,end],R)*R + dot(history[:,end],T)*T)
    gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

    lambda_history[:,1+nsteps] .= lambda
    u = copy(lambda[1:2])
    v = copy(lambda[3:4])
    
    RHSu::Vector{Float64} = zeros(2)
    RHSv::Vector{Float64} = zeros(2)
    RHS::Vector{Float64} = zeros(4)

    t = tf - dt
    # Discrete Adjoint Scheme
    for i in nsteps-1:-1:1
        utvt!(ut, vt, u, v, Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t, p, q, t, α)
        copy!(RHSu,u)
        axpy!(0.5*dt,ut,RHSu)

        copy!(RHSv,v)
        axpy!(0.5*dt,vt,RHSv)

        copyto!(RHS,1,RHSu,1,2)
        copyto!(RHS,3,RHSv,1,2)

        # NOTE: LHS and RHS Linear transformations use the SAME TIME

        LHS_map = LinearMap(
            x -> LHS_func(ut, vt, x[1:2], x[3:4], Ks_t, Ss_t, a_plus_adag_t, a_minus_adag_t, p, q, t, α, dt),
            4,4
        )

        gmres!(lambda, LHS_map, RHS, abstol=1e-15, reltol=1e-15)
        lambda_history[:,1+i] .= lambda
        u = lambda[1:2]
        v = lambda[3:4]
        t -= dt
    end

    grad = 0.0
    zero_mat = zeros(2,2)
    uv = zeros(4)
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



function eval_grad_forced(prob, target, dpda, dqda, α=1.0)
    # Get state vector history
    history = eval_forward(prob, α)

    ## Prepare forcing (-idH/dα ψ)
    # Prepare dH/dα
    
    # System hamiltonian is constant, falls out when taking derivative
    Ks::Matrix{Float64} = zeros(2,2)
    Ss::Matrix{Float64} = zeros(2,2)
    a_plus_adag = prob.a_plus_adag
    a_minus_adag = prob.a_minus_adag

    forcing_mat = zeros(4,1+prob.nsteps)
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
        copyto!(u,history[1:2,1+i])
        copyto!(v,history[3:4,1+i])

        utvt!(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, dpda, dqda, t, α)
        copyto!(forcing_vec,1,ut,1,2)
        copyto!(forcing_vec,3,vt,1,2)

        forcing_mat[:,1+i] .= forcing_vec

        t += dt
    end

    differentiated_prob = copy(prob) 
    # Get history of dψ/dα
    # Initial conditions for dψ/dα
    copyto!(differentiated_prob.u0, [0.0,0.0])
    copyto!(differentiated_prob.v0, [0.0,0.0])

    history_dψdα = eval_forward_forced(differentiated_prob, forcing_mat, α)

    dQda = history_dψdα[:,end]
    Q = history[:,end]
    R = copy(target)
    T = vcat(R[3:4], -R[1:2])

    gradient = -2*(dot(Q,R)*dot(dQda,R) + dot(Q,T)*dot(dQda,T))
    return gradient
end



"""
Evaluate gradient using differentiated/forced method approach.
Use forward diff for calculating gradients.
"""
function eval_grad_auto_forced(prob::SchrodingerProb, target, α=1.0)
    # dp/dα and dq/dα using auto (forward) differentiation
    p_wrapped(t_a_vec) = prob.p(t_a_vec[1], t_a_vec[2])
    q_wrapped(t_a_vec) = prob.q(t_a_vec[1], t_a_vec[2])
    dpda(t,α) = ForwardDiff.gradient(p_wrapped, [t,α])[2]
    dqda(t,α) = ForwardDiff.gradient(q_wrapped, [t,α])[2]

    return eval_grad_forced(prob, target, dpda, dqda, α)
end



function infidelity(ψ::Vector{Float64}, target::Vector{Float64})
    R = copy(target)
    T = vcat(R[3:4], -R[1:2])
    return 1 - (dot(ψ,R)^2 + dot(ψ,T)^2)
end

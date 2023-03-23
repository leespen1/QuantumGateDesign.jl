using LinearAlgebra
using IterativeSolvers
using LinearMaps
using Infiltrator
include("./auto_diff_example.jl")


struct SchrodingerProb
    tspan::Tuple{Float64, Float64}
    n_timesteps::Int64
    dt::Float64
    u0::Matrix{Float64}
    a::Float64
    S::Function
    K::Function
    St::Function
    Kt::Function
    Sa::Function
    Ka::Function
    M::Function
    Mt::Function
    Ma::Function
end


"""
Constructor
"""
function SchrodingerProb(
    tspan::Tuple{Float64, Float64},
    n_timesteps::Int64,
    u0::Matrix{Float64},
    a::Float64,
    S::Function,
    K::Function,
    St::Function,
    Kt::Function,
    Sa::Function,
    Ka::Function,
    )

    dt = (tspan[2]-tspan[1])/n_timesteps
    M(t,a) = [S(t,a) -K(t,a); K(t,a) S(t,a)]
    Mt(t,a) = [St(t,a) -Kt(t,a); Kt(t,a) St(t,a)]
    Ma(t,a) = [Sa(t,a) -Ka(t,a); Ka(t,a) Sa(t,a)]
    return SchrodingerProb(tspan, n_timesteps, dt, u0, a, S, K, St, Kt, Sa, Ka, M, Mt, Ma)
end

function complex_to_real(A)
    return vcat(real(A), imag(A))
end

function compute_utvt(S::AbstractMatrix{Float64},K::AbstractMatrix{Float64},
        u::Vector{Float64},v::Vector{Float64}
    )::Tuple{Vector{Float64},Vector{Float64}}

    ut = S*u - K*v
    vt = K*u + S*v
    return ut, vt
end


function compute_uttvtt(S::AbstractMatrix{Float64},K::AbstractMatrix{Float64},
        u::Vector{Float64},v::Vector{Float64},
        St::AbstractMatrix{Float64},Kt::AbstractMatrix{Float64},
        ut::Vector{Float64},vt::Vector{Float64}
    )::Tuple{Vector{Float64},Vector{Float64}}

    utt = St*u - Kt*v + S*ut - K*vt
    vtt = Kt*u + St*v + K*ut + S*vt
    return utt, vtt
end


function calc_infidelity(prob::SchrodingerProb, target::Matrix{Float64}, newparam::Float64)::Float64
    E = 2
    QN = eval_forward(prob, newparam)[:,:,end] # Get state vector at the end

    R = copy(target)
    T = vcat(target[3:4,:], -target[1:2,:])
    return 1 - (1/E^2)*(tr(QN'*R)^2 + tr(QN'*T)^2) # Could minus term be off by 4? EDIT: It should be 1/4 when evolving the *entire* basis
end


function eval_forward(prob::SchrodingerProb, newparam::Float64)::Array{Float64,3}
    t0 = prob.tspan[1]
    tf = prob.tspan[2]
    dt = prob.dt
    N = prob.n_timesteps
    a = newparam

    E = 2

    tn = NaN
    tnp1 = NaN # Initialize variable

    # Weights for fourth order Hermite-Rule
    wnp1 = [1,-0.5*dt*1/3] # Fourth Order
    #wnp1 = [1,0] # Second Order
    function lhs_action(prob::SchrodingerProb, Q::AbstractVector{Float64}, a::Float64)::Vector{Float64}
        dt = prob.dt

        S_mat  = prob.S(tnp1,a)
        St_mat = prob.St(tnp1,a)
        K_mat  = prob.K(tnp1,a)
        Kt_mat = prob.Kt(tnp1,a)
        u = Q[1:2]
        v = Q[3:4]

        ut, vt = compute_utvt(S_mat,K_mat,u,v)
        utt, vtt = compute_uttvtt(S_mat,K_mat,u,v,St_mat,Kt_mat,ut,vt)

        u = u .- 0.5*dt*(wnp1[1]*ut + wnp1[2]*utt) # Q = I + (1/2)Δt*M̃*Q
        v = v .- 0.5*dt*(wnp1[1]*vt + wnp1[2]*vtt)

        return vcat(u, v)
    end
    lhs_action_wrapper(Q) = lhs_action(prob, Q, a)
    LHS = LinearMap(lhs_action_wrapper, 2*E)

    # Weights for fourth order Hermite-Rule
    wn = [1,0.5*dt*1/3] # Fourth Order
    #wn = [1,0] # Second Order
    function rhs_action(prob::SchrodingerProb, Q::Vector{Float64}, a::Float64)::Vector{Float64}
        dt = prob.dt

        S_mat  = prob.S(tn,a)
        St_mat = prob.St(tn,a)
        K_mat  = prob.K(tn,a)
        Kt_mat = prob.Kt(tn,a)
        u = Q[1:2]
        v = Q[3:4]

        ut, vt = compute_utvt(S_mat,K_mat,u,v)
        utt, vtt = compute_uttvtt(S_mat,K_mat,u,v,St_mat,Kt_mat,ut,vt)

        u = u .+ 0.5*dt*(wn[1]*ut + wn[2]*utt) # Q = I + (1/2)Δt*M̃*Q
        v = v .+ 0.5*dt*(wn[1]*vt + wn[2]*vtt)

        return vcat(u, v)
    end

    Qs = zeros(size(prob.u0)..., N+1)
    Qs[:,:,1+0] .= prob.u0
    Q_col = zeros(size(prob.u0,1))
    RHS_col = zeros(size(prob.u0,1))
    num_RHS = size(prob.u0, 2)

    # Forward eval, saving all points
    for n in 0:N-1
        tn = t0 + n*dt
        tnp1 = t0 + (n+1)*dt

        for i in num_RHS:-1:1
            Q_col .= Qs[:,i,1+n]
            RHS_col = rhs_action(prob, Q_col, a)
            gmres!(Q_col, LHS, RHS_col, abstol=1e-15, reltol=1e-15)
            Qs[:,i,1+n+1] .= Q_col
        end
    end
    return Qs
end

function discrete_adjoint(prob::SchrodingerProb, newparam::Float64, target)
    Qs = eval_forward(prob, newparam)

    t0 = prob.tspan[1]
    tf = prob.tspan[2]
    dt = prob.dt
    N = prob.n_timesteps
    a = newparam

    E = 2

    tn = NaN

    # Weights for fourth order Hermite-Rule
    wnp1 = [1,-0.5*dt*1/3] # Fourth Order
    #wnp1 = [1,0] # Second Order
    function lhs_action(prob::SchrodingerProb, Q::AbstractVector{Float64}, a::Float64)::Vector{Float64}
        dt = prob.dt

        S_mat  = prob.S(tn,a)'
        St_mat = prob.St(tn,a)'
        K_mat  = -prob.K(tn,a)'
        Kt_mat = -prob.Kt(tn,a)'
        u = Q[1:2]
        v = Q[3:4]

        ut, vt = compute_utvt(S_mat,K_mat,u,v)
        utt, vtt = compute_uttvtt(S_mat,K_mat,u,v,St_mat,Kt_mat,ut,vt)

        u = u .- 0.5*dt*(wnp1[1]*ut + wnp1[2]*utt) # Q = I + (1/2)Δt*M̃*Q
        v = v .- 0.5*dt*(wnp1[1]*vt + wnp1[2]*vtt)

        return vcat(u, v)
    end
    lhs_action_wrapper(Q) = lhs_action(prob, Q, a)
    LHS = LinearMap(lhs_action_wrapper, 2*E)

    # Weights for fourth order Hermite-Rule
    wn = [1,0.5*dt*1/3] # Fourth Order
    #wn = [1,0] # Second Order
    function rhs_action(prob::SchrodingerProb, Q::Vector{Float64}, a::Float64)::Vector{Float64}
        dt = prob.dt

        S_mat  = prob.S(tn,a)'
        St_mat = prob.St(tn,a)'
        K_mat  = -prob.K(tn,a)'
        Kt_mat = -prob.Kt(tn,a)'
        u = Q[1:2]
        v = Q[3:4]

        ut, vt = compute_utvt(S_mat,K_mat,u,v)
        utt, vtt = compute_uttvtt(S_mat,K_mat,u,v,St_mat,Kt_mat,ut,vt)

        u = u .+ 0.5*dt*(wn[1]*ut + wn[2]*utt) # Q = I + (1/2)Δt*M̃*Q
        v = v .+ 0.5*dt*(wn[1]*vt + wn[2]*vtt)

        return vcat(u, v)
    end


    tn = t0 + N*dt
    QN = Qs[:,:,1+N]
    R = zeros(size(prob.u0)...)
    R .= target
    T = zeros(size(prob.u0)...)
    T[1:2,:] .= target[3:4,:]
    T[3:4,:] .= -target[1:2,:]
    Λs = zeros(size(prob.u0)..., N+1)

    terminal_RHS = (2/E^2)*( tr(QN'*R)*R + tr(QN'*T)*T )

    num_RHS = size(prob.u0, 2)
    Λ_col = zeros(size(prob.u0,1))
    RHS_col = zeros(size(prob.u0,1))
    for i in 1:num_RHS
        Λ_col .= QN[:,i]
        RHS_col .= terminal_RHS[:,i]
        gmres!(Λ_col, LHS, RHS_col, abstol=1e-15, reltol=1e-15)
        Λs[:,i,1+N] .= Λ_col
    end
    # Adjoint evolution, saving all points
    for n in N-1:-1:1
        tn = t0 + n*dt
        for i in 1:num_RHS
            Λ_col .= Λs[:,i,1+n+1]
            RHS_col = rhs_action(prob, Λ_col, a)
            gmres!(Λ_col, LHS, RHS_col, abstol=1e-15, reltol=1e-15)
            Λs[:,i,1+n] .= Λ_col
        end
    end
    
    Ma = prob.Ma
    gradient = 0.0
    for n in 0:N-1
        tn = t0 + n*dt
        tnp1 = t0 + (n+1)*dt
        gradient += tr((Ma(tn, a)*Qs[:,:,1+n] + Ma(tnp1, a)*Qs[:,:,1+n])'*Λs[:,:,1+n+1])
    end
    gradient *= -0.5*dt
    return gradient
end


function main(n_timesteps=100)
    tspan = (0.0, 1.0)
    Q0 = complex_to_real([1.0+1.0im, 1.0+1.0im])
    Q0 = Q0 / norm(Q0)

    p = 1.0

    ODE_prob = ODEProblem{true, SciMLBase.FullSpecialize}(schrodinger!, Q0, tspan, p) # Need this option for debug to work
    data_solution = solve(ODE_prob, saveat=1, abstol=1e-10, reltol=1e-10)
    # Convert solution to array
    data = Array(data_solution)
    Q_target = data[:,end]

    S(t,a) = [0.0 0.0;
               0.0 0.0]
    K(t,a) = [0.0 a*cos(t);
               a*cos(t) 1.0]
    St(t,a) = [0.0 0.0;
                 0.0 0.0]
    Kt(t,a) = [0.0 -a*sin(t);
               -a*sin(t) 0.0]
    Sa(t,a) = [0.0 0.0;
               0.0 0.0]
    Ka(t,a) = [0.0 cos(t);
               cos(t) 0.0]

    schroprob = SchrodingerProb(tspan, n_timesteps, Q0, p, S, K, St, Kt, Sa, Ka)

    infidelity = calc_infidelity(schroprob, Q_target, p)
    println("Infidelity: $infidelity")
    return infidelity
end


function test_discrete_adjoint()
    tspan = (0.0, 1.0)
    n_steps = 10
    dt_save = (tspan[2] - tspan[1])/n_steps

    Q0 = [1.0 0.0 0.0 0.0;
          0.0 1.0 0.0 0.0;
          0.0 0.0 1.0 0.0;
          0.0 0.0 0.0 1.0]
    num_RHS = size(Q0, 2)

    p = 1.0

    true_sol_ary = zeros(4,num_RHS,n_steps+1)
    for i in 1:num_RHS
        Q0_col = Q0[:,i]
        ODE_prob = ODEProblem{true, SciMLBase.FullSpecialize}(schrodinger!, Q0_col, tspan, p) # Need this option for debug to work
        data_solution = solve(ODE_prob, saveat=dt_save, abstol=1e-10, reltol=1e-10)
        data_solution_mat = Array(data_solution)
        true_sol_ary[:,i,:] .= data_solution_mat
    end

    Q_target = true_sol_ary[:,:,end]

    S(t,a) = [0.0 0.0;
               0.0 0.0]
    K(t,a) = [0.0 a*cos(t);
               a*cos(t) 1.0]
    St(t,a) = [0.0 0.0;
                 0.0 0.0]
    Kt(t,a) = [0.0 -a*sin(t);
               -a*sin(t) 0.0]
    Sa(t,a) = [0.0 0.0;
               0.0 0.0]
    Ka(t,a) = [0.0 cos(t);
               cos(t) 0.0]

    N = 10
    gradients = zeros(N+1)
    for i in 0:N
        schroprob = SchrodingerProb(tspan, 2^i, Q0, p, S, K, St, Kt, Sa, Ka)
        gradients[1+i] = discrete_adjoint(schroprob, p, Q_target)
    end
    ratios = zeros(N)
    for i in 1:N
        ratios[i] = log2(gradients[i] / gradients[i+1])
    end
    println("Gradients: $gradients")
    println("Log2 Ratios: $ratios")
end


"""
Testing convergence of state vector history
"""
function test1()
    tspan = (0.0, 1.0)
    n_steps = 10
    dt_save = (tspan[2] - tspan[1])/n_steps

    Q0 = [1.0 0.0 0.0 0.0;
          0.0 1.0 0.0 0.0;
          0.0 0.0 1.0 0.0;
          0.0 0.0 0.0 1.0]
    #Q0 = zeros(4,1)
    #Q0[2,1] = 1
    num_RHS = size(Q0, 2)

    p = 1.0

    true_sol_ary = zeros(4,num_RHS,n_steps+1)
    for i in 1:num_RHS
        Q0_col = Q0[:,i]
        ODE_prob = ODEProblem{true, SciMLBase.FullSpecialize}(schrodinger!, Q0_col, tspan, p) # Need this option for debug to work
        data_solution = solve(ODE_prob, saveat=dt_save, abstol=1e-10, reltol=1e-10)
        data_solution_mat = Array(data_solution)
        true_sol_ary[:,i,:] .= data_solution_mat
    end

    S(t,a) = [0.0 0.0;
               0.0 0.0]
    K(t,a) = [0.0 a*cos(t);
               a*cos(t) 1.0]
    St(t,a) = [0.0 0.0;
                 0.0 0.0]
    Kt(t,a) = [0.0 -a*sin(t);
               -a*sin(t) 0.0]
    Sa(t,a) = [0.0 0.0;
               0.0 0.0]
    Ka(t,a) = [0.0 cos(t);
               cos(t) 0.0]

    schroprob_10 = SchrodingerProb(tspan, 10, Q0, p, S, K, St, Kt, Sa, Ka)
    schroprob_20 = SchrodingerProb(tspan, 20, Q0, p, S, K, St, Kt, Sa, Ka)
    Qs_10 = eval_forward(schroprob_10, p)
    Qs_20 = eval_forward(schroprob_20, p)
    diff_10 = Qs_10[:,:,:] - true_sol_ary
    diff_20 = Qs_20[:,:,1:2:end] - true_sol_ary
    println("Ratio of Q_20-Q* to Q_10-Q*")
    display(diff_10 ./ diff_20)
    #@infiltrate
end

#=
"""
Testing convergence of infidelity
"""
function test2()
    tspan = (0.0, 1.0)
    n_steps = 10
    dt_save = (tspan[2] - tspan[1])/n_steps

    Q0 = [1.0 0.0 0.0 0.0;
          0.0 1.0 0.0 0.0;
          0.0 0.0 1.0 0.0;
          0.0 0.0 0.0 1.0]
    num_RHS = size(Q0, 2)

    p = 1.0

    true_sol_ary = zeros(4,num_RHS,n_steps+1)
    for i in 1:num_RHS
        Q0_col = Q0[:,i]
        ODE_prob = ODEProblem{true, SciMLBase.FullSpecialize}(schrodinger!, Q0_col, tspan, p) # Need this option for debug to work
        data_solution = solve(ODE_prob, saveat=dt_save, abstol=1e-10, reltol=1e-10)
        data_solution_mat = Array(data_solution)
        true_sol_ary[:,i,:] .= data_solution_mat
    end
    Q_target = true_sol_ary[:,:,end]

    S(t,a) = [0.0 0.0;
               0.0 0.0]
    K(t,a) = [0.0 a*cos(t);
               a*cos(t) 1.0]
    St(t,a) = [0.0 0.0;
                 0.0 0.0]
    Kt(t,a) = [0.0 -a*sin(t);
               -a*sin(t) 0.0]
    Sa(t,a) = [0.0 0.0;
               0.0 0.0]
    Ka(t,a) = [0.0 cos(t);
               cos(t) 0.0]

    schroprob_10 = SchrodingerProb(tspan, 10, Q0, p, S, K, St, Kt, Sa, Ka)
    schroprob_20 = SchrodingerProb(tspan, 20, Q0, p, S, K, St, Kt, Sa, Ka)
    infidelity_10 = calc_infidelity(schroprob_10, Q_target, p)
    infidelity_20 = calc_infidelity(schroprob_20, Q_target, p)
end
=#

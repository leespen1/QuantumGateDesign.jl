using LinearAlgebra
using IterativeSolvers
using LinearMaps
using Infiltrator
include("./auto_diff_example.jl")

const E = 2 # Hardcoding number of essential energy levels for now

#== To Do =====================================================================
# - Test infidelity using hand-picked matrices
# - Test discrete adjoint for 2nd order method
# - Test discrete adjoint for 4th order method
# - Add Filon Method, run comparisons (using similar structure should result in
# comparable runtimes)
# - Check memory allocation of methods (I suspect there is a lot in lhs/rhs_action)
#
# Note on performance, I think to have good performance I need all the matrix 
# functions, and the actions in LHS/RHS, to be mutating. That'll take some work.
== End To Do ================================================================#

mutable struct SchrodingerProb
    tspan::Tuple{Float64, Float64}
    n_timesteps::Int64
    u0::Matrix{Float64}
    a::Float64
    S::Function
    K::Function
    St::Function
    Kt::Function
    Stt::Function
    Ktt::Function
    Sa::Function
    Ka::Function
    M::Function
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
    Stt::Function,
    Ktt::Function,
    Sa::Function,
    Ka::Function,
    )

    M(t,a) = [S(t,a) -K(t,a); K(t,a) S(t,a)]
    Ma(t,a) = [Sa(t,a) -Ka(t,a); Ka(t,a) Sa(t,a)]
    return SchrodingerProb(tspan, n_timesteps, u0, a, S, K, St, Kt, Stt, Ktt, Sa, Ka, M, Ma)
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


function compute_utttvttt(S::AbstractMatrix{Float64},K::AbstractMatrix{Float64},
        u::Vector{Float64},v::Vector{Float64},
        St::AbstractMatrix{Float64},Kt::AbstractMatrix{Float64},
        ut::Vector{Float64},vt::Vector{Float64},
        Stt::AbstractMatrix{Float64},Ktt::AbstractMatrix{Float64},
        utt::Vector{Float64},vtt::Vector{Float64}
    )::Tuple{Vector{Float64},Vector{Float64}}

    uttt = Stt*u + St*ut - Ktt*v - Kt*vt + St*ut + S*utt - Kt*vt - K*vtt
    vttt = Ktt*u + Kt*ut + Stt*v + St*vt + Kt*ut + K*utt + St*vt + S*vtt
    return uttt, vttt
end


function calc_infidelity(QN::Matrix{Float64}, target::Matrix{Float64})::Float64
    R = copy(target)
    T = vcat(target[3:4,:], -target[1:2,:])
    return 1 - (1/E^2)*(tr(QN'*R)^2 + tr(QN'*T)^2)
end


function eval_forward(prob::SchrodingerProb, newparam::Float64; order::Int64=4)::Array{Float64,3}
    t0 = prob.tspan[1]
    tf = prob.tspan[2]
    N = prob.n_timesteps
    dt = (tf-t0)/N
    a = newparam

    nrows, ncols = size(prob.u0)

    tn = NaN
    tnp1 = NaN # Initialize variable

    # Determine Weights for Hermite-Rule
    if (order == 2)
        wn = [1,0,0] # Second Order
        wnp1 = [1,0,0] # Second Order
    elseif (order == 4)
        wn = [1,1/3,0] # Fourth Order
        wnp1 = [1,-1/3,0] # Fourth Order
    elseif (order == 6)
        wn = [1, 2/5, 1/15]
        wnp1 = [1, -2/5, 1/15]
    end

    # Is S_mat = prob.S(tnp1,a) allocating memory every time? If so, I need to put a stop to that
    function lhs_action(prob::SchrodingerProb, Q::AbstractVector{Float64}, a::Float64)::Vector{Float64}
        S_mat  = prob.S(tnp1,a)
        St_mat = prob.St(tnp1,a)
        Stt_mat = prob.Stt(tnp1,a)
        K_mat  = prob.K(tnp1,a)
        Kt_mat = prob.Kt(tnp1,a)
        Ktt_mat = prob.Ktt(tnp1,a)
        u = Q[1:2]
        v = Q[3:4]

        ut, vt = compute_utvt(S_mat,K_mat,u,v)
        utt, vtt = compute_uttvtt(S_mat,K_mat,u,v,St_mat,Kt_mat,ut,vt)
        uttt, vttt = compute_utttvttt(S_mat,K_mat,u,v,St_mat,Kt_mat,ut,vt,Stt_mat,Ktt_mat,utt,vtt)

        
        u = u .- 0.5*dt*(wnp1[1]*ut + 0.5*dt*(wnp1[2]*utt + 0.5*dt*wnp1[3]*uttt)) # Q = I + (1/2)Δt*M̃*Q
        v = v .- 0.5*dt*(wnp1[1]*vt + 0.5*dt*(wnp1[2]*vtt + 0.5*dt*wnp1[3]*vttt))

        return vcat(u, v)
    end

    lhs_action_wrapper(Q) = lhs_action(prob, Q, a)
    LHS = LinearMap(lhs_action_wrapper, 2*E)

    function rhs_action(prob::SchrodingerProb, Q::Vector{Float64}, a::Float64)::Vector{Float64}
        S_mat  = prob.S(tn,a)
        St_mat = prob.St(tn,a)
        Stt_mat = prob.Stt(tn,a)
        K_mat  = prob.K(tn,a)
        Kt_mat = prob.Kt(tn,a)
        Ktt_mat = prob.Ktt(tn,a)
        u = Q[1:2]
        v = Q[3:4]

        ut, vt = compute_utvt(S_mat,K_mat,u,v)
        utt, vtt = compute_uttvtt(S_mat,K_mat,u,v,St_mat,Kt_mat,ut,vt)
        uttt, vttt = compute_utttvttt(S_mat,K_mat,u,v,St_mat,Kt_mat,ut,vt,Stt_mat,Ktt_mat,utt,vtt)

        u = u .+ 0.5*dt*(wn[1]*ut + 0.5*dt*(wn[2]*utt + 0.5*dt*wn[3]*uttt)) # Q = I + (1/2)Δt*M̃*Q
        v = v .+ 0.5*dt*(wn[1]*vt + 0.5*dt*(wn[2]*vtt + 0.5*dt*wn[3]*vttt))

        return vcat(u, v)
    end

    Qs = zeros(nrows, ncols, N+1)
    Qs[:,:,1+0] .= prob.u0
    Q_col = zeros(nrows)
    RHS_col = zeros(nrows)
    num_RHS = ncols

    # Forward eval, saving all points
    # Perforrmance note, for 10 timestep method, 418 KiB of 423 allocated here! ()
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

function discrete_adjoint(prob::SchrodingerProb, newparam::Float64, target::Matrix{Float64}; order::Int64=4)::Float64
    Qs = eval_forward(prob, newparam)

    t0 = prob.tspan[1]
    tf = prob.tspan[2]
    N = prob.n_timesteps
    dt = (tf-t0)/N
    a = newparam

    nrows, ncols = size(prob.u0)

    tn = NaN

    # Weights for Hermite-Rule
    if (order == 2)
        wn = [1,0]
        wnp1 = [1,0]
    elseif (order == 4)
        wn = [1,0.5*dt*1/3]
        wnp1 = [1,-0.5*dt*1/3]
    end


    function lhs_action(prob::SchrodingerProb, Q::AbstractVector{Float64}, a::Float64)::Vector{Float64}
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

    function rhs_action(prob::SchrodingerProb, Q::Vector{Float64}, a::Float64)::Vector{Float64}
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
    R = zeros(nrows, ncols)
    R .= target
    T = zeros(nrows, ncols)
    T[1:2,:] .= target[3:4,:]
    T[3:4,:] .= -target[1:2,:]
    Λs = zeros(nrows, ncols, N+1)

    num_RHS = ncols
    Λ_col = zeros(nrows)
    RHS_col = zeros(nrows)

    # Handle terminal condition
    terminal_RHS = (2/E^2)*( tr(QN'*R)*R + tr(QN'*T)*T )
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
        gradient += tr((Ma(tn, a)*Qs[:,:,1+n] + Ma(tnp1, a)*Qs[:,:,1+n+1])'*Λs[:,:,1+n+1])
    end
    gradient *= -0.5*dt
    return gradient
end


"""
Modifying eval_forward for fair comparison
"""
function eval_forward_filon(prob::SchrodingerProb, newparam::Float64)::Array{Float64,3}
    t0 = prob.tspan[1]
    tf = prob.tspan[2]
    N = prob.n_timesteps
    dt = (tf-t0)/N
    a = newparam

    nrows, ncols = size(prob.u0)

    tn = NaN
    tnp1 = NaN # Initialize variable

    order = 4
    # Weights for Hermite-Rule
    if (order == 2)
        wn = [1,0] # Second Order
        wnp1 = [1,0] # Second Order
    elseif (order == 4)
        wn = [1,0.5*dt*1/3] # Fourth Order
        wnp1 = [1,-0.5*dt*1/3] # Fourth Order
    end

    # Is S_mat = prob.S(tnp1,a) allocating memory every time? If so, I need to put a stop to that
    function lhs_action(prob::SchrodingerProb, Q::AbstractVector{Float64}, a::Float64)::Vector{Float64}
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

    function rhs_action(prob::SchrodingerProb, Q::Vector{Float64}, a::Float64)::Vector{Float64}
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

    Qs = zeros(nrows, ncols, N+1)
    Qs[:,:,1+0] .= prob.u0
    Q_col = zeros(nrows)
    RHS_col = zeros(nrows)
    num_RHS = ncols

    # Forward eval, saving all points
    # Perforrmance note, for 10 timestep method, 418 KiB of 423 allocated here! ()
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




"""
Testing convergence of state vector history
"""
function test1()
    tspan = (0.0, 1.0)
    n_steps_ODEProblem = 10
    dt_save = (tspan[2] - tspan[1])/n_steps_ODEProblem

    Q0 = [1.0 0.0;
          0.0 1.0;
          0.0 0.0;
          0.0 0.0]
    #Q0 = zeros(4,1)
    #Q0[2,1] = 1
    num_RHS = size(Q0, 2)

    p = 1.0

    true_sol_ary = zeros(4,num_RHS,n_steps_ODEProblem+1)
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


    N = 5
    nsteps = n_steps_ODEProblem .* (2 .^ (0:N)) # Double the number of steps each time
    max_ratios = zeros(N)
    min_ratios = zeros(N)
    max_errors = zeros(N+1)

    schroprob = SchrodingerProb(tspan, nsteps[1], Q0, p, S, K, St, Kt, Sa, Ka)
    println(schroprob.n_timesteps)
    Qs_prev = eval_forward(schroprob, p)[:,:,1:nsteps[1]÷n_steps_ODEProblem:end]
    Q_diffs_prev = Qs_prev - true_sol_ary
    Q_diffs_prev = (Qs_prev - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
    max_errors[1] = max(Q_diffs_prev...)

    for i in 2:N+1
        schroprob.n_timesteps = nsteps[i]
        println(schroprob.n_timesteps)
        Qs_next = eval_forward(schroprob, p)[:,:,1:nsteps[i]÷n_steps_ODEProblem:end]
        Q_diffs_next = (Qs_next - true_sol_ary)[:,:,2:end]
        Q_diffs_next = (Qs_next - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
        ratios = log2.(abs.(Q_diffs_prev ./ Q_diffs_next))
        max_ratios[i-1] = max(ratios...)
        min_ratios[i-1] = min(ratios...)
        max_errors[i] = max(Q_diffs_next...)
        
        Qs_prev .= Qs_next
        Q_diffs_prev .= Q_diffs_next
    end

    pl1 = plot(nsteps[2:end], max_ratios, label="Max")
    plot!(pl1, nsteps[2:end], min_ratios, label="Min")
    plot!(pl1, xlabel="# Timesteps", ylabel="Log2 E(2*Δt)/E(Δt)")
    plot!(pl1, xscale=:log10)
    plot!(pl1, title="Convergence")
    pl2 = plot(nsteps, max_errors)
    plot!(pl2, xlabel="# Timesteps", ylabel="Max E(Δt) (per entry)")
    plot!(pl2, scale=:log10)
    plot!(pl2, title="Error")
    pl = plot(pl1, pl2, layout=(1,2))
    plot!(pl, plot_title="Error and Convergence (Per Entry)")
    return pl
    #=
    schroprob_10 = SchrodingerProb(tspan, 10, Q0, p, S, K, St, Kt, Sa, Ka)
    schroprob_20 = SchrodingerProb(tspan, 20, Q0, p, S, K, St, Kt, Sa, Ka)
    Qs_10 = eval_forward(schroprob_10, p)
    Qs_20 = eval_forward(schroprob_20, p)
    diff_10 = Qs_10[:,:,:] - true_sol_ary
    diff_20 = Qs_20[:,:,1:2:end] - true_sol_ary
    println("Ratio of Q_20-Q* to Q_10-Q*")
    ratio = (diff_10 ./ diff_20)
    display(ratio)
    return ratio
    =#
end

"""
Testing convergence of infidelity
"""
function test2()
    tspan = (0.0, 1.0)
    n_steps_ODEProblem = 10
    dt_save = (tspan[2] - tspan[1])/n_steps_ODEProblem

    Q0 = [1.0 0.0;
          0.0 1.0;
          0.0 0.0;
          0.0 0.0]
    num_RHS = size(Q0, 2)

    p = 1.0

    true_sol_ary = zeros(4,num_RHS,n_steps_ODEProblem+1)
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
    schroprob = SchrodingerProb(tspan, 10, Q0, p, S, K, St, Kt, Sa, Ka)

    N = 10
    infidelity_ary = zeros(N)
    nsteps = 2 .^ (1:N)
    for i in 1:N
        schroprob.n_timesteps = nsteps[i]
        Q_disc = eval_forward(schroprob, p)
        Q_final_disc = Q_disc[:,:,end]
        infidelity_ary[i] = calc_infidelity(Q_final_disc, Q_target)
    end
    pl = plot(nsteps, infidelity_ary)
    plot!(pl, xlabel="# Timesteps", ylabel="Infidelity")
    #plot!(scale=:log10)
    return pl
end

"""
Testing correctness of discrete adjoint gradient using discrete evolution as target
"""
function test3()
    tspan = (0.0, 1.0)
    Q0 = [1.0 0.0;
          0.0 1.0;
          0.0 0.0;
          0.0 0.0]
    num_RHS = size(Q0, 2)

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
    p = 1.0
    schroprob = SchrodingerProb(tspan, 10, Q0, p, S, K, St, Kt, Sa, Ka)
    Qs = eval_forward(schroprob, p)
    QN = Qs[:,:,end]

    grad = discrete_adjoint(schroprob, p, QN)
    println("Gradient (should be 0): $grad")
end

"""
Testing correctness of discrete adjoint gradient by convergence of finite diffrence
"""
function test4(plot_err=true)
    tspan = (0.0, 1.0)
    Q0 = [1.0 0.0;
          0.0 1.0;
          0.0 0.0;
          0.0 0.0]
    num_RHS = size(Q0, 2)

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
    p = 1.0
    schroprob = SchrodingerProb(tspan, 10000, Q0, p, S, K, St, Kt, Sa, Ka)
    Q_target = eval_forward(schroprob, p)[:,:,end]

    grad_dis_adj = discrete_adjoint(schroprob, p, Q_target)

    # Calculate gradient using finite difference method
    da = 1e-5
    QN_r = eval_forward(schroprob, p+da)[:,:,end]
    QN_l = eval_forward(schroprob, p-da)[:,:,end]
    infidelity_r = calc_infidelity(QN_r, Q_target)
    infidelity_l = calc_infidelity(QN_l, Q_target)
    grad_fin_dif = (infidelity_r - infidelity_l)/(2*da)

    println("Gradients")
    println("Finite Difference: $grad_fin_dif")
    println("Discrete Adjoint:  $grad_dis_adj")

    Nsamples = 7
    grad_fin_dif_ary = zeros(Nsamples)
    da_ary = zeros(Nsamples)
    for i in 1:Nsamples
        da = 10.0 ^ (-i)
        da_ary[i] = da

        QN_r = eval_forward(schroprob, p+da)[:,:,end]
        QN_l = eval_forward(schroprob, p-da)[:,:,end]
        infidelity_r = calc_infidelity(QN_r, Q_target)
        infidelity_l = calc_infidelity(QN_l, Q_target)

        grad_fin_dif = (infidelity_r - infidelity_l)/(2*da)
        if plot_err
            grad_fin_dif_ary[i] = abs(grad_fin_dif - grad_dis_adj)
        else
            grad_fin_dif_ary[i] = grad_fin_dif
        end
    end

    pl = plot(da_ary, grad_fin_dif_ary)
    plot!(pl, xlabel="dα")
    if plot_err
        plot!(ylabel="Absolute Error in ∇")
        plot!(pl, scale=:log10)
    else
        plot!(ylabel="Gradient (using Finite Difference)")
    end

    return pl
end

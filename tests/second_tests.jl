include("../src/hermite.jl")
using Plots

function this_prob(;ω::Float64=1.0, tf::Float64=1.0, nsteps::Int64=10)
    Ks::Matrix{Float64} = [0 0; 0 1]
    Ss::Matrix{Float64} = [0 0; 0 0]
    p(t,α) = α*cos(ω*t)
    q(t,α) = 0.0
    dpdt(t,α) = -α*ω*sin(ω*t)
    dqdt(t,α) = 0.0
    u0::Vector{Float64} = [1,0]
    v0::Vector{Float64} = [0,0]
    return SchrodingerProb(Ks,Ss,p,q,dpdt,dqdt,u0,v0,tf,nsteps)
end


#=
#Actually I don't really need this. ∂ψ/∂α is evolved according to the original
#schrodinger equation. The part with ∂H/∂α  is in the forcing term
"""
Evolve ∂ψ/∂α (in relation to the above problem). Will need forcing
System hamiltonian falls out, initial conditions are zero, and controls change
according to ∂/∂α
"""
function this_prob_grad(;ω::Float64=1.0, tf::Float64=1.0, nsteps::Int64=10)
    Ks::Matrix{Float64} = [0 0; 0 0]
    Ss::Matrix{Float64} = [0 0; 0 0]
    a_plus_adag::Matrix{Float64} = [0 1; 1 0]
    a_minus_adag::Matrix{Float64} = [0 1; -1 0]
    p(t,α) = cos(ω*t)
    q(t,α) = 0.0
    u0::Vector{Float64} = [0,0]
    v0::Vector{Float64} = [0,0]
    return SchrodingerProb(Ks,Ss,a_plus_adag,a_minus_adag,p,q,u0,v0,tf,nsteps)
end
=#

function convergence_test(α=1.0; order=2)

    # MAKE SURE TIMESTEPS ARE SMALL ENOUGH THAT SOLUTION ISNT OVERRESOLVED
    prob20 = this_prob(nsteps=20)
    history20 = eval_forward(prob20, α, order=order)
    # Change stride to match coarse timestep result
    history20 = history20[:,1:2:end]
    prob10 = this_prob(nsteps=10)
    history10 = eval_forward(prob10, α, order=order)

    # Use 1000 timesteps for "true solution"
    prob1000 = this_prob(nsteps=1000)
    history_true = eval_forward(prob1000, α, order=order)
    history_true = history_true[:,1:100:end]

    error10 = abs.(history_true - history10)
    error20 = abs.(history_true - history20)

    log_ratio = log2.(error10 ./ error20)
    println("Log₂ of error ratio between 10 step and 20 step methods")
    println("(Analytic solution used for 'true' value)")


    display(log_ratio)
    return log_ratio, error10, error20
end

function finite_difference(prob, α, target, dα=1e-5)
    # Centered Difference Approximation
    history_r = eval_forward(prob, α+dα)
    history_l = eval_forward(prob, α-dα)
    ψf_r = history_r[:,end]
    ψf_l = history_l[:,end]
    infidelity_r = infidelity(ψf_r, target)
    infidelity_l = infidelity(ψf_l, target)
    gradient = (infidelity_r - infidelity_l)/(2*dα)
    return gradient
end


"""
Obtain the gradient using the 'take derivative of schrodinger's equation'
method. 
"""
function eval_forward_grad_mat(target, α=1.0; nsteps=100)
    u0::Vector{Float64} = zeros(8)
    u0[1] = 1
    u = copy(u0)

    tf = 1.0
    dt = tf/nsteps

    # Need to write out LHS matrix to do implicit solve, or else use
    # abstract arrays and iterative solvers
    M(t, a) = [
    0.0 0.0 0.0 -a*cos(t)  0.0 0.0 0.0 0.0
    0.0 0.0 -a*cos(t) -1.0 0.0 0.0 0.0 0.0
    0.0 a*cos(t) 0.0 0.0   0.0 0.0 0.0 0.0
    a*cos(t) 1.0 0.0 0.0   0.0 0.0 0.0 0.0

    0.0 0.0 0.0 -cos(t)    0.0 0.0 0.0 -a*cos(t) 
    0.0 0.0 -cos(t) 0.0    0.0 0.0 -a*cos(t) -1.0
    0.0 cos(t) 0.0 0.0     0.0 a*cos(t) 0.0 0.0  
    cos(t) 0.0 0.0 0.0     a*cos(t) 1.0 0.0 0.0  
    ]
    for n in 0:nsteps-1
        tn = n*dt
        tnp1 = (n+1)*dt
        u = (I - 0.5*dt*M(tnp1, α)) \ (u + 0.5*dt*M(tn,α)*u)
    end

    Q = u[1:4,end]
    dQda = u[5:8,end]

    R = target[:]
    T = vcat(R[3:4], -R[1:2])

    #target_complex::Vector{ComplexF64} = target[1:2] - im*target[3:4]
    #Q_complex::Vector{ComplexF64} = Q[1:2] - im*Q[3:4]
    #dQda_complex::Vector{ComplexF64} = dQda[1:2] - im*dQda[3:4]

    ## The following two are equivalent, checked with ComplexF64 vectors
    #
    #grad_gargamel = -2*real((Q_complex'*target_complex)*(target_complex'*dQda_complex))
    grad_gargamel = -2*(dot(Q,R)*dot(dQda,R) + dot(Q,T)*dot(dQda,T))

    # rewrite gradient
    return grad_gargamel
end


function figure1()
    α = 1.0
    prob = this_prob()
    history = eval_forward(prob, α)
    target = history[:,end]

    dpda(t,a) = cos(t)
    dqda(t,a) = cos(t)

    N = 100
    alphas = LinRange(0,2,N)
    grads_fd = zeros(N)
    grads_diff_mat = zeros(N)
    grads_diff_forced = zeros(N)
    grads_diff_auto_forced = zeros(N)
    grads_da = zeros(N)
    for i in 1:N
        α = alphas[i]
        grads_fd[i] = finite_difference(prob, α, target)
        grads_diff_mat[i] = eval_forward_grad_mat(target, α)
        grads_diff_forced[i] = eval_grad_forced(target, α)
        grads_diff_auto_forced[i] = eval_grad_auto_forced(prob, target, α)
        grads_da[i] = discrete_adjoint(prob, target, dpda, dqda, α)
    end
    return alphas, grads_fd, grads_diff_mat, grads_diff_forced, grads_diff_auto_forced, grads_da
end

function plot_figure1(alphas, grads_fd, grads_diff_mat, grads_diff_forced, grads_diff_auto_forced, grads_da)
    # If all the gradients are working, this graph won't be much use
    pl1 = plot(alphas, grads_fd, label="Finite Difference", lw=2)
    plot!(pl1, alphas, grads_diff_mat, label="Differentiation (Matrix)", lw=2)
    plot!(pl1, alphas, grads_diff_forced, label="Differentiation (Forced)", lw=2)
    plot!(pl1, alphas, grads_diff_auto_forced, label="Differentiation (Auto Forced)", lw=2)
    plot!(pl1, alphas, grads_da, label="Discrete Adjoint", lw=2)
    plot!(pl1, xlabel="α", ylabel="Gradient")
    plot!(pl1, legendfontsize=14,guidefontsize=14,tickfontsize=14)

    # Use finite difference as the "true" value
    errs_diff_forced = abs.(grads_fd .- grads_diff_forced)
    errs_diff_auto_forced = abs.(grads_fd .- grads_diff_auto_forced)
    errs_diff_mat = abs.(grads_fd .- grads_diff_mat)
    errs_da = abs.(grads_fd .- grads_da)
    pl2 = plot(alphas, errs_diff_forced, label="Differentiaion (Forced)", lw=2)
    plot!(alphas, errs_diff_auto_forced, label="Differentiaion (Auto Forced)", lw=2)
    plot!(pl2, alphas, errs_diff_mat, label="Differentiaion (Matrix)", lw=2)
    plot!(pl2, alphas, errs_da, label="Discrete Adjoint", lw=2)
    plot!(pl2, legendfontsize=14,guidefontsize=14,tickfontsize=14)
    plot!(pl2, yscale=:log10)
    return pl1, pl2
end



function figure2!(prob::SchrodingerProb, α=1.0; order=2)
    N = 5
    final_states = zeros(4,N)
    base = 2
    for i in 1:N
        prob.nsteps = base^i
        history = eval_forward(prob, α, order=order)
        final_states[:,i] = history[:,end]
    end
    prob.nsteps = base^(N+1)
    history = eval_forward(prob, α, order=order)
    final_state_fine = history[:,end]

    step_sizes = zeros(N)
    for i in 1:N
        step_sizes[i] = prob.tf / (base^i)
    end
    sol_errs = [norm(final_states[:,i] - final_state_fine) for i in 1:N]
    infidelities = [infidelity(final_states[:,i],final_state_fine) for i in 1:N]
   
    return step_sizes, sol_errs, infidelities
end

function plot_figure2(step_sizes, sol_errs, infidelities)
    pl = plot(step_sizes, abs.(sol_errs), linewidth=2, marker=:circle, label="Error", scale=:log10)
    plot!(step_sizes, abs.(infidelities), linewidth=2, marker=:circle, label="Infidelities")
    plot!(step_sizes, step_sizes .^ 2, label="Δt^2", linestyle=:dash)
    plot!(step_sizes, step_sizes .^ 4, label="Δt^4", linestyle=:dash)
    plot!(step_sizes, step_sizes .^ 6, label="Δt^6", linestyle=:dash)
    plot!(legendfontsize=14,guidefontsize=14,tickfontsize=14)
    plot!(legend=:bottomright)
    plot!(xlabel="Δt")
    return pl
end



function eval_grad_forced(target, α=1.0; ω::Float64=1.0)
    # Get state vector history
    prob = this_prob(ω=ω)
    history = eval_forward(prob, α)

    ## Prepare forcing (-idH/dα ψ)
    # Prepare dH/dα
    Ks::Matrix{Float64} = [0 0; 0 0]
    Ss::Matrix{Float64} = [0 0; 0 0]
    a_plus_adag::Matrix{Float64} = [0 1; 1 0]
    a_minus_adag::Matrix{Float64} = [0 1; -1 0]
    p(t,α) = cos(ω*t)
    q(t,α) = 0.0

    forcing_mat = zeros(4,1+prob.nsteps)
    forcing_vec = zeros(4)

    u = zeros(2)
    v = zeros(2)
    ut = zeros(2)
    vt = zeros(2)

    nsteps = prob.nsteps
    t = 0
    dt = prob.tf/nsteps
    for i in 0:nsteps
        copyto!(u,history[1:2,1+i])
        copyto!(v,history[3:4,1+i])

        utvt!(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α)
        copyto!(forcing_vec,1,ut,1,2)
        copyto!(forcing_vec,3,vt,1,2)

        forcing_mat[:,1+i] .= forcing_vec

        t += dt
    end
    
    # Get history of dψ/dα
    # Initial conditions for dψ/dα
    copyto!(prob.u0, [0.0,0.0])
    copyto!(prob.v0, [0.0,0.0])

    history_dψdα = eval_forward_forced(prob, forcing_mat, α)

    dQda = history_dψdα[:,end]
    Q = history[:,end]
    R = copy(target)
    T = vcat(R[3:4], -R[1:2])

    gradient = -2*(dot(Q,R)*dot(dQda,R) + dot(Q,T)*dot(dQda,T))
    return gradient
end


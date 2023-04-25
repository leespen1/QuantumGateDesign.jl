include("../src/hermite.jl")
using Plots

function this_prob(;ω::Float64=1.0, tf::Float64=1.0, nsteps::Int64=10)
    Ks::Matrix{Float64} = [0 0; 0 1]
    Ss::Matrix{Float64} = [0 0; 0 0]
    p(t,α) = α*cos(ω*t)
    q(t,α) = 0.0
    dpdt(t,α) = -α*ω*sin(ω*t)
    dqdt(t,α) = 0.0
    dpda(t,α) = cos(ω*t)
    dqda(t,α) = 0.0
    d2p_dta(t,α) = -ω*sin(ω*t)
    d2q_dta(t,α) = 0.0
    u0::Vector{Float64} = [1,0]
    v0::Vector{Float64} = [0,0]
    return SchrodingerProb(Ks,Ss,
                           p,q,dpdt,dqdt,dpda,dqda,d2p_dta,d2q_dta,
                           u0,v0,tf,nsteps)
end



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

function finite_difference(prob, α, target, dα=1e-5; order=2)
    # Centered Difference Approximation
    history_r = eval_forward(prob, α+dα, order=order)
    history_l = eval_forward(prob, α-dα, order=order)
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


function figure1(α=1.0; order=2, nsteps=10)
    prob = this_prob(nsteps=nsteps)
    history = eval_forward(prob, α, order=order)
    target = history[:,end]

    N = 100
    alphas = LinRange(0.1,2,N)
    grads_fd = zeros(N)
    #grads_diff_mat = zeros(N)
    grads_diff_forced = zeros(N)
    grads_da = zeros(N)
    for i in 1:N
        α = alphas[i]
        grads_fd[i] = finite_difference(prob, α, target, order=order)
        #grads_diff_mat[i] = eval_forward_grad_mat(target, α)
        grads_da[i] = discrete_adjoint(prob, target, α, order=order)
        grads_diff_forced[i] = eval_grad_forced(prob, target, α, order=order)
    end
    #return alphas, grads_fd, grads_diff_mat, grads_diff_forced, grads_diff_auto_forced, grads_da
    return alphas, grads_fd, grads_diff_forced, grads_da
end

function plot_figure1(alphas, grads_fd, grads_diff_forced, grads_da)
    # If all the gradients are working, this graph won't be much use
    pl1 = plot(alphas, grads_fd, label="Finite Difference", lw=2)
    #plot!(pl1, alphas, grads_diff_mat, label="Differentiation (Matrix)", lw=2)
    plot!(pl1, alphas, grads_diff_forced, label="Differentiation / Forced", lw=2)
    plot!(pl1, alphas, grads_da, label="Discrete Adjoint", lw=2)
    plot!(pl1, xlabel="α", ylabel="Gradient")
    plot!(pl1, legendfontsize=14,guidefontsize=14,tickfontsize=14)

    # Use finite difference as the "true" value
    errs_diff_forced = abs.(grads_fd .- grads_diff_forced)
    #errs_diff_mat = abs.(grads_fd .- grads_diff_mat)
    errs_da = abs.(grads_fd .- grads_da)
    pl2 = plot(alphas, errs_diff_forced, label="Differentiation / Forced", lw=2)
    #plot!(pl2, alphas, errs_diff_mat, label="Differentiaion (Matrix)", lw=2)
    plot!(pl2, alphas, errs_da, label="Discrete Adjoint", lw=2)
    plot!(pl2, legendfontsize=14,guidefontsize=14,tickfontsize=14)
    plot!(pl2, yscale=:log10)
    plot!(pl2, xlabel="α", title="Deviation from Finite Difference Gradient")
    return pl1, pl2
end



function figure2(;α=1.0, tf=1.0, ω=1.0, base_nsteps=2)
    prob = this_prob(tf=tf, ω=ω)
    orders = [2,4]
    N = 5
    sol_errs = zeros(N, length(orders))
    infidelities = zeros(N, length(orders))

    step_sizes = zeros(N)
    for n in 1:N
        step_sizes[n] = prob.tf / (base_nsteps^n)
    end

    # Get 'true' solution, using most timesteps and highest order
    prob.nsteps = base_nsteps^(N+1)
    history = eval_forward(prob, α, order=max(orders...)) 
    final_state_fine = history[:,end]

    for j in 1:length(orders)
        order = orders[j]
        final_states = zeros(4,N)
        for i in 1:N
            prob.nsteps = base_nsteps^i
            history = eval_forward(prob, α, order=order)
            final_states[:,i] = history[:,end]
        end

        for i in 1:N
            sol_errs[i,j] = norm(final_states[:,i] - final_state_fine)
            infidelities[i,j] = infidelity(final_states[:,i], final_state_fine)
        end
    end
   
    return step_sizes, sol_errs, infidelities, orders
end

function plot_figure2(step_sizes, sol_errs, infidelities, orders)
    pl = plot()
    for i in 1:length(orders)
        plot!(pl, step_sizes, abs.(sol_errs[:,i]), linewidth=2, marker=:circle, label="Error (Order $(orders[i]))")
        plot!(pl, step_sizes, abs.(infidelities[:,i]), linewidth=2, marker=:circle, label="Infidelities (Order $(orders[i]))")
    end
    plot!(pl, step_sizes, step_sizes .^ 2, label="Δt^2", linestyle=:dash)
    plot!(pl, step_sizes, step_sizes .^ 4, label="Δt^4", linestyle=:dash)
    plot!(pl, step_sizes, step_sizes .^ 6, label="Δt^6", linestyle=:dash)
    plot!(pl, legendfontsize=14, guidefontsize=14, tickfontsize=14)
    plot!(pl, scale=:log10)
    plot!(pl, legend=:bottomright)
    plot!(pl, xlabel="Δt")
    return pl
end





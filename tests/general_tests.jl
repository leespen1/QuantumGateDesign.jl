using Plots
include("../src/hermite.jl")

function convergence_test!(prob::SchrodingerProb, α=missing; base_nsteps=2, N=5)
    orders = [2,4]
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



function plot_convergence_test(step_sizes, sol_errs, infidelities, orders)
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



function gradient_test(prob::SchrodingerProb, α, dα; order=2)
    history = eval_forward(prob, α, order=order)
    target = history[:,end]

    N = 100
    alphas = LinRange(0.1,2,N)
    grads_fd = zeros(N)
    grads_diff_forced = zeros(N)
    grads_da = zeros(N)
    for i in 1:N
        α = alphas[i]
        grads_fd[i] = eval_grad_finite_difference(prob, target, α, dα, order=order)
        grads_da[i] = discrete_adjoint(prob, target, α, order=order)
        grads_diff_forced[i] = eval_grad_forced(prob, target, α, order=order)
    end
    return alphas, grads_fd, grads_diff_forced, grads_da
end



function plot_gradient_test(alphas, grads_fd, grads_diff_forced, grads_da)
    # Plot gradient values
    pl1 = plot(alphas, grads_fd, label="Finite Difference", lw=2)
    plot!(pl1, alphas, grads_diff_forced, label="Differentiation / Forced", lw=2)
    plot!(pl1, alphas, grads_da, label="Discrete Adjoint", lw=2)
    plot!(pl1, xlabel="α", ylabel="Gradient")
    plot!(pl1, legendfontsize=14,guidefontsize=14,tickfontsize=14)

    # Use finite difference as the "true" value
    errs_diff_forced = abs.(grads_fd .- grads_diff_forced)
    errs_da = abs.(grads_fd .- grads_da)

    # Plot deviation from finite difference gradient
    pl2 = plot(alphas, errs_diff_forced, label="Differentiation / Forced", lw=2)
    plot!(pl2, alphas, errs_da, label="Discrete Adjoint", lw=2)
    plot!(pl2, legendfontsize=14,guidefontsize=14,tickfontsize=14)
    plot!(pl2, yscale=:log10)
    plot!(pl2, xlabel="α", title="Deviation from Finite Difference Gradient")
    return pl1, pl2
end



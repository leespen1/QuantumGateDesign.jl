function gradient_test(prob::SchrodingerProb, α, dα; order=2, cost_type=:Infidelity)
    history = eval_forward(prob, α, order=order)
    target = history[:,end]

    N = 100
    alphas = LinRange(0.1,2,N)
    grads_fd = zeros(N)
    grads_diff_forced = zeros(N)
    grads_da = zeros(N)
    for i in 1:N
        α = alphas[i]
        grads_fd[i] = eval_grad_finite_difference(prob, target, α, dα,
                                                  order=order, cost_type=cost_type)
        grads_diff_forced[i] = eval_grad_forced(prob, target, α,
                                                order=order, cost_type=cost_type)
        grads_da[i] = discrete_adjoint(prob, target, α,
                                       order=order, cost_type=cost_type)
    end
    return alphas, grads_fd, grads_diff_forced, grads_da
end


function plot_gradients(alphas, grads_fd, grads_diff_forced, grads_da)
    # Plot gradient values
    pl = plot(alphas, grads_fd, label="Finite Difference", lw=2)
    plot!(pl, alphas, grads_diff_forced, label="Differentiation / Forced", lw=2)
    plot!(pl, alphas, grads_da, label="Discrete Adjoint", lw=2)
    plot!(pl, xlabel="α", ylabel="Gradient")
    plot!(pl, legendfontsize=14,guidefontsize=14,tickfontsize=14)
    return pl
end

function plot_gradient_deviation(alphas, grads_fd, grads_diff_forced, grads_da)

    # Use finite difference as the "true" value
    errs_diff_forced = abs.(grads_fd .- grads_diff_forced)
    errs_da = abs.(grads_fd .- grads_da)

    # Plot deviation from finite difference gradient
    pl = plot(alphas, errs_diff_forced, label="Differentiation / Forced", lw=2)
    plot!(pl, alphas, errs_da, label="Discrete Adjoint", lw=2)
    plot!(pl, legendfontsize=14,guidefontsize=14,tickfontsize=14)
    plot!(pl, yscale=:log10)
    plot!(pl, xlabel="α", title="Deviation from Finite Difference Gradient")
    return pl
end

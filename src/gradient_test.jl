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

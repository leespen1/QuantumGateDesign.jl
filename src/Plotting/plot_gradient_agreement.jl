function plot_gradient_agreement(prob, controls, target; 
        orders=(2,4,6,8,10), cost_type=:Infidelity,
        n_runs=10, amax=5e-2, abstol=1e-15, reltol=1e-15
    )

    N_orders = length(orders)
    N_coeffs = get_number_of_control_parameters(controls)


    gradients = Array{Float64, 4}(undef, N_coeffs, 3, N_orders, n_runs)
    errors = Array{Float64, 3}(undef, n_runs, N_orders, 2)


    for i in 1:n_runs
        pcof = rand(MersenneTwister(i), N_coeffs) .* amax
        for (k, order) in enumerate(orders)
            # Check that gradients calculated using discrete adjoint and finite difference
            # methods agree to reasonable precision
            grad_disc_adj = discrete_adjoint(
                prob, controls, pcof, target, order=order,
                cost_type=cost_type, abstol=abstol=abstol, reltol=reltol
            )

            grad_forced = eval_grad_forced(
                prob, controls, pcof, target, order=order,
                cost_type=cost_type, abstol=abstol=abstol, reltol=reltol
            )

            grad_fin_diff = eval_grad_finite_difference(
                prob, controls, pcof, target, order=order,
                cost_type=cost_type, abstol=abstol=abstol, reltol=reltol
            )

            gradients[:,1,k,i] .= grad_disc_adj
            gradients[:,2,k,i] .= grad_forced
            gradients[:,3,k,i] .= grad_fin_diff

            errors[i, k, 1] = norm(grad_forced - grad_disc_adj)/norm(grad_disc_adj)
            errors[i, k, 2] = norm(grad_fin_diff - grad_disc_adj)/norm(grad_disc_adj)
        end
    end

    replace_zero_with_epsilon(x) = (x == 0.0) ? 1e-16 : x
    errors = replace_zero_with_epsilon.(errors)
    errors = log10.(errors)

    xticks = 1:n_runs
    yticks = -20:20

    display(errors)

    pl = Plots.plot(xlabel="Random Control Vector #", ylabel="Log₁₀(Rel Err in Gradient)",
                    legend=:outerright, xticks=xticks, yticks=yticks, size=(600,400))

    colors = Plots.theme_palette(:auto)
    marker_forced = :circle
    marker_fin_diff = :star5
    for (k, order) in enumerate(orders)
        Plots.scatter!(pl, errors[:, k, 1], color=colors[k], label="Order $order, Forced", marker=marker_forced, markersize=5)
        Plots.scatter!(pl, errors[:, k, 2], color=colors[k], label="Order $order, Fin Diff", marker=marker_fin_diff, markersize=5)
    end
    Plots.plot!(pl, yticks=-16:0)


    return pl, errors
end

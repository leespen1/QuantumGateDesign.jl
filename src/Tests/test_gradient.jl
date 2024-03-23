using Test: @test, @testset

function test_gradient_agreement(prob, control, pcof, target; 
        order=2, cost_type=:Infidelity,
        print_results=false
    )


    # Check that gradients calculated using discrete adjoint and finite difference
    # methods agree to reasonable precision
    grad_disc_adj = discrete_adjoint(
        prob, control, pcof, target, order=order,
        cost_type=cost_type
    )

    grad_forced = eval_grad_forced(
        prob, control, pcof, target, order=order,
        cost_type=cost_type
    )

    grad_fin_diff = eval_grad_finite_difference(
        prob, control, pcof, target, order=order,
        cost_type=cost_type
    )

    forced_atol = 1e-14
    fin_diff_atol = 1e-9

    @testset "Forced Method" begin
        for k in 1:length(grad_disc_adj)
            @test isapprox(grad_disc_adj[k], grad_forced[k], atol=forced_atol, rtol=forced_atol)
        end
    end

    @testset "Finite Difference" begin
        for k in 1:length(grad_disc_adj)
            @test isapprox(grad_disc_adj[k], grad_fin_diff[k], atol=fin_diff_atol, rtol=forced_atol)
        end
    end

    disc_adj_minus_forced = grad_disc_adj - grad_forced
    disc_adj_minus_fin_diff = grad_disc_adj - grad_fin_diff

    
    maximum(abs.(disc_adj_minus_fin_diff))


    if print_results
        println("Order=$order, cost_type=$cost_type")
        println("--------------------------------------------------")
        #println("Discrete Adjoint Gradient:\n", grad_disc_adj)
        #println("\nForced Method Gradient:\n", grad_forced)
        #println("\nFinite Difference Gradient:\n", grad_fin_diff)
        #println("\n\nDiscrete Adjoint - Forced Method:\n", disc_adj_minus_forced)
        #println("Discrete Adjoint - Finite Difference:\n", disc_adj_minus_fin_diff)
        println("||Discrete Adjoint - Forced Method||∞: ", maximum(abs.(disc_adj_minus_forced)))
        println("||Discrete Adjoint - Finite Difference||∞: ", maximum(abs.(disc_adj_minus_fin_diff)))
        println("\n")
    end


    return nothing
end

function plot_gradient_agreement(prob, controls, target; 
        orders=(2,4,6,8,10), cost_type=:Infidelity,
        n_runs=10, amax=5e-2
    )

    N_orders = length(orders)
    N_coeffs = get_number_of_control_parameters(controls)


    gradients = Array{Float64, 4}(undef, N_coeffs, 3, N_orders, n_runs)
    errors = Array{Float64, 3}(undef, n_runs, N_orders, 2)


    for i in 1:n_runs
        pcof = rand(N_coeffs) .* amax
        for (k, order) in enumerate(orders)
            # Check that gradients calculated using discrete adjoint and finite difference
            # methods agree to reasonable precision
            grad_disc_adj = discrete_adjoint(
                prob, controls, pcof, target, order=order,
                cost_type=cost_type
            )

            grad_forced = eval_grad_forced(
                prob, controls, pcof, target, order=order,
                cost_type=cost_type
            )

            grad_fin_diff = eval_grad_finite_difference(
                prob, controls, pcof, target, order=order,
                cost_type=cost_type
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


    return pl, errors

end

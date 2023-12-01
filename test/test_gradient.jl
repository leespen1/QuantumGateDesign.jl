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

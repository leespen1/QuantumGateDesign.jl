using Test: @test, @testset

function test_gradient_agreement(prob, control, pcof, target; 
        orders=[2, 4], cost_types=[:Infidelity, :Tracking, :Norm]
    )

    # Check that gradients calculated using discrete adjoint and finite difference
    # methods agree to reasonable precision
    @testset "Discrete Adjoint Vs Finite Difference Vs Forced Method" begin
    for cost_type in cost_types
        @testset "Cost function: $cost_type" begin
        for order in orders
            @testset "Order: $order" begin
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
                fin_diff_atol = 1e-9 # Might want to relax this to 1e-9.

                @testset "Forced Method" begin
                    for k in 1:length(grad_disc_adj)
                        @test isapprox(grad_disc_adj[k], grad_forced[k], atol=forced_atol)
                    end
                end

                @testset "Finite Difference" begin
                    for k in 1:length(grad_disc_adj)
                        @test isapprox(grad_disc_adj[k], grad_fin_diff[k], atol=fin_diff_atol)
                    end
                end

            end #@testset "Order: $order"
        end
        end #@testset "Cost function: $cost_type"
    end
    end # @testset "Discrete Adjoint Vs Finite Difference Vs Forced Method"
end

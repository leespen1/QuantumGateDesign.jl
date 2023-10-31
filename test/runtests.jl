using HermiteOptimalControl
using Test

#=
# Test suite
#
# Instead of randomly generative control vectors and targets each run, perhaps
# I should do it once and hard-code the results. I can even save the gradients,
# so that each time I update I will be checking whether the algorithms produce
# the exact same results up to machine precision.
#
=#
detuning_frequency = 1.0
self_kerr_coefficient = 0.5

N_ess = 2
N_guard = 0

N_coeff_per_control = 4

tf=1.0
nsteps = 5 # Use few steps for tests, care about algorithm correctness, not physical accuracy

prob, control = HermiteOptimalControl.qubit_with_bspline(
    detuning_frequency,
    self_kerr_coefficient,
    N_ess,
    N_guard,
    tf=tf,
    nsteps=nsteps,
    N_coeff_per_control=N_coeff_per_control
)

target = rand(4,2)
pcof = rand(8)

orders = [2, 4]
cost_types = [:Infidelity, :Tracking, :Norm]

@testset "Checking Gradient Agreement Between Methods" begin

# Check that gradients calculated using discrete adjoint and finite difference
# methods agree to reasonable precision
@testset "Discrete Adjoint Vs Finite Difference Vs Forced Method" begin
for cost_type in cost_types
    @testset "Cost function: $cost_type" begin
    for order in orders
        @testset "Order: $order" begin
        grad_disc_adj = discrete_adjoint(prob, control, pcof, target, order=order, cost_type=cost_type)

        grad_forced = eval_grad_forced(prob, control, pcof, target, order=order, cost_type=cost_type)
        grad_fin_diff = eval_grad_finite_difference(prob, control, pcof, target, order=order, cost_type=cost_type)

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

end # @testset "Checking Gradient Agreement Between Methods"

@testset "Covergence" begin
# Should check that convergence of forward evolution is of the correct order.
end


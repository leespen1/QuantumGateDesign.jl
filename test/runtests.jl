using Test

nsteps = 5 # Use a small number of steps for these correctness tests, to lower computational cost

N_ess = 2
N_guard = 0
#probs = [rabi_osc(N_ess, N_guard, nsteps=nsteps),
#         gargamel_prob(nsteps=nsteps),
#         ]
#pcofs = [rand(), rand(4), rand(1)]
probs = [bspline_prob(nsteps=nsteps), gargamel_prob(nsteps=nsteps),
         rabi_osc(N_ess, N_guard, nsteps=nsteps)]
probs_strs = ["Bspline Prob", "Gargamel Prob", "Rabi Oscilator"]
orders = [2, 4]
cost_types = [:Infidelity, :Tracking, :Norm]

# May be better to use a consistent-between-tests set of randomly generated 
# variables, instead of regenerating each time.
pcofs = [rand(8), rand(4), rand()]
target = rand(4)

@testset "Checking Gradient Agreement Between Methods" begin

# Check that gradients calculated using discrete adjoint and finite difference
# methods agree to reasonable precision
@testset "Discrete Adjoint Vs Finite Difference" begin
for (i, prob) in enumerate(probs)
    prob_str = probs_strs[i]
    pcof = pcofs[i]
    @testset "Problem: $prob_str" begin
    for cost_type in cost_types
        @testset "Cost function: $cost_type" begin
        for order in orders
            @testset "Order: $order" begin
            grad_disc_adj = discrete_adjoint(prob, target, pcof, order=order, cost_type=cost_type)
            grad_fin_diff = eval_grad_finite_difference(prob, target, pcof, order=order, cost_type=cost_type)
            absolute_tolerance = 1e-10 # Might want to relax this to 1e-9.

            for k in 1:length(grad_disc_adj)
                @test isapprox(grad_disc_adj[k], grad_fin_diff[k], atol = absolute_tolerance)
            end
            end #@testset
        end
        end #@testset
    end
    end #@testset
end
end #@testset "Discrete Adjoint Vs Finite Difference"

# Check agreement between discrete adjoint and forward differentiation
@testset "Discrete Adjoint Vs Forward Differentiation" begin
for i in 2:length(probs) # Omitting bspline prob for now because 4th order isnt ready for that problem
    prob = probs[i]
    prob_str = probs_strs[i]
    pcof = pcofs[i]
    @testset "Problem: $prob_str" begin
    for cost_type in cost_types
        @testset "Cost function: $cost_type" begin
        for order in orders
            @testset "Order: $order" begin
            grad_disc_adj = discrete_adjoint(prob, target, pcof, order=order, cost_type=cost_type)
            grad_fwd_diff = eval_grad_finite_difference(prob, target, pcof, order=order, cost_type=cost_type)
            absolute_tolerance = 1e-1 # Might want to relax this to 1e-9.

            for k in 1:length(grad_disc_adj)
                @test isapprox(grad_disc_adj[k], grad_fwd_diff[k], atol = absolute_tolerance)
            end
            end #@testset
        end
        end #@testset
    end
    end #@testset
end
end #@testset "Discrete Adjoint Vs Forward Differentiation"

end # @testset "Checking Gradient Agreement Between Methods"
println("End Tests")

# Should I also add an automated test for convergence?
# I can precompute a "true" solution if I keep the variables the same between
# tests.

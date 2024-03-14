using QuantumGateDesign
using Test: @testset, @test


#=
# Test suite
#
# Instead of randomly generative control vectors and targets each run, perhaps
# I should do it once and hard-code the results. I can even save the gradients,
# so that each time I update I will be checking whether the algorithms produce
# the exact same results up to machine precision.
#
=#
@testset "Constructing Typical Problem Without Throwing Exception" begin
end


@testset "Constructing Control Without Throwing Exception" begin
end


N_ess = 2
N_guard = 2
N_tot_levels = N_ess + N_guard

detuning_frequency = 1.0
self_kerr_coefficient = 0.5

tf=1.0
nsteps = 5

prob = rotating_frame_qubit(
    N_ess,
    N_guard,
    tf = tf,
    nsteps = nsteps,
    detuning_frequency = detuning_frequency,
    self_kerr_coefficient = self_kerr_coefficient
)


D1 = 4
carrier_wave_freqs = [-k*self_kerr_coefficient for k in 0:2]
control = bspline_control(tf, D1, carrier_wave_freqs)

N_tot_levels = N_ess + N_guard
target = rand(2*N_tot_levels, N_ess)
pcof = rand(control.N_coeff)

orders = (2, 4)
cost_types = (:Infidelity, :Tracking, :Norm)


# Smoke test, just make sure the main functions run without throwing exceptions
@testset "Gradient Calculations Run Without Throwing Exceptions" begin
for order in orders
    for cost_type in cost_types
        eval_forward(prob, control, pcof, order=order)

        discrete_adjoint(prob,            control, pcof, target, order=order, cost_type=cost_type)
        eval_grad_finite_difference(prob, control, pcof, target, order=order, cost_type=cost_type)
        eval_grad_forced(prob,            control, pcof, target, order=order, cost_type=cost_type)
    end
end
@test true
end


@testset "Gradient Agreement Between Methods" begin
for order in orders
    @testset "Order: $order" begin
    for cost_type in cost_types
        @testset "Cost Function: $cost_type" begin
        QuantumGateDesign.test_gradient_agreement(
            prob, control, pcof, target, order=order, cost_type=cost_type,
            print_results=true
        )
        end
    end
    end
end
end # @testset "Checking Gradient Agreement Between Methods"

@testset "Covergence" begin
# Should check that convergence of forward evolution is of the correct order.
end

@testset "Eval Forward Hard-Coded" begin
end

@testset "Eval Adjoint Hard-Coded" begin
end

@testset "Gradients Hard-Coded" begin
end

include("./hardcoded_derivatives.jl")

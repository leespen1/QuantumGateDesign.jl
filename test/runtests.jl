using HermiteOptimalControl
using Test: @testset

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
# EDIT: Forced and Discrete Adjoint gradients seem to deviate as number of
# steps increase. Is this numerical saturation / double precision breakdown, or
# something worse?

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

@testset "Gradient Agreement Between Methods" begin
    HermiteOptimalControl.test_gradient_agreement(
        prob, control, pcof, target, orders=orders, cost_types=cost_types
    )
end # @testset "Checking Gradient Agreement Between Methods"

@testset "Covergence" begin
# Should check that convergence of forward evolution is of the correct order.
end




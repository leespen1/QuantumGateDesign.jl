using QuantumGateDesign
using Test: @testset, @test

@testset "All Tests" begin
#include("./hardcoded_derivatives.jl")
# Test that Control Functions are working properly
include("./ControlFunctionTests/quintic_hermite_interpolation.jl")
include("./ControlFunctionTests/test_control_derivatives.jl")
include("./ControlFunctionTests/test_control_gradients.jl")
include("./ControlFunctionTests/hermite_polynomial.jl")
# Test that IVPs are converging with correct order
include("./ConvergenceTests/forward_convergence.jl")
# Test that gradients are being computed consistently
include("./GradientTests/compare_gradients.jl")
# Check that the gate design/optimization problem is behaving properly
include("./OptimizationTests/optimization_rabi_osc_SWAP.jl")
end

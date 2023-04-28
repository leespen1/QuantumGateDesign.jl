module HermiteOptimalControl

using LinearAlgebra, LinearMaps, IterativeSolvers, Plots

export SchrodingerProb, eval_forward, eval_forward_forced
export eval_grad_finite_difference, eval_grad_forced, discrete_adjoint
export gradient_test, plot_gradients, plot_gradient_deviation
export convergence_test!, plot_convergence_test
export rabi_osc

include("hermite.jl")
include("gradient_test.jl")
include("convergence_test.jl")
include("../tests/rabi_hermite_tests.jl")

end # module HermiteOptimalControl

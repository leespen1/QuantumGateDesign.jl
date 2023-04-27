module HermiteOptimalControl

using LinearAlgebra, LinearMaps, IterativeSolvers, Plots
#using ForwardDiff

export SchrodingerProb, eval_forward, eval_forward_forced
export eval_grad_finite_difference, eval_grad_forced, discrete_adjoint
export gradient_test, plot_gradient_test
export convergence_test!, plot_convergence_test

include("hermite.jl")
include("gradient_test.jl")
include("convergence_test.jl")

end # module HermiteOptimalControl

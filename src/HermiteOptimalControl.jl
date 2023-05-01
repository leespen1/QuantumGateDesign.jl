module HermiteOptimalControl

using LinearAlgebra, LinearMaps, IterativeSolvers, Plots

export SchrodingerProb, eval_forward, eval_forward_forced
export eval_grad_finite_difference, eval_grad_forced, discrete_adjoint
export gradient_test, plot_gradients, plot_gradient_deviation
export convergence_test!, plot_convergence_test
export rabi_osc
export gargamel_prob
#export daniel_prob, daniel_test

include("SchrodingerProb.jl")
include("hermite.jl")
include("forward_evolution.jl")
include("eval_grad.jl")
include("gradient_test.jl")
include("convergence_test.jl")
include("../tests/rabi_prob.jl")
include("../tests/gargamel_prob.jl")
#include("../tests/daniel_prob.jl")

end # module HermiteOptimalControl

module HermiteOptimalControl


import LinearMaps, IterativeSolvers, Plots, Ipopt
using LinearAlgebra: mul!, axpy!, dot, tr, norm

# Export schrodinger problem definition and forward evolution methods
export SchrodingerProb, VectorSchrodingerProb, eval_forward, eval_forward_forced

# Export gradient evaulation methods
export eval_grad_finite_difference, eval_grad_forced, discrete_adjoint

# Export bspline functions
export bcparams, bcarrier2, bcarrier2_dt, gradbcarrier2, gradbcarrier2_dt, gradbcarrier2!, gradbcarrier2_dt!

# Export tests
export gradient_test, plot_gradients, plot_gradient_deviation
export convergence_test!, plot_convergence_test

# Export specific problem definitions
export qubit_with_bspline, bspline_control

export infidelity

export optimize_gate

export get_populations, target_helper, plot_populations

include("SchrodingerProb.jl")
include("bsplines.jl")
include("Control.jl")
include("convergence_test.jl")
include("eval_grad.jl")
include("forward_evolution.jl")
include("gradient_test.jl")
include("hamiltonian_construction.jl")
include("hermite.jl")
include("ipopt_optimal_control.jl")
include("OptimizationParameters.jl")
include("state_vector_helpers.jl")

#include("ExampleProblems/rabi_prob.jl")
#include("ExampleProblems/gargamel_prob.jl")
include("ExampleProblems/bspline_prob.jl")

include("../test/test_gradient.jl")

end # module HermiteOptimalControl

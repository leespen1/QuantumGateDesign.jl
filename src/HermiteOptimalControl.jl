module HermiteOptimalControl


import LinearMaps, IterativeSolvers, Plots, Ipopt, ForwardDiff
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
export single_qubit_prob_with_bspline_control, bspline_control

export infidelity

export optimize_gate

export get_populations, target_helper, plot_populations, real_to_complex, complex_to_real

export Control, AbstractControl, BSplineControl

export rotating_frame_qubit
export initial_basis


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
include("ExampleProblems/bspline_control.jl")
include("ExampleProblems/sincos_control.jl")
include("ExampleProblems/standard_prob.jl")

include("../test/test_gradient.jl")
include("../test/test_convergence.jl")

end # module HermiteOptimalControl

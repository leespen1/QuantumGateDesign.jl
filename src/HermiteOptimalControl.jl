module HermiteOptimalControl

using LinearAlgebra, LinearMaps, IterativeSolvers, Plots
import Ipopt

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
export rabi_osc, gargamel_prob, bspline_prob

export infidelity

export optimize_gate

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

#include("ExampleProblems/rabi_prob.jl")
#include("ExampleProblems/gargamel_prob.jl")
include("ExampleProblems/bspline_prob.jl")

end # module HermiteOptimalControl

module HermiteOptimalControl


import LinearMaps, IterativeSolvers, Plots, Ipopt, ForwardDiff
using LinearAlgebra: mul!, axpy!, dot, tr, norm

# Export schrodinger problem definition and forward evolution methods
export SchrodingerProb, VectorSchrodingerProb
export eval_forward, eval_forward_forced, eval_forward_arbitrary_order

# Export gradient evaulation methods
export eval_grad_finite_difference, eval_grad_forced, discrete_adjoint

# Export optimization callback
export optimize_gate

# Export helper for functions for dealing state vectors and histories
export get_populations, target_helper, plot_populations, real_to_complex, complex_to_real

export Control, AbstractControl, BSplineControl


# Export example problems and problem construction helpers
export lowering_operator, raising_operator, subsytem_lowering_operator
export composite_system_lowering_operators
export rotating_frame_qubit, dahlquist_problem

# Export testing functions
export plot_history_convergence, plot_history_convergence_new


export initial_basis
export infidelity


include("SchrodingerProb.jl")


include("Controls/Control.jl")
include("Controls/bspline_backend.jl")
include("Controls/bspline_control.jl")
include("Controls/grape_control.jl")

include("hermite.jl")

include("forward_evolution.jl")

include("infidelity.jl")

include("eval_grad_discrete_adjoint.jl")
include("eval_grad_finite_difference.jl")
include("eval_grad_forced.jl")


include("ipopt_optimal_control.jl")
include("state_vector_helpers.jl")

include("ProblemConstructors/lowering_operators.jl")
include("ProblemConstructors/rotating_frame_qubit.jl")
include("ProblemConstructors/dahlquist_problem.jl")

include("Tests/test_gradient.jl")
include("Tests/test_convergence.jl")

end # module HermiteOptimalControl

module HermiteOptimalControl


import LinearMaps, IterativeSolvers, Plots, Ipopt, ForwardDiff, LinearAlgebra
using LinearAlgebra: mul!, axpy!, dot, tr, norm

# Export schrodinger problem definition and forward evolution methods
export SchrodingerProb, VectorSchrodingerProb
export eval_forward, eval_forward_forced, eval_forward_arbitrary_order, eval_adjoint_arbitrary_order
export convert_old_ordering_to_new

# Export gradient evaulation methods
export eval_grad_finite_difference, eval_grad_forced, discrete_adjoint
export discrete_adjoint_arbitrary_order, compute_terminal_condition

# Export optimization callback
export optimize_gate

# Export helper for functions for dealing state vectors and histories
export get_populations, target_helper, plot_populations, real_to_complex, complex_to_real

# Export control types and constructors
export AbstractControl, BSplineControl, GRAPEControl, GeneralGRAPEControl
export bspline_control, ZeroControl
export eval_p, eval_q, eval_p_derivative, eval_q_derivative, eval_grad_p_derivative, eval_grad_q_derivative


# Export example problems and problem construction helpers
export lowering_operator, raising_operator, subsytem_lowering_operator
export composite_system_lowering_operators
export rotating_frame_qubit, dahlquist_problem

# Export testing functions
export plot_history_convergence, plot_history_convergence_new

# Export plotting functions
export plot_control


export initial_basis
export infidelity

export hermite_interp_poly

include("SchrodingerProb.jl")
include("../Daniel/hermite_map.jl")


include("Controls/Control.jl")
include("Controls/bspline_backend.jl")
include("Controls/bspline_control.jl")
include("Controls/grape_control.jl")
include("Controls/hermite_control.jl")
include("Controls/sincos_control.jl")
include("Controls/zero_control.jl")
include("Controls/generalized_grape_control.jl")


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

include("Plotting/plot_control.jl")

include("extension_compatibility.jl")

end # module HermiteOptimalControl

module QuantumGateDesign


import LinearMaps, IterativeSolvers, Plots, Ipopt, ForwardDiff, LinearAlgebra
import BenchmarkTools, SparseArrays, Dates, OrderedCollections, JLD2, Random
import BSplines, BasicBSpline
using Printf: @printf, @sprintf
using LinearAlgebra: mul!, axpy!, dot, tr, norm
using Random: rand, MersenneTwister
using Base.Iterators: product
using BasicBSpline: BSplineDerivativeSpace, BSplineSpace

# Export derivative computation functions
export compute_derivatives!, compute_adjoint_derivatives!, compute_partial_derivative!, apply_hamiltonian!

# Export schrodinger problem definition and forward evolution methods
export SchrodingerProb, VectorSchrodingerProb
export eval_forward, eval_forward_forced
export convert_old_ordering_to_new

# Export gradient evaulation methods
export eval_grad_finite_difference, eval_grad_forced, discrete_adjoint
export eval_hessian
export discrete_adjoint, compute_terminal_condition
export eval_grad_forced

export control_ops, basis_state, create_initial_conditions, guard_projector, create_gate
export lowering_operator_subsystem, lowering_operators_system
export rotation_matrix

# Export optimization callback
export optimize_gate

# Export helper for functions for dealing state vectors and histories
export get_populations, target_helper, plot_populations, real_to_complex, complex_to_real

# Export control types and constructors
export AbstractControl, BSplineControl, GRAPEControl, GeneralGRAPEControl, HermiteControl, HermiteCarrierControl
export bspline_control, ZeroControl, GeneralBSplineControl, CarrierControl
export eval_p, eval_q, eval_p_derivative, eval_q_derivative, eval_grad_p_derivative, eval_grad_q_derivative, eval_grad_p_derivative!, eval_grad_q_derivative!

export DispersiveProblem, JaynesCummingsProblem


# Export example problems and problem construction helpers
export lowering_operator, raising_operator, subsytem_lowering_operator
export composite_system_lowering_operators
export rotating_frame_qubit, dahlquist_problem

# Export testing functions
export get_history_convergence, plot_history_convergence, plot_history_convergence_new

# Export plotting functions
export plot_control


export initial_basis
export infidelity

export hermite_interp_poly

export convert_juqbox

export MySplineControl

include("preconditioners.jl")
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
include("Controls/hermite_carrier.jl")
include("Controls/GeneralBSplineControl.jl")
include("Controls/CarrierControl.jl")
#include("Controls/BasicBSplineControl.jl")
include("Controls/FortranBSpline.jl")


include("hermite.jl")

include("forward_evolution.jl")

include("infidelity.jl")

include("eval_grad_discrete_adjoint.jl")
include("eval_grad_finite_difference.jl")
include("eval_grad_forced.jl")

include("eval_hessian.jl")

include("ipopt_optimal_control.jl")
include("gradient_descent.jl")

include("state_vector_helpers.jl")


include("ProblemConstructors/multi_qudit_systems.jl")
include("ProblemConstructors/rotating_frame_qubit.jl")
include("ProblemConstructors/dahlquist_problem.jl")
include("ProblemConstructors/juqbox_converter.jl")
include("ProblemConstructors/rabi_oscillator.jl")
include("ProblemConstructors/random_problem.jl")


include("Tests/test_convergence.jl")

include("Plotting/plot_control.jl")
include("Plotting/plot_populations.jl")
include("Plotting/plot_gradient_agreement.jl")
include("Plotting/plot_states.jl")
include("calculate_timestep.jl")


# Define functions without methods, so that extensions can override them
include("extension_compatibility.jl")
export visualize_control
export construct_ODEProb
export convert_to_numpy, Qobj, unpack_Qobj, simulate_prob_no_control

export get_number_of_control_parameters
export multi_qudit_hamiltonian
export control_ops
export eval_p_single, eval_q_single

export get_histories

export FortranBSplineControl

end # module QuantumGateDesign

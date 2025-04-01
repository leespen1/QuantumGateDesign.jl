module QuantumGateDesign

import LinearMaps, IterativeSolvers, Plots, Ipopt, LinearAlgebra,
       BenchmarkTools, SparseArrays, Dates, OrderedCollections, JLD2, Random,
       DelimitedFiles, Juqbox

using Printf: @printf, @sprintf
using LinearAlgebra: mul!, axpy!, dot, tr, norm, issymmetric, Diagonal,
      Bidiagonal, diagm
using Random: rand, MersenneTwister
using Base.Iterators: product
using LoopVectorization: @turbo
using SparseArrays: sparse



include("common.jl")
export AbstractControl, ControlsType

include("preconditioners.jl")
export IdentityPreconditioner, LUPreconditioner, DiagonalHamiltonianPreconditioner
# Defining Schrodinger Optimal Control Problems
include("SchrodingerProb.jl")
export SchrodingerProb, VectorSchrodingerProb

# Computing derivatives, Hermite quadrature
include("hermite.jl")

# Forward and Adjoint Evolution
include("forward_evolution.jl")
export eval_forward, eval_forward_forced
export GMRESTracker, avg_N_iterations, avg_residual

include("infidelity.jl")
export infidelity

# Gradient evaulation methods
include("eval_grad_discrete_adjoint.jl")
export discrete_adjoint
include("eval_grad_finite_difference.jl")
export eval_grad_finite_difference
include("eval_grad_forced.jl")
export eval_grad_forced

# IPOPT interface
include("ipopt_optimal_control.jl")
export optimize_gate

# Common Operators
include("common_operators.jl")
export lower_op, raise_op, number_op, identity_op, basis_state,
       compsys_basis_state, rot_frame_op, compsys_rot_frame_op,
       promote_subsys_op, gate_initial_states, guard_projector_op

include("state_indexing.jl")
export index_to_state, state_to_index

# Controls
include("Controls/Control.jl")
include("Controls/grape_control.jl")
include("Controls/sincos_control.jl")
include("Controls/zero_control.jl")
include("Controls/FortranBSpline.jl")
include("Controls/CarrierControl.jl")
export AbstractControl, CarrierControl, FortranBSplineControl,
       FortranBSplineControl2, GRAPEControl, ZeroControl
export eval_p, eval_q, eval_p_derivative, eval_q_derivative,
       eval_grad_p_derivative, eval_grad_q_derivative, eval_grad_p_derivative!,
       eval_grad_q_derivative!, get_number_of_control_parameters

include("eval_hessian.jl")
export eval_hessian

include("gradient_descent.jl")
# Helper for functions for dealing state vectors and histories
include("state_vector_helpers.jl")
export get_populations, target_helper, plot_populations, real_to_complex,
       complex_to_real

include("richardson_extrapolation.jl")
export RichardsonExtrapolation

include("cnot3_setup.jl")
export get_controls, get_D1, setup_cnot3

include("plotting.jl")
export plot_control

include("calculate_timestep.jl")


include("ProblemConstructors/dispersive_qudits.jl")
export dispersive_qudits_problem
include("ProblemConstructors/jaynes_cummings_qudits.jl")
export jaynes_cummings_qudits_problem
include("ProblemConstructors/jaynes_cummings_plus_kerr_qudits.jl")
export jaynes_cummings_plus_kerr_qudits_problem
include("ProblemConstructors/rotating_frame_qudit.jl")
export rotating_frame_qubit_problem
include("ProblemConstructors/dahlquist_problem.jl")
export dahlquist_problem
include("ProblemConstructors/rabi_oscillator.jl")
export rabi_oscillator_problem
include("ProblemConstructors/juqbox_converter.jl")
export convert_juqbox
include("ProblemConstructors/random_problem.jl")
export random_problem

include("common_gates.jl")
export PauliX_gate, PauliY_gate, PauliZ_gate, Hadamard_gate, Phase_gate,
       T_gate, CNOT_gate, SWAP_gate, ControlledZ_gate, QFT_gate, Toffoli_gate


# Testing Functions (not for CI, but for personal use)
include("Tests/test_convergence.jl")
export get_history_convergence, plot_history_convergence, plot_history_convergence_new



# Define functions without methods, so that extensions can override them
include("extension_compatibility.jl")
export visualize_control
export construct_ODEProb
export convert_to_numpy, Qobj, unpack_Qobj, simulate_prob_no_control

export control_ops
export eval_p_single, eval_q_single

export get_histories

export FortranBSplineControl

end # module QuantumGateDesign

#=
# Script for testing that setting up cnot3 matches the results of juqbox.
# (which I assume to be correct)
=#

using QuantumGateDesign
using LinearAlgebra: Symmetric

N_osc_levels = 10
Tmax = 550.0
nsteps = 1

cnot3ret = setup_cnot3(N_osc_levels = N_osc_levels, Tmax=Tmax)
juqbox_params = cnot3ret.juqbox_params

subsystem_sizes = (N_osc_levels, 4, 4)
essential_subsystem_sizes = (1, 2, 2) # Maybe reverse?

## Frequences from Juqbox
fa = 4.10595
fb = 4.81526  # official
fs = 7.8447 # storage   # official
rot_freq = [fa, fb, fs] # rotational frequencies
xa = 2 * 0.1099
xb = 2 * 0.1126 # official
xs = 0.002494^2/xa # 2.8298e-5 # official
xab = 1.0e-6 # 1e-6 official
xas = sqrt(xa*xs) # 2.494e-3 # official
xbs = sqrt(xb*xs) # 2.524e-3 # official

transition_freqs = (fs, fb, fa)
rotation_freqs = transition_freqs

kerr_coeffs = Symmetric(
    [xs   xbs   xas;
     xbs  xb    xab;
     xas  xab   xa],
    :U
)

prob = dispersive_qudits_problem(
    subsystem_sizes,
    essential_subsystem_sizes,
    transition_freqs,
    rotation_freqs,
    kerr_coeffs,
    Tmax,
    nsteps,
)

rot_op_T = compsys_rot_frame_op(rotation_freqs, subsystem_sizes, Tmax)

lab_target = (prob.u0 + prob.v0)*CNOT_gate()
rot_target = rot_op_T*lab_target

@show prob.u0 == juqbox_params.Uinit
@show iszero(prob.v0)

@show isapprox(prob.system_sym, juqbox_params.Hconst)

@show prob.sym_operators[1] == juqbox_params.Hsym_ops[3]
@show prob.sym_operators[2] == juqbox_params.Hsym_ops[2]
@show prob.sym_operators[3] == juqbox_params.Hsym_ops[1]

@show prob.asym_operators[1] == juqbox_params.Hanti_ops[3]
@show prob.asym_operators[2] == juqbox_params.Hanti_ops[2]
@show prob.asym_operators[3] == juqbox_params.Hanti_ops[1]

@show isapprox(rot_op_T, cnot3ret.rot1*cnot3ret.rot2*cnot3ret.rot3)

@show isapprox(rot_target, juqbox_params.Utarget_r + im*juqbox_params.Utarget_i)


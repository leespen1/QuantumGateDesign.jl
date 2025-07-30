#=
# Script for testing that setting up cnot3 matches the results of juqbox.
# (which I assume to be correct)
=#

using QuantumGateDesign
using LinearAlgebra: Symmetric

N_osc_levels = 10
Tmax = 550.0
nsteps = 1000

subsystem_sizes = (N_osc_levels, 4, 4)
essential_subsystem_sizes = (1, 2, 2)

fa = 4.10595
fb = 4.81526 
fs = 7.8447
rot_freq = [fa, fb, fs] # rotational frequencies
xa = 2 * 0.1099
xb = 2 * 0.1126 
xs = 0.002494^2/xa
xab = 1.0e-6
xas = sqrt(xa*xs)
xbs = sqrt(xb*xs)

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

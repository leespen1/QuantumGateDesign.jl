#=
# Script for testing that setting up cnot3 matches the results of juqbox.
# (which I assume to be correct)
=#

using QuantumGateDesign
using QuantumGateDesign: get_shortest_period
using LinearAlgebra: Symmetric

N_osc_levels = 10
Tmax = 1.0
nsteps = 1

cnot3_subsystem_sizes = (N_osc_levels, 4, 4)
cnot3_essential_subsystem_sizes = (1, 2, 2) # Maybe reverse?

## Frequencies from Juqbox
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

cnot3_transition_freqs = (fs, fb, fa)
cnot3_rotation_freqs = cnot3_transition_freqs

cnot3_kerr_coeffs = Symmetric(
    [xs   xbs   xas;
     xbs  xb    xab;
     xas  xab   xa],
    :U
)

cnot3_prob = dispersive_qudits_problem(
    cnot3_subsystem_sizes,
    cnot3_essential_subsystem_sizes,
    cnot3_transition_freqs,
    cnot3_rotation_freqs,
    cnot3_kerr_coeffs,
    Tmax,
    nsteps,
)
cnot3_w = get_shortest_period(cnot3_prob, zeros(cnot3_prob.N_operators))
println("CNOT3 Shortest Wavelength $(cnot3_prob.real_system_size): $cnot3_w")

# Can try fully connected, then spin chain
for nq in 1:5
    subsystem_sizes = ntuple(x -> 7, nq)
    essential_subsystem_sizes = subsystem_sizes
    transition_freqs = ntuple(x -> 4.80, nq)
    rotation_freqs = transition_freqs
    self_kerr =  0.22
    #cross_kerr = 1.0e-6
    cross_kerr = xas
    kerr_coeffs = [(j == k) ? self_kerr : cross_kerr for j in 1:nq, k in 1:nq]

    prob = dispersive_qudits_problem(
        subsystem_sizes,
        essential_subsystem_sizes,
        transition_freqs,
        rotation_freqs,
        kerr_coeffs,
        Tmax,
        nsteps,
    )

    max_amps = 1.00 * ones(prob.N_operators)
    w = get_shortest_period(prob, max_amps)
    println("$nq connected qubits shortest wavelength $(prob.real_system_size): $w")
    
    
end


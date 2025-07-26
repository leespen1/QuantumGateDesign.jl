# Remember to activate environment
using QuantumGateDesign, HssMatrices, Random
using QuantumGateDesign: setup_cnot3, get_D1, get_controls

degree = 14
D1 = 15
nsteps = 64
N_osc_levels = 10
t = 0.0
order = 10

cnot3ret = QuantumGateDesign.setup_cnot3(N_osc_levels=N_osc_levels)

prob = cnot3ret.qgd_prob
prob.nsteps = nsteps
controls = get_controls(14, 16, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)
N_coeff = get_number_of_control_parameters(controls)
dt = prob.tf / prob.nsteps

pcof = cnot3ret.amax * 2 * (0.5 .- rand(MersenneTwister(0), N_coeff))
pcof0 = zeros(N_coeff)

A_nonzero = QuantumGateDesign.form_LHS(prob, controls, t, pcof, dt, order)
A_zero = QuantumGateDesign.form_LHS(prob, controls, t, pcof0, dt, order)

hss_A_nonzero = hss(A_nonzero, atol=1e-6, rtol=1e-6)
hss_A_zero = hss(A_zero, atol=1e-6, rtol=1e-6)

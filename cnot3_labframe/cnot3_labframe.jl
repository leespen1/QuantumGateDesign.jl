#=
# Script for testing that setting up cnot3 matches the results of juqbox.
# (which I assume to be correct)
=#

using QuantumGateDesign, DelimitedFiles
using LinearAlgebra: Symmetric

N_osc_levels = 10
Tmax = 550.0
nsteps = 1000

subsystem_sizes = (N_osc_levels, 4, 4)
essential_subsystem_sizes = (1, 2, 2)

fa = 4.10595
fb = 4.81526
fs = 7.8447
xa = 2 * 0.1099
xb = 2 * 0.1126
xs = 0.002494^2/xa
xab = 1.0e-6
xas = sqrt(xa*xs)
xbs = sqrt(xb*xs)

transition_freqs = (fs, fb, fa)
rotation_freqs = transition_freqs # Rotating Frame

kerr_coeffs = Symmetric(
    [xs   xbs   xas;
     xbs  xb    xab;
     xas  xab   xa],
    :U
)


prob_rot = dispersive_qudits_problem(
    subsystem_sizes,
    essential_subsystem_sizes,
    transition_freqs,
    rotation_freqs,
    kerr_coeffs,
    Tmax,
    nsteps,
    rot_frame=true,
    gmres_abstol=1e-15,
    gmres_reltol=1e-15,
)

prob_lab = dispersive_qudits_problem(
    subsystem_sizes,
    essential_subsystem_sizes,
    transition_freqs,
    rotation_freqs,
    kerr_coeffs,
    Tmax,
    nsteps,
    rot_frame=false,
    gmres_abstol=1e-15,
    gmres_reltol=1e-15,
)

rot_op_T = compsys_rot_frame_op(rotation_freqs, subsystem_sizes, Tmax)

lab_target = (prob_lab.u0 + im.*prob_lab.v0)*CNOT_gate()
rot_target = rot_op_T*lab_target

seed=0
atol=1e-15
rtol=1e-15
degree = 14
D1 = 15

cnot3ret = QuantumGateDesign.setup_cnot3(
    seed=seed,
    atol=atol,
    rtol=rtol,
    D1=D1,
    N_osc_levels=N_osc_levels,
    Tmax=Tmax
)

Nctrl = 3
Nfreq = 3
Cfreq = zeros(Nctrl,Nfreq)
Cfreq[1:2,2] .= -2.0*pi*xa # carrier freq's for ctrl Hamiltonian 1 & 2
Cfreq[1:2,3] .= -2.0*pi*xb # carrier freq's for ctrl Hamiltonian 1 & 2
Cfreq[3,2] = -2.0*pi*xas # carrier freq 2 for ctrl Hamiltonian #3
Cfreq[3,3] = -2.0*pi*xbs # carrier freq 2 for ctrl Hamiltonian #3

Cfreq_lab_shift = repeat(collect(rotation_freqs), 1, Nfreq) .* 2.0*pi
Cfreq_lab = Cfreq .+ Cfreq_lab_shift

rot_controls = get_controls(degree, D1, Cfreq, Tmax)
lab_controls = [DoubleRealPartControl(control)
                for control in get_controls(degree, D1, Cfreq_lab, Tmax)]

# Alternative implementation, which should be slower
lab_controls2 = [LabFrameControl(rot_controls[i], 2.0*pi*rotation_freqs[i])
                 for i in 1:3]

pcof_csv = readdlm("targetError=1e-7_cnot3OptimizationTest_order=6_degree=14_seed=5_nsteps=5409_atol=1.0e-15_rtol=1.0e-15_D1=16_time=6.0_maxiter=10000_nthreads=4_costType=GeneralizedInfidelity_gateDuration=550.0_nCavityLevels=10_pcof.csv", ',')
pcof = pcof_csv[end,:]
# Dummy run to compile
prob_rot.nsteps = 1
prob_lab.nsteps = 1
history_rot = eval_forward(prob_rot, rot_controls, pcof, order=6)
history_lab = eval_forward(prob_lab, lab_controls, pcof, order=6)

# Actual run
prob_rot.nsteps = 100
prob_lab.nsteps = 100
#@time history_rot = eval_forward(prob_rot, rot_controls, pcof, order=6)
@time history_lab = eval_forward(prob_lab, lab_controls, pcof, order=6)


include("stepsize_copy.jl")
order = 6
max_walltime = 1 # In hours
filename_base = "lab_frame"
nsaves = 10
initial_nsteps = 50_000 # Nyquist limit should be about here
collect_data(prob_lab, lab_controls, pcof, order, max_walltime, filename_base,
             nsaves, initial_nsteps)

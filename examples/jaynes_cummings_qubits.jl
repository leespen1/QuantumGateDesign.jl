# Set up the SchrodingerProb

# Two qubits with guard levels
subsystem_sizes = (8,8)
essential_subsystem_sizes = (2,2)

transition_freqs = [5.17839, 5.06323] .* 2pi
# Use the average transition frequency as the frequency of rotation
rotation_freq = sum(transition_freqs) / length(transition_freqs)

kerr_coeffs = 2pi .* [0.3411 0
                      0      0.3413]
jayne_cummings_coeffs = 2pi .* [0        1.995e-3
                                1.995e-3 0]
tf = 250.0
nsteps = 20_750
sparse_rep=true


prob = JaynesCummingsProblem(
    subsystem_sizes, essential_subsystem_sizes,
    transition_freqs, rotation_freq, kerr_coeffs, jayne_cummings_coeffs,
    tf, nsteps;
    sparse_rep
)

# Set up the Controls

# Same carrier wave frequencies for control on each qubit
carrier_wave_freqs = [5.7611e-2, -5.7611e-2] .* 2pi
bspline_degree = 2
N_knots = 5

bspline_control = GeneralBSplineControl(bspline_degree, N_knots, tf)
bcarrier_control = CarrierControl(bspline_control, carrier_wave_freqs)
# Use the same control for each qubit (but they will have different control vectors)
controls = [bcarrier_control, bcarrier_control]

N_coeff = get_number_of_control_parameters(controls)
pcof_init = rand(N_coeff)

# Set up target gate
CNOT_target = create_gate(
    subsystem_sizes, essential_subsystem_sizes,
    [(1,0) => (1,1),
     (1,1) => (1,0)]
)

# Transform target into rotating frame
rot_mat = rotation_matrix(subsystem_sizes, rotation_freqs, tf)
CNOT_target_complex = real_to_complex(CNOT_target)
CNOT_target_rotating_frame_complex = prod(rot_mat) * CNOT_target_complex
CNOT_target = complex_to_real(CNOT_target_rotating_frame_complex)


#ret = optimize_gate(prob, controls, pcof_init, CNOT_target, order=4, maxIter=50)

# Setup Juqbox version of the problem
Cfreq = vcat(carrier_wave_freqs, carrier_wave_freqs)
params = QuantumGateDesign.convert_to_juqbox(
    prob,
    essential_subsystem_sizes,
    subsystem_sizes .- essential_subsystem_sizes,
    Cfreq,
    200,
    CNOT_target_rotating_frame_complex
)


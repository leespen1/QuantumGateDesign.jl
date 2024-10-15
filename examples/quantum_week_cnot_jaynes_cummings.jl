using QuantumGateDesign
# Set up the SchrodingerProb

# Two qubits with guard levels
subsystem_sizes = (4,4)
essential_subsystem_sizes = (2,2)

transition_freqs = [5.17839, 5.06323] .* 2pi
# Use the average transition frequency as the frequency of rotation
rotation_freq = sum(transition_freqs) / length(transition_freqs)
rotation_freqs = fill(rotation_freq, length(transition_freqs))

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
    sparse_rep,
    gmres_reltol=1e-12
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
max_amplitude = 0.04*2pi
pcof0 = rand(N_coeff) .* max_amplitude ./ 100

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


# Compare step size vs relative error to get an idea of how many timesteps should be used
run_convergence_test = true
if run_convergence_test
    N_iterations = 9
    base_nsteps = 10

    ret_dict = get_histories(prob, controls, pcof0, N_iterations, base_nsteps=base_nsteps)

    pl1 = QuantumGateDesign.plot_stepsize_convergence(ret_dict)
    pl2 = QuantumGateDesign.plot_timing_convergence(ret_dict)
end

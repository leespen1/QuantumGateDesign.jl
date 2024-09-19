# Set up the SchrodingerProb
subsystem_sizes = (4,4)
essential_subsystem_sizes = (2,2)
transition_freqs = [4.10595, 4.81526] .* 2pi
rotation_freqs = copy(transition_freqs)
# 0.1 for artificially large fast coupling. Actual value is 1e-6
kerr_coeffs = 2pi .* [2*0.1099   0.1
                      0.1        2*0.1126]
tf = 50.0
nsteps = 100
sparse_rep=true

prob = DispersiveProblem(
    subsystem_sizes, essential_subsystem_sizes,
    transition_freqs, rotation_freqs, kerr_coeffs,
    tf, nsteps;
    sparse_rep=sparse_rep
)

# Set up the Controls
carrier_wave_freqs = transition_freqs .- rotation_freqs
bspline_degree = 2
N_knots = 5

D1 = 10
fa = sum(transition_freqs) / length(transition_freqs)
om1 = 2pi .* [transition_freqs[1] - rotation_freqs[1],
              transition_freqs[1] - rotation_freqs[1] - kerr_coeffs[1,2],
             ]
om2 = 2pi .* [transition_freqs[2] - rotation_freqs[2],
              transition_freqs[2] - rotation_freqs[2] - kerr_coeffs[1,2],
             ]

control1 =  BSplineControl(tf, D1, om1)
control2 =  BSplineControl(tf, D1, om2)

controls = [control1, control2]

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

convergence_dict = QuantumGateDesign.get_histories(prob, controls, pcof0, 5, base_nsteps=100, orders=(2,4))

#ret = optimize_gate(prob, controls, pcof_init, CNOT_target, order=4, maxIter=50)

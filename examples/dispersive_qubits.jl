# Set up the SchrodingerProb
subsystem_sizes = (2,2)
essential_subsystem_sizes = (2,2)
transition_freqs = [0.5, 0.75]
rotation_freqs = [0.0, 0]
kerr_coeffs = [0 0
               0 0]
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

ret = optimize_gate(prob, controls, pcof_init, CNOT_target, order=4, maxIter=1)

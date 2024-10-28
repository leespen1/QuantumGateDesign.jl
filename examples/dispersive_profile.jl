using QuantumGateDesign
using Profile
using PProf

subsystem_sizes = (4,4,4)
essential_subsystem_sizes = (2,2,2)
transition_freqs = 2pi .* [4.10595, 4.81526, 7.8447]
rotation_freqs = transition_freqs
xa = 2 * 0.1099
xb = 2 * 0.1126
xs = 0.002494^2/xa
xab = 1e-6
xas = sqrt(xa*xs)
xbs = sqrt(xb*xs)
kerr_coeffs = 2pi .* [xa  xab xas
                      xab xb  xbs
                      xas xbs xs]
tf = 500.0
nsteps = 2_000

prob = DispersiveProblem(
    subsystem_sizes,
    essential_subsystem_sizes,
    transition_freqs,
    rotation_freqs,
    kerr_coeffs,
    tf,
    nsteps,
    sparse_rep=false
)

D1 = 10
controls = [MySplineControl(tf, D1) for i in 1:prob.N_operators]
pcof = rand(get_number_of_control_parameters(controls))

# Run once to get compilation out of the way
history = eval_forward(prob, controls, pcof, order=4)

Profile.clear()
@profile eval_forward(prob, controls, pcof, order=4)

pprof()

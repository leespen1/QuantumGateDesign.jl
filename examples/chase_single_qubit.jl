using Distributions

nqubits = 1

center = 4.5
std_dev = 0.1
gauss_distribution = Normal(center, std_dev)


H_control_real = [0.0 1;
1 0]
H_control_imag = [0.0 1;
-1 0]
real_control_ops = [H_control_real]
imag_control_ops = [H_control_imag]
U0 = [1 0; 0 1]
t0 = 0.0
T = 50.0 # time in nanoseconds
nsteps = 200
sym_ops = [H_control_real]
asym_ops = [H_control_imag]


omegas = rand(gauss_distribution, 2)
deltas = omegas .- center

probs = [SchrodingerProb(Float64[0 0; 0 delta], real_control_ops, imag_control_ops, U0, T, nsteps)
         for delta in deltas]

# N_GRAPE_amplitudes = 2
# control = GRAPEControl(N_GRAPE_amplitudes, T)
max_control_parameter = 0.1
pcof_l = -max_control_parameter
pcof_u = max_control_parameter

degree = 2
D1 = 4
control = FortranBSplineControl(degree, D1, T)
pcof0 = (0.5 .- rand(control.N_coeff)) .* max_control_parameter

x_gate = [0 1; 1 0]
y_gate = [0 -im; im 0]
z_gate = [1 0; 0 -1]

x_gate_prob = abs2.(x_gate)
y_gate_prob = abs2.(y_gate)
z_gate_prob = abs2.(z_gate)


opt_gate = x_gate
opt_gate_prob = abs2.(x_gate)

## Single problem
#println("Optimizing single problem")
#opt_ret_single = optimize_prob(probs[1], control, pcof0, opt_gate, pcof_lbound=pcof_l, pcof_ubound=pcof_u, cost_type=:Infidelity, ipopt_options=["max_iter" => 100])

# Multiple Problems
println("Optimizing 'risk-netural' multiple problem")
opt_ret_multiple = optimize_prob(probs, control, pcof0, opt_gate, pcof_lbound=pcof_l, pcof_ubound=pcof_u, cost_type=:Infidelity, ipopt_options=["max_iter" => 100])




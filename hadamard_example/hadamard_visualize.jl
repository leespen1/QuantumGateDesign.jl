using QuantumGateDesign, GLMakie

σx = [0.0 1;1 0]
Z = [0.0 0;0 0]
σz = [1.0 0;0 -1] .* (0.1/2)
U0 = [1 0;0 1]
tf = 50.0
nsteps = 1000
N_ess_levels = 2

degree = 4
D1 = 8

prob = SchrodingerProb(σz, [σx], [Z], U0, tf, nsteps, N_ess_levels)
control = FortranBSplineControl(degree, D1, tf)
#control = CarrierControl(control, [maximum(eigvals(σz))])
hadamard_gate = [1 1;1 -1] ./ sqrt(2)

pcof_l = -0.1
pcof_u = 0.1
pcof_l = ones(control.N_coeff)*pcof_l
pcof_u = ones(control.N_coeff)*pcof_u
pcof_l[begin] = pcof_l[end] = pcof_u[begin] = pcof_u[end] = 0
pcof_l[div(end,2)+1] = pcof_l[div(end,2)] = pcof_u[div(end,2)+1] = pcof_u[div(end,2)] = 0

#pcof0 = zeros(control.N_coeff)
pcof0 = (0.5 .- rand(control.N_coeff)) .* pcof_u

ret = optimize_gate(prob, control, pcof0, hadamard_gate, pcof_lbound=pcof_l, pcof_ubound=pcof_u, cost_type=:GeneralizedInfidelity, ipopt_options=["max_iter" => 100])
pcof_f = ret.x

## Plotting
#visualize_control(control,prob=prob, pcof_init=pcof0, target=hadamard_gate)
visualize_control(control,prob=prob, pcof_init=pcof_f, target=hadamard_gate)

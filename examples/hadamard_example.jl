#=
# Script for testing that setting up cnot3 matches the results of juqbox.
# (which I assume to be correct)
=#

using QuantumGateDesign, Random

σx = [0.0 1;1 0]
Z = [0.0 0;0 0]
σz = [1.0 0;0 -1] .* (0.1/2)
U0 = [1 0;0 1]
tf = 50.0
nsteps = 1000

prob = SchrodingerProb(σz, [σx], [Z], U0, tf, nsteps)

degree = 4
D1 = 8
control = FortranBSplineControl(degree, D1, tf)

#control = CarrierControl(control, [maximum(eigvals(σz))])
hadamard_gate = [1 1;1 -1] ./ sqrt(2)

max_control_parameter = 0.1
pcof_l = -max_control_parameter
pcof_u = max_control_parameter
pcof0 = (0.5 .- rand(control.N_coeff)) .* max_control_parameter 

opt_ret = optimize_prob(prob, control, pcof0, hadamard_gate, pcof_lbound=pcof_l, pcof_ubound=pcof_u, cost_type=:Infidelity, ipopt_options=["max_iter" => 100])
pcof_optimal = opt_ret.x

history_optimal = eval_forward(prob, control, pcof_optimal)
pop_plots = plot_populations(history_optimal, tf, labels=["|00>", "|01>", "|10>", "|11>"])
control_plot = plot_controls(control, pcof_optimal)

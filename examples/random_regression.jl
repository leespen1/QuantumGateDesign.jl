using QuantumGateDesign
using Random
using Dates
using JLD2
using ProfileView

#complex_system_size = 4
complex_system_size = 64
N_operators = 3
prob = QuantumGateDesign.construct_rand_prob(complex_system_size, N_operators)

#degree = 16
degree = 2
N_knots = 20
tf = prob.tf
N_frequencies = 3
base_controls = [GeneralBSplineControl(degree, N_knots, tf) for i in 1:N_operators]
#controls = [CarrierControl(base_controls[i], rand(MersenneTwister(i), N_frequencies)) for i in 1:N_operators]
controls = [CarrierControl(MySplineControl(tf, N_knots), rand(MersenneTwister(i), N_frequencies)) for i in 1:N_operators]

pcof = rand(MersenneTwister(0), get_number_of_control_parameters(controls))
prob.nsteps = 1_000
#prob.nsteps = 100

dummy_terminal_condition = vcat(prob.u0, prob.v0)
dummy_target = prob.u0 + im*prob.v0

order=12
#order=4

#=
history = eval_forward(prob, controls, pcof, order=order)
println("Finished Forward Eval")
lambda_history = QuantumGateDesign.eval_adjoint(prob, controls, pcof, dummy_terminal_condition, order=order)
println("Finished Adjoint Eval")
#grad = discrete_adjoint(prob, controls, pcof, dummy_target, order=order)
println("Finished Gradient")
@time history = eval_forward(prob, controls, pcof, order=order)
@time lambda_history = QuantumGateDesign.eval_adjoint(prob, controls, pcof, dummy_terminal_condition, order=order)
#@time grad = discrete_adjoint(prob, controls, pcof, dummy_target, order=order)

dict = load("original2.jld2")
println("History agrees: ", history == dict["history"])
println("Lambda agrees: ", lambda_history == dict["lambda_history"])
println("Gradient agrees: ", grad == dict["grad"])
=#

@profview history = eval_forward(prob, controls, pcof, order=order)

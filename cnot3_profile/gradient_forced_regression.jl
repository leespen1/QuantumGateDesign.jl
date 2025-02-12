using QuantumGateDesign, Random
using QuantumGateDesign: setup_cnot3, get_controls
using DelimitedFiles, Dates

clean_NaN(A) = replace(x -> isnan(x) ? 0 : x, A)
abs_errors(A,B) = abs.(A .- B)
rel_errors(A,B) = abs_errors(A,B) ./ abs.(B)
max_abs_errors(A,B) = maximum(abs_errors(A,B))
max_rel_errors(A,B) = maximum(clean_NaN(rel_errors(A,B)))

atol = 1e-15
rtol = 1e-15
seed = 0
D1 = 15
degree = 14
order = 2
target_error = 1e-1

target_error_int = round(Int64, log10(target_error))
target_error_index = abs(target_error_int)
order_index = div(order, 2)


cnot3ret = setup_cnot3(seed=0, atol=NaN, rtol=NaN, D1=D1)
controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

fortran_bspline = QuantumGateDesign.FortranBSpline(degree, D1)
new_bspline_control = QuantumGateDesign.FortranBSplineControl2(fortran_bspline, controls[1].tf) 
new_carrier_controls = [CarrierControl(new_bspline_control, freqs) for freqs in eachrow(cnot3ret.juqbox_params.Cfreq)]

cnot3ret = QuantumGateDesign.setup_cnot3(seed=seed, atol=atol, rtol=rtol, D1=D1)
controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)
N_coeff = QuantumGateDesign.get_number_of_control_parameters(controls)
pcof = cnot3ret.amax * 2* (0.5 .- rand(MersenneTwister(seed), N_coeff))


cnot3ret.qgd_prob.nsteps = 3
grad_da = discrete_adjoint(cnot3ret.qgd_prob, controls, pcof, cnot3ret.target, order=order)
grad_forced = eval_grad_forced(cnot3ret.qgd_prob, controls, pcof, cnot3ret.target, order=order)
grad_finite = eval_grad_finite_difference(cnot3ret.qgd_prob, controls, pcof, cnot3ret.target, order=order)

writedlm("OG_3step_grad_da_order=$(order)_targetError=$(target_error_int).dlm", grad_da)
writedlm("OG_3step_grad_forced_order=$(order)_targetError=$(target_error_int).dlm", grad_forced)
writedlm("OG_3step_grad_finite_order=$(order)_targetError=$(target_error_int).dlm", grad_finite)

println("\n")
println("Gradient Regression Testing:")
println("This Code:")
println("\tMaximum Absolute Errors:")
println("\t\tDiscrete Adjoint vs Forced: ",            max_abs_errors(grad_da,     grad_forced))
println("\t\tDiscrete Adjoint vs Finite Difference: ", max_abs_errors(grad_da,     grad_finite))
println("\t\tForced vs Finite Difference: ",           max_abs_errors(grad_forced, grad_finite))
println("\tMaximum Relative Errors:")
println("\t\tDiscrete Adjoint vs Forced: ",            max_rel_errors(grad_da,     grad_forced))
println("\t\tDiscrete Adjoint vs Finite Difference: ", max_rel_errors(grad_da,     grad_finite))
println("\t\tForced vs Finite Difference: ",           max_rel_errors(grad_forced, grad_finite))

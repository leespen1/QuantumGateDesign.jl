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
NSTEPS_MATRIX= [
  7433 825 379 183 175 92
  25350 1944 705 388 297 191
  80064 3535 1153 610 388 287
  246583 6302 1702 822 507 352
  745377 11214 2508 1105 643 433
  2314385 19946 3686 1478 814 530
  7186131 35471 5414 1975 1030 645
]
target_nsteps = NSTEPS_MATRIX[target_error_index, order_index]



cnot3ret = setup_cnot3(seed=0, atol=NaN, rtol=NaN, D1=D1)
controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

fortran_bspline = QuantumGateDesign.FortranBSpline(degree, D1)
new_bspline_control = QuantumGateDesign.FortranBSplineControl2(fortran_bspline, controls[1].tf) 
new_carrier_controls = [CarrierControl(new_bspline_control, freqs) for freqs in eachrow(cnot3ret.juqbox_params.Cfreq)]

cnot3ret = QuantumGateDesign.setup_cnot3(seed=seed, atol=atol, rtol=rtol, D1=D1)
controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)
N_coeff = QuantumGateDesign.get_number_of_control_parameters(controls)
pcof = cnot3ret.amax * 2* (0.5 .- rand(MersenneTwister(seed), N_coeff))


println("\n\n")
println("Doing dummy run")
cnot3ret.qgd_prob.nsteps = 1
dummy_grad = discrete_adjoint(cnot3ret.qgd_prob, controls, pcof, cnot3ret.target, order=order)

println("Doing real run")
cnot3ret.qgd_prob.nsteps = target_nsteps
grad = discrete_adjoint(cnot3ret.qgd_prob, new_carrier_controls, pcof, cnot3ret.target, order=order)

now_str = now()
#writedlm("OG_accumulate_grad_order=$(order)_targetError=$(target_error_int).dlm", grad)
#writedlm("accumulate_grad_order=$(order)_targetError=$(target_error_int).dlm", grad)
grad_reg = reshape(readdlm("OG_accumulate_grad_order=$(order)_targetError=$(target_error_int).dlm"), :)

println("\n")
println("Gradient Regression Testing:")
println("Maximum Absolute Errors:")
println("\tNew vs Reg: ", max_abs_errors(grad, grad_reg))
println("Maximum Relative Errors:")
println("\tNew vs Reg: ", max_rel_errors(grad, grad_reg))

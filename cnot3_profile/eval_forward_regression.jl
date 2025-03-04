using QuantumGateDesign, Random
using QuantumGateDesign: setup_cnot3, get_controls
using DelimitedFiles, Dates
using LinearAlgebra: norm

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
order = 6
#target_error = 1e-1
target_error = 1e-3

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



cnot3ret = QuantumGateDesign.setup_cnot3(seed=seed, atol=atol, rtol=rtol, D1=D1)
controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

fortran_bspline = QuantumGateDesign.FortranBSpline(degree, D1)
new_bspline_control = QuantumGateDesign.FortranBSplineControl2(fortran_bspline, controls[1].tf) 
new_carrier_controls = [CarrierControl(new_bspline_control, freqs) for freqs in eachrow(cnot3ret.juqbox_params.Cfreq)]

controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)
N_coeff = QuantumGateDesign.get_number_of_control_parameters(controls)
pcof = cnot3ret.amax * 2* (0.5 .- rand(MersenneTwister(seed), N_coeff))

#controls = new_carrier_controls

println("\n\n")
println("Doing dummy run")
cnot3ret.qgd_prob.nsteps = 1
dummy_history = eval_forward(cnot3ret.qgd_prob, controls, pcof, order=order)

println("Doing real run")
cnot3ret.qgd_prob.nsteps = target_nsteps
history = eval_forward(cnot3ret.qgd_prob, controls, pcof, order=order)
history_dlm = reshape(history, :, size(history, 1))

@show any(isnan.(history))
@show size(history_dlm)

now_str = now()
#writedlm("OG_history_order=$(order)_targetError=$(target_error_int).dlm", history_dlm)
#writedlm("history_order=$(order)_targetError=$(target_error_int).dlm", history_dlm)
history_reg = readdlm("OG_history_order=$(order)_targetError=$(target_error_int).dlm", '\t', ComplexF64)

@show size(history_reg)

println("\n")
println("History Regression Testing:")
println("Maximum Absolute Errors:")
println("\tNew vs Reg: ", max_abs_errors(history_dlm, history_reg))
println("Maximum Relative Errors:")
println("\tNew vs Reg: ", max_rel_errors(history_dlm, history_reg))
println("Overall Absolute Error: ", norm(history_dlm - history_reg))
println("Overall Relative Error: ", norm(history_dlm - history_reg)/norm(history_dlm))

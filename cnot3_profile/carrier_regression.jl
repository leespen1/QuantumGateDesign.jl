using QuantumGateDesign, DelimitedFiles, Dates
using QuantumGateDesign: setup_cnot3, get_controls, fill_p_mat!, fill_q_mat!
using Random: MersenneTwister
using Profile, PProf, BenchmarkTools

clean_NaN(A) = replace(x -> isnan(x) ? 0 : x, A)
abs_errors(A,B) = abs.(A .- B)
rel_errors(A,B) = abs_errors(A,B) ./ abs.(B)
max_abs_errors(A,B) = maximum(abs_errors(A,B))
max_rel_errors(A,B) = maximum(clean_NaN(rel_errors(A,B)))

write_dlm = false

degree = 14
D1 = 15
order = 2

Nderiv = 13+1

cnot3ret = setup_cnot3(seed=0, atol=NaN, rtol=NaN, D1=D1)
controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

fortran_bspline = QuantumGateDesign.FortranBSpline(degree, D1)
new_bspline_control = QuantumGateDesign.FortranBSplineControl2(fortran_bspline, controls[1].tf) 
new_carrier_controls = [CarrierControl(new_bspline_control, freqs) for freqs in eachrow(cnot3ret.juqbox_params.Cfreq)]

N_coeff = QuantumGateDesign.get_number_of_control_parameters(controls)
N_controls = length(controls)
N_points = 1_001
t_range = LinRange(0, cnot3ret.tf, N_points)
pcof = cnot3ret.amax * 2* (0.5 .- rand(MersenneTwister(0), N_coeff))


### Control Value test


println("\n\n")

control_vals_mat = fill(NaN, Nderiv, length(controls))
vals_array_old = fill(NaN, Nderiv, N_controls, 2, N_points)
vals_array_new = fill(NaN, Nderiv, N_controls, 2, N_points)

println("Evaluating controls (old style) ...")
@time for (i,t) in enumerate(t_range)
    fill_p_mat!(control_vals_mat, controls, t, pcof)
    vals_array_old[:,:,1,i] .= control_vals_mat
    fill_q_mat!(control_vals_mat, controls, t, pcof)
    vals_array_old[:,:,2,i] .= control_vals_mat
end

println("Evaluating controls (new style) ...")
@time for (i,t) in enumerate(t_range)
    fill_p_mat!(control_vals_mat, new_carrier_controls, t, pcof)
    vals_array_new[:,:,1,i] .= control_vals_mat
    fill_q_mat!(control_vals_mat, new_carrier_controls, t, pcof)
    vals_array_new[:,:,2,i] .= control_vals_mat
end

vals_dlm_mat_old = reshape(vals_array_old, Nderiv, :)
vals_dlm_mat_new = reshape(vals_array_new, Nderiv, :)
vals_dlm_mat_reg = readdlm("OG_vals.dlm", '\t', Float64)
if write_dlm
    now_str= now()
    writedlm("old_vals_$(now_str).dlm", vals_dlm_mat_old)
    writedlm("new_vals_$(now_str).dlm", vals_dlm_mat_new)
end

println("\n")
println("Control Values Testing")
println("Maximum Absolute Errors:")
println("\tOld vs New: ", max_abs_errors(vals_array_old, vals_array_new))
println("\tOld vs Reg: ", max_abs_errors(vals_dlm_mat_old, vals_dlm_mat_reg))
println("\tNew vs Reg: ", max_abs_errors(vals_dlm_mat_new, vals_dlm_mat_reg))
println("Maximum Relative Errors:")
println("\tOld vs New: ", max_rel_errors(vals_array_old, vals_array_new))
println("\tOld vs Reg: ", max_rel_errors(vals_dlm_mat_old, vals_dlm_mat_reg))
println("\tNew vs Reg: ", max_rel_errors(vals_dlm_mat_new, vals_dlm_mat_reg))


### Gradient Regression


println("\n\n")

single_N_coeff = controls[1].N_coeff
grad_array_old = fill(NaN, single_N_coeff, Nderiv, 2, N_points)
grad_array_new = fill(NaN, single_N_coeff, Nderiv, 2, N_points)
grad_vec = fill(NaN, single_N_coeff)
empty_grad_mat = fill(NaN, single_N_coeff, Nderiv)

println("Evaluating gradients (old style) ...")
@time for (i, t) in enumerate(t_range)
    # Put in two more matrices
    for order in 0:13
        eval_grad_p_derivative!(grad_vec, controls[1], t, pcof, order)
        grad_array_old[:,1+order,1,i] .= grad_vec
        eval_grad_q_derivative!(grad_vec, controls[1], t, pcof, order)
        grad_array_old[:,1+order,2,i] .= grad_vec
    end
end

println("Evaluating gradients (new style) ...")
@time for (i, t) in enumerate(t_range)
    # Put in two more matrices
    for order in 0:13
        eval_grad_p_derivative!(grad_vec, new_carrier_controls[1], t, pcof, order)
        grad_array_new[:,1+order,1,i] .= grad_vec
        eval_grad_q_derivative!(grad_vec, new_carrier_controls[1], t, pcof, order)
        grad_array_new[:,1+order,2,i] .= grad_vec
    end
end

grad_dlm_mat_old = reshape(grad_array_old, single_N_coeff, :)
grad_dlm_mat_new = reshape(grad_array_new, single_N_coeff, :)
grad_dlm_mat_reg = readdlm("OG_gradients.dlm", '\t', Float64)
if write_dlm
    now_str= now()
    writedlm("old_gradients_$(now_str).dlm", grad_dlm_mat_old)
    writedlm("new_gradients_$(now_str).dlm", grad_dlm_mat_new)
end


println("\n")
println("Gradient Testing")
println("Maximum Absolute Errors:")
println("\tOld vs New: ", max_abs_errors(grad_array_old, grad_array_new))
println("\tOld vs Reg: ", max_abs_errors(grad_dlm_mat_old, grad_dlm_mat_reg))
println("\tNew vs Reg: ", max_abs_errors(grad_dlm_mat_new, grad_dlm_mat_reg))
println("Maximum Relative Errors:")
println("\tOld vs New: ", max_rel_errors(grad_array_old, grad_array_new))
println("\tOld vs Reg: ", max_rel_errors(grad_dlm_mat_old, grad_dlm_mat_reg))
println("\tNew vs Reg: ", max_rel_errors(grad_dlm_mat_new, grad_dlm_mat_reg))

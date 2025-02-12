using QuantumGateDesign, Random
using QuantumGateDesign: setup_cnot3, get_controls

D1 = 15
degree = 14

cnot3ret = setup_cnot3(seed=0, atol=NaN, rtol=NaN, D1=D1)
controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

fortran_bspline = QuantumGateDesign.FortranBSpline(degree, D1)
new_bspline_control = QuantumGateDesign.FortranBSplineControl2(fortran_bspline, controls[1].tf) 
new_carrier_controls = [CarrierControl(new_bspline_control, freqs) for freqs in eachrow(cnot3ret.juqbox_params.Cfreq)]

N_coeff1 = QuantumGateDesign.get_number_of_control_parameters(controls)
N_coeff2 = QuantumGateDesign.get_number_of_control_parameters(new_carrier_controls)

@assert N_coeff1 == N_coeff2
println("N_coeff = ", N_coeff1)
println("N_coef per bspline = ", new_carrier_controls[1].base_control.N_coeff)

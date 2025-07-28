using QuantumGateDesign
using Random
degree = 4
N_basis_functions = 6
tf = 1.0
t = 0.5
fortran_control = FortranBSplineControl(degree, N_basis_functions, tf)
pcof = rand(MersenneTwister(0), fortran_control.N_coeff)
eval_p_derivative(fortran_control, t, pcof, 0)

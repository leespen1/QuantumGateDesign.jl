using QuantumGateDesign
using Random
#===============================================================================
#
# Visually compare the Bspline implemented by Juqbox (my adaptation) with the
# Bspline implemented by pppack.
#
===============================================================================#

tf = 1.0
D1 = 7 # Number of basis functions in each spline
juqbox_control = MySplineControl(tf, D1)

degree = 2
order = degree + 1
#N_basis_functions = N_knots + order - 2 #(N_nonrepeating_knots + (order-1) + (order-1) - order)
#N_knots = D1 + 2 - order
N_knots = D1-1
fortran_control = FortranBSplineControl(degree, N_knots, tf)
package_bspline = GeneralBSplineControl(degree, N_knots, tf)

@assert juqbox_control.N_coeff == fortran_control.N_coeff == package_bspline.N_coeff
#@assert fortran_control.N_coeff == package_bspline.N_coeff
N_coeff = fortran_control.N_coeff
pcof = rand(MersenneTwister(0), N_coeff)
#pcof = ones(N_coeff)
#pcof[div(N_coeff,2)] = pcof[1] = 0
pcof = repeat(pcof, 3)

controls = [juqbox_control, fortran_control, package_bspline]
QuantumGateDesign.plot_controls(controls, pcof, derivative_orders=0:3, control_indices=1:3)

# Results seem the same, except that the juqbox version differs a little at the
# endpoints (and the number of knots there might be different)

# Also, I need to fix chain rule in derivatives of bspline controls (since I 
# scale the interval to [0,1]) (I think I may have put it in wrong.
# Keep in mind I need to do it in eval_derivative, fill_vec, and eval_grad)

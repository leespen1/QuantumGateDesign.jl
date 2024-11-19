using QuantumGateDesign
using Random
#===============================================================================
#
# Visually compare the Bspline implemented by Juqbox (my adaptation) with the
# Bspline implemented by pppack.
#
===============================================================================#

tf = 100.0
D1 = 10 # Number of basis functions in each spline
juqbox_control = MySplineControl(tf, D1,)

degree = 2
order = degree + 1
#N_basis_functions = N_knots + order - 2 #(N_nonrepeating_knots + (order-1) + (order-1) - order)
N_knots = D1 + 2 - order
fortran_control = FortranBSplineControl(degree, tf, N_knots)

@assert juqbox_control.N_coeff == fortran_control.N_coeff
N_coeff = juqbox_control.N_coeff
pcof = rand(MersenneTwister(0), N_coeff)
pcof = repeat(pcof, 2)

controls = [juqbox_control, fortran_control]
QuantumGateDesign.plot_controls(controls, pcof, derivative_orders=0:3)

# Results seem the same, except that the juqbox version differs a little at the
# endpoints (and the number of knots there might be different)

# Also, I need to fix chain rule in derivatives of bspline controls (since I 
# scale the interval to [0,1]) (I think I may have put it in wrong.
# Keep in mind I need to do it in eval_derivative, fill_vec, and eval_grad)

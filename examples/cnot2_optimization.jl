using QuantumGateDesign
using Random
using Profile
using PProf

#==============================================================================
# Set up Problem
==============================================================================#

subsystem_sizes = (2,2)
essential_subsystem_sizes = (2,2)
transition_freqs = 2pi .* [4.10595, 4.81526]
rotation_freqs = transition_freqs
xa = 2 * 0.1099
xb = 2 * 0.1126
xab = 1e-2 # 1e-6 official value
kerr_coeffs = 2pi .* [xa  xab
                      xab xb ]
                      
tf = 100.0
nsteps = 100

prob = DispersiveProblem(
    subsystem_sizes,
    essential_subsystem_sizes,
    transition_freqs,
    rotation_freqs,
    kerr_coeffs,
    tf,
    nsteps,
    sparse_rep=true
)

target = zeros(prob.N_tot_levels, prob.N_initial_conditions)
for i in 1:prob.N_tot_levels
    target[i,i] = 1
end

#==============================================================================
# Set up Controls
==============================================================================#
degree = 2
N_knots = 10
#controls_package = [GeneralBSplineControl(degree, N_knots,tf) for i in 1:prob.N_operators]
controls_fortran = [FortranBSplineControl(degree, N_knots, tf) for i in 1:prob.N_operators]
pcof = 1e-2*(0.5 .- rand(MersenneTwister(1),
                        get_number_of_control_parameters(controls_fortran)))


#==============================================================================
# Run relevant functions once to get compilation out of the way
==============================================================================#
history = eval_forward(prob, controls_fortran, pcof, order=2)
dummy_terminal_condition = vcat(prob.u0, prob.v0)
dummy_target = prob.u0 + im*prob.v0
lambda_history = QuantumGateDesign.eval_adjoint(prob, controls_fortran, pcof, dummy_terminal_condition, order=2)
grad = discrete_adjoint(prob, controls_fortran, pcof, target, order=2)
ret = optimize_gate(prob, controls_fortran, pcof, target, order=2,
                    maxIter=1, print_level=0);

#==============================================================================
# Perform Optimization
==============================================================================#
ret = optimize_gate(prob, controls_fortran, pcof, target, order=4,
                    maxIter=70, pcof_L=-50e-2, pcof_U=50e-2, filename="test.jld2");

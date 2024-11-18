using QuantumGateDesign
using Random
using Profile
using PProf

subsystem_sizes = (4,4,4)
essential_subsystem_sizes = (2,2,2)
transition_freqs = 2pi .* [4.10595, 4.81526, 7.8447]
rotation_freqs = transition_freqs
xa = 2 * 0.1099
xb = 2 * 0.1126
xs = 0.002494^2/xa
xab = 1e-6
xas = sqrt(xa*xs)
xbs = sqrt(xb*xs)
kerr_coeffs = 2pi .* [xa  xab xas
                      xab xb  xbs
                      xas xbs xs]
tf = 100.0
nsteps = 50

prob = DispersiveProblem(
    subsystem_sizes,
    essential_subsystem_sizes,
    transition_freqs,
    rotation_freqs,
    kerr_coeffs,
    tf,
    nsteps,
    sparse_rep=false
)

degree = 2
N_knots = 10
#controls = [GeneralBSplineControl(degree, N_knots,tf) for i in 1:prob.N_operators]
controls = [FortranBSplineControl(degree, tf, N_knots) for i in 1:prob.N_operators]

D1 = 10
#controls = [MySplineControl(tf, D1) for i in 1:prob.N_operators]

pcof = rand(MersenneTwister(0), get_number_of_control_parameters(controls))

# Run once to get compilation out of the way
history = eval_forward(prob, controls, pcof, order=8)
#display(history[:,end,end])

dummy_terminal_condition = vcat(prob.u0, prob.v0)
dummy_target = prob.u0 + im*prob.v0

lambda_history = QuantumGateDesign.eval_adjoint(prob, controls, pcof, dummy_terminal_condition, order=8)
#display(lambda_history[:,1,2,end])

@time history = eval_forward(prob, controls, pcof, order=8);
@time lambda_history = QuantumGateDesign.eval_adjoint(prob, controls, pcof, dummy_terminal_condition, order=8);
@time grad = discrete_adjoint(prob, controls, pcof, dummy_target, order=8);

#=
# Collect an allocation profile
Profile.Allocs.clear()
Profile.Allocs.@profile QuantumGateDesign.eval_adjoint(prob, controls, pcof, dummy_terminal_condition, order=8)
# Export pprof allocation profile and open interactive profiling web interface.
PProf.Allocs.pprof()
=#

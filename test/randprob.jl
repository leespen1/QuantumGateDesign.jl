using Test
using Random
using LinearAlgebra
using QuantumGateDesign

function random_sym_matrix(N)
    rand_mat = rand(N, N)
    return rand_mat + rand_mat'
end

function random_asym_matrix(N)
    rand_mat = rand(N, N)
    return rand_mat - rand_mat'
end

complex_system_size = 5
N_operators = 2


Random.seed!(42)

system_sym = random_sym_matrix(complex_system_size)
system_asym = random_asym_matrix(complex_system_size)
sym_operators = [random_sym_matrix(complex_system_size) for i in 1:N_operators]
asym_operators = [random_asym_matrix(complex_system_size) for i in 1:N_operators]
# In Quantum Computing, state vectors should be unit vectors, but that is not a
# requirement of our algorithm.
u0 = rand(complex_system_size, complex_system_size)
v0 = rand(complex_system_size, complex_system_size)

tf = 100.0
nsteps = 1000 # Should change this to resolve shortest wavelength or something.

N_ess_levels = complex_system_size
N_guard_levels = 0

prob = SchrodingerProb(
    system_sym, system_asym, sym_operators, asym_operators, u0, v0,
    tf, nsteps, N_ess_levels, N_guard_levels
)

prob.guard_subspace_projector .= 0

# Use two controls, make sure aliasing pcof works correctly
control = QuantumGateDesign.SinCosControl(prob.tf)
controls = [control, control]

N_control_params = QuantumGateDesign.get_number_of_control_parameters(controls)
pcof = rand(N_control_params)

# Should check that algorithm does not depend
target = rand(2*complex_system_size, N_ess_levels)

#TODO add subspace projector
#TODO change N_ess_levels to N_initial_cond, add essential_subspace_size
#TODO remove N_guard_levels

orders = (2, 4, 6, 8)
for order in orders
    println("Order = ", order)
    grad_fd = eval_grad_finite_difference(prob, controls, pcof, target, order=order);
    grad_forced = eval_grad_forced(prob, controls, pcof, target, order=order);
    grad_da = discrete_adjoint(prob, controls, pcof, target, order=order);

    #TODO change these to tests
    println("FD Rel Err: ", norm(grad_fd - grad_forced)/ norm(grad_forced))
    println("DA Rel Err: ", norm(grad_da - grad_forced)/ norm(grad_forced))
end





using QuantumGateDesign, LinearAlgebra
using QuantumGateDesign: FortranBSpline, FortranBSplineControl2
using Tables, CSV, Plots

"""
Raising / Creation Operator
"""
function create(N)
    A = zeros(N, N)
    for i in 1:N-1
        A[i+1,i] = sqrt(i)
    end
    return A
end

"""
Lowering / Destruction Operator
"""
function destroy(N)
    A = zeros(N, N)
    for i in 1:N-1
        A[i,i+1] = sqrt(i)
    end
    return A
end

"""
Identity operator
"""
function Id(N)
    A = Diagonal(ones(N))
    return A
end

function number(N)
    A = Diagonal(0:N-1)
    return A
end

function basis(level, system_size)
    ψ = zeros(system_size)
    ψ[1+level] = 1 # Energy level assumed to be given starting from 0
    return ψ
end

function state(levels, system_sizes)
    individual_states = [basis(level, system_size) for (level, system_size) in zip(levels, system_sizes)]
    return reduce(kron, individual_states)
end



## System Parameters - Frequencies will be multiplied by 2pi to get angular frequencies in Hamiltonian!
# Subsystem sizes
n1 = 2
n2 = 2
# n3 = 4
N_ess_levels = 4 
# Gate duration
tf = 550.0
nsteps = 1_000
# Transition frequencies
ω1 = 4.10595
ω2 = 4.81526
#ω3 = 7.8447
# Frequencies to use for rotating frame transformation
ω1_r = ω1
ω2_r = ω2
#ω3_r = ω3
# Self-Kerr Coefficients
ξ1 = 2*0.1099
ξ2 =2*0.1126
#ξ3 = 0.002494^2/ξ1
# Cross-Kerr Coefficients
#ξ12 = 2e-6
ξ12 = 1e-2
#ξ13 = sqrt(ξ1*ξ3)
#ξ23 = sqrt(ξ2*ξ3)

# Operators for each subsystem
A1 = destroy(n1)
A2 = destroy(n2)
#A3 = destroy(n3)

N1 = number(n1)
N2 = number(n2)
#N3 = number(n3)

I1 = Id(n1)
I2 = Id(n2)
#I3 = Id(n3)

# Operators in full subsystem
a1 = kron(I2, A1)
a2 = kron(A2, I1)
#a3 = kron(A3, I2, I1)

# Hamiltonians for each subsystem - Lab Frame
H1_lab = ω1*a1'*a1 - 0.5*ξ1*a1'*a1'*a1*a1
H2_lab = ω2*a2'*a2 - 0.5*ξ2*a2'*a2'*a2*a2
#H3_lab = ω3*a3'*a3 - 0.5*ξ3*a3'*a3'*a3*a3

H1_rft = (ω1-ω1_r)*a1'*a1 - 0.5*ξ1*a1'*a1'*a1*a1
H2_rft = (ω2-ω2_r)*a2'*a2 - 0.5*ξ2*a2'*a2'*a2*a2
#H3_rft = (ω3-ω3_r)*a3'*a3 - 0.5*ξ3*a3'*a3'*a3*a3

# Interaction Hamiltonians
H12_lab = -ξ12*a1'*a1*a2'*a2
#H13_lab = -ξ13*a1'*a1*a3'*a3
#H23_lab = -ξ23*a2'*a2*a3'*a3

# H_sys = H1_rft + H2_rft + H3_rft + H12_lab + H13_lab + H23_lab
H_sys = H1_rft + H2_rft + H12_lab
H_sys .*= 2pi # To put frequencies in angular units

# Construct rotating frame transformation operator
# Although we can contruct the rotating frame Hamiltonians directly, we still want
# R(T) to get the target gate in the rotating frame
R1_tf = exp(im*tf*2pi*ω1_r*N1)
R2_tf = exp(im*tf*2pi*ω2_r*N2)
#R3_tf = exp(im*tf*2pi*ω3_r*N3)
# R_tf = kron(R3_tf, R2_tf, R1_tf)
R_tf = kron(R2_tf, R1_tf)

# Note: it is convenient to make the oscillator the first in the list, since
# the indices there vary most slowly.
state_00 = state((0,0), (n2,n1))
state_01 = state((0,1), (n2,n1))
state_10 = state((1,0), (n2,n1))
state_11 = state((1,1), (n2,n1))

U0 = hcat(state_00, state_01, state_10, state_11)

# |00⟩ -> |00⟩, |01⟩ -> |01⟩, |10⟩ -> |11⟩, |11⟩ -> |10⟩
target_unitary = hcat(state_00, state_01, state_11, state_10)
target_unitary_rtf = R_tf*target_unitary

Hsym_ops = [a1 + a1', a2 + a2']
Hasym_ops = [a1 - a1', a2 - a2']
# Hsym_ops = [a1 + a1', a2 + a2', a3 + a3']
# Hasym_ops = [a1 - a1', a2 - a2', a3 - a3']

prob = SchrodingerProb(H_sys, Hsym_ops, Hasym_ops, U0, tf, nsteps, N_ess_levels)

degree = 8
N_basis_functions = 14 # Number of bspline wavelets, I think it needs to be greater than the degree
bspline = FortranBSpline(degree, N_basis_functions)
base_control = FortranBSplineControl2(bspline, tf)
carrier_frequencies = [
    [ξ1,0.99*ξ1,1.01*ξ1],
    [ξ2,0.99*ξ2,1.01*ξ2],
]
controls = [CarrierControl(base_control, freqs) for freqs in carrier_frequencies]

N_coeff = get_number_of_control_parameters(controls)

pcof = randn(N_coeff)
prob.nsteps = 600
order = 6
pcof_ubound = 0.3
pcof_lbound = -0.3

maxiter = 100

ipopt_options = (
    "max_iter" => maxiter,
    #"max_wall_time" => 60.0*60*time,
    #"derivative_test" => "first-order",
    "limited_memory_max_history" => 50,
    #"output_file" => filename * ".txt"
)


optimization_history = optimize_gate(
    prob, controls, pcof, target_unitary_rtf, order=order,
    pcof_ubound=pcof_ubound, pcof_lbound=pcof_lbound,
    savename="gargamel",
    ipopt_options = ipopt_options
)


M = CSV.read("gargamel.csv", Tables.matrix, header=1)
plot(M[:,9],M[:,3],lw=2)


optimized_trajectory = eval_forward(prob, controls, optimization_history.x, order=4)
optimized_final_unitary = optimized_trajectory[:,end,:]
final_infidelity = 1 - (1/N_ess_levels^2)*abs(dot(optimized_final_unitary, target_unitary_rtf))^2


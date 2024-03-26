using LinearAlgebra

"""
It should be assumed that the hamiltonian is *complex-valued*, and hence should
be converted when passing to Schrodinger Prob.

kerr_coeffs should be a lower triangular matrix. The q-th diagonal entry should
give ω_q, and then entry kerr_coeff[p,q] (p > q) should
give ξ_pq. 

ω_q is the ground state transition frequency of subsystem q
ξ_q is the self-kerr coefficient of subsystem q
ξ_pq is the cross-kerr coefficient between subsystems p and q
g_pq is the Jaynes-Cummings coupling coefficient between subsystems p and q

Examples of frequencies/coefficients:
ω_q/2π = 4.416 GHz (seen as high as 6 GHz, I think for the ground state transition frequency of the cavity.)
ξ_q/2π = 230 MHz
T₁ = 93.79 μs
T₂ = 102.52 μs, 25 μs (for cavity)

A cavity appears to be treated the same as a qudit, just with many levels.

Maybe I should add a separate thing for rotating frequencies?
"""
function hamiltonian(subsystem_sizes::AbstractVector{Int}, transition_freqs::AbstractVector{<: Real},
        kerr_coeffs::AbstractMatrix{<: Real})

    @assert length(transition_freqs) == size(kerr_coeffs, 1) == size(kerr_coeffs, 2)

    Q = length(subsystem_sizes)
    full_system_size = prod(subsystem_sizes)
    H = zeros(full_system_size, full_system_size)

    lowering_ops = lowering_operators(subsystem_sizes)

    for q in 1:Q
        a_q = lowering_ops[q]
        H .+= transition_freqs[q] .* (a_q' * a_q)
        H .-= 0.5*kerr_coeffs[q,q] .* (a_q' * a_q' * a_q * a_q)
        for p in (q+1):Q
            a_p = lowering_ops[p]
            H .-= kerr_coeffs[p,q] .* (a_p' * a_p* a_q' * a_q)
            # Ignore Jaynes-Cummings coupling for now
        end
    end

    return H
end

function control_ops(subsystem_sizes::AbstractVector{Int})
    lower_ops = lowering_operators(subsystem_sizes)
    sym_ops = [a + a' for a in lower_ops]
    asym_ops = [a - a' for a in lower_ops]
    return sym_ops, asym_ops
end


"""
Given a base qubit transition frequency, and base self/cross-kerr coefficients,
perturb them to create a problem for an ensemble of semi-random qubits.
"""
function multi_qubit_system(subsystem_sizes::AbstractVector{Int},
        base_transition_frequency, base_self_kerr, base_cross_kerr; perturbation_ratio=0.0)
    N = length(subsystem_sizes)

    # Function for 'perturbing' a value (uniform distribution)
    perturb(val) = val + val*rand()*rand([-1,1])*perturbation_ratio

    transition_freqs = ones(N) .* base_transition_frequency
    transition_freqs = perturb.(transition_freqs)

    kerr_coeffs = zeros(N,N)
    for i in 1:N
        kerr_coeffs[i,i] = perturb(base_self_kerr)
        for j in (i+1):N
            kerr_coeffs[j,i] = perturb(base_cross_kerr)
        end
    end

    # Convert to angular
    transition_freqs .*= 2pi
    kerr_coeffs .*= 2pi

    H = hamiltonian(subsystem_sizes, transition_freqs, kerr_coeffs)
    sym_ops, asym_ops = control_ops(subsystem_sizes)

    tf = 50.0
    nsteps = 10

    complex_system_size = size(H, 1)

    U0 = diagm(ones(complex_system_size))

    # Assume all states are essential
    N_ess = complex_system_size
    N_guard = 0

    prob = SchrodingerProb(real(H), imag(H), sym_ops, asym_ops, real(U0), imag(U0),
        tf, nsteps, N_ess)

    return prob
end



"""
Form the identity matrix of sixe N×N. (Actually forming the matrix, so we can
take kronecker products).
"""
function identity(N)
    return Matrix{Int64}(LinearAlgebra.I, N, N)
end


"""
Construct the lowering operator for a single system.
"""
function lowering_operator(system_size::Int)
    return sqrt.(LinearAlgebra.diagm(1 => 1:(system_size-1)))
end

"""
Construct the lowering operators for each subsystem of a larger system.
"""
function lowering_operators(subsystem_sizes::AbstractVector{Int})
    full_system_size = prod(subsystem_sizes)

    # Holds the identity matrix for each subsystem 
    subsystem_identity_matrices = [identity(n) for n in subsystem_sizes] 
    # Holds the matrices we will take the kronecker product of to form the lowering operators for the whole system.
    kronecker_prod_vec = Vector{Matrix{Float64}}(undef, length(subsystem_sizes))

    # The full-system lowering operators we will return.
    lowering_operators_vec = Vector{Matrix{Float64}}(undef, length(subsystem_sizes))

    for i in eachindex(subsystem_sizes)
        subsys_size = subsystem_sizes[i]
        # Set all matrices to the identity matrix for that subssystem
        kronecker_prod_vec .= subsystem_identity_matrices
        # Replace i-th matrix with the lowering operator for that subsystem
        kronecker_prod_vec[i] = lowering_operator(subsys_size)
        # Take the kronecker product of all the matrices (all identity matrices except for the one lowering operator)
        lowering_operators_vec[i] = kron(kronecker_prod_vec...)
    end

    return lowering_operators_vec
end

function collapse_operators(subsystem_sizes::AbstractVector{Int}, T1_times::AbstractVector{<: Real})
    @assert length(subsystem_sizes) == length(T1_times)
    return lowering_operators(subsystem_sizes) ./ (sqrt.(T1_times))
end

function dephasing_operators(subsystem_sizes::AbstractVector{Int}, T2_times::AbstractVector{<: Real})
    @assert length(subsystem_sizes) == length(T2_times)
    lowering_ops = lowering_operators(subsystem_sizes)
    return (adjoint.(lowering_ops) .* lowering_ops) ./ (sqrt.(T2_times))
end

"""
Given an operator for a closed system, convert it to the equivalent operator
for the vectorized open system.

E.g. in the open-system (ρ is a matrix) we have
[H,ρ] = Hρ - ρH
and in the closed-system embedding (ρ is a vector) we have
(I⊗ H - Hᵀ⊗ I)ρ

This is how commutator operations, as in the system and control hamiltonians,
is handled.

For an anticommuator operations, as in the Linbladian terms, we go from the
open-system formulation
{L, ρ} = Lρ + ρL
to the closed-system embedding
(I⊗ L + L⊗ I)ρ

We handle that seperately, because it only comes up in the linbladian terms, and
theres also a LρL^†  term. So I would rather just do the Linbladian directly in
a function which sets up the closed-system embedding.

In fact, I should have a function which converts a closed-system schrodinger
prob into an open-system one.
"""
function closed_operator_to_open(op::AbstractMatrix)
    @assert size(op, 1) == size(op, 2)
    I_N = identity(size(op, 1))
    return kron(I_N, op) - kron(transpose(op), I_n)
end


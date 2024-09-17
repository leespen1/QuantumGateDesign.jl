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
function multi_qudit_hamiltonian(
        subsystem_sizes::AbstractVector{Int},
        transition_freqs::AbstractVector{<: Real},
        rotation_freqs::AbstractVector{<: Real},
        kerr_coeffs::AbstractMatrix{<: Real};
        sparse_rep=true)

    @assert length(transition_freqs) == size(kerr_coeffs, 1) == size(kerr_coeffs, 2)
    @assert issymmetric(kerr_coeffs)

    Q = length(subsystem_sizes)
    full_system_size = prod(subsystem_sizes)
    H_sym = zeros(full_system_size, full_system_size)
    H_asym = zeros(full_system_size, full_system_size)

    lowering_ops = QuantumGateDesign.lowering_operators_system(subsystem_sizes)

    for q in 1:Q
        a_q = lowering_ops[q]
        H_sym .+= (transition_freqs[q] - rotation_freqs[q]) .* (a_q' * a_q)
        H_sym .-= 0.5*kerr_coeffs[q,q] .* (a_q' * a_q' * a_q * a_q)
        for p in (q+1):Q
            a_p = lowering_ops[p]
            H_sym .-= kerr_coeffs[p,q] .* (a_p' * a_p* a_q' * a_q)
            # Ignore Jaynes-Cummings coupling for now
        end
    end

    if sparse_rep
        return SparseArrays.sparse(H_sym), SparseArrays.sparse(H_asym)
    end

    return H_sym, H_asym
end

function control_ops(subsystem_sizes::AbstractVector{Int}; sparse_rep=true)
    lower_ops = QuantumGateDesign.lowering_operators_system(subsystem_sizes)
    sym_ops = [a + a' for a in lower_ops]
    asym_ops = [a - a' for a in lower_ops]
    if sparse_rep
        sparse_sym_ops = [SparseArrays.sparse(op) for op in sym_ops]
        sparse_asym_ops = [SparseArrays.sparse(op) for op in asym_ops]
        return sparse_sym_ops, sparse_asym_ops
    end

    return sym_ops, asym_ops
end


"""
Construct a multi-qudit hamilotonian with kerr and Jaynes-cummings terms.
We require that the same rotation frequency be used in all subsystems, so that
the system Hamiltonian is time-independent.

To get the lab frame hamiltonian, take `transition_freq=0`
"""
function multi_qudit_hamiltonian_jayne(
        subsystem_sizes::AbstractVector{Int},
        transition_freqs::AbstractVector{<: Real},
        rotation_freq::Real,
        kerr_coeffs::AbstractMatrix{<: Real},
        jayne_cummings_coeffs::AbstractMatrix{<: Real};
        sparse_rep=true
    )
    @assert issymmetric(kerr_coeffs)
    @assert issymmetric(jayne_cummings_coeffs)
    @assert iszero(diag(jayne_cummings_coeffs)) # No self JC coupling

    Q = length(subsystem_sizes)
    full_system_size = prod(subsystem_sizes)
    H_sym = SparseArrays.spzeros(Float64, full_system_size, full_system_size)
    H_asym = SparseArrays.spzeros(Float64, full_system_size, full_system_size)

    lowering_ops = lowering_operators_system(subsystem_sizes)

    for q in 1:Q
        a_q = lowering_ops[q]
        H_sym .+= (transition_freqs[q] - rotation_freq) .* (a_q' * a_q)
        H_sym .-= 0.5*kerr_coeffs[q,q] .* (a_q' * a_q' * a_q * a_q)
        for p in (q+1):Q
            a_p = lowering_ops[p]
            H_sym .-= kerr_coeffs[p,q] .* (a_p' * a_p* a_q' * a_q)
            # No time dependence in Jayne-Cummings because we use the same rotational frequency in all subsystems
            H_sym .+= jayne_cummings_coeffs[p,q]*(a_q'*a_p + a_q*a_p')
        end
    end

    if sparse_rep
        return SparseArrays.sparse(H_sym), SparseArrays.sparse(H_asym)
    end

    return H_sym, H_asym
end



"""
Make a Jaynes-Cummings SchrodingerProb

I should really have a hard-coded example to test correctness. 3 qubits should be enough to test.
"""
function JaynesCummingsProblem(
        subsystem_sizes::AbstractVector{Int}, # Maybe I should allow any iterable? I think tuples are reasonable
        essential_subsystem_sizes::AbstractVector{Int}, # Should I have a default? E.g. [2,2,2...]?
        transition_freqs::AbstractVector{<: Real},
        rotation_freq::Real,
        kerr_coeffs::AbstractMatrix{<: Real},
        jayne_cummings_coeffs::AbstractMatrix{<: Real},
        tf,
        nsteps;
        sparse_rep=true,
        bitstring_ordered=true
        # What else do I need? Final time? Guard penalty? Preconditioner?
        # (it would be good to have the preconditioner determined here)
    )

    system_sym, system_asym = multi_qudit_hamiltonian_jayne(
        subsystem_sizes,
        transition_freqs,
        rotation_freq,
        kerr_coeffs, 
        jayne_cummings_coeffs
    )

    sym_ops, asym_ops = control_ops(subsystem_sizes)

    guard_subspace_projector = guard_projector(subsystem_sizes, essential_subsystem_sizes)

    N_ess_levels = prod(essential_subsystem_sizes) # Pretty sure this is correct

    u0, v0 = create_intial_conditions(subsystem_sizes, essential_subsystem_sizes, bitstring_ordered)

    return SchrodingerProb(
        system_sym,
        system_asym,
        sym_ops,
        asym_ops,
        u0,
        v0,
        tf,
        nsteps,
        N_ess_levels,
        guard_subspace_projector,
    )
end


"""
Construct a basis state for a composite system. Ordering of the subsystems is
the same as in the bitstring.

`bitstring_ordered` determines whether the subsystems are provided in the natural
Julia ordering (where the first entry changes most rapidly), or in the opposite
ordering (where the last entry changes most rapidly), like is done in the bra-ket
notation of quantum computing.
"""
function basis_state(subsystem_sizes, subsystem_indices, bitstring_ordered=true)
    # Go from zero-based indices (used in quantum computing) to one-based indices (used in Julia)
    subsystem_indices_one_indexed = subsystem_indices .+ 1

    # Check that indices are valid
    if any(subsystem_indices_one_indexed .> subsystem_sizes)
        throw(ArgumentError("Subsystem indices $(subsystem_indices) are invalid for subsystem sizes $(subsystem_sizes)."))
    end

    # 
    if bitstring_ordered
        subsystem_sizes = reverse(subsystem_sizes)
        subsystem_indices_one_indexed = reverse(subsystem_indices_one_indexed)
    end

    # Construct state as a tensor to make it easy to work with, then convert to vector
    state_tensor = zeros(subsystem_sizes...)
    state_tensor[subsystem_indices_one_indexed...] = 1
    # Convert to state vector
    state_vector = reshape(state_tensor, :) 
    return state_vector
end

"""
Create iniitial conditions. Now works!
"""
function create_initial_conditions(subsystem_sizes, essential_subsystem_sizes, bitstring_ordered=true)
    system_size = prod(subsystem_sizes)
    essential_system_size = prod(essential_subsystem_sizes)

    u0 = zeros(system_size, essential_system_size)
    v0 = zeros(system_size, essential_system_size)
    
    # E.g. if essential_subsystem_sizes is (2,3,4), get (0:1, 0:2, 0:4)
    essential_index_ranges = ntuple(
        i -> 0:essential_index_ranges[i]-1, length(essential_subsystem_sizes)
    )

    if bitstring_ordered
        essential_index_ranges = reverse(essential_index_ranges)
    end

    # product iterates over (0:1, 0:1) as (0,0), (1,0), (0,1), (1,1)
    for (i, subsystem_indices) in enumerate(product(essential_index_ranges...))
        if bitstring_ordered
            subsystem_indices = reverse(subsystem_indices)
        end
        u0[:,i] .= basis_state(subsystem_sizes, subsystem_indices, bitstring_ordered)
    end

    return u0, v0
end

"""
Given the size of each subsystem and the number of essential levels in each subsystem,
return a matrix which projects the state vector onto the guarded subspace.

Note that the first subsystem corresponds to the leftmost bit of the quantum bitstring.

E.g.

``|n_0 n_1 n_2 \\rangle = |n_0\\rangle \\otimes |n_1\\rangle \\otimes |n_2\\rangle``

Examples
≡≡≡≡≡≡≡≡

```
julia> guard_projector([3], [2])
6×6 SparseMatrixCSC{Int64, Int64} with 6 stored entries:
 0  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  0  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  0  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  0  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  1

julia> guard_projector([2,2], [2,1])
8×8 SparseMatrixCSC{Int64, Int64} with 8 stored entries:
 0  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  0  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  0  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  0  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1
 ```
"""
function guard_projector(subsystem_sizes, essential_subsystem_sizes, bitstring_ordered=true)
    system_size = prod(subsystem_sizes)
    essential_system_size = prod(essential_subsystem_sizes)

    G = SparseArrays.spzeros(system_size, system_size)
    Z = SparseArrays.spzeros(system_size, system_size)
    
    # E.g. if subsystem_sizes is (2,3,4), get (0:1, 0:2, 0:3)
    subsystem_index_ranges = ntuple(
        i -> 0:subsystem_sizes[i]-1, length(subsystem_sizes)
    )

    if bitstring_ordered
        subsystem_index_ranges = reverse(subsystem_index_ranges)
    end

    # product iterates over (0:1, 0:1) as (0,0), (1,0), (0,1), (1,1)
    for (i, subsystem_indices) in enumerate(product(subsystem_index_ranges...))
        # If this state is in the essential space, then this column should stay as all zeros
        if all(subsystem_indices .< essential_subsystem_sizes)
            continue
        end

        if bitstring_ordered
            subsystem_indices = reverse(subsystem_indices)
        end
        G[:,i] .= basis_state(subsystem_sizes, subsystem_indices, bitstring_ordered)
    end

    real_valued_guard_projector = [G Z;
                                   Z G]

    return real_valued_guard_projector
end


"""
Construct the lowering operator for a single subsystem.
"""
function lowering_operator_subsystem(subsystem_size::Int)
    return sqrt.(LinearAlgebra.diagm(1 => 1:(subsystem_size-1)))
end


"""
Construct the lowering operators for each subsystem of a larger system, as
applied to the larger system.
"""
function lowering_operators_system(subsystem_sizes::AbstractVector{Int}, bitstring_ordered=true)
    if !bitstring_ordered
        throw(ArgumentError("bitstring_ordered=false not yet supported."))
    end

    full_system_size = prod(subsystem_sizes)

    # Holds the identity matrix for each subsystem 
    subsystem_identity_matrices = [Matrix(LinearAlgebra.I, n, n) for n in subsystem_sizes] 
    # Holds the matrices we will take the kronecker product of to form the lowering operators for the whole system.
    kronecker_prod_vec = Vector{Matrix{Float64}}(undef, length(subsystem_sizes))

    # The full-system lowering operators we will return.
    lowering_operators_vec = Vector{Matrix{Float64}}(undef, length(subsystem_sizes))

    for i in eachindex(subsystem_sizes)
        subsys_size = subsystem_sizes[i]
        # Set all matrices to the identity matrix for that subssystem
        kronecker_prod_vec .= subsystem_identity_matrices
        # Replace i-th matrix with the lowering operator for that subsystem
        kronecker_prod_vec[i] = lowering_operator_subsystem(subsys_size)
        # Take the kronecker product of all the matrices (all identity matrices except for the one lowering operator)
        lowering_operators_vec[i] = reduce(kron, kronecker_prod_vec)
    end

    return lowering_operators_vec
end

# Unfinished work for open systems
#=
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
=#

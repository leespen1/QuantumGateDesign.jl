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
function lowering_operators_system(subsystem_sizes::AbstractVector{Int})
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


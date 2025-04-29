"""
    lower_op(N)

Return the lowering/annihilation operator for a system of size `N`.
"""
function lower_op(N::Integer)
    return sqrt.(diagm(1 => 1:(N-1)))
end

"""
    raise_op(N)

Return the raising/creation operator for a system of size `N`.
"""
function raise_op(N::Integer)
    return sqrt.(diagm(-1 => 1:(N-1)))
end

"""
    number_op(N)

Return the number operator for a system of size `N`.
"""
function number_op(N::Integer)
    return Diagonal(0:N-1)
end

"""
    identity_op(N)

Return the identity operator for a system of size `N`.
"""
function identity_op(N::Integer)
    Matrix(LinearAlgebra.I, N, N)
end

"""
    basis_state(i, N)

Return the `i`-th basis state (in the standard basis) for a system of size `N`.

`i` is zero indexed, to align with common quantum computing notation. E.g. the
vector representing |2⟩ for a 4-level qudit is given by `basis_state(2,4)`.
"""
function basis_state(i::Integer, N::Integer)
    state = zeros(Int64, N)
    state[1+i] = 1
    return state
end

"""
    compsys_basis_state(levels, subsystem_sizes)

For a composite system, return the computational basis (non-entangled) state
where the `i`-th subsystem is in the `levels[i]`-th basis state, and the size of
that subsystem is `subsystem_sizes[i]`.

|n₃n₂n₁⟩ = |n₃⟩⊗|n₂⟩⊗|n₁⟩

E.g. `compsys_basis_state((1,0), (2,2))` constructs the state |10⟩ for a system
of 2 qubits. The vector representation is (0,0,1,0)^T. The state |10⟩ for a system
of one qubit and one 3-level qudit, respectively, is (0,0,0,1,0,0)^T.

In contrast to the column-major ordering of arrays in Julia, for this function
the last subsystem (i.e. the last bit in the bitstring) changes the most rapidly.
That way the ordering of the states is |00⟩, |01⟩, |10⟩, |11⟩.
"""
function compsys_basis_state(subsystem_sizes::IntegersType, levels::IntegersType)
    individual_states = [basis_state(i, N) for (i, N) in zip(levels, subsystem_sizes)]
    return reduce(kron, individual_states)
end

"""
    subsys_rot_frame_op(w, N, t)

Get the operator which applies the rotating transformation with frequency `w` 
to a system of size `N` at time `t`.

For applying a rotating transformation to a composite system (with potentially
different frequencies for each subsystem), see `compsys_rot_frame_op`.
"""
function rot_frame_op(w::Real, N::Integer, t::Real)
    diag_exponents = collect(2pi*im*w*t .* (0:N-1))
    return Diagonal(exp.(diag_exponents))
end

"""
    compsys_rot_frame_op(frequencies, subsystem_sizes, t)


Get the operator which applies the rotating transformation with frequency
`frequencies[i]` to the `i`-th subsystem (which has size `subsystem_sizes[i]`)
at time `t`.
"""
function compsys_rot_frame_op(frequencies::RealsType,
        subsystem_sizes::IntegersType, t::Real
    )

    if (axes(frequencies) != axes(subsystem_sizes))
        throw(DimensionMismatch("Number of frequencies does not match number of subsystem sizes."))
    end

    # 
    subsys_rotating_frame_ops = [rotating_frame_op(w, N, t) 
                                 for (w, N) in zip(frequencies, subsystem_sizes)]
    # (A ⊗ I)*(I ⊗ B) = (A ⊗ B)
    return reduce(kron, subsys_rotating_frame_ops)
end


"""
    promote_subsystem(op, subsystem_sizes, subsystem_index)

"Promote" an operator acting on a subsystem to the corresponding operator acting
on a composite system. 

- op: the operator that acts on the subsystem.
- subsystem_sizes: the sizes of each subsystem of the composite system.
- subsystem_index: the index of the subsystem 'op' acts on. 
"""
function promote_subsys_op(op::AbstractMatrix,
        subsystem_sizes::IntegersType,
        subsystem_index::Integer
    )

    if !(subsystem_index in eachindex(subsystem_sizes))
        throw(ArgumentError("subsystem_index is not a valid index of subsystem_sizes."))
    end

    if !(size(op, 1) == size(op, 2) == subsystem_sizes[subsystem_index])
        throw(DimensionMismatch("Size of operator does not match the corresponding subsystem size."))
    end

    kronecker_prod_vec = Vector{Union{typeof(op), Matrix{Bool}}}(undef, length(subsystem_sizes))
    for i in eachindex(subsystem_sizes)
        if (i == subsystem_index)
            kronecker_prod_vec[i] = op
        else
            n = subsystem_sizes[i]
            kronecker_prod_vec[i] = identity_op(n)
        end
    end
       
    return reduce(kron, kronecker_prod_vec)
end

"""
    gate_initial_states(subsystem_sizes, essential_subsystem_sizes)

Given a composite system whose susbsystems have essential and
guard levels, get a matrix where the `i`-th column is the `i`-th essential
computational basis state of the composite system.

A computational basis state of the composite system is essential when each
subsystem is an essential state.

The `k`-th subsystem of the composite system has size `subsystem_sizes[k]` and
consists of `essential_subsystem_sizes[k]` "essential" levels and
`subsystem_sizes[k]-essential_subsystem_sizes[k]` "guard levels" (ordered in
that way).
"""
function gate_initial_states(subsystem_sizes::IntegersType, 
        essential_subsystem_sizes::IntegersType
    )
    # Was previously a function argument, but I will always have this as true
    bitstring_ordered = true

    system_size = prod(subsystem_sizes)
    essential_system_size = prod(essential_subsystem_sizes)

    U0 = zeros(ComplexF64, system_size, essential_system_size)
    
    # E.g. if essential_subsystem_sizes is (2,3,4), get (0:1, 0:2, 0:4)
    essential_index_ranges = ntuple(
        i -> 0:essential_subsystem_sizes[i]-1, length(essential_subsystem_sizes)
    )

    if bitstring_ordered
        essential_index_ranges = reverse(essential_index_ranges)
    end

    # product iterates over (0:1, 0:1) as (0,0), (1,0), (0,1), (1,1)
    for (i, subsystem_indices) in enumerate(product(essential_index_ranges...))
        if bitstring_ordered
            subsystem_indices = reverse(subsystem_indices)
        end
        U0[:,i] .= compsys_basis_state(subsystem_sizes, subsystem_indices)
    end

    return U0
end


"""
Given the size of each subsystem and the number of essential levels in each
subsystem, return a matrix which projects the (real-valued) state vector onto
the guarded subspace.

Note that the first subsystem corresponds to the leftmost bit of the quantum
bitstring.

E.g.

``|n_0 n_1 n_2 \\rangle = |n_0\\rangle \\otimes |n_1\\rangle \\otimes |n_2\\rangle``

Examples
≡≡≡≡≡≡≡≡

```
julia> guard_projector_op([3], [2])
6×6 SparseMatrixCSC{Int64, Int64} with 6 stored entries:
 0  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  0  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  0  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  0  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  1

julia> guard_projector_op([2,2], [2,1])
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
function guard_projector_op(subsystem_sizes::IntegersType,
        essential_subsystem_sizes::IntegersType
    )
    # Was previously a function argument, but I will always have this as true
    bitstring_ordered = true

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
        G[:,i] .= compsys_basis_state(subsystem_sizes, subsystem_indices)
    end

    real_valued_guard_projector = [G Z;
                                   Z G]

    return real_valued_guard_projector
end

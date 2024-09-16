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
function guard_projector(subsystem_sizes::AbstractVector{Int}, essential_levels_vec::AbstractVector{Int}; use_sparse=true)
    system_size = prod(subsystem_sizes)
    vecs = [zeros(subsystem_size) for subsystem_size in subsystem_sizes]

    if use_sparse
        guard_projector = SparseArrays.spzeros(system_size, 0)
        zero_vec = SparseArrays.spzeros(system_size)
    else
        guard_projector = zeros(system_size, 0)
        zero_vec = zeros(system_size)
    end

    # The reverse and end+i-1 are done so that the initial conditions are
    # ordered from lowest state to highest state. 
    for I in CartesianIndices(Tuple(reverse(subsystem_sizes)))
        I_tup = Tuple(I)

        state_is_essential = true
        for (i, val) in enumerate(I_tup)
            vecs[end+1-i] .= 0
            vecs[end+1-i][val] = 1
            # Check if subsystem state is essential or not
            if (val > essential_levels_vec[end+1-i])
                state_is_essential = false
            end
        end

        # If the state is essential, column should be zero
        if state_is_essential
            column = zero_vec
        else
            column = reduce(kron, vecs)
        end

        guard_projector = hcat(guard_projector, column)
    end

    Re = real(guard_projector)
    Im = imag(guard_projector)
    real_guard_projector = [Re -Im; Im Re]
    return real_guard_projector
end

"""
Create iniitial conditions.
"""
function create_intial_conditions(subsystem_sizes, essential_levels_vec)
    vecs = [zeros(subsystem_size) for subsystem_size in subsystem_sizes]
    essential_states = Vector{Float64}[]
    # The reverse and end+i-1 are done so that the initial conditions are
    # ordered from lowest state to highest state. 
    for I in CartesianIndices(Tuple(reverse(essential_levels_vec)))
        I_tup = Tuple(I)

        for (i, val) in enumerate(I_tup)
            vecs[end+1-i] .= 0
            vecs[end+1-i][val] = 1
        end
        essential_state = reduce(kron, vecs)
        push!(essential_states, essential_state)
    end

    essential_states_matrix = reduce(hcat, essential_states)
    u0 = real(essential_states_matrix)
    v0 = imag(essential_states_matrix)

    #BUGBUGBUGBUG
    
    return u0, v0
end

"""
Make a Jaynes-Cummings SchrodingerProb

I should really have a hard-coded example to test correctness. 3 qubits should be enough to test.
"""
function JaynesCummingsProblem(
        subsystem_sizes::AbstractVector{Int}, # Maybe I should allow any iterable? I think tuples are reasonable
        essential_levels_vec::AbstractVector{Int}, # Should I have a default? E.g. [2,2,2...]?
        transition_freqs::AbstractVector{<: Real},
        rotation_freq::Real,
        kerr_coeffs::AbstractMatrix{<: Real},
        jayne_cummings_coeffs::AbstractMatrix{<: Real},
        tf,
        nsteps;
        sparse_rep=true
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

    guard_subspace_projector = guard_projector(subsystem_sizes, essential_levels_vec)

    N_ess_levels = prod(essential_levels_vec) # Pretty sure this is correct

    u0, v0 = create_intial_conditions(subsystem_sizes, essential_levels_vec)

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

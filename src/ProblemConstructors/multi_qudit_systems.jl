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

    lowering_ops = lowering_operators_system(subsystem_sizes)

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
    lower_ops = lowering_operators(subsystem_sizes)
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
"""
function guard_projector(subsystem_sizes::AbstractVector{Int}, essential_levels_vec::AbstractVector{Int})
    system_size = prod(subsystem_sizes)
    guard_levels_vec = subsystem_sizes - essential_levels_vec

    forbidden_slices = [1+n_ess_levels:n_ess_levels+n_guard_levels
                        for (n_ess_levels, n_guard_levels)
                        in zip(essential_levels_vec, guard_levels_vec)]
    guard_projector_tensor = zeros(Int64, subsystem_sizes...)
    for (subsystem_index, forbidden_slice) in enumerate(forbidden_slices)
        selectdim(guard_projector_tensor, subsystem_index, forbidden_slice) .= 1
    end

    complex_guard_projector = SparseArrays.spdiagm(reshape(guard_projector_tensor, :))

    Re = real(complex_guard_projector)
    Im = imag(complex_guard_projector)
    real_guard_projector = [Re -Im; Im Re]
    return real_guard_projector
end

"""
Make a Jaynes-Cummings SchrodingerProb
"""
function JaynesCummingsProblem(
        subsystem_sizes::AbstractVector{Int},
        transition_freqs::AbstractVector{<: Real},
        rotation_freq::Real,
        kerr_coeffs::AbstractMatrix{<: Real},
        jayne_cummings_coeffs::AbstractMatrix{<: Real};
        sparse_rep=true
        # What else do I need? Final time? Guard penalty? Preconditioner?
        # (it would be good to have the preconditioner determined here)
    )

end

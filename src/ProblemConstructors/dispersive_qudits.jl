"""

H = ∑ⱼΔaⱼ'aⱼ - (ξⱼ/2)aⱼ'aⱼ'aⱼaⱼ - ∑₍ₖ> ⱼ₎ ξⱼₖ aₖ'aₖ aⱼ'aⱼ

It is assumed that the coefficients are given in units of 1/2π. Therefore, the
above Hamiltonian will be multiplied by 2π in this function.

kerr_coeffs should be symmetric matrix. The q-th diagonal entry should
give ξ_q, and then entry kerr_coeff[p,q] (p > q) should give ξ_pq. 

ω_q is the ground state transition frequency of subsystem q
ξ_q is the self-kerr coefficient of subsystem q
ξ_pq is the cross-kerr coefficient between subsystems p and q

Examples of frequencies/coefficients:
ω_q/2π = 4.416 GHz (seen as high as 6 GHz, I think for the ground state
transition frequency of the cavity.)
ξ_q/2π = 230 MHz
Typical Decoherence times:
T₁ = 93.79 μs
T₂ = 102.52 μs, 25 μs (for cavity)

A cavity appears to be treated the same as a qudit, just with many levels.

Maybe I should add a separate thing for rotating frequencies?
"""
function dispersive_qudits_problem(
        subsystem_sizes::IntegersType,
        essential_subsystem_sizes::IntegersType,
        transition_freqs::RealsType,
        rotation_freqs::RealsType,
        kerr_coeffs::AbstractMatrix{<: Real},
        tf::Real,
        nsteps::Integer;
        sparse_rep::Bool=true,
        gmres_abstol::Real=1e-10,
        gmres_reltol::Real=1e-10,
        preconditioner_type::Type=DiagonalHamiltonianPreconditioner,
        rot_frame::Bool = true,
    )

    @assert length(transition_freqs) == size(kerr_coeffs, 1) == size(kerr_coeffs, 2)
    @assert issymmetric(kerr_coeffs)


    lowering_ops = [promote_subsys_op(lower_op(N), subsystem_sizes, i)
                    for (i, N) in enumerate(subsystem_sizes)]

    # Construct System Hamiltonian
    Q = length(subsystem_sizes)
    full_system_size = prod(subsystem_sizes)
    Hsys = zeros(ComplexF64, full_system_size, full_system_size)
    for q in 1:Q
        a_q = lowering_ops[q]

        if rot_frame
            Hsys .+= (transition_freqs[q] - rotation_freqs[q]) .* (a_q' * a_q)
        else # lab frame
            Hsys .+= (transition_freqs[q]) .* (a_q' * a_q)
        end

        Hsys .-= 0.5*kerr_coeffs[q,q] .* (a_q' * a_q' * a_q * a_q)
        for p in (q+1):Q
            a_p = lowering_ops[p]
            Hsys .-= kerr_coeffs[p,q] .* (a_p' * a_p * a_q' * a_q)
        end
    end
    Hsys .*= 2pi # Assume frequencies are given in units of GHz/2pi

    # Construct Control Hamiltonians
    if rot_frame
        sym_ops = [a + a' for a in lowering_ops]
        asym_ops = [a - a' for a in lowering_ops]
    else # lab frame
        sym_ops = [a + a' for a in lowering_ops]
        asym_ops = [zeros(full_system_size, full_system_size) for a in lowering_ops]
    end

    if sparse_rep
        Hsys = sparse(Hsys)
        sym_ops = [sparse(op) for op in sym_ops]
        asym_ops = [sparse(op) for op in asym_ops]
    end

    # Initial Conditions
    U0 = gate_initial_states(subsystem_sizes, essential_subsystem_sizes)

    N_ess_levels = prod(essential_subsystem_sizes)

    # Guard Projector
    guard_subspace_projector = guard_projector_op(subsystem_sizes, essential_subsystem_sizes)

    return SchrodingerProb(
        Hsys,
        sym_ops,
        asym_ops,
        U0,
        tf,
        nsteps,
        N_ess_levels,
        guard_subspace_projector,
        gmres_abstol=gmres_abstol,
        gmres_reltol=gmres_reltol,
        preconditioner_type=preconditioner_type
    )
end

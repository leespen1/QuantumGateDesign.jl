"""
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
function dispersive_qudits_problem(
        subsystem_sizes::IntegersType,
        essential_subsystem_sizes::IntegersType,
        transition_freqs::AbstractVector{<: Real},
        rotation_freqs::AbstractVector{<: Real},
        kerr_coeffs::AbstractMatrix{<: Real},
        tf::Real,
        nsteps::Integer;
        sparse_rep::Bool=true,
        gmres_abstol::Real=1e-10,
        gmres_reltol::Real=1e-10,
        preconditioner_type::Type=DiagonalHamiltonianPreconditioner,
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
        Hsys .+= (transition_freqs[q] - rotation_freqs[q]) .* (a_q' * a_q)
        Hsys .-= 0.5*kerr_coeffs[q,q] .* (a_q' * a_q' * a_q * a_q)
        for p in (q+1):Q
            a_p = lowering_ops[p]
            Hsys .-= kerr_coeffs[p,q] .* (a_p' * a_p* a_q' * a_q)
        end
    end

    # Construct Control Hamiltonians
    sym_ops = [a + a' for a in lower_ops] 
    asym_ops = [a - a' for a in lower_ops] 

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
        system_hamiltonian,
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

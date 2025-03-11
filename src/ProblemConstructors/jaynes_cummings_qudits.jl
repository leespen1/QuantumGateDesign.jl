"""
Make a Jaynes-Cummings SchrodingerProb

I should really have a hard-coded example to test correctness. 3 qubits should
be enough to test.

Construct a multi-qudit hamilotonian with kerr and Jaynes-cummings terms.
We require that the same rotation frequency be used in all subsystems, so that
the system Hamiltonian is time-independent.

To get the lab frame hamiltonian, take `transition_freq=0`
"""
function jaynes_cummings_qudits_problem(
        subsystem_sizes::IntegersType, 
        essential_subsystem_sizes::IntegersType, 
        transition_freqs::AbstractVector{<: Real},
        rotation_freq::Real,
        jayne_cummings_coeffs::AbstractMatrix{<: Real},
        tf::Real,
        nsteps::Integer;
        sparse_rep::Bool=true,
        preconditioner_type::Type=LUPreconditioner,
        gmres_abstol::Real=1e-10,
        gmres_reltol::Real=1e-10,
        # What else do I need? Final time? Guard penalty? Preconditioner?
        # (it would be good to have the preconditioner determined here)
    )

    @assert issymmetric(kerr_coeffs)
    @assert issymmetric(jayne_cummings_coeffs)
    @assert iszero(diag(jayne_cummings_coeffs)) # No self JC coupling


    lowering_ops = [promote_subsys_op(lower_op(N), subsystem_sizes, i)
                    for (i, N) in enumerate(subsystem_sizes)]

    Q = length(subsystem_sizes)
    full_system_size = prod(subsystem_sizes)
    Hsys = SparseArrays.spzeros(ComplexF64, full_system_size, full_system_size)
    for q in 1:Q
        a_q = lowering_ops[q]
        Hsys .+= (transition_freqs[q] - rotation_freq) .* (a_q' * a_q)
        for p in (q+1):Q
            a_p = lowering_ops[p]
            # No time dependence in Jayne-Cummings because we use the same rotational frequency in all subsystems
            Hsys .+= jayne_cummings_coeffs[p,q]*(a_q'*a_p + a_q*a_p')
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
        u0,
        v0,
        tf,
        nsteps,
        N_ess_levels,
        guard_subspace_projector,
        gmres_abstol=gmres_abstol,
        gmres_reltol=gmres_reltol,
        preconditioner_type=preconditioner_type
    )
end

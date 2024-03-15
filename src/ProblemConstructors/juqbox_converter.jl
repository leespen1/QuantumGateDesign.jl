function convert_juqbox(juqbox_params)
    system_sym     = copy(juqbox_params.Hconst)
    system_asym    = similar(system_sym)
    system_asym .= 0 # Juqbox assumes no antisymmetric (imaginary) part in system hamiltonian
    # I used `similar` because the operators may be stored as sparse matrices.

    sym_operators  = [copy(op) for op in juqbox_params.Hsym_ops]
    asym_operators = [copy(op) for op in juqbox_params.Hanti_ops]

    @assert length(juqbox_params.Hunc_ops) == 0

    # The state is always stored as a dense matrix, so no need for `similar` here.
    u0 = copy(juqbox_params.Uinit)
    v0 = zeros(size(u0))

    tf = juqbox_params.T
    nsteps = juqbox_params.nsteps

    # Assume every level is essential for now.
    N_ess_levels = juqbox_params.N
    N_guard_levels = 0

    prob = SchrodingerProb(
        system_sym,
        system_asym,
        sym_operators,
        asym_operators,
        u0,
        v0,
        tf,
        nsteps,
        N_ess_levels,
    )
end

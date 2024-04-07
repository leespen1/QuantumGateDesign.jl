function convert_juqbox(juqbox_params)
    system_sym     = copy(juqbox_params.Hconst)
    if isa(system_sym, SparseArrays.SparseMatrixCSC)
        system_asym = SparseArrays.spzeros(size(system_sym))
    else 
        system_asym = similar(system_sym)
        system_asym .= 0
    end


    sym_operators  = [copy(op) for op in juqbox_params.Hsym_ops]
    asym_operators = [copy(op) for op in juqbox_params.Hanti_ops]

    @assert length(juqbox_params.Hunc_ops) == 0

    # The state is always stored as a dense matrix, so no need for `similar` here.
    u0 = copy(juqbox_params.Uinit)
    v0 = zeros(size(u0))

    tf = juqbox_params.T
    nsteps = juqbox_params.nsteps

    N_ess_levels = juqbox_params.N

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

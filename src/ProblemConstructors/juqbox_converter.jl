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

    guard_subspace_projector = guard_projector(
        juqbox_params.Ne .+ juqbox_params.Ng, juqbox_params.Ne 
    )

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
        guard_subspace_projector
    )
end

"""
`Ne` is the number of energy levels for each subsystem.
`Ng` is the nubmer of guard levels for each subsystem.

"""
function convert_to_juqbox(prob::SchrodingerProb, Ne, Ng, Cfreq, nCoeff, target_complex; use_sparse=true)
    complex_system_size = div(prob.real_system_size, 2)

    Tmax = prob.tf
    nsteps = prob.nsteps

    #U0 = vcat(prob.u0, -prob.v0)
    U0 = prob.u0 -im*prob.v0

    #Utarget = vcat(target[1:real_system_size,:], -target[real_system_size+1:end, :])

    Rfreq = fill(NaN, prob.N_operators) # We don't use uncoupled ops in QGD, so just pass NaN here

    Hconst = [prob.system_asym -prob.system_sym
             prob.system_sym   prob.system_asym]

    Hsym_ops = prob.sym_operators
    Hanti_ops = prob.asym_operators



    params = Juqbox.objparams(
        Ne,
        Ng,
        Tmax,
        nsteps,
        Uinit = U0,
        Utarget = target_complex,
        Cfreq = Cfreq,
        Rfreq = Rfreq,
        Hconst = Hconst,
        Hsym_ops = Hsym_ops,
        Hanti_ops = Hanti_ops,
        use_sparse = use_sparse
    )

end

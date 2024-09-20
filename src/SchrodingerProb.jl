"""
    SchrodingerProb(system_sym, system_asym, sym_operators, asym_operators, u0, v0, tf, nsteps, N_ess_levels)

Set up an object containing the data that defines the physics and numerics
of the problem (the Hamiltonians, initial conditions, number of timesteps, etc).

# Arguments
- `system_sym::M`: the symmetric/real part of the system Hamiltonian.
- `system_asym::M`: the antisymmetric/imaginary part of the system Hamiltonian.
- `sym_operators::Vector{M}`: a vector whose i-th entry is the symmetric part of the i-th control Hamiltonian.
- `asym_operators::Vector{M}`: a vector whose i-th entry is the antisymmetric part of the i-th control Hamiltonian.
- `u0::M`: the real part of the initial conditions. The i-th column corresponds to the i-th initial state in the gate basis.
- `v0::M`: the imaginary part of the initial conditions. The i-th column corresponds to the i-th initial state in the gate basis.
- `tf::Real`: duration of the gate.
- `nsteps::Int64`: number of timesteps to take.
- `N_ess_levels::Int64`: number of levels in the 'essential' subspace, i.e. the part of the subspace actually used for computation.
- `guard_subspace_projector::Union{M, missing}=missing`: matrix projecting a state vector in to the 'guard' subspace.
where `M <: AbstractMatrix{Float64}`

TODO: allow different types for each operator, to allow better specializations (e.g. for symmetric or tridiagonal matrices).
"""
mutable struct SchrodingerProb{M, VM, P} 
    system_sym::M
    system_asym::M
    sym_operators::Vector{M} # a + a^†
    asym_operators::Vector{M} # a - a^†
    u0::VM
    v0::VM
    guard_subspace_projector::M
    tf::Float64
    nsteps::Int64
    N_initial_conditions::Int64
    N_ess_levels::Int64
    N_tot_levels::Int64
    N_operators::Int64 # Number of "control Hamiltonians"
    real_system_size::Int64
    gmres_abstol::Float64
    gmres_reltol::Float64
    forward_preconditioner::P
    adjoint_preconditioner::P
    """
    SchrodingerProb inner constructor, for when all necessary information is
    provided to do forward evolution and gradient calculation to any
    implemented order.
    """
    function SchrodingerProb(
            system_sym::M,
            system_asym::M,
            sym_operators::Vector{M},
            asym_operators::Vector{M},
            u0::VM,
            v0::VM,
            tf::Real,
            nsteps::Int64,
            N_ess_levels::Int64,
            guard_subspace_projector::Union{M, Missing}=missing;
            gmres_abstol=1e-10,
            gmres_reltol=1e-10,
            forward_preconditioner::P=IterativeSolvers.Identity(),
            adjoint_preconditioner::P=IterativeSolvers.Identity(),
        ) where {M<:AbstractMatrix{Float64}, VM<:AbstractVecOrMat{Float64}, P}

        N_tot_levels = size(u0, 1)
        N_initial_conditions = size(u0, 2)

        tf = convert(Float64, tf) # Make sure final time is a float

        # Check dimensions of all matrices and vectors
        @assert size(u0) == size(v0)
        @assert size(u0,1) == size(v0,1) == N_tot_levels
        @assert length(sym_operators) == length(asym_operators)
        @assert size(system_sym,1) == size(system_sym,2) == N_tot_levels
        @assert size(system_asym,1) == size(system_asym,2) == N_tot_levels
        N_operators = length(sym_operators)
        real_system_size = 2*N_tot_levels
        for i in eachindex(sym_operators)
            sym_op = sym_operators[i]
            asym_op = asym_operators[i]
            @assert size(sym_op,1) == size(sym_op,2) == N_tot_levels
            @assert size(asym_op,1) == size(asym_op,2) == N_tot_levels
        end

        # Currently hardcoded for a single qubit. I should make a default here,
        # and an assertion that the projector is symmetric (which is really the
        # only requirement we have)
        if ismissing(guard_subspace_projector)
            guard_subspace_projector = SparseArrays.spzeros(real_system_size, real_system_size)
        end

        @assert isprojection(guard_subspace_projector)

        # Copy arrays when creating a Schrodinger problem
        new{M, VM, P}(
            system_sym, system_asym,
            sym_operators, asym_operators,
            u0, v0,
            guard_subspace_projector,
            tf, nsteps,
            N_initial_conditions, N_ess_levels, N_tot_levels,
            N_operators, real_system_size, 
            gmres_abstol, gmres_reltol,
            forward_preconditioner,
            adjoint_preconditioner
        )
    end
end



function Base.copy(prob::SchrodingerProb)
    # Mutable parameters are copied in the constructor, don't need to copy them again
    return SchrodingerProb(
        copy(prob.system_sym), copy(prob.system_asym),
        deepcopy(prob.sym_operators), deepcopy(prob.asym_operators),
        copy(prob.u0), copy(prob.v0),
        prob.tf, prob.nsteps,
        prob.N_ess_levels,
        copy(prob.guard_subspace_projector),
        gmres_abstol=prob.gmres_abstol, gmres_reltol=prob.gmres_reltol,
        forward_preconditioner=prob.forward_preconditioner,
        adjoint_preconditioner=prob.adjoint_preconditioner
    )
end



"""
Given a Schrodinger problem whose states are matrices (e.g. multiple initial conditions)
"""
function VectorSchrodingerProb(
        prob::SchrodingerProb{M1, M2, P}, initial_condition_index::Int64
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}, P}

    return SchrodingerProb(
        prob.system_sym, prob.system_asym,
        prob.sym_operators, prob.asym_operators,
        prob.u0[:,initial_condition_index], prob.v0[:,initial_condition_index],
        prob.tf, prob.nsteps,
        prob.N_ess_levels,
        prob.guard_subspace_projector,
        gmres_abstol=prob.gmres_abstol,
        gmres_reltol=prob.gmres_reltol,
        forward_preconditioner=prob.forward_preconditioner,
        adjoint_preconditioner=prob.adjoint_preconditioner
    )
end

function VectorSchrodingerProb2(
        prob::SchrodingerProb{M1, M2, P}, initial_condition_index::Int64
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}, P}

    return SchrodingerProb(
        prob.system_sym, prob.system_asym,
        prob.sym_operators, prob.asym_operators,
        reshape(prob.u0[:,initial_condition_index], :, 1), 
        reshape(prob.v0[:,initial_condition_index], :, 1),
        prob.tf, prob.nsteps,
        prob.N_ess_levels,
        prob.guard_subspace_projector,
        gmres_abstol=prob.gmres_abstol,
        gmres_reltol=prob.gmres_reltol,
        forward_preconditioner=prob.forward_preconditioner,
        adjoint_preconditioner=prob.adjoint_preconditioner
    )
end

"""
Show/display problem parameters in a human readable format.
(Considering getting rid of the matrix displays, they are cumbersome. Maybe have another function for seeing those.)
"""
function Base.show(io::IO, ::MIME"text/plain", prob::SchrodingerProb{M, VM, P}) where {M, VM, P}
    println(io, typeof(prob))
    println(io, "Type of operators: ", M)
    print(io, "Type of states: ", VM)
    if (VM <: AbstractVector)
        print(io, " (state transfer problem)")
    elseif (VM <: AbstractMatrix)
        print(io, " (gate design problem)")
    end

    println(io, "\n\nSystem symmetric operator:")
    show(io, "text/plain", prob.system_sym)

    println(io, "\n\nSystem asymmetric operator:")
    show(io, "text/plain", prob.system_asym)

    println(io, "\n\nControl symmetric operators:")
    for op in prob.sym_operators
        println(io)
        show(io, "text/plain", op)
    end

    println(io, "\n\nControl asymmetric operators:")
    for op in prob.asym_operators
        println(io)
        show(io, "text/plain", op)
    end

    println(io, "\n\nGuard supspace projector:")
    show(io, "text/plain", prob.guard_subspace_projector)

    println(io, "\n\nReal part of initial state(s):")
    show(io, "text/plain",  prob.u0)
    println(io, "\n\nImaginary part of initial state(s):")
    show(io, "text/plain",  prob.v0)

    println(io, "\n\nFinal time: ", prob.tf)
    println(io, "Number of timesteps: ", prob.nsteps)
    println(io, "Number of initial condtions: ", prob.N_initial_conditions)
    println(io, "Number of essential levels: ", prob.N_ess_levels)
    println(io, "Total number of levels: ", prob.N_tot_levels)
    println(io, "Number of control Hamiltonians: ", prob.N_operators)
    print(io, "Size of real-valued system: ", prob.real_system_size)
    print(io, "GMRES abstol: ", prob.gmres_abstol)
    print(io, "GMRES reltol: ", prob.gmres_reltol)
    print(io, "Forward preconditioner: ", prob.forward_preconditioner)
    print(io, "Adjoint preconditioner: ", prob.adjoint_preconditioner)


    return nothing
end



function isprojection(A::AbstractMatrix)
    return isapprox(A*A, A, rtol=1e-15)
end

function change_preconditioners(prob::SchrodingerProb,
        forward_preconditioner=prob.forward_preconditioner,
        adjoint_preconditioner=prob.adjoint_preconditioner,
    )
    return SchrodingerProb(
            prob.system_sym,
            prob.system_asym,
            prob.sym_operators,
            prob.asym_operators,
            prob.u0,
            prob.v0,
            prob.tf,
            prob.nsteps,
            prob.N_ess_levels,
            prob.guard_subspace_projector;
            gmres_abstol=prob.gmres_abstol,
            gmres_reltol=prob.gmres_reltol,
            forward_preconditioner=forward_preconditioner,
            adjoint_preconditioner=adjoint_preconditioner,
    )
end

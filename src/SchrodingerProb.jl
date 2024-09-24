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
# Keyword Arguments
- `gmres_abstol::Float64`: absolute tolerance to use when solving linear systems with GMRES.
- `gmres_reltol::Float64`: relative tolerance to use when solving linear systems with GMRES.
- ``preconditioner_type`: the type of preconditioner to use in the algorithm.

where `M <: AbstractMatrix{Float64}`
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
    """
    Inner constructor for `SchrodingerProb` type. Performs input validation on
    the operators: checks sizes, makes sure they are symmetric or
    anti-symmetric, etc. 

    Also computes some problem properties (E.g. the system size, the number of
    initial conditions, etc.) based on the arguments.
    """
    function SchrodingerProb(
        system_sym::M,
        system_asym::M,
        sym_operators::Vector{M}, # a + a^†
        asym_operators::Vector{M}, # a - a^†
        u0::VM,
        v0::VM,
        guard_subspace_projector::M,
        tf::Float64,
        nsteps::Int64,
        N_ess_levels::Int64,
        gmres_abstol::Float64,
        gmres_reltol::Float64,
        preconditioner_type::Type = IdentityPreconditioner
    ) where {M <: AbstractMatrix{Float64}, VM <: AbstractVecOrMat{Float64}}

        complex_system_size = size(system_sym)
        N_tot_levels = size(system_sym, 1)
        real_system_size = 2*N_tot_levels
        N_initial_conditions = size(u0, 2)
        N_operators = length(sym_operators)


        # Check symmetry of operators
        if (system_sym != transpose(system_sym))
            throw(ArgumentError("Real part of system Hamiltonian is not symmetric."))
        end
        for (i, op) in enumerate(sym_operators)
            if (op != transpose(op))
                throw(ArgumentError("Symmetric operator $i is not symmetric."))
            end
        end

        # Check anti-symmetry of operators
        if (system_asym != -transpose(system_asym))
            throw(ArgumentError("Imaginary part of system Hamiltonian is not anti-symmetric."))
        end
        for (i, op) in enumerate(asym_operators)
            if (op != -transpose(op))
                throw(ArgumentError("Anti-symmetric operator $i is not anti-symmetric."))
            end
        end

        # Check size/shape of operators
        if size(system_sym, 1) != size(system_sym, 2)
            throw(ArgumentError("Real part of system Hamiltonian is not square."))
        end


        system_asym_size = size(system_asym)
        if (system_asym_size != complex_system_size)
            throw(ArgumentError("Size $system_asym_size of imaginary part of Hamiltonian does not match size $complex_system_size real part of Hamiltonian."))
        end

        for (i, op) in enumerate(sym_operators)
            op_size = size(op)
            if op_size != complex_system_size
                throw(ArgumentError("Size $op_size of symmetric operator $i does match size $complex_system_size of system Hamiltonian."))
            end
        end

        for (i, op) in enumerate(asym_operators)
            op_size = size(op)
            if op_size != complex_system_size
                throw(ArgumentError("Size $op_size of anti-symmetric operator $i does match size $complex_system_size of system Hamiltonian."))
            end
        end

        # Check size/shape of intiial conditions
        u0_size = size(u0)
        v0_size = size(v0)
        if (u0_size != v0_size)
            throw(ArgumentError("Size $u0_size of the real part of the initial condition does not match the size $v0_size of the imaginary part of the initial condition."))
        end

        if (size(u0, 1) != N_tot_levels)
            throw(ArgumentError("Number of levels $N_tot_levels in initial condition is inconsistent with the size $complex_system_size of system Hamiltonian."))
        end

        # Check number of 
        N_sym = length(sym_operators)
        N_asym = length(asym_operators)
        if N_sym != N_asym
            throw(ArgumentError("Number of symmetric operators $N_sym does not match number of anti-symmetric operators $N_asym."))
        end

        # Check size of guard subspace projector, and check that it is really a projection
        if !isprojection(guard_subspace_projector)
            throw(ArgumentError("Guard subspace projector is not a projector (A*A != A)."))
        end

        guard_subspace_projector_size = size(guard_subspace_projector)
        if guard_subspace_projector_size != (real_system_size, real_system_size)
            throw(ArgumentError("Guard subspace projector size $guard_subspace_projector_size should be twice the size $complex_system_size of the complex-valued system."))
        end

        # Check number of essential levels makes sense
        if N_ess_levels > N_tot_levels
            throw(ArgumentError("Number of essential levels $N_ess_levels cannot be greater than the total number of levels $N_tot_levels."))
        end

        if !(preconditioner_type <: AbstractQGDPreconditioner)
            throw(ArgumentError("preconditioner_type is not an AbstractQGDPreconditioner."))
        end
        
        new{M, VM, preconditioner_type}(
            system_sym, system_asym, sym_operators, asym_operators,
            u0, v0,
            guard_subspace_projector,
            tf, nsteps,
            N_initial_conditions, N_ess_levels, N_tot_levels, N_operators,
            real_system_size, gmres_abstol, gmres_reltol
        )
    end
end

function SchrodingerProb(
        system_hamiltonian::AbstractMatrix{<: Number},
        sym_operators::Vector{<: AbstractMatrix{<: Real}},
        asym_operators::Vector{<: AbstractMatrix{<: Real}},
        U0::AbstractVecOrMat{<: Number},
        tf::Real,
        nsteps::Integer,
        N_ess_levels::Integer,
        guard_subspace_projector::Union{AbstractMatrix{<: Number}, Missing}=missing;
        gmres_abstol::Real=1e-10,
        gmres_reltol::Real=1e-10,
        preconditioner_type::Type=IdentityPreconditioner
    )

    # Seprate initial condition into real and imaginary parts, use Float64 representation
    u0 = real(U0)
    v0 = imag(U0)
    u0 = map(Float64, u0)
    v0 = map(Float64, v0)

    if !LinearAlgebra.ishermitian(system_hamiltonian) 
        throw(ArgumentError("System Hamiltonian is not Hermitian."))
    end

    system_sym = real(system_hamiltonian)
    system_asym = imag(system_hamiltonian)


    # Convert all operators to a common type, using Float64 as the numerical type.
    # (But the elements may be stored in dense or sparse matrices, etc.)
    system_sym = map(Float64, system_sym)
    OpType = typeof(system_sym)
    system_sym = convert(OpType, system_sym)
    system_asym = convert(OpType, system_asym)
    sym_operators = convert.(OpType, sym_operators)
    asym_operators = convert.(OpType, asym_operators)

    real_system_size = 2*size(system_sym, 1)

    if ismissing(guard_subspace_projector)
        guard_subspace_projector = similar(
            system_sym,
            real_system_size,
            real_system_size
        )
        guard_subspace_projector .= 0
        if SparseArrays.issparse(guard_subspace_projector)
            SparseArrays.dropzeros!(guard_subspace_projector) 
        end
    end


    # Convert other arguments to the correct type for storage in SchrodingerProb
    nsteps = convert(Int64, nsteps)
    N_ess_levels = convert(Int64, N_ess_levels)
    tf = convert(Float64, tf)
    gmres_abstol = convert(Float64, gmres_abstol)
    gmres_reltol = convert(Float64, gmres_reltol)

    return SchrodingerProb(
        system_sym, system_asym,
        sym_operators, asym_operators,
        u0, v0,
        guard_subspace_projector,
        tf, nsteps,
        N_ess_levels,
        gmres_abstol, gmres_reltol,
        preconditioner_type
    )
end



function Base.copy(prob::SchrodingerProb{M, VM, P}) where {M, VM, P}
    # Mutable parameters are copied in the constructor, don't need to copy them again
    return SchrodingerProb(
        copy(prob.system_sym), copy(prob.system_asym),
        deepcopy(prob.sym_operators), deepcopy(prob.asym_operators),
        copy(prob.u0), copy(prob.v0),
        copy(prob.guard_subspace_projector),
        prob.tf, prob.nsteps,
        prob.N_ess_levels,
        prob.gmres_abstol,
        prob.gmres_reltol,
        P
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
        prob.guard_subspace_projector,
        prob.tf, prob.nsteps,
        prob.N_ess_levels,
        prob.gmres_abstol,
        prob.gmres_reltol,
        P
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

    println(io, "\n\nSystem anti-symmetric operator:")
    show(io, "text/plain", prob.system_asym)

    println(io, "\n\nControl symmetric operators:")
    for op in prob.sym_operators
        show(io, "text/plain", op)
        println(io, "\n")
    end

    println(io, "\n\nControl anti-symmetric operators:")
    for op in prob.asym_operators
        show(io, "text/plain", op)
        println(io, "\n")
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
    println(io, "Size of real-valued system: ", prob.real_system_size)
    println(io, "GMRES abstol: ", prob.gmres_abstol)
    println(io, "GMRES reltol: ", prob.gmres_reltol)
    println(io, "Preconditioner type: ", P)


    return nothing
end


function isprojection(A::AbstractMatrix)
    return isapprox(A*A, A, rtol=1e-15)
end

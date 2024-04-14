"""
Struct containing all the necessary information needed (except the value of the
control vector and target gate) to evolve a state vector according to
Schrodinger's equation and compute gradients.

TODO: if I add more types, i.e. one for system sym, one for system_asym, allow
sym_ops to be a tuple of vectors, then I think we may be able to do some great
things with Diagonal, TriDiagonal, etc types. It would be especially good to have
the system_sym be Diagonal type (probably better than sparse), and system_asym
be some 'empty' type.
"""
mutable struct SchrodingerProb{M, VM} 
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
            tf::Float64,
            nsteps::Int64,
            N_ess_levels::Int64,
            guard_subspace_projector::Union{M, Missing}=missing
        ) where {M<:AbstractMatrix{Float64}, VM<:AbstractVecOrMat{Float64}}

        N_tot_levels = size(u0, 1)
        N_initial_conditions = size(u0, 2)

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

        # Copy arrays when creating a Schrodinger problem
        new{M, VM}(
            system_sym, system_asym,
            sym_operators, asym_operators,
            u0, v0,
            guard_subspace_projector,
            tf, nsteps,
            N_initial_conditions, N_ess_levels, N_tot_levels,
            N_operators, real_system_size
        )
    end
end



function Base.copy(prob::SchrodingerProb{T}) where T
    # Mutable parameters are copied in the constructor, don't need to copy them again
    return SchrodingerProb(
        copy(prob.system_sym), copy(prob.system_asym),
        deepcopy(prob.sym_operators), deepcopy(prob.asym_operators),
        copy(prob.u0), copy(prob.v0),
        prob.tf, prob.nsteps,
        prob.N_ess_levels,
        copy(prob.guard_subspace_projector)
    )
end



"""
Given a Schrodinger problem whose states are matrices (e.g. multiple initial conditions)
"""
function VectorSchrodingerProb(
        prob::SchrodingerProb{M1, M2}, initial_condition_index::Int64
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    return SchrodingerProb(
        prob.system_sym, prob.system_asym,
        prob.sym_operators, prob.asym_operators,
        prob.u0[:,initial_condition_index], prob.v0[:,initial_condition_index],
        prob.tf, prob.nsteps,
        prob.N_ess_levels,
        prob.guard_subspace_projector
    )
end

function VectorSchrodingerProb2(
        prob::SchrodingerProb{M1, M2}, initial_condition_index::Int64
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    return SchrodingerProb(
        prob.system_sym, prob.system_asym,
        prob.sym_operators, prob.asym_operators,
        reshape(prob.u0[:,initial_condition_index], :, 1), 
        reshape(prob.v0[:,initial_condition_index], :, 1),
        prob.tf, prob.nsteps,
        prob.N_ess_levels,
        prob.guard_subspace_projector
    )
end

"""
Show/display problem parameters in a human readable format.
"""
function Base.show(io::IO, ::MIME"text/plain", prob::SchrodingerProb{M, VM}) where {M, VM}
    println(io, typeof(prob))
    println(io, "Type of operators: ", M)
    print(io, "Type of states: ", VM)
    if (VM <: AbstractVector)
        print(io, " (state transfer problem)")
    elseif (VM <: AbstractMatrix)
        print(io, " (gate design problem)")
    end
    print(io, "\n")

    println(io, "System symmetric operator: \n", prob.system_sym)
    println(io, "System asymmetric operator: \n", prob.system_asym)

    println(io, "Control symmetric operators:")
    for op in prob.sym_operators
        println(io, "\t", op)
    end

    println(io, "Control asymmetric operators:")
    for op in prob.asym_operators
        println(io, "\t", op)
    end

    println(io, "Guard supspace projector: \n", prob.guard_subspace_projector)

    println(io, "Real part of initial state(s): ", prob.u0)
    println(io, "Imaginary part of initial state(s): ", prob.v0)

    println(io, "Final time: ", prob.tf)
    println(io, "Number of timesteps: ", prob.nsteps)
    println(io, "Number of initial condtions: ", prob.N_initial_conditions)
    println(io, "Number of essential levels: ", prob.N_ess_levels)
    println(io, "Total number of levels: ", prob.N_tot_levels)
    println(io, "Number of control Hamiltonians: ", prob.N_operators)
    print(io, "Size of real-valued system: ", prob.real_system_size)

    return nothing
end


"""
Given a SchrodingerProb, return version where all the arrays are dense.
"""
function DenseSchrodingerProb(prob::SchrodingerProb)
    dense_prob =  SchrodingerProb(
        Array(prob.system_sym),
        Array(prob.system_asym),
        Array.(prob.sym_operators),
        Array.(prob.asym_operators),
        Array(prob.u0),
        Array(prob.v0),
        prob.tf,
        prob.nsteps,
        prob.N_ess_levels,
    )
    return dense_prob
end

#=
"""
Create the matrix that projects a state vector onto the guarded subspace.

Currently hardcoded for a single qubit. Shouldn't be a challenge to do it for
multiple qubits using kronecker products.
"""
function create_guard_subspace_projector(N_ess_levels, N_tot_levels)
    @assert length(N_ess_levels) == length(N_tot_levels)
    W = zeros(N_tot_levels, N_tot_levels)
    for i in (N_ess_levels+1):N_tot_levels
        W[i,i] = 1
    end

    # Take kronecker product with 2x2 identity to get real-valued system version
    identity_2by2 = [1 0; 0 1]
    W_realsystem = LinearAlgebra.kron(identity_2by2, W)

    return W_realsystem
end
=#

function create_guard_subspace_projector(N_tot_levels, essential_levels=[])
    real_system_size = 2*N_tot_levels

    guard_subspace_projector = SparseArrays.spzeros(real_system_size, real_system_size)
    for level in 1:N_tot_levels
        if !(level in essential_levels)
            guard_subspace_projector[level, level] = 1
            guard_subspace_projector[N_tot_levels + level, N_tot_levels + level] = 1
        end
    end

    return guard_subspace_projector
end



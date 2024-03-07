"""
Struct containing all the necessary information needed (except the value of the
control vector and target gate) to evolve a state vector according to
Schrodinger's equation and compute gradients.
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
    N_ess_levels::Int64
    N_guard_levels::Int64
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
            N_guard_levels::Int64,
        ) where {M<:AbstractMatrix{Float64}, VM<:AbstractVecOrMat{Float64}}
        N_tot_levels = N_ess_levels + N_guard_levels
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
        # Wait, do I want to project onto essential or forbidden subspace?
        essential_subspace_projector = create_essential_subspace_projector(N_ess_levels, N_tot_levels)

        # Copy arrays when creating a Schrodinger problem
        new{M, VM}(
            copy(system_sym), copy(system_asym),
            deepcopy(sym_operators), deepcopy(asym_operators),
            copy(u0), copy(v0),
            essential_subspace_projector,
            tf, nsteps,
            N_ess_levels, N_guard_levels, N_tot_levels,
            N_operators, real_system_size
        )
    end
end



function Base.copy(prob::SchrodingerProb{T}) where T
    # Mutable parameters are copied in the constructor, don't need to copy them again
    return SchrodingerProb(
        prob.system_sym, prob.system_asym,
        prob.sym_operators, prob.asym_operators,
        prob.u0, prob.v0,
        prob.tf, prob.nsteps,
        prob.N_ess_levels, prob.N_guard_levels
    )
end



function VectorSchrodingerProb(
        prob::SchrodingerProb{M1, M2}, initial_condition_index::Int64
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    return SchrodingerProb(
        prob.system_sym, prob.system_asym,
        prob.sym_operators, prob.asym_operators,
        prob.u0[:,initial_condition_index], prob.v0[:,initial_condition_index],
        prob.tf, prob.nsteps,
        prob.N_ess_levels, prob.N_guard_levels
    )
end


"""
For compatibility in eval_grad_forced (should refactor code)
"""
function VectorSchrodingerProb(
        prob::SchrodingerProb{M, V}, initial_condition_index::Int64
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}
    @assert initial_condition_index == 1
    return copy(prob)
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

    println(io, "System symmetric operator: ", prob.system_sym)
    println(io, "System asymmetric operator: ", prob.system_asym)

    println(io, "Control symmetric operators:")
    for op in prob.sym_operators
        println(io, "\t", op)
    end

    println(io, "Control asymmetric operators:")
    for op in prob.asym_operators
        println(io, "\t", op)
    end

    println(io, "Real part of initial state(s): ", prob.u0)
    println(io, "Imaginary part of initial state(s): ", prob.v0)

    println(io, "Final time: ", prob.tf)
    println(io, "Number of timesteps: ", prob.nsteps)
    println(io, "Number of essential levels: ", prob.N_ess_levels)
    println(io, "Number of guard levels: ", prob.N_guard_levels)
    println(io, "Total number of levels: ", prob.N_tot_levels)
    println(io, "Number of control Hamiltonians: ", prob.N_operators)
    print(io, "Size of real-valued system: ", prob.real_system_size)

    return nothing
end

"""
Create the matrix that projects a state vector onto the guarded subspace.

Currently hardcoded for a single qubit. Shouldn't be a challenge to do it for
multiple qubits using kronecker products.
"""
function create_essential_subspace_projector(N_ess_levels::Int64, N_tot_levels::Int64)
    W = zeros(N_tot_levels, N_tot_levels)
    for i in (N_ess_levels+1):N_tot_levels
        W[i,i] = 1
    end

    # Take kronecker product with 2x2 identity to get real-valued system version
    identity_2by2 = [1 0; 0 1]
    W_realsystem = LinearAlgebra.kron(identity_2by2, W)

    return W_realsystem
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
        prob.N_guard_levels,
    )
    return dense_prob
end

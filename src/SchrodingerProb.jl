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

        # Copy arrays when creating a Schrodinger problem
        new{M, VM}(
            copy(system_sym), copy(system_asym),
            deepcopy(sym_operators), deepcopy(asym_operators),
            copy(u0), copy(v0),
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
Return a Schrodinger Problem which is a copy of the one provided, but with the
difference that the system operators and initial conditions are all zero, as
they are constant in the original problem and therefore go to zero when we take
the partial derivative with respect to a control parameter.

Could do this in a non-copying way, but I am not worried about the performance
of this right now. Especially since I only expect to use this method in
eval_grad_forced.
"""
function differentiated_prob(prob::SchrodingerProb)
    diff_prob = copy(prob)
    diff_prob.system_sym .= 0
    diff_prob.system_asym .= 0
    diff_prob.u0 .= 0
    diff_prob.v0 .= 0
    return diff_prob
end

function time_diff_prob(prob::SchrodingerProb)
    diff_prob = copy(pprob)
    diff_prob.u0 .= 0
    diff_prob.v0 .= 0
    return diff_prob
end

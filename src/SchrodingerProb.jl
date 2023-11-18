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
            N_ess_levels, N_guard_levels, N_tot_levels
        )
    end
end



function Base.copy(prob::SchrodingerProb{T}) where T
    return SchrodingerProb(
        copy(prob.system_sym), copy(prob.system_asym),
        deepcopy(prob.sym_operators), deepcopy(prob.asym_operators),
        copy(prob.u0), copy(prob.v0),
        prob.tf, prob.nsteps,
        prob.N_ess_levels, prob.N_guard_levels
    )
end



function VectorSchrodingerProb(
        prob::SchrodingerProb{M1, M2}, initial_condition_index::Int64
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    return SchrodingerProb(
        copy(prob.system_sym), copy(prob.system_asym),
        deepcopy(prob.sym_operators), deepcopy(prob.asym_operators),
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

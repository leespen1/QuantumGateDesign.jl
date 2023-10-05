"""
Struct containing all the necessary information needed (except the value of the
control vector and target gate) to evolve a state vector according to
Schrodinger's equation and compute gradients.

Should eventually change p_operator/q_operator to be more like Juqbox's coupled
and uncoupled operators.
"""
mutable struct SchrodingerProb{M, VM} 
    Ks::M
    Ss::M
    p_operator::M # a + a^†
    q_operator::M # a - a^†
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
            Ks::M,
            Ss::M,
            p_operator::M,
            q_operator::M,
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
        @assert size(Ks,1) == size(Ks,2) == N_tot_levels
        @assert size(Ss,1) == size(Ss,2) == N_tot_levels
        @assert size(p_operator,1) == size(p_operator,2) == N_tot_levels
        @assert size(q_operator,1) == size(q_operator,2) == N_tot_levels

        # Copy arrays when creating a Schrodinger problem
        new{M, VM}(copy(Ks), copy(Ss), copy(p_operator), copy(q_operator),
            copy(u0), copy(v0),
            tf, nsteps,
            N_ess_levels, N_guard_levels, N_tot_levels)
    end
end



function Base.copy(prob::SchrodingerProb{T}) where T
    return SchrodingerProb(
        copy(prob.Ks), copy(prob.Ss), copy(prob.p_operator), copy(prob.q_operator),
        copy(prob.u0), copy(prob.v0),
        prob.tf, prob.nsteps,
        prob.N_ess_levels, prob.N_guard_levels
    )
end



function VectorSchrodingerProb(
        prob::SchrodingerProb{M1, M2}, initial_condition_index::Int64
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    return SchrodingerProb(
        copy(prob.Ks), copy(prob.Ss), copy(prob.p_operator), copy(prob.q_operator),
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

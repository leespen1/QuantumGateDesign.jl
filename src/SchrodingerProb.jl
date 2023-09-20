"""
Struct containing all the necessary information needed (except the value of the
control vector and target gate) to evolve a state vector according to
Schrodinger's equation and compute gradients.
"""
mutable struct SchrodingerProb{M, VM} 
    Ks::M
    Ss::M
    a_plus_adag::M # a + a^†
    a_minus_adag::M # a - a^†
    p::Function
    q::Function
    dpdt::Function
    dqdt::Function
    dpda::Function
    dqda::Function
    d2p_dta::Function
    d2q_dta::Function
    u0::VM
    v0::VM
    tf::Float64
    nsteps::Int64
    N_ess_levels::Int64
    N_guard_levels::Int64
    N_tot_levels::Int64
    nCoeff::Int64
    """
    SchrodingerProb inner constructor, for when all necessary information is
    provided to do forward evolution and gradient calculation to any
    implemented order.

    Note that a_plus_adag and a_minus_adag don't necessarily have to be those
    operators. I should think of a name change.
    """
    function SchrodingerProb(
            Ks::M,
            Ss::M,
            a_plus_adag::M,
            a_minus_adag::M,
            p::Function,
            q::Function,
            dpdt::Function,
            dqdt::Function,
            dpda::Function,
            dqda::Function,
            d2p_dta::Function,
            d2q_dta::Function,
            u0::VM,
            v0::VM,
            tf::Float64,
            nsteps::Int64,
            N_ess_levels::Int64,
            N_guard_levels::Int64,
            nCoeff::Int64=2
        ) where {M<:AbstractMatrix{Float64}, VM<:AbstractVecOrMat{Float64}}
        N_tot_levels = N_ess_levels + N_guard_levels
        # Check dimensions of all matrices and vectors
        @assert size(u0) == size(v0)
        @assert size(u0,1) == size(v0,1) == N_tot_levels
        @assert size(Ks,1) == size(Ks,2) == N_tot_levels
        @assert size(Ss,1) == size(Ss,2) == N_tot_levels
        @assert size(a_plus_adag,1) == size(a_plus_adag,2) == N_tot_levels
        @assert size(a_minus_adag,1) == size(a_minus_adag,2) == N_tot_levels

        # Copy arrays when creating a Schrodinger problem
        new{M, VM}(copy(Ks), copy(Ss), copy(a_plus_adag), copy(a_minus_adag),
            p, q, dpdt, dqdt, dpda, dqda, d2p_dta, d2q_dta,
            copy(u0), copy(v0),
            tf, nsteps,
            N_ess_levels, N_guard_levels, N_tot_levels, nCoeff)
    end
end



function Base.copy(prob::SchrodingerProb{T}) where T
    return SchrodingerProb(
        copy(prob.Ks), copy(prob.Ss), copy(prob.a_plus_adag), copy(prob.a_minus_adag),
        prob.p, prob.q, prob.dpdt, prob.dqdt,
        prob.dpda, prob.dqda, prob.d2p_dta, prob.d2q_dta,
        copy(prob.u0), copy(prob.v0),
        prob.tf, prob.nsteps,
        prob.N_ess_levels, prob.N_guard_levels, prob.nCoeff
    )
end


"""
Constructor for when only p and q, and not their derivatives, are given.

Can choose to assign derivatives as `missing`, or to compute them using
forward-mode automatic differentiation, using the `auto_diff` keyword.
"""
function SchrodingerProb(Ks, Ss, a_plus_adag, a_minus_adag,
        p, q, u0, v0,
        tf, nsteps, 
        N_ess_levels, N_guard_levels; auto_diff=false)
    if auto_diff
        return SchrodingerProb_AutoDiff(Ks, Ss, a_plus_adag, a_minus_adag,
                                        p, q, u0, v0, tf, nsteps,
                                        N_ess_levels, N_guard_levels)
    end
    return SchrodingerProb_MissingDiff(Ks, Ss, a_plus_adag, a_minus_adag,
                                       p, q, u0, v0, tf, nsteps,
                                       N_ess_levels, N_guard_levels)
end

"""
If no derivatives given, assign them as missing, so they are easy to spot.
"""
function SchrodingerProb_MissingDiff(Ks, Ss, a_plus_adag, a_minus_adag, 
        p, q, u0, v0, tf, nsteps, N_ess_levels, N_guard_levels)

    dpdt(t,a) = missing
    dqdt(t,a) = missing
    dpda(t,a) = missing
    dqda(t,a) = missing
    d2p_dta(t,a) = missing
    d2q_dta(t,a) = missing

    return SchrodingerProb(Ks, Ss, a_plus_adag, a_minus_adag,
                           p, q, dpdt, dqdt, dpda, dqda, d2p_dta, d2q_dta,
                           u0, v0, tf,nsteps,
                           N_ess_levels, N_guard_levels)
end


"""
If no derivatives given, compute them using (forward-mode) automatic differentiation.
"""
function SchrodingerProb_AutoDiff(Ks, Ss, a_plus_adag, a_minus_adag,
        p, q, u0, v0, tf, nsteps, N_ess_levels, N_guard_levels)

    dpdt(t,a) = ForwardDiff.derivative(x -> p(x,a), t)
    dqdt(t,a) = ForwardDiff.derivative(x -> q(x,a), t)
    dpda(t,a) = ForwardDiff.derivative(x -> p(t,x), a)
    dqda(t,a) = ForwardDiff.derivative(x -> q(t,x), a)
    d2p_dta(t,a) = ForwardDiff.derivative(x -> dpdt(t,x), a)
    d2q_dta(t,a) = ForwardDiff.derivative(x -> dqdt(t,x), a)

    return SchrodingerProb(Ks, Ss, a_plus_adag, a_minus_adag,
                           p, q, dpdt, dqdt, dpda, dqda, d2p_dta, d2q_dta,
                           u0, v0, tf, nsteps,
                           N_ess_levels, N_guard_levels)
end


function VectorSchrodingerProb(
        prob::SchrodingerProb{M1, M2}, initial_condition_index::Int64
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    return SchrodingerProb(
        copy(prob.Ks), copy(prob.Ss), copy(prob.a_plus_adag), copy(prob.a_minus_adag),
        prob.p, prob.q, prob.dpdt, prob.dqdt,
        prob.dpda, prob.dqda, prob.d2p_dta, prob.d2q_dta,
        prob.u0[:,initial_condition_index], prob.v0[:,initial_condition_index],
        prob.tf, prob.nsteps,
        prob.N_ess_levels, prob.N_guard_levels, prob.nCoeff
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

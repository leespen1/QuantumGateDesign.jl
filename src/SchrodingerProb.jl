mutable struct SchrodingerProb
    Ks::AbstractMatrix{Float64}
    Ss::AbstractMatrix{Float64}
    a_plus_adag::AbstractMatrix{Float64} # a + a^†
    a_minus_adag::AbstractMatrix{Float64} # a - a^†
    p::Function
    q::Function
    dpdt::Function
    dqdt::Function
    dpda::Function
    dqda::Function
    d2p_dta::Function
    d2q_dta::Function
    u0::AbstractVector{Float64}
    v0::AbstractVector{Float64}
    tf::Float64
    nsteps::Int64
    N_ess_levels::Int64
    N_guard_levels::Int64
    N_tot_levels::Int64
    nCoeff::Int64
    function SchrodingerProb(
            Ks::AbstractMatrix{Float64},
            Ss::AbstractMatrix{Float64},
            a_plus_adag::AbstractMatrix{Float64},
            a_minus_adag::AbstractMatrix{Float64},
            p::Function,
            q::Function,
            dpdt::Function,
            dqdt::Function,
            dpda::Function,
            dqda::Function,
            d2p_dta::Function,
            d2q_dta::Function,
            u0::AbstractVector{Float64},
            v0::AbstractVector{Float64},
            tf::Float64,
            nsteps::Int64,
            N_ess_levels::Int64,
            N_guard_levels::Int64,
            nCoeff::Int64=2
        )
        N_tot_levels = N_ess_levels + N_guard_levels
        # Check dimensions of all matrices and vectors
        @assert length(u0) == length(v0) == N_tot_levels
        @assert size(Ks,1) == size(Ks,2) == N_tot_levels
        @assert size(Ss,1) == size(Ss,2) == N_tot_levels
        @assert size(a_plus_adag,1) == size(a_plus_adag,2) == N_tot_levels
        @assert size(a_minus_adag,1) == size(a_minus_adag,2) == N_tot_levels

        # Copy arrays when creating a Schrodinger problem
        new(copy(Ks), copy(Ss), copy(a_plus_adag), copy(a_minus_adag),
            p, q, dpdt, dqdt, dpda, dqda, d2p_dta, d2q_dta,
            copy(u0), copy(v0),
            tf, nsteps,
            N_ess_levels, N_guard_levels, N_tot_levels, nCoeff)
    end
end



function Base.copy(prob::SchrodingerProb)
    return SchrodingerProb(prob.Ks, prob.Ss, prob.a_plus_adag, prob.a_minus_adag,
                           prob.p, prob.q, prob.dpdt, prob.dqdt,
                           prob.dpda, prob.dqda, prob.d2p_dta, prob.d2q_dta,
                           prob.u0, prob.v0,
                           prob.tf, prob.nsteps,
                           prob.N_ess_levels, prob.N_guard_levels, prob.nCoeff)
end



"""
Handle when only p and q are given, with no derivatives.
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





mutable struct SchrodingerProb
    Ks::Matrix{Float64}
    Ss::Matrix{Float64}
    a_plus_adag::Matrix{Float64} # a + a^†
    a_minus_adag::Matrix{Float64} # a - a^†
    p::Function
    q::Function
    dpdt::Function
    dqdt::Function
    dpda::Function
    dqda::Function
    d2p_dta::Function
    d2q_dta::Function
    u0::Vector{Float64}
    v0::Vector{Float64}
    tf::Float64
    nsteps::Int64
    function SchrodingerProb(
            Ks::Matrix{Float64},
            Ss::Matrix{Float64},
            a_plus_adag::Matrix{Float64},
            a_minus_adag::Matrix{Float64},
            p::Function,
            q::Function,
            dpdt::Function,
            dqdt::Function,
            dpda::Function,
            dqda::Function,
            d2p_dta::Function,
            d2q_dta::Function,
            u0::Vector{Float64},
            v0::Vector{Float64},
            tf::Float64,
            nsteps::Int64
        )
        # Copy arrays when creating a Schrodinger problem
        new(copy(Ks), copy(Ss), copy(a_plus_adag), copy(a_minus_adag),
            p, q, dpdt, dqdt, dpda, dqda, d2p_dta, d2q_dta,
            copy(u0), copy(v0), tf, nsteps)
    end
end



function Base.copy(prob::SchrodingerProb)
    return SchrodingerProb(prob.Ks, prob.Ss, prob.a_plus_adag, prob.a_minus_adag,
                           prob.p, prob.q, prob.dpdt, prob.dqdt,
                           prob.dpda, prob.dqda, prob.d2p_dta, prob.d2q_dta,
                           prob.u0, prob.v0,
                           prob.tf, prob.nsteps)
end



"""
Handle when only p and q are given, with no derivatives.
"""
function SchrodingerProb(Ks, Ss, a_plus_adag, a_minus_adag, p, q, u0, v0, tf, nsteps; auto_diff=false)
    if auto_diff
        return SchrodingerProb_AutoDiff(Ks, Ss, a_plus_adag, a_minus_adag,
                                        p, q, u0, v0, tf, nsteps)
    end
    return SchrodingerProb_MissingDiff(Ks, Ss, a_plus_adag, a_minus_adag,
                                       p, q, u0, v0, tf, nsteps)
end

"""
If no derivatives given, assign them as missing, so they are easy to spot.
"""
function SchrodingerProb_MissingDiff(Ks, Ss, a_plus_adag, a_minus_adag, p, q, u0, v0, tf,
        nsteps)

    dpdt(t,a) = missing
    dqdt(t,a) = missing
    dpda(t,a) = missing
    dqda(t,a) = missing
    d2p_dta(t,a) = missing
    d2q_dta(t,a) = missing

    return SchrodingerProb(Ks, Ss, a_plus_adag, a_minus_adag,
                           p, q, dpdt, dqdt, dpda, dqda, d2p_dta, d2q_dta,
                           u0,v0,
                           tf,nsteps)
end


"""
If no derivatives given, compute them using (forward-mode) automatic differentiation.
"""
function SchrodingerProb_AutoDiff(Ks, Ss, a_plus_adag, a_minus_adag, p, q, u0, v0, tf,
        nsteps)

    dpdt(t,a) = ForwardDiff.derivative(x -> p(x,a), t)
    dqdt(t,a) = ForwardDiff.derivative(x -> q(x,a), t)
    dpda(t,a) = ForwardDiff.derivative(x -> p(t,x), a)
    dqda(t,a) = ForwardDiff.derivative(x -> q(t,x), a)
    d2p_dta(t,a) = ForwardDiff.derivative(x -> dpdt(t,x), a)
    d2q_dta(t,a) = ForwardDiff.derivative(x -> dqdt(t,x), a)

    return SchrodingerProb(Ks, Ss, a_plus_adag, a_minus_adag,
                           p, q, dpdt, dqdt, dpda, dqda, d2p_dta, d2q_dta,
                           u0,v0,
                           tf,nsteps)
end





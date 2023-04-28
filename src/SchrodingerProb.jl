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

        a_plus_adag::Matrix{Float64} = [0.0 1.0; 1.0 0.0]
        a_minus_adag::Matrix{Float64} = [0.0 1.0; -1.0 0.0]

        new(copy(Ks), copy(Ss), a_plus_adag, a_minus_adag,
            p, q, dpdt, dqdt, dpda, dqda, d2p_dta, d2q_dta,
            copy(u0), copy(v0), tf, nsteps)
    end
end


function SchrodingerProb(Ks,Ss,p,q,u0,v0,tf,nsteps)
    dpdt(t,a) = nothing
    dqdt(t,a) = nothing
    dpda(t,a) = nothing
    dqda(t,a) = nothing
    d2p_dta(t,a) = nothing
    d2q_dta(t,a) = nothing
    return SchrodingerProb(Ks, Ss,
                           p, q, dpdt, dqdt, dpda, dqda, d2p_dta, d2q_dta,
                           u0,v0,
                           tf,nsteps)
end



function Base.copy(prob::SchrodingerProb)
    return SchrodingerProb(prob.Ks, prob.Ss,
                           prob.p, prob.q, prob.dpdt, prob.dqdt,
                           prob.dpda, prob.dqda, prob.d2p_dta, prob.d2q_dta,
                           prob.u0, prob.v0,
                           prob.tf, prob.nsteps)
end

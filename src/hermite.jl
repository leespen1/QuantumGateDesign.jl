function utvt!(ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        Ks::AbstractMatrix{Float64}, Ss::AbstractMatrix{Float64},
        a_plus_adag::AbstractMatrix{Float64}, a_minus_adag::AbstractMatrix{Float64},
        p::Function, q::Function, t, α)
    # Call the version of utvt! which uses the values of p and q
    utvt!(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p(t,α), q(t,α))

    return nothing
end

"""
Values of p(t,α) and q(t,α) provided
"""
function utvt!(ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        Ks::AbstractMatrix{Float64}, Ss::AbstractMatrix{Float64},
        a_plus_adag::AbstractMatrix{Float64}, a_minus_adag::AbstractMatrix{Float64},
        p::Float64, q::Float64)
    # Non-Memory-Allocating Version (test performance)
    # ut = (Ss + q(t)(a-a†))u - (Ks + p(t)(a+a†))v
    mul!(ut, Ss, u)
    mul!(ut, a_minus_adag, u, q, 1)
    mul!(ut, Ks, v, -1, 1)
    mul!(ut, a_plus_adag, v, -p, 1)

    # vt = (Ss + q(t)(a-a†))v + (Ks + p(t)(a+a†))u
    mul!(vt, Ss, v)
    mul!(vt, a_minus_adag, v, q, 1)
    mul!(vt, Ks, u, 1, 1)
    mul!(vt, a_plus_adag,  u, p, 1)

    return nothing
end

function uttvtt!(utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        Ks::AbstractMatrix{Float64}, Ss::AbstractMatrix{Float64},
        a_plus_adag::AbstractMatrix{Float64}, a_minus_adag::AbstractMatrix{Float64},
        p::Function, q::Function,
        dpdt::Function, dqdt::Function,
        t, α)
    uttvtt!(utt, vtt, ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag,
            p(t,α), q(t,α), dpdt(t,α), dqdt(t,α))

    return nothing
end

function uttvtt!(utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        Ks::AbstractMatrix{Float64}, Ss::AbstractMatrix{Float64},
        a_plus_adag::AbstractMatrix{Float64}, a_minus_adag::AbstractMatrix{Float64},
        p::Float64, q::Float64,
        dpdt::Float64, dqdt::Float64
        )
    ## Make use of utvt!
    utvt!(utt, vtt, ut, vt, Ks, Ss, a_plus_adag, a_minus_adag, p, q)

    mul!(utt, a_minus_adag, u, dqdt, 1)
    mul!(utt, a_plus_adag, v, -dpdt, 1)

    mul!(vtt, a_plus_adag,  u, dpdt, 1)
    mul!(vtt, a_minus_adag, v, dqdt, 1)

    return nothing
end



"""
In the Hermite timestepping method, we arrive at an equation like

LHS*uvⁿ⁺¹ = RHS*uvⁿ

This function computes the action of LHS on a vector uv.
That is, given an input uv (as two arguments u and v), return LHS*uv
"""
function LHS_func(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α, dt, N_tot)
    utvt!(ut, vt, u, v,
          Ks, Ss, a_plus_adag, a_minus_adag,
          p, q, t, α)
    
    LHSu = copy(u)
    axpy!(-0.5*dt,ut,LHSu)
    LHSv = copy(v)
    axpy!(-0.5*dt,vt,LHSv)

    LHS_uv = zeros(Float64, 2*N_tot)
    copyto!(LHS_uv,1,LHSu,1,N_tot)
    copyto!(LHS_uv,1+N_tot,LHSv,1,N_tot)

    return LHS_uv
end

function LHS_func_order4(utt, vtt, ut, vt, u, v,
        Ks, Ss, a_plus_adag, a_minus_adag,
        p, q, dpdt, dqdt, t, α, dt, N_tot)

    utvt!(ut, vt, u, v,
          Ks, Ss, a_plus_adag, a_minus_adag,
          p, q, t, α)
    uttvtt!(utt, vtt, ut, vt, u, v,
            Ks, Ss, a_plus_adag, a_minus_adag,
            p, q, dpdt, dqdt, t, α)

    weights = [1,-1/3]
    
    LHSu = copy(u)
    axpy!(-0.5*dt*weights[1],ut,LHSu)
    axpy!(-0.25*dt^2*weights[2],utt,LHSu)
    LHSv = copy(v)
    axpy!(-0.5*dt*weights[1],vt,LHSv)
    axpy!(-0.25*dt^2*weights[2],vtt,LHSv)

    LHS = zeros(Float64, 2*N_tot)
    copyto!(LHS,1,LHSu,1,N_tot)
    copyto!(LHS,1+N_tot,LHSv,1,N_tot)

    return LHS
end

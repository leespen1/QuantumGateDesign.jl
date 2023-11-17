#==============================================================================
#
# I should take a look at this again and give things more informative names
#
==============================================================================#
function utvt!(ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        Ks::AbstractMatrix{Float64}, Ss::AbstractMatrix{Float64},
        a_plus_adag::AbstractMatrix{Float64}, a_minus_adag::AbstractMatrix{Float64},
        control::AbstractControl, t::Float64, pcof::AbstractArray{Float64})

    pval = eval_p(control, t, pcof)
    qval = eval_q(control, t, pcof)

    utvt!(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, pval, qval)

    return nothing
end

"""
Values of p(t,pcof) and q(t,pcof) provided. Mutates ut and vt, leaves all other variables untouched.
"""
function utvt!(ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        Ks::AbstractMatrix{Float64}, Ss::AbstractMatrix{Float64},
        a_plus_adag::AbstractMatrix{Float64}, a_minus_adag::AbstractMatrix{Float64},
        p_val::Float64, q_val::Float64)
    # Non-Memory-Allocating Version (test performance)
    # ut = (Ss + q(t)(a-a†))u + (Ks + p(t)(a+a†))v
    mul!(ut, Ss, u)
    mul!(ut, a_minus_adag, u, q_val, 1)
    mul!(ut, Ks, v, 1, 1)
    mul!(ut, a_plus_adag, v, p_val, 1)

    # vt = (Ss + q(t)(a-a†))v - (Ks + p(t)(a+a†))u
    mul!(vt, Ss, v)
    mul!(vt, a_minus_adag, v, q_val, 1)
    mul!(vt, Ks, u, -1, 1)
    mul!(vt, a_plus_adag,  u, -p_val, 1)

    return nothing
end


"""
Assumes that ut and vt have already been computed

ψtt = Ht*ψ + H*ψt = Ht*ψ + H²*ψ

Hψ is already calculated and stored in ut/vt. Therefore the call to utvt! with ut/vt in
place of u/v computes H²ψ.

The rest of the function computes Ht*ψb

"""
function uttvtt!(utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        Ks::AbstractMatrix{Float64}, Ss::AbstractMatrix{Float64},
        a_plus_adag::AbstractMatrix{Float64}, a_minus_adag::AbstractMatrix{Float64},
        p_val::Float64, q_val::Float64,
        dpdt_val::Float64, dqdt_val::Float64
        )
    ## Make use of utvt! to compute H²ψ, make use of ψt already computed
    utvt!(utt, vtt, ut, vt, Ks, Ss, a_plus_adag, a_minus_adag, p_val, q_val)

    # Add Ht*ψ. System/drift hamiltonian is time-independent, falls out in Ht
    mul!(utt, a_minus_adag, u, dqdt_val, 1)
    mul!(utt, a_plus_adag, v, dpdt_val, 1)

    mul!(vtt, a_plus_adag,  u, -dpdt_val, 1)
    mul!(vtt, a_minus_adag, v, dqdt_val, 1)

    return nothing
end

function uttvtt!(utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        Ks::AbstractMatrix{Float64}, Ss::AbstractMatrix{Float64},
        a_plus_adag::AbstractMatrix{Float64}, a_minus_adag::AbstractMatrix{Float64},
        control::AbstractControl, t::Float64, pcof::AbstractArray{Float64})

    p_val = eval_p(control, t, pcof)
    q_val = eval_q(control, t, pcof)
    pt_val = eval_pt(control, t, pcof)
    qt_val = eval_qt(control, t, pcof)

    uttvtt!(utt, vtt, ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag,
            p_val, q_val, pt_val, qt_val)

    return nothing
end



"""
In the Hermite timestepping method, we arrive at an equation like

LHS*uvⁿ⁺¹ = RHS*uvⁿ

This function computes the action of LHS on a vector uv.
That is, given an input uv (as two arguments u and v), return LHS*uv
"""
function LHS_func(ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        Ks::AbstractMatrix{Float64}, Ss::AbstractMatrix{Float64}, 
        a_plus_adag::AbstractMatrix{Float64}, a_minus_adag::AbstractMatrix{Float64},
        control::AbstractControl, t::Float64, pcof::AbstractVector{Float64},
        dt::Float64, N_tot::Int64)

    utvt!(ut, vt, u, v,
          Ks, Ss, a_plus_adag, a_minus_adag,
          control, t, pcof)
    
    LHSu = copy(u)
    axpy!(-0.5*dt,ut,LHSu)
    LHSv = copy(v)
    axpy!(-0.5*dt,vt,LHSv)

    LHS_uv = zeros(Float64, 2*N_tot)
    copyto!(LHS_uv,1,LHSu,1,N_tot)
    copyto!(LHS_uv,1+N_tot,LHSv,1,N_tot)

    return LHS_uv
end

function LHS_func_order4(utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        Ks::AbstractMatrix{Float64}, Ss::AbstractMatrix{Float64}, 
        a_plus_adag::AbstractMatrix{Float64}, a_minus_adag::AbstractMatrix{Float64},
        control::AbstractControl, t::Float64,
        pcof::AbstractVector{Float64}, dt::Float64, N_tot::Int64)

    utvt!(ut, vt, u, v,
          Ks, Ss, a_plus_adag, a_minus_adag,
          control, t, pcof)
    uttvtt!(utt, vtt, ut, vt, u, v,
            Ks, Ss, a_plus_adag, a_minus_adag,
            control, t, pcof)

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

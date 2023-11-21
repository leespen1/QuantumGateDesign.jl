#==============================================================================
#
# I should take a look at this again and give things more informative names
#
==============================================================================#
"""
Values of p(t,pcof) and q(t,pcof) provided. Mutates ut and vt, leaves all other variables untouched.
"""
function utvt!(ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        system_sym::AbstractMatrix{Float64}, system_asym::AbstractMatrix{Float64},
        control_op_sym::AbstractMatrix{Float64}, control_op_asym::AbstractMatrix{Float64},
        p_val::Float64, q_val::Float64)
    # Non-Memory-Allocating Version (test performance)
    # ut = (Ss + q(t)(a-a†))u + (Ks + p(t)(a+a†))v
    mul!(ut, system_asym, u)
    mul!(ut, control_op_asym, u, q_val, 1)
    mul!(ut, system_sym, v, 1, 1)
    mul!(ut, control_op_sym, v, p_val, 1)

    # vt = (Ss + q(t)(a-a†))v - (Ks + p(t)(a+a†))u
    mul!(vt, system_asym, v)
    mul!(vt, control_op_asym, v, q_val, 1)
    mul!(vt, system_sym, u, -1, 1)
    mul!(vt, control_op_sym,  u, -p_val, 1)

    return nothing
end

"""
Multiple control version
"""
function utvt!(ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{Float64})

    # Non-Memory-Allocating Version (test performance)
    # ut = (Ss + q(t)(a-a†))u + (Ks + p(t)(a+a†))v
    # vt = (Ss + q(t)(a-a†))v - (Ks + p(t)(a+a†))u
    
    mul!(ut, prob.system_asym, u)
    mul!(ut, prob.system_sym, v, 1, 1)

    mul!(vt, prob.system_asym, v)
    mul!(vt, prob.system_sym, u, -1, 1)

    for i in 1:length(controls)
        control = controls[i]
        sym_op = prob.sym_operators[i]
        asym_op = prob.asym_operators[i]
        this_pcof = get_control_vector_slice(pcof, controls, i)

        mul!(ut, asym_op, u, eval_q(control, t, this_pcof), 1)
        mul!(ut, sym_op, v, eval_p(control, t, this_pcof), 1)

        mul!(vt, asym_op, v, eval_q(control, t, this_pcof), 1)
        mul!(vt, sym_op,  u, -eval_p(control, t, this_pcof), 1)
    end

    return nothing
end

"""
Evaluate first derivative in adjoint equation.

In the real-valued formulation `y = Ax`, the adjoint equation is `y = Aᵀx`.
Because the Hamiltonian is Hermitian, for the adjoint equation we only need to
do 
[S K; -K S]ᵀ = [-S -K; K -S]
"""
function utvt_adj!(ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{Float64})

    # System / Drift Part of Hamiltonian
    mul!(ut, prob.system_asym, u)
    ut .*= -1 # Want to do ut=Sᵀu = -Su, this is the best way to get the negative without allocating memory
    mul!(ut, prob.system_sym, v, -1, 1)

    mul!(vt, prob.system_asym, v)
    vt .*= -1 # Get negative without allocating memory
    mul!(vt, prob.system_sym, u, 1, 1)

    for i in 1:length(controls)
        control = controls[i]
        sym_op = prob.sym_operators[i]
        asym_op = prob.asym_operators[i]

        # Get part of control vector corresponding to this control
        this_pcof = get_control_vector_slice(pcof, controls, i)

        mul!(ut, asym_op, u, -eval_q(control, t, this_pcof), 1)
        mul!(ut, sym_op, v, -eval_p(control, t, this_pcof), 1)

        mul!(vt, sym_op,  u, eval_p(control, t, this_pcof), 1)
        mul!(vt, asym_op, v, -eval_q(control, t, this_pcof), 1)
    end

    return nothing
end


"""
Assumes that ut and vt have already been computed

Overwrites utt and vtt, leaves everything else untouched.

ψtt = Ht*ψ + H*ψt = Ht*ψ + H²*ψ

Hψ is already calculated and stored in ut/vt. Therefore the call to utvt! with ut/vt in
place of u/v computes H²ψ.

The rest of the function computes Ht*ψb

"""
function uttvtt!(utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        system_sym::AbstractMatrix{Float64}, system_asym::AbstractMatrix{Float64},
        control_op_sym::AbstractMatrix{Float64}, control_op_asym::AbstractMatrix{Float64},
        p_val::Float64, q_val::Float64,
        dpdt_val::Float64, dqdt_val::Float64
        )
    ## Make use of utvt! to compute H²ψ, make use of ψt already computed
    utvt!(utt, vtt, ut, vt, system_sym, system_asym, control_op_sym, control_op_asym, p_val, q_val)

    # Add Ht*ψ. System/drift hamiltonian is time-independent, falls out in Ht
    mul!(utt, control_op_asym, u, dqdt_val, 1)
    mul!(utt, control_op_sym,  v, dpdt_val, 1)

    mul!(vtt, control_op_sym,  u, -dpdt_val, 1)
    mul!(vtt, control_op_asym, v, dqdt_val,  1)

    return nothing
end

function uttvtt!(utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{Float64})

    # Calculate H²ψ
    utvt!(utt, vtt, ut, vt, prob, controls, t, pcof)

    # Calculate Hₜψ (time derivative means drift hamiltonian goes away)
    for i in 1:length(controls)
        control = controls[i]
        sym_op = prob.sym_operators[i]
        asym_op = prob.asym_operators[i]

        # Get part of control vector corresponding to this control
        this_pcof = get_control_vector_slice(pcof, controls, i)

        mul!(utt, asym_op, u, eval_qt(control, t, this_pcof), 1)
        mul!(utt, sym_op,  v, eval_pt(control, t, this_pcof), 1)

        mul!(vtt, sym_op,  u, -eval_pt(control, t, this_pcof), 1)
        mul!(vtt, asym_op, v, eval_qt(control, t, this_pcof),  1)
    end

end

function uttvtt_adj!(utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{Float64})

    # Calculate H²ψ
    utvt_adj!(utt, vtt, ut, vt, prob, controls, t, pcof)

    # Calculate Hₜψ (time derivative means drift hamiltonian goes away)
    for i in 1:length(controls)
        control = controls[i]
        sym_op = prob.sym_operators[i]
        asym_op = prob.asym_operators[i]

        # Get part of control vector corresponding to this control
        this_pcof = get_control_vector_slice(pcof, controls, i)

        mul!(ut, asym_op, u, -eval_qt(control, t, this_pcof), 1)
        mul!(ut, sym_op,  v, -eval_pt(control, t, this_pcof), 1)

        mul!(vt, sym_op,  u, eval_pt(control,  t, this_pcof), 1)
        mul!(vt, asym_op, v, -eval_qt(control, t, this_pcof), 1)
    end

end

"""
In the Hermite timestepping method, we arrive at an equation like

LHS*uvⁿ⁺¹ = RHS*uvⁿ

This function computes the action of LHS on a vector uv.
That is, given an input uv (as two arguments u and v), return LHS*uv
"""
function LHS_func(ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        system_sym::AbstractMatrix{Float64}, system_asym::AbstractMatrix{Float64}, 
        control_op_sym::AbstractMatrix{Float64}, control_op_asym::AbstractMatrix{Float64},
        control::AbstractControl, t::Float64, pcof::AbstractVector{Float64},
        dt::Float64, N_tot::Int64)

    utvt!(ut, vt, u, v,
          system_sym, system_asym, control_op_sym, control_op_asym,
          control, t, pcof)
    
    LHSu = copy(u)
    axpy!(-0.5*dt,ut,LHSu)
    LHSv = copy(v)
    axpy!(-0.5*dt,vt,LHSv)

    LHS_uv = zeros(Float64, 2*N_tot)
    copyto!(LHS_uv, 1,       LHSu, 1, N_tot)
    copyto!(LHS_uv, 1+N_tot, LHSv, 1, N_tot)

    return LHS_uv
end

function LHS_func!(LHS_uv::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{Float64},
        dt::Float64, N_tot::Int64)

    utvt!(ut, vt, u, v, prob, controls, t, pcof)

    LHS_u = view(LHS_uv, 1:N_tot)
    LHS_v = view(LHS_uv, 1+N_tot:2*N_tot)
    
    LHS_u .= u
    axpy!(-0.5*dt, ut, LHS_u)
    LHS_v .= v
    axpy!(-0.5*dt, vt, LHS_v)

    return nothing
end

function LHS_func_adj!(LHS_uv::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{Float64},
        dt::Float64, N_tot::Int64)

    utvt_adj!(ut, vt, u, v, prob, controls, t, pcof)
    
    LHS_u = view(LHS_uv, 1:N_tot)
    LHS_v = view(LHS_uv, 1+N_tot:2*N_tot)

    LHS_u .= u
    axpy!(-0.5*dt, ut, LHS_u)
    LHS_v .= v
    axpy!(-0.5*dt, vt, LHS_v)

    return nothing
end



function LHS_func_order4(utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        system_sym::AbstractMatrix{Float64}, system_asym::AbstractMatrix{Float64}, 
        control_op_sym::AbstractMatrix{Float64}, control_op_asym::AbstractMatrix{Float64},
        control::AbstractControl, t::Float64,
        pcof::AbstractVector{Float64}, dt::Float64, N_tot::Int64)

    utvt!(ut, vt, u, v,
          system_sym, system_asym, control_op_sym, control_op_asym,
          control, t, pcof)
    uttvtt!(utt, vtt, ut, vt, u, v,
            system_sym, system_asym, control_op_sym, control_op_asym,
            control, t, pcof)

    weights = (1,-1/3)
    
    LHSu = copy(u)
    axpy!(-0.5*dt*weights[1],    ut,  LHSu)
    axpy!(-0.25*dt^2*weights[2], utt, LHSu)
    LHSv = copy(v)
    axpy!(-0.5*dt*weights[1],    vt,  LHSv)
    axpy!(-0.25*dt^2*weights[2], vtt, LHSv)

    LHS = zeros(Float64, 2*N_tot)
    copyto!(LHS, 1,       LHSu, 1, N_tot)
    copyto!(LHS, 1+N_tot, LHSv, 1, N_tot)

    return LHS
end

function LHS_func_order4!(LHS_uv::AbstractVector{Float64},
        utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{Float64},
        dt::Float64, N_tot::Int64)

    utvt!(ut, vt, u, v, prob, controls, t, pcof)
    uttvtt!(utt, vtt, ut, vt, u, v, prob, controls, t, pcof)

    weights = (1,-1/3)

    LHS_u = view(LHS_uv, 1:N_tot)
    LHS_v = view(LHS_uv, 1+N_tot:2*N_tot)
    
    LHS_u .= u
    axpy!(-0.5*dt*weights[1],    ut,  LHS_u)
    axpy!(-0.25*dt^2*weights[2], utt, LHS_u)
    LHS_v .= v
    axpy!(-0.5*dt*weights[1],    vt,  LHS_v)
    axpy!(-0.25*dt^2*weights[2], vtt, LHS_v)

    return nothing
end

function LHS_func_order4_adj!(LHS_uv::AbstractVector{Float64},
        utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{Float64},
        dt::Float64, N_tot::Int64)

    utvt_adj!(ut, vt, u, v, prob, controls, t, pcof)
    uttvtt_adj!(utt, vtt, ut, vt, u, v, prob, controls, t, pcof)

    weights = (1,-1/3)

    LHS_u = view(LHS_uv, 1:N_tot)
    LHS_v = view(LHS_uv, 1+N_tot,2*N_tot)
    
    LHS_u .= u
    axpy!(-0.5*dt*weights[1],    ut,  LHS_u)
    axpy!(-0.25*dt^2*weights[2], utt, LHS_u)
    LHS_v .= v
    axpy!(-0.5*dt*weights[1],    vt,  LHS_v)
    axpy!(-0.25*dt^2*weights[2], vtt, LHS_v)

    return nothing
end

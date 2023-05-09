function utvt!(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α)
    #Should I provide the functions p and q? or the values at (t,α)?

    #mul!(Y, A, B) -> Y
    #Calculates the matrix-matrix or matrix-vector product AB and stores the result in Y, overwriting the existing value of Y. Note that Y must not be aliased with either A or B.
    #mul!(C, A, B, α, β) -> C
    #Combined inplace matrix-matrix or matrix-vector multiply-add A B α + C β. The result is stored in C by overwriting it. Note that C must not be aliased with either A or B.
    
    # Non-Memory-Allocating Version (test performance)
    # ut = (Ss + q(t)(a-a†))u - (Ks + p(t)(a+a†))v
    mul!(ut, Ss, u)
    mul!(ut, a_minus_adag, u, q(t,α), 1)
    mul!(ut, Ks, v, -1, 1)
    mul!(ut, a_plus_adag, v, -p(t,α), 1)

    # vt = (Ss + q(t)(a-a†))v + (Ks + p(t)(a+a†))u
    mul!(vt, Ss, v)
    mul!(vt, a_minus_adag, v, q(t,α), 1)
    mul!(vt, Ks, u, 1, 1)
    mul!(vt, a_plus_adag,  u, p(t,α), 1)
end

function uttvtt!(utt, vtt, ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, dpdt, dqdt, t, α)
    ## Make use of utvt!
    utvt!(utt, vtt, ut, vt, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α)
    ## Replicate utvt! explicitly 
    #mul!(utt, Ss, ut)
    #mul!(utt, a_minus_adag, ut, q(t,α), 1)
    #mul!(utt, Ks, vt, -1, 1)
    #mul!(utt, a_plus_adag, vt, -p(t,α), 1)

    mul!(utt, a_minus_adag, u, dqdt(t,α), 1)
    mul!(utt, a_plus_adag, v, -dpdt(t,α), 1)

    ### Replicate utvt! explicitly 
    #mul!(vtt, Ss, vt)
    #mul!(vtt, a_minus_adag, vt, q(t,α), 1)
    #mul!(vtt, Ks, ut, 1, 1)
    #mul!(vtt, a_plus_adag, ut, p(t,α), 1)

    mul!(vtt, a_plus_adag,  u, dpdt(t,α), 1)
    mul!(vtt, a_minus_adag, v, dqdt(t,α), 1)
end


function LHS_func(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α, dt, N_tot)
    utvt!(ut, vt, u, v,
          Ks, Ss, a_plus_adag, a_minus_adag,
          p, q, t, α)
    
    LHSu = copy(u)
    axpy!(-0.5*dt,ut,LHSu)
    LHSv = copy(v)
    axpy!(-0.5*dt,vt,LHSv)

    LHS_uv = zeros(2*N_tot)
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

    LHS = zeros(2*N_tot)
    copyto!(LHS,1,LHSu,1,N_tot)
    copyto!(LHS,1+N_tot,LHSv,1,N_tot)

    return LHS
end

# Maybe I should add a RHS func as well? Just for consistency?

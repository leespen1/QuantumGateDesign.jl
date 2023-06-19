include("./second_tests.jl")
#include("../high_order_2d.jl")
function f(t=0.0, dt=0.1)
    u=[1.0,2.0]
    v=[3.0,4.0]
    ut = zeros(2)
    utt = zeros(2)
    vt = zeros(2)
    vtt = zeros(2)

    prob = this_prob()
    α = 1.0
    t = 0.0

    utvt!(ut,vt, u, v,
          prob.Ks, prob.Ss, prob.a_plus_adag, prob.a_minus_adag,
          prob.p, prob.q, t, α)
    uttvtt!(utt, vtt, ut,vt, u, v,
            prob.Ks, prob.Ss, prob.a_plus_adag, prob.a_minus_adag,
            prob.p, prob.q, prob.dpdt, prob.dqdt, t, 1.0)

    println("New Way")
    println("u, v, ut, vt, utt, vtt")
    display(hcat(u,v,ut,vt,utt,vtt))

    RHSu::Vector{Float64} = zeros(2)
    RHSv::Vector{Float64} = zeros(2)
    RHS_uv::Vector{Float64} = zeros(4)

    weights = [1,1/3]
    copy!(RHSu,u)
    axpy!(0.5*dt*weights[1],ut,RHSu)
    axpy!(0.25*dt^2*weights[2],utt,RHSu)

    copy!(RHSv,v)
    axpy!(0.5*dt*weights[1],vt,RHSv)
    axpy!(0.25*dt^2*weights[2],vtt,RHSv)

    copyto!(RHS_uv,1,RHSu,1,2)
    copyto!(RHS_uv,3,RHSv,1,2)

    println("RHS $RHS_uv")

    LHS = LHS_func_order4(utt, vtt, ut, vt, u, v,
                          prob.Ks, prob.Ss, prob.a_plus_adag, prob.a_minus_adag,
                          prob.p, prob.q, prob.dpdt, prob.dqdt, t, α, dt)
    println("LHS (applied to psi_n, which we wouldn't actually do)")
    println(LHS)
    println()

    # Clear old data, make sure it's not being reused
    u=[1.0,2.0]
    v=[3.0,4.0]
    ut = zeros(2)
    utt = zeros(2)
    vt = zeros(2)
    vtt = zeros(2)

    K_mat(t,α) = prob.Ks + prob.p(t,α)*prob.a_plus_adag
    S_mat(t,α) = prob.Ss + prob.q(t,α)*prob.a_minus_adag
    Kt_mat(t,α) = prob.dpdt(t,α)*prob.a_plus_adag
    St_mat(t,α) = prob.dqdt(t,α)*prob.a_minus_adag
    ut, vt = compute_utvt(S_mat(t, α), K_mat(t, α),u,v)
    utt, vtt = compute_uttvtt(S_mat(t, α), K_mat(t,α), u, v, St_mat(t,α), Kt_mat(t,α), ut, vt)
    println("Old Way")
    println("u, v, ut, vt, utt, vtt")
    display(hcat(u,v,ut,vt,utt,vtt))

    wn = [1,1/3,0] # Fourth Order
    RHSu = u .+ 0.5*dt*(wn[1]*ut + 0.5*dt*(wn[2]*utt)) # Q = I + (1/2)Δt*M̃*Q
    RHSv = v .+ 0.5*dt*(wn[1]*vt + 0.5*dt*(wn[2]*vtt))
    RHS_uv = vcat(RHSu,RHSv)
    println("RHS $RHS_uv \n")

    wnp1 = [1,-1/3,0] # Fourth Order
    LHSu = u .- 0.5*dt*(wnp1[1]*ut + 0.5*dt*(wnp1[2]*utt)) # Q = I + (1/2)Δt*M̃*Q
    LHSv = v .- 0.5*dt*(wnp1[1]*vt + 0.5*dt*(wnp1[2]*vtt))
    LHS = vcat(LHSu, LHSv)
    println("LHS (applied to psi_n, which we wouldn't actually do)")
    println(LHS)
    println()



end

## BEGIN OLD SECTION

function compute_utvt(S::AbstractMatrix{Float64},K::AbstractMatrix{Float64},
        u::Vector{Float64},v::Vector{Float64}
    )::Tuple{Vector{Float64},Vector{Float64}}

    ut = S*u - K*v
    vt = K*u + S*v
    return ut, vt
end


function compute_uttvtt(S::AbstractMatrix{Float64},K::AbstractMatrix{Float64},
        u::Vector{Float64},v::Vector{Float64},
        St::AbstractMatrix{Float64},Kt::AbstractMatrix{Float64},
        ut::Vector{Float64},vt::Vector{Float64}
    )::Tuple{Vector{Float64},Vector{Float64}}

    utt = St*u - Kt*v + S*ut - K*vt
    vtt = Kt*u + St*v + K*ut + S*vt
    return utt, vtt
end


# BEGIN NEW SECTION


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
    mul!(vt, a_plus_adag, u, p(t,α), 1)
end

function uttvtt!(utt, vtt, ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, dpdt, dqdt, t, α)
    ## Make use of utvt!
    #utvt!(utt, vtt, ut, vt, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α)
    # Replicate utvt! explicitly 
    mul!(utt, Ss, ut)
    mul!(utt, a_minus_adag, ut, q(t,α), 1)
    mul!(utt, Ks, vt, -1, 1)
    mul!(utt, a_plus_adag, vt, -p(t,α), 1)

    mul!(utt, a_minus_adag, u, dqdt(t,α), 1)
    mul!(utt, a_plus_adag, v, -dpdt(t,α), 1)

    mul!(vtt, Ss, vt)
    mul!(vtt, a_minus_adag, vt, q(t,α), 1)
    mul!(vtt, Ks, ut, 1, 1)
    mul!(vtt, a_plus_adag, ut, p(t,α), 1)

    mul!(vtt, a_minus_adag, v, dqdt(t,α), 1)
    mul!(vtt, a_plus_adag, u, dpdt(t,α), 1)
end

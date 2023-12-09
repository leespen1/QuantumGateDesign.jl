#=
#
# transition frequency  ω₁/2pi = 4.8 GHz
# Self-Kerr coefficient ξ₁/2pi = 0.22 GHz
# Detuning              Δ₁ = 0
#
# Amplitude Restrictions: maxₜ|d(t;α)| ≤ c_∞, c_∞/2pi = 9 MHz (in rotating frame)
#
# For d-level SWAP gate, carrier wave frequencies are
# Ω₁,ₖ = (k-1)(-ξ₁), k = 1,2,…,N_f,  N_f = d
# (one frequency for each of the essential states)
#
# d T
# 3 140
# 4 215
# 5 265
# 6 425
#
# Number of spline segments D₁ per frequency
# D₁ = 10 for d=3,4,5
# D₁ = 20 for d=6
#
=#

using HermiteOptimalControl
using LinearMaps
using IterativeSolvers
using Plots
using DifferentialEquations
using TimerOutputs
using LinearAlgebra
using BenchmarkTools

include("hermite_map.jl")

function evalhinterp(x,ucof)
    deg_p1 = length(ucof)
    u = zeros(size(x))
    z = ones(size(x))
    for d = 0:deg_p1-1
        u += ucof[1+d]*z
        z = z.*x
    end
    return u
end

function my_sin(t,omega,dt,m)

    nt = length(t)-1
    P_tay = zeros(m+1,nt+1)
    for it = 0:nt
        df = 1.0
        for i = 0:m
            if mod(i,4) == 0
                P_tay[1+i,1+it] =  df*sin(omega*t[it+1])
            elseif mod(i,4) == 1
                P_tay[1+i,1+it] =  df*cos(omega*t[it+1])
            elseif mod(i,4) == 2
                P_tay[1+i,1+it] = -df*sin(omega*t[it+1])
            elseif mod(i,4) == 3
                P_tay[1+i,1+it] = -df*cos(omega*t[it+1])
            end
            df = omega*df*dt/(i+1)
        end
    end
    return P_tay
end

function tpolymul!(c,a,b,q)
    #=
    !
    !  truncated multiplication of degree q polynomials
    !  with coefficients a,b. On return c will contain the
    !  coefficients of the product truncated at degree q
    !
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: q
    INTEGER :: j,k
    DOUBLE PRECISION, DIMENSION(0:q), INTENT(IN) :: a,b
    DOUBLE PRECISION, DIMENSION(0:q), INTENT(OUT) :: c
    !
    c=0.d0
    DO j=0,q
    DO k=0,j
    c(j)=c(j)+a(k)*b(j-k)
    END DO
    END DO
    END SUBROUTINE tpolymul
    =#
    c .= 0.0
    for j = 0:q
        for k = 0:j
            c[1+j]=c[1+j]+a[1+k]*b[1+j-k]
        end
    end
    
end


function H_carrier(nt,nt_ctrl,m)
    
    # This function studies the approximation of
    # sin(w_low*t)*sin(w_high*t)
    # when sin(w_low*t) is approxmated by a Hermite interpolant
    # and sin(w_high*t) is known
    w_high = 50.0
    w_low  = 1.0

    tend = 2*pi
    dt = tend/nt
    dt_ctrl = tend/nt_ctrl

    t_ctrl = zeros(nt_ctrl+1)
    th_ctrl = zeros(nt_ctrl)
    for i = 0:nt_ctrl
        t_ctrl[1+i] = i*dt_ctrl
    end
    th_ctrl .= t_ctrl[1:nt_ctrl] .+ 0.5*dt_ctrl

    A_tay = my_sin(t_ctrl,w_low,dt_ctrl,m)    
    tloc_ctrl = zeros(nt+1)
    idx_ctrl = zeros(Int64,nt+1)
    # We find the map between the timestep index and the control array index
    j = 1
    for it = 0:nt
        t = it*dt
        for j = 1:nt_ctrl
            t_ctrl = j*dt_ctrl
            if (j-1)*dt_ctrl <= t && t < j*dt_ctrl
                idx_ctrl[1+it] = j
            end
        end
    end
    # protect for roundoff
    idx_ctrl[nt+1] = nt_ctrl
    # find the normalized local coordinate (between -1/2 and 1/2)
    for it = 0:nt
        t = it*dt
        t_c = (idx_ctrl[1+it]-0.5)*dt_ctrl
        tloc_ctrl[1+it] = (t-t_c)/dt_ctrl
    end
    #
    #
    Hmat = zeros(2*m+2,2*m+2)
    Hermite_map!(Hmat,m,0.0,1.0,0.5,0)
    neval = 31
    
    tt = collect(LinRange(0.0,tend,1001))
    # 
    pl = plot(tt,sin.(w_low.*tt).*sin.(w_high.*tt),color=:lightblue,lw=2,label = "Function")
    #pl = plot(tt,sin.(w_low.*tt),color=:lightblue,lw=4,label = "Function")
    a_int = zeros(2*m+2)
    b_tay = zeros(2*m+2)
    c_tay = zeros(2*m+2)
    ucof = zeros(2*m+2)
    ploc = zeros(2*m+2)
    ctrl_scl = zeros(2*m+2)
    df = 1.0
    for i = 0:2*m+1
        ctrl_scl[1+i] = df
        df = df/(dt_ctrl/dt)
    end
    for it = 0:nt
        view(ucof,1:m+1) .= view(A_tay,:,idx_ctrl[1+it])
        view(ucof,m+2:2*m+2) .= view(A_tay,:,idx_ctrl[1+it]+1)
        a_int = Hmat*ucof
        extrapolate!(a_int,tloc_ctrl[1+it],2*m+1,ploc)
        b_tay = my_sin(it*dt,w_high,dt_ctrl,2*m+1)
        tpolymul!(c_tay,a_int,b_tay,2*m+1)
        z = collect(LinRange(-0.5*dt/dt_ctrl,0.5*dt/dt_ctrl,neval))
        uplot = evalhinterp(z,c_tay)
        plot!(pl,it*dt .+ dt_ctrl*z,uplot,color=:black,lw=1,label = :none,xlims=(0.0,tend))
        plot!(pl,[it*dt],[sin.(w_low.*it*dt).*sin.(w_high.*it*dt)],color=:red,lw=0,label = :none,xlims=(0.0,tend),marker = :star)
    end
    return pl
end

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

function w_to_F!(F,W,w,H0,P,Q,p_tay,q_tay,m,wjp)
    view(W,:,1+0,) .= w
    for j = 0:m-1
        wjp .= 0.0
        for i = 0:j
            mul!(wjp,P,view(W,:,1+i),p_tay[1+j-i],1.0)
            mul!(wjp,Q,view(W,:,1+i),q_tay[1+j-i],1.0)
        end
        mul!(wjp,H0,view(W,:,1+j),1.0/(1+j),1.0/(1+j))
        view(W,:,2+j) .= wjp
    end
    F .= view(W,:,2:m+1)
end


function compute_b!(b,w,F,c,dt,m)
    b .= c[1].*w
    dft = dt
    for j = 1:m
        b .= b .+ (dft*c[1+j]).*view(F,:,j)
        dft = dft*dt
    end
end

function Aw(w,H0,P,Q,p_tay,q_tay,m,c,dt,F,W,F1sum)
    w_to_F!(F,W,w,H0,P,Q,p_tay,q_tay,m,F1sum)
    df = -1.0
    dft = dt
    F1sum .= 0.0
    for j = 1:m
        F1sum .= F1sum .+ (dft*df*c[1+j]).*view(F,:,j)
        df *= -1.0
        dft = dft*dt
    end
    F1sum .+= w
    return F1sum
end


function HIT_solve(nt,nt_ctrl,m ; d=6, N_guard=1)

    N_ess_levels = d+1
    tf = 2.0*pi
    nsteps = nt
    # Frequencies in GHz, *not angular*
    detuning_frequency = 0.0
    self_kerr_coefficient =  0.22
    prob = rotating_frame_qubit(
        N_ess_levels,
        N_guard,
        tf=tf,
        detuning_frequency=detuning_frequency,
        self_kerr_coefficient=self_kerr_coefficient,
        nsteps = nsteps
    )

    H0 = [prob.system_asym prob.system_sym
          -prob.system_sym prob.system_asym]
    Q = [prob.asym_operators[1] 0*prob.asym_operators[1]
         0*prob.asym_operators[1] prob.asym_operators[1]]
    P = [0*prob.sym_operators[1] prob.sym_operators[1]
         -prob.sym_operators[1] 0*prob.sym_operators[1]]

    n_w = 2*prob.N_tot_levels
    tend = tf
    dt = tend/nt
    dt_ctrl = tend/nt_ctrl

    # initial data
    kk = 1
    w0 = zeros(n_w)
    w0[kk] = 1.0
    # Coefficients in method
    c = zeros(m+1)
    for j = 0:m
        c[1+j] = (factorial(m)*factorial(2*m-j))/(factorial(2*m)*factorial(m-j))
    end

    # These arrays hold the controls
    # For now we use the convention that these are Taylor coefficients that are
    # scaled with dt_ctrl^k/k! just as in Chides
    P_tay = zeros(m+1,nt_ctrl+1)
    Q_tay = zeros(m+1,nt_ctrl+1)
    # we consider the taylor expansion of the functions
    # p(t) = sin(t)
    # q(t) = cos(t)
    for it = 0:nt_ctrl
        t = it*dt_ctrl
        df = 1.0
        for i = 0:m
            if mod(i,4) == 0
                Q_tay[1+i,1+it] =  df*cos(t)
                P_tay[1+i,1+it] =  df*sin(t)
            elseif mod(i,4) == 1
                Q_tay[1+i,1+it] = -df*sin(t)
                P_tay[1+i,1+it] =  df*cos(t)
            elseif mod(i,4) == 2
                Q_tay[1+i,1+it] = -df*cos(t)
                P_tay[1+i,1+it] = -df*sin(t)
            elseif mod(i,4) == 3
                Q_tay[1+i,1+it] =  df*sin(t)
                P_tay[1+i,1+it] = -df*cos(t)
            end
            df = df*dt_ctrl/(i+1)
        end
    end
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
        t_ctrl = (idx_ctrl[1+it]-0.5)*dt_ctrl
        tloc_ctrl[1+it] = (t-t_ctrl)/dt_ctrl
    end
    # memory for local expansions
    q_tay = zeros(m+1)
    p_tay = zeros(m+1)
    # Scaling for removing timestep scaling to conform with
    # solver scaling
    ctrl_scl = zeros(m+1)
    df = 1.0
    for i = 0:m
        ctrl_scl[1+i] = df
        df = df/dt_ctrl
    end
    #
    Hmat = zeros(2*m+2,2*m+2)
    Hermite_map!(Hmat,m,0.0,1.0,0.5,0)
    # Solution array
    W = zeros(n_w,m+1,nt+1)
    T = LinRange(0,tend,nt+1)

    q = 2*m+1 # for extrapolation
    # work arrays
    ucof = zeros(2*m+2)
    uint = zeros(2*m+2)
    ploc = zeros(q+1)
    F0 = zeros(n_w,m)
    Wtay = zeros(n_w,m+1)
    w_work = zeros(n_w)
    F_work = zeros(n_w,m)
    W_tay_work = zeros(n_w,m+1)
    b = zeros(n_w)

    tval, t1, bytes, gctime, memallocs = @timed begin
        # Interpolate and shift the control before entering time-loop
        it = 0
        view(ucof,1:m+1) .= view(Q_tay,:,idx_ctrl[1+it])
        view(ucof,m+2:2*m+2) .= view(Q_tay,:,idx_ctrl[1+it]+1)
        mul!(uint,Hmat,ucof)
        extrapolate!(uint,tloc_ctrl[1+it],q,ploc)
        q_tay .= ctrl_scl.*view(uint,1:m+1)
        view(ucof,1:m+1) .= view(P_tay,:,idx_ctrl[1+it])
        view(ucof,m+2:2*m+2) .= view(P_tay,:,idx_ctrl[1+it]+1)
        mul!(uint,Hmat,ucof)
        extrapolate!(uint,tloc_ctrl[1+it],q,ploc)
        p_tay .= ctrl_scl.*view(uint,1:m+1)

        w_to_F!(F0,Wtay,w0,H0,P,Q,p_tay,q_tay,m,w_work)
        W[:,:,1+it] .= Wtay
        compute_b!(b,w0,F0,c,dt,m)
        # Linear map
        Awcb(z) = Aw(z,H0,P,Q,p_tay,q_tay,m,c,dt,F_work,W_tay_work,w_work)
        HM_MAT = LinearMap(Awcb,length(w0))
        w1 = copy(w0)
        for it = 1:nt
            t = it*dt
            # Interpolate and shift the control
            view(ucof,1:m+1) .= view(Q_tay,:,idx_ctrl[1+it])
            view(ucof,m+2:2*m+2) .= view(Q_tay,:,idx_ctrl[1+it]+1)
            mul!(uint,Hmat,ucof)
            extrapolate!(uint,tloc_ctrl[1+it],q,ploc)
            q_tay .= ctrl_scl.*view(uint,1:m+1)
            view(ucof,1:m+1) .= view(P_tay,:,idx_ctrl[1+it])
            view(ucof,m+2:2*m+2) .= view(P_tay,:,idx_ctrl[1+it]+1)
            mul!(uint,Hmat,ucof)
            extrapolate!(uint,tloc_ctrl[1+it],q,ploc)
            p_tay .= ctrl_scl.*view(uint,1:m+1)
            df = dt
            # Experimented with a better starting guess than zero
            # Reduces the number of iterations significantly when dt is small
            for idt = 1:m
                w1 .+= df*view(W,:,1+idt,it)
                df *= dt
            end
            w1,history1 = gmres!(w1,HM_MAT,b,abstol = 1e-14,reltol=1e-14,log=true,initially_zero=false)
            # Plain vanilla solve
            # w1,history2 = gmres(HM_MAT,b,abstol = 1e-14,reltol=1e-14,log=true)
            # println(n_w," : ",history1.iters," : ",history2.iters," : ")
            # Shuffle timesteps and compute stuff neded in the linear system
            w0 .= w1
            w_to_F!(F0,Wtay,w0,H0,P,Q,p_tay,q_tay,m,w_work)
            W[:,:,1+it] .= Wtay
            compute_b!(b,w0,F0,c,dt,m)
        end
    end
    return T,W,H0,Q,P,w0,tend,t1
end

function get_err(nt,nt_ctrl,m)

    T,W,H0,Q,P,w0,tend,t1 = HIT_solve(nt,nt_ctrl,m)
    # Compare with DifferentialEquations.jl
    f(w, p, t) = H0*w .+ sin(t).*(P*w) .+ cos(t).*(Q*w)
    #f(w, p, t) = H0*w .+ (P*w) .+ (Q*w)
    tspan = (0.0, tend)
    w0 .= 0.0
    w0[1] = 1.0
    prob_DEQ = ODEProblem(f, w0, tspan)
    sol = solve(prob_DEQ,Tsit5(), reltol = 1e-14, abstol = 1e-14)

    dt = tend / nt

    scale = zeros(Float64,2*(m+1),2*(m+1))
    df = 1.0
    for i = 0:2*m+1
        scale[1+i,1+i] = df
        df = df*dt
    end

    neval = 101
    z = collect(LinRange(-0.5,0.5,neval))
    if false
        pl = plot(sol,lw=2,color=:yellow,label = :none)
        Hmat = zeros(2*m+2,2*m+2)
        Hermite_map!(Hmat,m,0.0,1.0,0.5,0)
        ucof = zeros(2*m+2)
        for j = 1:5
            for i = 1:nt
                ifield = j
                ucof[1:m+1] = scale[1:m+1,1:m+1]*W[ifield,:,i]
                ucof[m+2:2*m+2] = scale[1:m+1,1:m+1]*W[ifield,:,i+1]
                uint = Hmat*ucof
                uplot = evalhinterp(z,uint)
                ifield = j+5
                ucof[1:m+1] = scale[1:m+1,1:m+1]*W[ifield,:,i]
                ucof[m+2:2*m+2] = scale[1:m+1,1:m+1]*W[ifield,:,i+1]
                uint = Hmat*ucof
                vplot = evalhinterp(z,uint)
                plot!(pl,T[i] .+ dt*(0.5 .+ z),uplot,color=:black,lw=1,label = :none)
                plot!(pl,T[i] .+ dt*(0.5 .+ z),vplot,color=:black,lw=1,label = :none)
            end
        end
    end

    if false
        pl1 = plot(sol.t,sol[1,:],color=:red,lw=2,label = :none)
        j = 1
        for i = 1:nt
            ifield = j
            ucof[1:m+1] = scale[1:m+1,1:m+1]*W[ifield,:,i]
            ucof[m+2:2*m+2] = scale[1:m+1,1:m+1]*W[ifield,:,i+1]
            uint = Hmat*ucof
            uplot = evalhinterp(z,uint)
            plot!(pl1,T[i] .+ dt*(0.5 .+ z),uplot,color=:black,lw=1,label = :none)
        end
    end
    maxerr = maximum(abs.(sol[:,end] .- W[:,1,end]))
    #display(maxerr)
    #display(pl)
    return maxerr,t1
end

function main(nt_ctrl)
    pl = plot(xaxis=:log, yaxis=:log)
    c = [5.0, 2.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]
    for m = 1:8
        e = [];
        dt = [];
        for nt = 5:75
            err, = get_err(nt,nt_ctrl,m)
            e = [e ; err]
            dt = [dt ; 2*pi/nt]
        end
        plot!(pl,dt,e,color = :red,
              xaxis=:log, yaxis=:log,lw=2,label = :none)
        plot!(pl,dt,c[m]*dt.^(2*m),
              linestyle=:dash,lw=1,color = :black,label = :none,
              xaxis=:log, yaxis=:log , ylims=(1e-16,10))
    end
    return pl
end

function main_timing()
    tt = []
    nsamples = 10
    ee = []
    for i = 1:nsamples
        err1,t1 = get_err(20000,50,1)
        err2,t2 = get_err(230,50,2)
        err3,t3 = get_err(67,50,3)
        err4,t4 = get_err(35,50,4)
        err5,t5 = get_err(23,50,5)
        err6,t6 = get_err(17,50,6)
        err7,t7 = get_err(14,50,7)
        err8,t8 = get_err(12,50,8)
        ttt = [t1, t2, t3, t4, t5, t6, t7, t8]
        ee = [err1, err2, err3, err4, err5, err6, err7, err8]
        tt = [tt ; ttt]
    end
    return mean(reshape(tt,8,nsamples),dims=2),ee
end

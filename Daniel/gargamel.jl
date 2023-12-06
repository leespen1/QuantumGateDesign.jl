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

function w_to_F(w,H0,P,Q,p_tay,q_tay,m)
    W = zeros(length(w),m+1)
    W[:,1+0] .= w
    for j = 0:m-1
        wjp = zeros(size(w))
        for i = 0:j
            wjp .+= p_tay[1+j-i].*P*W[:,1+i]
            wjp .+= q_tay[1+j-i].*Q*W[:,1+i]
        end
        W[:,2+j] .= 1/(1+j).*(H0*W[:,1+j] .+ wjp)
    end
    return W[:,2:m+1],W
end

function compute_b(w,F,c,dt,m)
    b = c[1]*w
    dft = dt;
    for j = 1:m
        b .= b .+ dft*F[:,j]*c[1+j]
        dft = dft*dt
    end
    return b
end

function Aw(w,H0,P,Q,p_tay,q_tay,m,c,dt)
    F, = w_to_F(w,H0,P,Q,p_tay,q_tay,m)
    F1sum = zeros(length(w))
    df = -1.0
    dft = dt
    for j = 1:m
        F1sum .= F1sum .+ dft*df*F[:,j]*c[1+j]
        df *= -1.0
        dft = dft*dt
    end
    return w .+ F1sum
end


function main(nt,nt_ctrl,m ; d=3, N_guard=1)

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
    kk = 1
    w0 = zeros(n_w)
    w0[kk] = 1.0
    

    f(w, p, t) = H0*w .+ sin(t).*(P*w) .+ cos(t).*(Q*w)
    tspan = (0.0, tend)
    prob_DEQ = ODEProblem(f, w0, tspan)
    sol = solve(prob_DEQ,Tsit5(), reltol = 1e-14, abstol = 1e-14)
    pl0 = plot(sol)
    
    c = zeros(m+1)
    for j = 0:m
        c[1+j] = (factorial(m)*factorial(2*m-j))/(factorial(2*m)*factorial(m-j))
    end
    
    dt_ctrl = tend/nt_ctrl
    # These arrays hold the controls 
    # For now we use the convention that these are Taylor coefficients that are
    # scaled with dt_ctrl^k/k! just as in Chides
    P_tay = zeros(m+1,nt_ctrl+1)
    Q_tay = zeros(m+1,nt_ctrl+1)

    #P_tay = randn(Float64,(m+1,nt_ctrl+1))
    #Q_tay = randn(Float64,(m+1,nt_ctrl+1))

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
        t_ctrl = j*dt_ctrl
        if t > t_ctrl
            j = j+1
        end
        idx_ctrl[1+it] = j
    end
    # protect for roundoff
    idx_ctrl[nt+1] = nt_ctrl
    # find the normalized local coordinate (between -1/2 and 1/2)
    for it = 0:nt
        t = it*dt
        t_ctrl = (idx_ctrl[1+it]-0.5)*dt_ctrl
        tloc_ctrl[1+it] = (t-t_ctrl)/dt_ctrl
    end
    
    q_tay = zeros(m+1)
    p_tay = zeros(m+1)
        
    ctrl_scl = zeros(m+1)
    df = 1.0
    for i = 0:m
        ctrl_scl[1+i] = df
        df = df/dt_ctrl
    end

    Hmat = zeros(2*m+2,2*m+2)
    Hermite_map!(Hmat,m,0.0,1.0,0.5,0)

    W = zeros(length(w0),m+1,nt+1)
    T = LinRange(0,tend,nt+1)
        
    q = 2*m+1
    ucof = zeros(2*m+2)
    it = 0
    ucof[1:m+1]     = Q_tay[:,idx_ctrl[1+it]]
    ucof[m+2:2*m+2] = Q_tay[:,idx_ctrl[1+it]+1]
    uint = Hmat*ucof
    extrapolate!(uint,tloc_ctrl[1+it],q)
    q_tay .= ctrl_scl.*uint[1:m+1]
    ucof[1:m+1]     = P_tay[:,idx_ctrl[1+it]]
    ucof[m+2:2*m+2] = P_tay[:,idx_ctrl[1+it]+1]
    uint = Hmat*ucof
    extrapolate!(uint,tloc_ctrl[1+it],q)
    p_tay .= ctrl_scl.*uint[1:m+1]

    F0,Wtay = w_to_F(w0,H0,P,Q,p_tay,q_tay,m)
    W[:,:,1] .= Wtay
    b = compute_b(w0,F0,c,dt,m)

    Awcb(z) = Aw(z,H0,P,Q,p_tay,q_tay,m,c,dt)
    HM_MAT = LinearMap(Awcb,length(w0))
    
    for it = 1:nt
        t = it*dt
        ucof[1:m+1]     = Q_tay[:,idx_ctrl[1+it]]
        ucof[m+2:2*m+2] = Q_tay[:,idx_ctrl[1+it]+1]
        uint = Hmat*ucof
        extrapolate!(uint,tloc_ctrl[1+it],q)
        q_tay .= ctrl_scl.*uint[1:m+1]
        ucof[1:m+1]     = P_tay[:,idx_ctrl[1+it]]
        ucof[m+2:2*m+2] = P_tay[:,idx_ctrl[1+it]+1]
        uint = Hmat*ucof
        extrapolate!(uint,tloc_ctrl[1+it],q)

        p_tay .= ctrl_scl.*uint[1:m+1]
        w1,history = gmres(HM_MAT,b,reltol=1e-12,log=true)
        # Shuffle timesteps and compute stuff neded in the linear system
        w0 = w1
        F0,Wtay = w_to_F(w0,H0,P,Q,p_tay,q_tay,m)
        W[:,:,1+it] .= Wtay
        b = compute_b(w0,F0,c,dt,m)
    end

    scale = zeros(Float64,2*(m+1),2*(m+1))
    df = 1.0
    for i = 0:2*m+1
        scale[1+i,1+i] = df
        df = df*dt
    end

    neval = 101
    pl = plot(T,0.0 .* sqrt.(W[1,1,:].^2+W[6,1,:].^2),lw=2,marker=:star,label=:none)
    

    z = collect(LinRange(-0.5,0.5,neval))
    for j = 1:prob.N_tot_levels
        for i = 1:nt
            ifield = j
            ucof[1:m+1] = scale[1:m+1,1:m+1]*W[ifield,:,i]
            ucof[m+2:2*m+2] = scale[1:m+1,1:m+1]*W[ifield,:,i+1]
            uint = Hmat*ucof
            uplot = evalhinterp(z,uint)
            ifield = j+prob.N_tot_levels
            ucof[1:m+1] = scale[1:m+1,1:m+1]*W[ifield,:,i]
            ucof[m+2:2*m+2] = scale[1:m+1,1:m+1]*W[ifield,:,i+1]
            uint = Hmat*ucof
            vplot = evalhinterp(z,uint)
            #plot!(pl,T[i] .+ dt*(0.5 .+ z),sqrt.(uplot.^2 .+ vplot.^2),color=:black,lw=1,label = :none)
            plot!(pl,T[i] .+ dt*(0.5 .+ z),uplot,color=:black,lw=1,label = :none)
            plot!(pl,T[i] .+ dt*(0.5 .+ z),vplot,color=:red,lw=1,label = :none)
        end
    end


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
    maxerr = maximum(abs.(sol[:,end] .- W[:,1,end]))
    return T,W,sol,pl,pl0,pl1,maxerr
end

function get_err(nt,nt_ctrl,m)
    #nt = 50
    #nt_ctrl = 2
    #m = 1
    T,W,sol,pl,pl0,pl1,maxerr = main(nt,nt_ctrl,m)
    #pl = main()
    display(pl1)
    println("Maximum error is: ",maxerr)
    return nothing
end

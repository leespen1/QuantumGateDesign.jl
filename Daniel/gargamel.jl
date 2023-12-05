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


function main(;d=3, N_guard=1, D1=missing, tf=missing, nsteps=missing)

    N_ess_levels = d+1
    # Set up problem (for d = 1, 2, I just loosely followed pattern for nsteps and duration)

    # If everything else is in GHz, then I think times should be in ns
    duration_defaults = [50.0, 100.0, 140.0, 215.0, 265.0, 425.0]
    if ismissing(tf)
        tf = duration_defaults[d]
    end
    nsteps_defaults = [3500, 7000, 14787, 37843, 69962, 157082] # Defaults, as in Juqbox
    if ismissing(nsteps)
        nsteps = nsteps_defaults[d]
    end

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
    
    m = 7
    H0 = [prob.system_asym prob.system_sym
          -prob.system_sym prob.system_asym]
    Q = [prob.asym_operators[1] 0*prob.asym_operators[1]
         0*prob.asym_operators[1] prob.asym_operators[1]]
    P = [0*prob.sym_operators[1] prob.sym_operators[1]
         -prob.sym_operators[1] 0*prob.sym_operators[1]]
    
    c = zeros(m+1)
    for j = 0:m
        c[1+j] = (factorial(m)*factorial(2*m-j))/(factorial(2*m)*factorial(m-j))
    end

    n_w = 2*prob.N_tot_levels
    tend = tf
    nt = 11
    dt = tend/nt
    w0 = zeros(n_w)
    w0[1] = 1.0
    W = zeros(length(w0),m+1,nt+1)
    T = LinRange(0,tend,nt+1)
    
    q_tay = zeros(m+1)
    p_tay = zeros(m+1)
    p_tay[1] = 1.0/2.0/pi
    #=    
    ctrl_scl = zeros(m+1)
    df = 1.0
    for i = 0:m
        ctrl_scl[1+i] = df
        df = df/tend
    end
    
    t = 0.0
    t_int = -0.5 + t/tend
    
    q = 2*m+1
    ucof = copy(a_ctrl_I)
    extrapolate!(ucof,t_int,q)
    
    a_tay[1:m+1] .= (ctrl_scl.*ucof[1:m+1])   
=#
    F0,Wtay = w_to_F(w0,H0,P,Q,p_tay,q_tay,m)
    W[:,:,1] .= Wtay
    b = compute_b(w0,F0,c,dt,m)

    Awcb(z) = Aw(z,H0,P,Q,p_tay,q_tay,m,c,dt)
    HM_MAT = LinearMap(Awcb,length(w0))
    
    for it = 1:nt
        t = it*dt
        #t_int = -0.5 + t/tend
        #q = 2*m+1
        #ucof = copy(a_ctrl_I)
        #extrapolate!(ucof,t_int,q)
        #a_tay[1:m+1] .= (ctrl_scl.*ucof[1:m+1])   
        w1,history = gmres(HM_MAT,b,reltol=1e-12,log=true)
        # Shuffle timesteps and compute stuff neded in the linear system
        w0 = w1
        F0,Wtay = w_to_F(w0,H0,P,Q,p_tay,q_tay,m)
        W[:,:,1+it] .= Wtay
        b = compute_b(w0,F0,c,dt,m)
    end

    Hmat = zeros(2*m+2,2*m+2)
    Hermite_map!(Hmat,m,0.0,1.0,0.5,0)
    #
    scale = zeros(Float64,2*(m+1),2*(m+1))
    df = 1.0
    for i = 0:2*m+1
        scale[1+i,1+i] = df
        df = df*dt
    end

    neval = 101
    pl = plot(T,sqrt.(W[1,1,:].^2+W[6,1,:].^2),lw=2,marker=:star)
    
    ucof = zeros(2*m+2)
    z = collect(LinRange(-0.5,0.5,neval))
    for i = 1:nt
        ifield = 1
        ucof[1:m+1] = scale[1:m+1,1:m+1]*W[ifield,:,i]
        ucof[m+2:2*m+2] = scale[1:m+1,1:m+1]*W[ifield,:,i+1]
        uint = Hmat*ucof
        uplot = evalhinterp(z,uint)
        ifield = 6
        ucof[1:m+1] = scale[1:m+1,1:m+1]*W[ifield,:,i]
        ucof[m+2:2*m+2] = scale[1:m+1,1:m+1]*W[ifield,:,i+1]
        uint = Hmat*ucof
        vplot = evalhinterp(z,uint)
        plot!(pl,T[i] .+ dt*(0.5 .+ z),sqrt.(uplot.^2 .+ vplot.^2),color=:black,lw=1,label = :none)
    end
    return T,W,pl
end

T,W,pl = main()
display(pl)

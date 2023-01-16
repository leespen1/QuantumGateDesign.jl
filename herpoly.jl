using Polynomials
using PyPlot
using TaylorSeries
using LinearAlgebra
using LinearMaps
using IterativeSolvers
using DifferentialEquations

mutable struct H_stuff
    S
    K
    KKerr
    p
    pt
    ptt
    q
    qt
    qtt
    wl
    wr
    dt
    function H_stuff(S,K,KKerr,p,pt,ptt,q,qt,qtt,wl,wr,dt)
        new(S,K,KKerr,p,pt,ptt,q,qt,qtt,wl,wr,dt)
    end
end


function build_paux(m)
    # Build the auxiliary polynomials
    ptmp = fromroots(1*ones(Int64,m+1))/(-2)^(m+1)
    paux = [ptmp]
    for k = 1:m
        paux=[paux; Polynomial(fromroots(-ones(Int64,k))/factorial(k))*ptmp]
    end
    return paux
end

function build_HLp(m)
    paux = build_paux(m)
    HLp = Vector{Polynomial{Float64}}(undef,m+1)
    HLp[1+m] = paux[1+m]
    # Loop backwards in k
    for k = m-1:-1:0
        HLp[1+k] = paux[1+k]
        for nu = k+1:m
            val = Polynomials.derivative(paux[1+k],nu)(-1.0)
            HLp[1+k] = HLp[1+k] - val*HLp[1+nu]
        end
    end
    return HLp
end


function generate_Hermite_weights(m)
    P = build_HLp(m)
    w = zeros(m+1)
    for i = 1:m+1
        z = Polynomials.integrate(P[i])
        w[i] = z(1.0)-z(-1.0);
    end
    return w
end


function compute_utvt(u,v,p,q,S,K,KKerr)
    ut = S*(q*u)-KKerr*v-K*(p*v)
    vt = S*(q*v)+KKerr*u+K*(p*u)
    return ut,vt
end

function compute_uttvtt(u,v,ut,vt,p,q,pt,qt,S,K,KKerr)
    utt = S*(qt*u+q*ut)-KKerr*vt-K*(pt*v+p*vt)
    vtt = S*(qt*v+q*vt)+KKerr*ut+K*(pt*u+p*ut)
    return utt,vtt
end

function compute_utttvttt(u,v,ut,vt,utt,vtt,p,q,pt,qt,ptt,qtt,S,K,KKerr)
    uttt = S*(qtt*u+2*qt*ut+q*utt)-KKerr*vtt-K*(ptt*v+2*pt*vt+p*vtt)
    vttt = S*(qtt*v+2*qt*vt+q*vtt)+KKerr*utt+K*(ptt*u+2*pt*ut+p*utt)
    return uttt,vttt
end

function left_matrix_action(w,Hp)

    dt = Hp.dt
    wl = Hp.wl
    wr = Hp.wr
    p = Hp.p
    q = Hp.q
    pt = Hp.pt
    qt = Hp.qt
    ptt = Hp.ptt
    qtt = Hp.qtt
    S = Hp.S
    K = Hp.K
    KKerr = Hp.KKerr
    
    n = Int(length(w)/2)
    u = w[1:n]
    v = w[n+1:2*n]

    ut,vt = compute_utvt(u,v,p,q,S,K,KKerr)
    utt,vtt = compute_uttvtt(u,v,ut,vt,p,q,pt,qt,S,K,KKerr)
    uttt,vttt = compute_utttvttt(u,v,ut,vt,utt,vtt,p,q,pt,qt,ptt,qtt,S,K,KKerr)
    u = u.+0.5*dt*(wl[1]*ut .+ wl[2]*utt + wl[3]*uttt)
    v = v.+0.5*dt*(wl[1]*vt .+ wl[2]*vtt + wl[3]*vttt)
    return [u;v]
    
end

function right_matrix_action(w,Hp)

    dt = Hp.dt
    wl = Hp.wl
    wr = Hp.wr
    p = Hp.p
    q = Hp.q
    pt = Hp.pt
    qt = Hp.qt
    ptt = Hp.ptt
    qtt = Hp.qtt
    S = Hp.S
    K = Hp.K
    KKerr = Hp.KKerr

    n = Int(length(w)/2)
    u = w[1:n]
    v = w[n+1:2*n]

    ut,vt = compute_utvt(u,v,p,q,S,K,KKerr)
    utt,vtt = compute_uttvtt(u,v,ut,vt,p,q,pt,qt,S,K,KKerr)
    uttt,vttt = compute_utttvttt(u,v,ut,vt,utt,vtt,p,q,pt,qt,ptt,qtt,S,K,KKerr)
    u = u .- 0.5*dt*(wr[1]*ut .+ wr[2]*utt + wr[3]*uttt)
    v = v .- 0.5*dt*(wr[1]*vt .+ wr[2]*vtt + wr[3]*vttt)
    return [u;v]
    
end

function odefun(w,par,t)
    
    n = Int(length(w)/2)
    u = w[1:n]
    v = w[n+1:2*n]
    p_amp = 1.0
    p = p_amp*sin(t)
    q_amp = 1.0
    q = q_amp*sin(t)
    ut,vt = compute_utvt(u,v,p,q,par.S,par.K,par.KKerr)
    return [ut;vt]
    
end

function get_qall(sol)

    N = length(sum.(sol.u[1]))
    Nt = length(sol.t)
    qall = zeros(N,Nt)
    for i = 1:length(sol.t)
        w = sum.(sol.u[i])
        qall[:,i] = w
    end
    return qall
end

function get_matrix(nt)

    m = 2
    T = 100.0
    dt = T/nt

    # wl = [1.0, 0.4, 2.0/30]
    wl = generate_Hermite_weights(m)
    wr = wl.*(-1).^(0:m)
    for k = 1:m
        wr[1+k] = wr[1+k]*(dt/2)^k
        wl[1+k] = wl[1+k]*(dt/2)^k
    end

    Nstates = 3
    xi = 0.2
    
    amat = Bidiagonal(zeros(Nstates),sqrt.(collect(1:Nstates-1)),:U)
    adag = amat'
    
    K = amat+adag
    KKerr = -xi/2*adag*adag*amat*amat
    S = amat-adag

    Hpar = H_stuff(S,K,KKerr,0.0,0.0,0.0,0.0,0.0,0.0,wl,wr,dt)
    par = (S=S,K=K,KKerr=KKerr)
    
    p_amp = 1.0
    pfun(t) = p_amp*sin(t)
    ptfun(t) = p_amp*cos(t)
    pttfun(t) = -p_amp*sin(t)
    
    q_amp = 1.0
    qfun(t) = q_amp*sin(t)
    qtfun(t) = q_amp*cos(t)
    qttfun(t) = -q_amp*sin(t)
    
    w = zeros(2*Nstates)
    w[1] = 1.0
    wp = w
    wpl = zeros(2*Nstates,nt+1)
    wpl[:,1] = w
    for it = 1:nt
        t = (it-1)*dt
        Hpar.p = pfun(t)
        Hpar.q = qfun(t)
        Hpar.pt = ptfun(t)
        Hpar.qt = qtfun(t)
        Hpar.ptt = pttfun(t)
        Hpar.qtt = qttfun(t)
        
        b = left_matrix_action(w,Hpar)
        t = t+dt
        Hpar.p = pfun(t)
        Hpar.q = qfun(t)
        Hpar.pt = ptfun(t)
        Hpar.qt = qtfun(t)
        Hpar.ptt = pttfun(t)
        Hpar.qtt = qttfun(t)

        right_matrix_action_wrap(w) = right_matrix_action(w,Hpar)
        AA = LinearMap(right_matrix_action_wrap,2*Nstates) 
        
        wp,gmres_h = gmres!(wp,AA,b,log=true,reltol=1e-12)
        w = wp
        wpl[:,1+it] = w
    end
    tt = dt*collect(0:nt)

    
    q0 = zeros(2*Nstates)
    q0[1] = 1.0
    tspan = (0.0,T)
    prob = ODEProblem(odefun,q0,tspan,par)
    sol = solve(prob, Midpoint(), reltol=1e-8, abstol=1e-8,saveat=dt)
    sol2 = solve(prob, Midpoint(), reltol=1e-8, abstol=1e-8)
    println("Midpoint uses ",length(sol2.t), " timesteps")
    #qall = get_qall(sol)

    #=
    HHerm = zeros(nt+1)
    HODE = zeros(length(sol.t))
    for it = 1:nt+1
        t = tt[it]
        SS = qfun(t)*S
        KK = KKerr+pfun(t)*K
        uu = wpl[1:Nstates,it]
        vv = wpl[Nstates+1:2*Nstates,it]
        HHerm[it] = uu'*SS*vv+0.5*uu'*KK*uu+0.5*vv'*KK*vv
    end
    for it = 1:length(sol.t)
        t = sol.t[it]
        SS = qfun(t)*S
        KK = KKerr+pfun(t)*K
        uu = qall[1:Nstates,it]
        vv = qall[Nstates+1:2*Nstates,it]
        HODE[it] = uu'*SS*vv+0.5*uu'*KK*uu+0.5*vv'*KK*vv
    end
    =#

    return sol,wpl,tt
end

# Perform a symmetric projection scheme by using a simplified Newton iteration
function SymmProj!(wp_newt,w,maxiter,tol,Hpar,AA,b)
    μ = 1e-7
    copy!(wp_newt,w)
    w_tmp = (1.0+2.0*μ).*w
    gmres!(w_tmp,AA,b,log=false,reltol=1e-12)
    wp_tmp = (1.0-2.0*μ).*wp_newt
    ctr = 1
    dpsi = zeros(length(w))
    while (sqrt(norm(w_tmp.-wp_tmp)^2 + abs2(1-norm(wp_newt)^2)) > tol && ctr < maxiter)
        nm = sum(abs2.(wp_newt))
        inm = 1.0/nm
        
        # Compute Newton updates
        dpsi .= (w_tmp.-wp_tmp) .+ inm.*sum(wp_newt.*(wp_tmp.-w_tmp)).*wp_newt .+ 0.5.*wp_newt.*inm.*(1.0-nm)
        dmu = 0.25*sum((wp_tmp.-w_tmp).*wp_newt).*inm + 0.125*(1.0-nm)*inm

        # Update solution
        wp_newt .+= dpsi
        μ += dmu

        # Compute residual for next iteration
        w_tmp .= (1.0+2.0*μ).*w
        wp_tmp .= (1.0-2.0*μ).*wp_newt
        b = left_matrix_action(w_tmp,Hpar)
        gmres!(w_tmp,AA,b,log=false,reltol=1e-12)
        ctr+=1
    end
end

# Perform a symmetric projection scheme by using a simplified Newton iteration
function SymmProj_bwd!(wp_newt,w,maxiter,tol,Hpar,AA,b)
    μ = 1e-6
    copy!(wp_newt,w)
    w_tmp = (1.0+2.0*μ).*w
    gmres!(w_tmp,AA,b,log=false,reltol=1e-12)

    wp_tmp = (1.0-2.0*μ).*wp_newt
    ctr = 0
    dpsi = zeros(length(w))
    while (sqrt(norm(w_tmp.-wp_tmp)^2 + abs2(1-norm(wp_newt)^2)) > tol && ctr < maxiter)
        nm = sum(abs2.(wp_newt))
        inm = 1.0/nm
        
        # Compute Newton updates
        dpsi .= (w_tmp.-wp_tmp) .+ inm.*sum(wp_newt.*(wp_tmp.-w_tmp)).*wp_newt .+ 0.5.*wp_newt.*inm.*(1.0-nm)
        dmu = 0.25*sum((wp_tmp.-w_tmp).*wp_newt).*inm + 0.125*(1.0-nm)*inm

        # Update solution
        wp_newt .+= dpsi
        μ += dmu

        # Compute residual for next iteration
        w_tmp .= (1.0+2.0*μ).*w
        wp_tmp .= (1.0-2.0*μ).*wp_newt
        b = right_matrix_action(w_tmp,Hpar)
        gmres!(w_tmp,AA,b,log=false,reltol=1e-12)
        ctr+=1
    end
end


function reverse_me(nt)

    m = 2
    T = 100.0
    dt = T/nt

    wl = [1.0, 0.4, 2.0/30]
    wr = wl.*(-1).^(0:m)
    for k = 1:m
        wr[1+k] = wr[1+k]*(dt/2)^k
        wl[1+k] = wl[1+k]*(dt/2)^k
    end

    Nstates = 3
    xi = 0.2
    
    amat = Bidiagonal(zeros(Nstates),sqrt.(collect(1:Nstates-1)),:U)
    adag = amat'
    
    K = amat+adag
    KKerr = -xi/2*adag*adag*amat*amat
    S = amat-adag

    Hpar = H_stuff(S,K,KKerr,0.0,0.0,0.0,0.0,0.0,0.0,wl,wr,dt)
    par = (S=S,K=K,KKerr=KKerr)
    
    p_amp = 1.0
    pfun(t) = p_amp*sin(t)
    ptfun(t) = p_amp*cos(t)
    pttfun(t) = -p_amp*sin(t)
    
    q_amp = 1.0
    qfun(t) = q_amp*sin(t)
    qtfun(t) = q_amp*cos(t)
    qttfun(t) = -q_amp*sin(t)
    
    w = zeros(2*Nstates)
    w[1] = 1.0
    wp = w
    wpl = zeros(2*Nstates,nt+1)
    wpl[:,1] = w
    for it = 1:nt
        t = (it-1)*dt
        Hpar.p = pfun(t)
        Hpar.q = qfun(t)
        Hpar.pt = ptfun(t)
        Hpar.qt = qtfun(t)
        Hpar.ptt = pttfun(t)
        Hpar.qtt = qttfun(t)
        
        b = left_matrix_action(w,Hpar)
        t = t+dt
        Hpar.p = pfun(t)
        Hpar.q = qfun(t)
        Hpar.pt = ptfun(t)
        Hpar.qt = qtfun(t)
        Hpar.ptt = pttfun(t)
        Hpar.qtt = qttfun(t)

        right_matrix_action_wrap(w) = right_matrix_action(w,Hpar)
        AA = LinearMap(right_matrix_action_wrap,2*Nstates) 
        gmres_h = gmres!(wp,AA,b,log=true,reltol=1e-12)

        w = wp
        wpl[:,1+it] = w
    end
    tt = dt*collect(0:nt)
    
    wpl_bwd = zeros(2*Nstates,nt+1)
    wpl_bwd[:,nt+1] = w
    for it = nt:-1:1
        t = (it)*dt
        
        Hpar.p = pfun(t)
        Hpar.q = qfun(t)
        Hpar.pt = ptfun(t)
        Hpar.qt = qtfun(t)
        Hpar.ptt = pttfun(t)
        Hpar.qtt = qttfun(t)
        
        b = right_matrix_action(w,Hpar)
        
        t = t-dt
        Hpar.p = pfun(t)
        Hpar.q = qfun(t)
        Hpar.pt = ptfun(t)
        Hpar.qt = qtfun(t)
        Hpar.ptt = pttfun(t)
        Hpar.qtt = qttfun(t)

        left_matrix_action_wrap(w) = left_matrix_action(w,Hpar)
        AA = LinearMap(left_matrix_action_wrap,2*Nstates) 
        
        wp,gmres_h = gmres!(wp,AA,b,log=true,reltol=1e-12)

        w = wp
        wpl_bwd[:,it] = w
    end


    return wpl,wpl_bwd,tt
end



function reverse_me_SymmProj(nt)

    m = 2
    T = 100.0
    dt = T/nt

    wl = [1.0, 0.4, 2.0/30]
    wr = wl.*(-1).^(0:m)
    for k = 1:m
        wr[1+k] = wr[1+k]*(dt/2)^k
        wl[1+k] = wl[1+k]*(dt/2)^k
    end

    Nstates = 3
    xi = 0.2
    
    amat = Bidiagonal(zeros(Nstates),sqrt.(collect(1:Nstates-1)),:U)
    adag = amat'
    
    K = amat+adag
    KKerr = -xi/2*adag*adag*amat*amat
    S = amat-adag

    Hpar = H_stuff(S,K,KKerr,0.0,0.0,0.0,0.0,0.0,0.0,wl,wr,dt)
    par = (S=S,K=K,KKerr=KKerr)
    
    p_amp = 1.0
    pfun(t) = p_amp*sin(t)
    ptfun(t) = p_amp*cos(t)
    pttfun(t) = -p_amp*sin(t)
    
    q_amp = 1.0
    qfun(t) = q_amp*sin(t)
    qtfun(t) = q_amp*cos(t)
    qttfun(t) = -q_amp*sin(t)
    
    w = zeros(2*Nstates)
    w[1] = 1.0
    wp = w
    wpl = zeros(2*Nstates,nt+1)
    wpl[:,1] = w
    wpl_newt = copy(wpl)
    tmp = zeros(2*Nstates)
    wp_tmp = zeros(2*Nstates)
    Axb = zeros(2*Nstates)
    wp_newt = copy(w)
    for it = 1:nt
        t = (it-1)*dt
        Hpar.p = pfun(t)
        Hpar.q = qfun(t)
        Hpar.pt = ptfun(t)
        Hpar.qt = qtfun(t)
        Hpar.ptt = pttfun(t)
        Hpar.qtt = qttfun(t)
        
        b = left_matrix_action(w,Hpar)
        t = t+dt
        Hpar.p = pfun(t)
        Hpar.q = qfun(t)
        Hpar.pt = ptfun(t)
        Hpar.qt = qtfun(t)
        Hpar.ptt = pttfun(t)
        Hpar.qtt = qttfun(t)

        right_matrix_action_wrap(w) = right_matrix_action(w,Hpar)
        AA = LinearMap(right_matrix_action_wrap,2*Nstates) 

        # Perform a symmetric projection scheme by using a simplified Newton iteration
        # NOTE: Here we have to reset the values to the previous time-step as we update
        # the RHS each Newton iteration
        maxiter = 10
        tol = 1e-14
        t = (it-1)*dt
        Hpar.p = pfun(t)
        Hpar.q = qfun(t)
        Hpar.pt = ptfun(t)
        Hpar.qt = qtfun(t)
        Hpar.ptt = pttfun(t)
        Hpar.qtt = qttfun(t)
        
        b = left_matrix_action(w,Hpar)
        SymmProj!(wp_newt,w,maxiter,tol,Hpar,AA,b)
        copy!(w,wp_newt)
        wpl[:,1+it] = wp_newt
    end
    tt = dt*collect(0:nt)
    
    wpl_bwd = zeros(2*Nstates,nt+1)
    wpl_bwd[:,nt+1] = w
    wp_newt = copy(w)
    for it = nt:-1:1
        t = (it)*dt
        
        Hpar.p = pfun(t)
        Hpar.q = qfun(t)
        Hpar.pt = ptfun(t)
        Hpar.qt = qtfun(t)
        Hpar.ptt = pttfun(t)
        Hpar.qtt = qttfun(t)
        
        b = right_matrix_action(w,Hpar)
        
        t = t-dt
        Hpar.p = pfun(t)
        Hpar.q = qfun(t)
        Hpar.pt = ptfun(t)
        Hpar.qt = qtfun(t)
        Hpar.ptt = pttfun(t)
        Hpar.qtt = qttfun(t)

        left_matrix_action_wrap(w) = left_matrix_action(w,Hpar)
        AA = LinearMap(left_matrix_action_wrap,2*Nstates) 
        
        maxiter = 10
        tol = 1e-14
        SymmProj_bwd!(wp_newt,w,maxiter,tol,Hpar,AA,b)
        copy!(w,wp_newt)
        wpl_bwd[:,it] = w
    end


    return wpl,wpl_bwd,tt
end


# wpl1,wpl_bwd,tt = reverse_me(400)
nsteps = 800
wpl1,wpl_bwd,tt = reverse_me_SymmProj(nsteps)


sol,wpl,tt = get_matrix(nsteps)
qall = get_qall(sol)

# Reversibility plot
plot(tt,sqrt.(sum(abs2.(wpl1.-wpl_bwd),dims=1))[:])
PyPlot.grid(true,which="both",alpha=0.5)
PyPlot.yscale("log");
PyPlot.legend(loc="best")
PyPlot.xlabel("t",fontsize=16)
PyPlot.ylim([1e-16,1e-13])
PyPlot.title("Reversibility Error",fontsize=18)

# Uncomment to output plot
# PyPlot.savefig("reversibility_error.pdf",dpi=150,format="pdf")

# Unitarity plot
len = dropdims(sum(qall.^2,dims=1),dims=1)
lenerr = abs.(sqrt.(len).-1.0).+1e-16

len_newt = dropdims(sum(wpl1.^2,dims=1),dims=1)
lenerr_new = abs.(sqrt.(len_newt).-1.0).+1e-16

PyPlot.display_figs()
sleep(2.0)
PyPlot.clf()
plot(tt,lenerr,"cornflowerblue",label="Base scheme")
plot(tt,lenerr_new,"orangered",label="Symmetric Projection")
PyPlot.grid(true,which="both",alpha=0.5)
PyPlot.yscale("log");
PyPlot.legend(loc="best")
PyPlot.xlabel("t",fontsize=16)
PyPlot.title("Unit Length Error",fontsize=18)

# Uncomment to output plot
# PyPlot.savefig("unitarity_comparison.pdf",dpi=150,format="pdf")


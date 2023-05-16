using HermiteOptimalControl
using Plots
using Infiltrator
include("../src/bsplines.jl")



function main(;coupled=true)
    if coupled
        T = 1.0
        D1 = 4 # Number of B-spline coefficients per control function
        omega = [[0.0]] # 1 frequency for 1 pair of coupled controls (p and q)
        pcof = ones(2*D1)
        # Use simplest constructor
        b = bcparams(T, D1,omega, pcof) 
    else
        T = 1.0
        D1 = 4 # Number of B-spline coefficients per control function
        Ncoupled = 0
        Nunc = 2
        Nfreq = [1,1]
        
        omega = [[1.0], [1.0]] # 2 frequencies for 2 uncoupled controls
        pcof = ones(4*D1)
        # Use simplest constructor
        b = bcparams(T, D1, Ncoupled, Nunc, Nfreq, omega, pcof) 
    end

    N = 1001
    t = LinRange(0,T,N)
    p = zeros(N)
    grad = zeros(2*D1)

    grad_p_hist = zeros(2*D1, N)
    grad_q_hist = zeros(2*D1, N)

    grad_pt_hist = zeros(2*D1, N)
    grad_qt_hist = zeros(2*D1, N)

    grad_p_hist_fd = zeros(2*D1, N)
    grad_q_hist_fd = zeros(2*D1, N)

    grad_pt_hist_fd = zeros(2*D1, N)
    grad_qt_hist_fd = zeros(2*D1, N)

    q = zeros(N)
    pt = zeros(N)
    qt = zeros(N)
    pt_fd = zeros(N)
    qt_fd = zeros(N)
    dt = 1e-5
    da = 1e-5
    for i=1:N
        p[i] = bcarrier2(t[i], b, 0, pcof)
        q[i] = bcarrier2(t[i], b, 1, pcof)

        pt[i] = bcarrier2_dt(t[i], b, 0, pcof)
        qt[i] = bcarrier2_dt(t[i], b, 1, pcof)

        pt_fd[i]= (bcarrier2(t[i]+dt, b, 0, pcof) - bcarrier2(t[i]-dt, b, 0, pcof))/(2*dt)
        qt_fd[i]= (bcarrier2(t[i]+dt, b, 1, pcof) - bcarrier2(t[i]-dt, b, 1, pcof))/(2*dt)


        grad .= 0
        gradbcarrier2!(t[i], b, 0, grad)
        grad_p_hist[:,i] .= grad
        grad .= 0
        gradbcarrier2!(t[i], b, 1, grad)
        grad_q_hist[:,i] .= grad

        grad .= 0
        gradbcarrier2_dt!(t[i], b, 0, grad)
        grad_pt_hist[:,i] .= grad
        grad .= 0
        gradbcarrier2_dt!(t[i], b, 1, grad)
        grad_qt_hist[:,i] .= grad

        for j=1:2*D1
            # p
            pcof_r = copy(pcof)
            pcof_l = copy(pcof)
            pcof_r[j] += da
            pcof_l[j] -= da

            grad_p_hist_fd[j,i] = (bcarrier2(t[i], b, 0, pcof_r) - bcarrier2(t[i], b, 0, pcof_l))/(2*(da))
            grad_q_hist_fd[j,i] = (bcarrier2(t[i], b, 1, pcof_r) - bcarrier2(t[i], b, 1, pcof_l))/(2*(da))

            grad_pt_hist_fd[j,i] = (bcarrier2_dt(t[i], b, 0, pcof_r) - bcarrier2_dt(t[i], b, 0, pcof_l))/(2*(da))
            grad_qt_hist_fd[j,i] = (bcarrier2_dt(t[i], b, 1, pcof_r) - bcarrier2_dt(t[i], b, 1, pcof_l))/(2*(da))
        end


    end

    pl = plot(t, pt)
    fd_approx = zeros(N)
    # Finite difference approxication
    fd_approx[2:N-1] = (p[3:N] - p[1:N-2])/(2*(t[2]-t[1]))
    plot!(pl, t, fd_approx)

    println("grad_p_hist")
    display(grad_p_hist)
    println("grad_q_hist")
    display(grad_q_hist)

    #@infiltrate
    println("Low-Res pt")
    display(log10.(abs.(fd_approx - pt)))
    println("pt")
    display(log10.(abs.(pt_fd - pt)))
    println("qt")
    display(log10.(abs.(qt_fd - qt)))
    println("grad_p")
    display(log10.(abs.(grad_p_hist - grad_p_hist_fd)))
    println("grad_q")
    display(log10.(abs.(grad_q_hist - grad_q_hist_fd)))
    println("grad_pt")
    display(log10.(abs.(grad_pt_hist - grad_pt_hist_fd)))
    println("grad_qt")
    display(log10.(abs.(grad_qt_hist - grad_qt_hist_fd)))
    #@infiltrate
    return pl
    #plot!(pl, t, q)
end

function main2(i=1)
    T = 1.0
    D1 = 8 # Number of B-spline coefficients per control function
    omega = [[1.0]] # 1 frequency for 1 pair of coupled controls (p and q)
    pcof = zeros(2*D1)
    #pcof[i] = 1
    # Use simplest constructor
    b = bcparams(T, D1,omega, pcof) 

    N = 1001
    t = LinRange(0,T,N)
    p = zeros(N)
    q = zeros(N)
    pt = zeros(N)
    qt = zeros(N)
    for i in 1:N
        p[i] = bcarrier2(t[i], b, 0)
        q[i] = bcarrier2(t[i], b, 1)
        pt[i] = bcarrier2_dt(t[i], b, 0)
        qt[i] = bcarrier2_dt(t[i], b, 1)
    end
    pl = plot(xlabel="t")
    plot!(pl, t, p,  label="p(t)")
    plot!(pl, t, q,  label="q(t)")
    #plot!(pl, t, pt, label="pt(t)")
    #plot!(pl, t, qt, label="qt(t)")
    #@infiltrate
    return pl
end

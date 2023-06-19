using Juqbox
using Plots

function main()
    T = 1.0
    D1 = 8 # Number of B-spline coefficients per control function
    omega = [[1.0]] # 1 frequency for 1 pair of coupled controls (p and q)
    pcof = zeros(2*D1)
    pcof[2] = 1
    # Use simplest constructor
    b = bcparams(T, D1,omega, pcof) 

    N = 1001
    t = LinRange(0,T,N)
    p = zeros(N)
    q = zeros(N)
    for i in 1:N
        p[i] = bcarrier2(t[i], b, 0)
        q[i] = bcarrier2(t[i], b, 1)
    end
    pl = plot(xlabel="t")
    plot!(pl, t, p,  label="p(t)")
    plot!(pl, t, q,  label="q(t)")
    return pl
end

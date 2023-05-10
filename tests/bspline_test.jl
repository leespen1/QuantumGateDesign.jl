include("../src/bsplines.jl")

function main(k)
    T::Float64 = 1.0
    D1::Int64 = 8
    Ncoupled::Int64 = 1
    Nunc::Int64 = 0
    Nfreq::Vector{Int64} = [1]
    omega::Vector{Vector{Float64}} = [[100.0]]
    pcof::Array{Float64,1} = zeros(2*D1*sum(Nfreq))
    pcof[k] = 1
    b = bcparams(T, D1, Ncoupled, Nunc, Nfreq, omega, pcof)

    N = 1001
    t = LinRange(0,T,N)
    p = zeros(N)
    q = zeros(N)
    pt = zeros(N)
    qt = zeros(N)
    for i=1:N
        p[i] = bcarrier2(t[i],b,0)
        q[i] = bcarrier2(t[i],b,1)
        pt[i] = bcarrier2_dt(t[i],b,0)
        qt[i] = bcarrier2_dt(t[i],b,1)
    end

    pl = plot(t, pt)
    v = zeros(N)
    v[2:N-1] = (p[3:N] - p[1:N-2])/(2*(t[2]-t[1]))
    plot!(pl, t, v)
    return pl
    #plot!(pl, t, q)
end

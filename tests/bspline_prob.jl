include("../src/bsplines.jl")

function my_p(t,α,bcpar)
    return bcarrier2(t,bcpar,0,α)
end

function my_dpdt(t,α,bcpar)
    return bcarrier2_dt(t,bcpar,0,α)
end

function my_dpda(t,α,bcpar)
    grad = zeros(bcpar.Ncoeff)
    gradbcarrier2!(t, bcpar, 0, grad)  
    N = div(bcpar.Ncoeff,2)
    return grad[1:N]
end

function my_d2pdta(t,α,bcpar)
    grad = zeros(bcpar.Ncoeff)
    gradbcarrier2_dt!(t, bcpar, 0, grad)  
    N = div(bcpar.Ncoeff,2)
    return grad[1:N]
end

function my_q(t,α,bcpar)
    return bcarrier2(t,bcpar,1,α)
end

function my_dqdt(t,α,bcpar)
    return bcarrier2_dt(t,bcpar,1,α)
end

function my_dqda(t,α,bcpar)
    grad = zeros(bcpar.Ncoeff)
    gradbcarrier2!(t, bcpar, 1, grad)  
    N = div(bcpar.Ncoeff,2)
    return grad[N+1:bcpar.Ncoeff]
end

function my_d2qdta(t,α,bcpar)
    grad = zeros(bcpar.Ncoeff)
    gradbcarrier2_dt!(t, bcpar, 1, grad)  
    N = div(bcpar.Ncoeff,2)
    return grad[N+1:bcpar.Ncoeff]
end


function bspline_prob(;tf::Float64=1.0, nsteps::Int64=10)
    Ks::Matrix{Float64} = [0 0; 0 1]
    Ss::Matrix{Float64} = [0 0; 0 0]
    a_plus_adag::Matrix{Float64} = [0.0 1.0; 1.0 0.0]
    a_minus_adag::Matrix{Float64} = [0.0 1.0; -1.0 0.0]

    T = 1.0
    D1 = 4 # Number of B-spline coefficients per control function
    omega = [[0.0]] # 1 frequency for 1 pair of coupled controls (p and q)
    pcof = ones(2*D1)
    # Use simplest constructor
    bcpar = bcparams(T, D1,omega, pcof) 

    p(t,α) = my_p(t,α,bcpar)
    q(t,α) = my_q(t,α,bcpar)
    dpdt(t,α) = my_dpdt(t,α,bcpar)
    dqdt(t,α) = my_dqdt(t,α,bcpar)
    dpda(t,α) = my_dpda(t,α,bcpar)
    dqda(t,α) = my_dqda(t,α,bcpar)
    d2p_dta(t,α) = my_d2pdta(t,α,bcpar)
    d2q_dta(t,α) = my_d2qdta(t,α,bcpar)

    u0::Vector{Float64} = [1,0]
    v0::Vector{Float64} = [0,0]
    N_essential = 2
    N_guard = 0
    return SchrodingerProb(Ks,Ss, a_plus_adag, a_minus_adag,
                           p,q,dpdt,dqdt,dpda,dqda,d2p_dta,d2q_dta,
                           u0,v0,tf,nsteps,N_essential,N_guard)
end

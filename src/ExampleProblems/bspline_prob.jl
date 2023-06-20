function bspline_p(t,α,bcpar)
    return bcarrier2(t,bcpar,0,α)
end

function bspline_dpdt(t,α,bcpar)
    return bcarrier2_dt(t,bcpar,0,α)
end

function bspline_dpda(t,α,bcpar)
    grad = zeros(bcpar.Ncoeff)
    gradbcarrier2!(t, bcpar, 0, grad)  
    return grad
end

function bspline_d2pdta(t,α,bcpar)
    grad = zeros(bcpar.Ncoeff)
    gradbcarrier2_dt!(t, bcpar, 0, grad)  
    return grad
end

function bspline_q(t,α,bcpar)
    return bcarrier2(t,bcpar,1,α)
end

function bspline_dqdt(t,α,bcpar)
    return bcarrier2_dt(t,bcpar,1,α)
end

function bspline_dqda(t,α,bcpar)
    grad = zeros(bcpar.Ncoeff)
    gradbcarrier2!(t, bcpar, 1, grad)  
    return grad
end

function bspline_d2qdta(t,α,bcpar)
    grad = zeros(bcpar.Ncoeff)
    gradbcarrier2_dt!(t, bcpar, 1, grad)  
    return grad
end


# Expects pcof/α of length 8
"""
    prob = bspline_prob(ω::Float64; tf, nsteps)

Construct a problem for a 2-level qubit in the rotating frame with a B-spline
control using a real-valued control vector of length 8 (the first half is the
real part and the second half is the imaginary part in the
complex-formulation).
"""
function bspline_prob(ω::Float64=0.0; tf::Float64=1.0, nsteps::Int64=10)
    Ks::Matrix{Float64} = [0 0; 0 1]
    Ss::Matrix{Float64} = [0 0; 0 0]
    a_plus_adag::Matrix{Float64} = [0.0 1.0; 1.0 0.0]
    a_minus_adag::Matrix{Float64} = [0.0 1.0; -1.0 0.0]

    T = 1.0
    D1 = 4 # Number of B-spline coefficients per control function
    omega = [[ω]] # 1 frequency for 1 pair of coupled controls (p and q)
    pcof = ones(2*D1)
    # Use simplest constructor
    bcpar = bcparams(T, D1,omega, pcof) 

    p(t,α) = bspline_p(t,α,bcpar)
    q(t,α) = bspline_q(t,α,bcpar)
    dpdt(t,α) = bspline_dpdt(t,α,bcpar)
    dqdt(t,α) = bspline_dqdt(t,α,bcpar)
    dpda(t,α) = bspline_dpda(t,α,bcpar)
    dqda(t,α) = bspline_dqda(t,α,bcpar)
    d2p_dta(t,α) = bspline_d2pdta(t,α,bcpar)
    d2q_dta(t,α) = bspline_d2qdta(t,α,bcpar)

    u0::Vector{Float64} = [1,0]
    v0::Vector{Float64} = [0,0]
    N_essential = 2
    N_guard = 0
    return SchrodingerProb(Ks,Ss, a_plus_adag, a_minus_adag,
                           p,q,dpdt,dqdt,dpda,dqda,d2p_dta,d2q_dta,
                           u0,v0,tf,nsteps,N_essential,N_guard)
end

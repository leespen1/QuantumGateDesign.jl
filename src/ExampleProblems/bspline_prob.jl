"""
One thing I really want to be careful of is whether 'p' and 'q' are the real
parts of the bspline with carrier wave, or just of the bspline. In the paper,
they are the real parts of the bspline only, but I think in Juqbox (where I
got the bsplines from) they include the carrier wave as well.
"""
function bspline_control(bcpar)
    p_vec = [
        (t, pcof) -> bcarrier2(t,    bcpar, 0, pcof),
        (t, pcof) -> bcarrier2_dt(t, bcpar, 0, pcof)
    ]
    q_vec = [
        (t, pcof) -> bcarrier2(t,    bcpar, 1, pcof),
        (t, pcof) -> bcarrier2_dt(t, bcpar, 1, pcof)
    ]
    # Gradients for bcarrier actually don't depend on pcof.
    grad_p_vec = [
        (t, pcof) -> grad_bcarrier2(t, bcpar, 0),
        (t, pcof) -> grad_bcarrier2_dt(t, bcpar, 0)
    ]
    grad_q_vec = [
        (t, pcof) -> grad_bcarrier2(t, bcpar, 1),
        (t, pcof) -> grad_bcarrier2_dt(t, bcpar, 1)
    ]

    N_coeff = length(bcpar.pcof)
    return Control(p_vec, q_vec, grad_p_vec, grad_q_vec, N_coeff)
end

function qubit_with_bspline(ground_state_transition_frequency,
        self_kerr_coefficient, N_ess_levels, N_guard_levels;
        N_coeff_per_control = 4, tf=1.0, nsteps=10
    )

    prob = single_transmon_qubit_rwa(
        ground_state_transition_frequency, self_kerr_coefficient, N_ess_levels,
        N_guard_levels, tf, nsteps
    )

    omega::Vector{Vector{Float64}} = [[ground_state_transition_frequency]] # 1 frequency for 1 pair of coupled controls (p and q)
    pcof = ones(2*N_coeff_per_control) # Put in dummy pcof, just so the length is known
    # Use simplest constructor
    bcpar = bcparams(prob.tf, N_coeff_per_control, omega, pcof) 

    control = bspline_control(bcpar)

    return prob, control
end
#=
# Expects pcof of length 8
"""
    prob = bspline_prob(ω::Float64; tf, nsteps)

Construct a problem for a 2-level qubit in the rotating frame with a B-spline
control using a real-valued control vector of length 8 (the first half is the
real part and the second half is the imaginary part in the
complex-formulation).
"""
function bspline_prob(ω::Float64=0.0; n_levels::Int=1, tf::Float64=1.0, nsteps::Int=10)
    n_subsystems = 1
    system_hamiltonian = get_system_hamiltonian(n_subsystems, n_levels)

    Ss::Matrix{Float64} = [0 0; 0 0]
    a_plus_adag::Matrix{Float64} = [0.0 1.0; 1.0 0.0]
    a_minus_adag::Matrix{Float64} = [0.0 1.0; -1.0 0.0]

    T = 1.0
    D1 = 4 # Number of B-spline coefficients per control function
    omega = [[ω]] # 1 frequency for 1 pair of coupled controls (p and q)
    pcof = ones(2*D1)
    # Use simplest constructor
    bcpar = bcparams(T, D1,omega, pcof) 

    N_essential = 2
    N_guard = 0
    return SchrodingerProb(Ks,Ss, a_plus_adag, a_minus_adag,
                           p,q,dpdt,dqdt,dpda,dqda,d2p_dta,d2q_dta,
                           u0,v0,tf,nsteps,N_essential,N_guard)
end

function bspline_prob_vec(ω::Float64=0.0; tf::Float64=1.0, nsteps::Int64=10)
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
=#

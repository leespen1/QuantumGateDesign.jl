include("../src/hermite.jl")

function this_prob(;ω::Float64=1.0, tf::Float64=1.0, nsteps::Int64=10)
    Ks::Matrix{Float64} = [0 0; 0 1]
    Ss::Matrix{Float64} = [0 0; 0 0]
    a_plus_adag::Matrix{Float64} = [0 1; 1 0]
    a_minus_adag::Matrix{Float64} = [0 1; -1 0]
    p(t,α) = α*cos(ω*t)
    q(t,α) = 0.0
    u0::Vector{Float64} = [1,0]
    v0::Vector{Float64} = [0,0]
    return SchrodingerProb(Ks,Ss,a_plus_adag,a_minus_adag,p,q,u0,v0,tf,nsteps)
end


#=
#Actually I don't really need this. ∂ψ/∂α is evolved according to the original
#schrodinger equation. The part with ∂H/∂α  is in the forcing term
"""
Evolve ∂ψ/∂α (in relation to the above problem). Will need forcing
System hamiltonian falls out, initial conditions are zero, and controls change
according to ∂/∂α
"""
function this_prob_grad(;ω::Float64=1.0, tf::Float64=1.0, nsteps::Int64=10)
    Ks::Matrix{Float64} = [0 0; 0 0]
    Ss::Matrix{Float64} = [0 0; 0 0]
    a_plus_adag::Matrix{Float64} = [0 1; 1 0]
    a_minus_adag::Matrix{Float64} = [0 1; -1 0]
    p(t,α) = cos(ω*t)
    q(t,α) = 0.0
    u0::Vector{Float64} = [0,0]
    v0::Vector{Float64} = [0,0]
    return SchrodingerProb(Ks,Ss,a_plus_adag,a_minus_adag,p,q,u0,v0,tf,nsteps)
end
=#

function convergence_test!(prob::SchrodingerProb, α=1.0)

    prob.nsteps = 200
    history200 = eval_forward(prob, α)
    # Change stride to match 100 timestep result
    history200 = history200[:,1:2:end]
    prob.nsteps = 100
    prob100 = this_prob(nsteps=100)
    history100 = eval_forward(prob100, α)

    # Use 10000 timesteps for "true solution"
    prob10000 = this_prob(nsteps=10000)
    history_true = eval_forward(prob10000, α)
    history_true = history_true[:,1:100:end]

    error100 = abs.(history_true - history100)
    error200 = abs.(history_true - history200)

    log_ratio = log2.(error100 ./ error200)
    println("Log₂ of error ratio between 100 step and 200 step methods")
    println("(Analytic solution used for 'true' value)")


    display(log_ratio)
    return log_ratio, error100, error200
end

function finite_difference(prob, α, target, dα=1e-5)
    history_r = eval_forward(prob, α+dα)
    history_l = eval_forward(prob, α-dα)
    ψf_r = history_r[:,end]
    ψf_l = history_l[:,end]
    infidelity_r = infidelity(ψf_r, target)
    infidelity_l = infidelity(ψf_l, target)
    gradient = (infidelity_r - infidelity_l)/dα
    return gradient
end


"""
Obtain the gradient using the 'take derivative of schrodinger's equation'
method. 
"""
function eval_forward_grad_mat(target, α=1.0; nsteps=100)
    u0::Vector{Float64} = zeros(8)
    u0[1] = 1
    u = copy(u0)

    tf = 1.0
    dt = tf/nsteps

    # Need to write out LHS matrix to do implicit solve, or else use
    # abstract arrays and iterative solvers
    M(t, a) = [
    0.0 0.0 0.0 -a*cos(t)  0.0 0.0 0.0 0.0
    0.0 0.0 -a*cos(t) -1.0 0.0 0.0 0.0 0.0
    0.0 a*cos(t) 0.0 0.0   0.0 0.0 0.0 0.0
    a*cos(t) 1.0 0.0 0.0   0.0 0.0 0.0 0.0

    0.0 0.0 0.0 -cos(t)    0.0 0.0 0.0 -a*cos(t) 
    0.0 0.0 -cos(t) 0.0    0.0 0.0 -a*cos(t) -1.0
    0.0 cos(t) 0.0 0.0     0.0 a*cos(t) 0.0 0.0  
    cos(t) 0.0 0.0 0.0     a*cos(t) 1.0 0.0 0.0  
    ]
    for n in 0:prob.nsteps-1
        tn = n*dt
        tnp1 = (n+1)*dt
        u = (I - 0.5*dt*M(tnp1, α)) \ (u + 0.5*dt*M(tn,α)*u)
    end

    R = target[:]
    T = vcat(R[3:4], -R[1:2])
    Q = u[1:4,end]
    dQda = u[5:8,end]

    #grad_gargamel = 2*sum(Q .* dQda)
    grad_gargamel = -0.5*((Q'*R)*(dQda'*R) + (Q'*T)*(dQda'*T))
    return grad_gargamel
end


#=
function eval_grad_fwd(target, α=1.0; ω::Float64=1.0)
    # Get state vector history
    prob = this_prob(ω=ω)
    history = eval_forward(prob, α)

    ## Prepare forcing (-idH/dα ψ)
    # Prepare dH/dα
    Ks::Matrix{Float64} = [0 0; 0 0]
    Ss::Matrix{Float64} = [0 0; 0 0]
    a_plus_adag::Matrix{Float64} = [0 1; 1 0]
    a_minus_adag::Matrix{Float64} = [0 1; -1 0]
    p(t,α) = cos(ω*t)
    q(t,α) = 0.0

    forcing_mat = zeros(4,1+prob.nsteps)
    forcing_vec = zeros(4)

    u = zeros(2)
    v = zeros(2)
    ut = zeros(2)
    vt = zeros(2)

    nsteps = prob.nsteps
    t = 0
    dt = prob.tf/nsteps
    for i in 0:nsteps
        copyto!(u,history[1:2,1+i])
        copyto!(v,history[3:4,1+i])

        utvt!(ut, vt, u, v, Ks, Ss, a_plus_adag, a_minus_adag, p, q, t, α)
        copyto!(forcing_vec,1,ut,1,2)
        copyto!(forcing_vec,3,vt,1,2)

        forcing_mat[:,1+i] .= forcing_vec

        t += dt
    end
    
    # Get history of dψ/dα
    # Initial conditions for dψ/dα
    copyto!(prob.u0, [0.0,0.0])
    copyto!(prob.v0, [0.0,0.0])

    history_dψdα = eval_forward_forced(prob, forcing_mat, α)

    dψdα_T = history_dψdα[:,end]
    ψ_T = history[:,end]
    R = copy(target)
    T = vcat(R[3:4], -R[1:2])

    gradient = -0.5*(dot(ψ_T,R)*dot(dψdα_T,R) + dot(ψ_T,T)*dot(dψdα_T,T))
    #grad_gargamel = -0.5*((Q'*R)*(dQda'*R) + (Q'*T)*(dQda'*T))
    return gradient
end
=#

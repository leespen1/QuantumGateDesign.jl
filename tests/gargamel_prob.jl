function gargamel_prob(;ω::Float64=1.0, tf::Float64=1.0, nsteps::Int64=10)
    Ks::Matrix{Float64} = [0 0; 0 1]
    Ss::Matrix{Float64} = [0 0; 0 0]
    p(t,α) = α*cos(ω*t)
    q(t,α) = 0.0
    dpdt(t,α) = -α*ω*sin(ω*t)
    dqdt(t,α) = 0.0
    dpda(t,α) = cos(ω*t)
    dqda(t,α) = 0.0
    d2p_dta(t,α) = -ω*sin(ω*t)
    d2q_dta(t,α) = 0.0
    u0::Vector{Float64} = [1,0]
    v0::Vector{Float64} = [0,0]
    return SchrodingerProb(Ks,Ss,
                           p,q,dpdt,dqdt,dpda,dqda,d2p_dta,d2q_dta,
                           u0,v0,tf,nsteps)
end



function gargamel_gradient_test(α=1.0; dα=1e-5, order=2, nsteps=10, tf=1.0, ω=1.0)
    prob = gargamel_prob(nsteps=nsteps, tf=tf, ω=ω)
    return gradient_test(prob, α, dα, order=order)
end



function gargamel_convergence_test(;α=1.0, tf=1.0, ω=1.0, base_nsteps=2, N=5)
    prob = gargamel_prob(tf=tf, ω=ω)
    return convergence_test!(prob, α, base_nsteps=base_nsteps, N=N)
end


function array_convergence_test(α=1.0; order=2)
    # MAKE SURE TIMESTEPS ARE SMALL ENOUGH THAT SOLUTION ISNT OVERRESOLVED
    prob20 = gargamel_prob(nsteps=20)
    history20 = eval_forward(prob20, α, order=order)
    # Change stride to match coarse timestep result
    history20 = history20[:,1:2:end]
    prob10 = gargamel_prob(nsteps=10)
    history10 = eval_forward(prob10, α, order=order)

    # Use 1000 timesteps for "true solution"
    prob1000 = gargamel_prob(nsteps=1000)
    history_true = eval_forward(prob1000, α, order=order)
    history_true = history_true[:,1:100:end]

    error10 = abs.(history_true - history10)
    error20 = abs.(history_true - history20)

    log_ratio = log2.(error10 ./ error20)
    println("Log₂ of error ratio between 10 step and 20 step methods")

    display(log_ratio)
    return log_ratio, error10, error20
end


#=
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
    for n in 0:nsteps-1
        tn = n*dt
        tnp1 = (n+1)*dt
        u = (I - 0.5*dt*M(tnp1, α)) \ (u + 0.5*dt*M(tn,α)*u)
    end

    Q = u[1:4,end]
    dQda = u[5:8,end]

    R = target[:]
    T = vcat(R[3:4], -R[1:2])

    #target_complex::Vector{ComplexF64} = target[1:2] - im*target[3:4]
    #Q_complex::Vector{ComplexF64} = Q[1:2] - im*Q[3:4]
    #dQda_complex::Vector{ComplexF64} = dQda[1:2] - im*dQda[3:4]

    ## The following two are equivalent, checked with ComplexF64 vectors
    #
    #grad_gargamel = -2*real((Q_complex'*target_complex)*(target_complex'*dQda_complex))
    grad_gargamel = -2*(dot(Q,R)*dot(dQda,R) + dot(Q,T)*dot(dQda,T))

    # rewrite gradient
    return grad_gargamel
end
=#

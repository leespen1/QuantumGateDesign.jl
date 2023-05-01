include("../Daniel/da_test.jl")

# The equation we solve is
# du/dt = (a*cos(t)-1)*u
# u(0) = 1
# This has the solution 
# u(t) = exp(a*sin(t)-t)
# 
# The cost function is J(a) = u(T)^2/2, where T is the final time.
# 
# The scheme is
# u(t+dt) = u(t) + dt/2*(P(t+dt)*u(t+dt)+M(t)*u(t)
# u(0) = 1
# 
# The adjoint scheme is
# (1-dt/2*P(t))*lam(t) = (1+dt/2*M(t))*lam(t+dt) 
# (1-dt/2*P(T))*lam(T) = -u(T)
# 
# The gradient is
# 
# \sum_{i=0}^{N-1} (dP/da((i+1)*dt)*u((it+1)*dt)+dM/da(it*dt)*u(it*dt))*lam((it+1)*dt)

"""
Construct a 'SchrodingerProb' corresponding to Daniel's test:

In our problem, we recreate this in the first component of u by taking K(t)=0,
S(t) = [a*cos(t)-1 0; 0 0]

"""
function daniel_prob(;tf::Float64=1.2, nsteps::Int64=100)
    Ks::Matrix{Float64} = [0 0; 0 0]
    Ss::Matrix{Float64} = [-1 0; 0 0]
    p(t,α) = 0.0
    q(t,α) = α*cos(t)
    dpdt(t,α) = 0.0
    dqdt(t,α) = -α*sin(t)
    dpda(t,α) = 0.0
    dqda(t,α) = cos(t)
    d2p_dta(t,α) = 0.0
    d2q_dta(t,α) = -sin(t)
    u0::Vector{Float64} = [1,0]
    v0::Vector{Float64} = [0,0]

    prob = SchrodingerProb(Ks,Ss,
                           p,q,dpdt,dqdt,dpda,dqda,d2p_dta,d2q_dta,
                           u0,v0,tf,nsteps)

    # Do not really use a+a†, this is a hack to to recreate a scalar equation
    prob.a_plus_adag = [0 0; 0 0] 
    prob.a_minus_adag = [1 0; 0 0]
    return prob
end

function daniel_discrete_adjoint(a,nsteps,T)
    # Function to compute the gradient via the discrete adjoint
    
    # Forward solve
    u_arr = zeros(nsteps+1)
    lam_arr = zeros(nsteps+1)
    u = 1
    u_arr[1] = u
    t = 0
    dt = T/nsteps
    for it = 1:nsteps
        tnp = t+dt 
        up = 1/(1-0.5*dt*PM(tnp,a,dt))*(1+0.5*dt*MM(t,a,dt))*u
        u = up
        t = tnp
        u_arr[it+1] = u
    end

    # Adjoint solve
    t = T
    lam = -u/(1-0.5*dt*PM(t,a,dt))
    lam_arr[nsteps+1] = lam
    for it = 1:nsteps
        tm = t-dt 
        lam_m = 1/(1-0.5*dt*PM(tm,a,dt))*(1+0.5*dt*MM(tm,a,dt))*lam
        lam = lam_m
        t = tm
        lam_arr[nsteps+1-it] = lam
    end

    # Accumulation of gradient
    dLda = 0
    t = 0
    for it=1:nsteps
        dLda += (dPMda(t+dt,a,dt)*u_arr[it+1]+dMMda(t,a,dt)*u_arr[it])*lam_arr[it+1]
        #println(0.5*dt*dMMda(t,a,dt), " ", 0.5*dt*dPMda(t+dt,a,dt))
        #println("Daniel: ", 0.5*dt*(dPMda(t+dt,a,dt)*u_arr[it+1]+dMMda(t,a,dt)*u_arr[it])*lam_arr[it+1])
        t = t+dt
    end
    dLda *= -0.5*dt 
    return dLda, u_arr, lam_arr
end


function check_convergence(a,T,n)
    prob = daniel_prob(tf=T, nsteps=n)
    cost(x) = 0.5*(eval_forward(prob, x, order=4)[1,end])^2

    uex = exp(a*sin(T)-T)

    nsteps = [10 20 40 80 160]
    errs = zeros(5)
    for i in 1:5
        prob.nsteps = nsteps[i]
        errs[i] = abs(eval_forward(prob,a, order=4)[1,end] - uex)
    end
    println("Errors\n",errs,"\n")
    println("Rates of convergence\n",log2.(errs[1:4]./errs[2:5]))
end

function compare_discrete_adjoints(a::Float64,n::Int64,T::Float64)
    prob = daniel_prob(tf=T, nsteps=n)
    dummy_target::Vector{Float64} = [NaN, NaN, NaN, NaN]

    grad_s, history_s, lambda_history_s = discrete_adjoint(prob, dummy_target, a, order=4,
                                                return_lambda_history=true,
                                                cost_type=:Norm)
    grad_d, history_d, lambda_history_d = daniel_discrete_adjoint(a,n,T)
    return grad_s, grad_d, history_s, history_d, lambda_history_s, lambda_history_d
end

function daniel_test()
    a = 1.1
    T = 1.2
    n = 100

    check_convergence(a,T,n)

    uex = exp(a*sin(T)-T)

    cost(x)  = 0.5*get_u(x,n,T)^2
    cost_grad(x)  = sin(T)*get_u(x,n,T)^2 

    prob = daniel_prob(tf=T, nsteps=n)
    dummy_target::Vector{Float64} = [NaN, NaN, NaN, NaN]
    cost_spencer(x) = 0.5*(eval_forward(prob, x)[1,end])^2
    discrete_adjoint_spencer(x) = discrete_adjoint(prob, dummy_target, x, order=4, cost_type=:Norm)

    aa = LinRange(-2,2,100)
    J = cost.(aa)
    dJda_ex = cost_grad.(aa)

    dJda = ForwardDiff.derivative.(cost, aa)
    dJda_da = main.(aa,n,T)

    dJda_da_spencer = cost_spencer.(aa)

    # plot of gradients
    pl1 = plot(aa,dJda,lw=2)
    plot!(pl1,aa,dJda_ex,lw=2)
    plot!(pl1,aa,dJda_da,lw=2)
    # plot of differences between gradients
    pl2 = plot(aa,abs.(dJda-dJda_ex),lw=2)
    # 
    pl3 = plot(aa,abs.(dJda_da-dJda_ex),lw=2)
    # This last one should be machine precision
    pl4 = plot(aa,abs.(dJda-dJda_da),lw=2)

    pl5 = plot(aa,abs.(dJda-dJda_da_spencer),lw=2)
    return pl1, pl2, pl3, pl4, pl5
end

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

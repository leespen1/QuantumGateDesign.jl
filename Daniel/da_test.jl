using Plots, ForwardDiff

# The equation we solve is
# du/dt = (a*cos(t)-1)*u
# u(0) = 1
# This has the solution 
# u(t) = exp(a*sin(t)-t)
#
#
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
#
# Where N*dt = T


# Matrix free version, helper function
function get_ut_utt(u,t,a)
    ut = (a*cos(t)-1)*u
    utt = (a*cos(t)-1)*ut-a*sin(t)*u
    return ut,utt
end
# Matrix free version, solver
function get_u_mf(a,nsteps,T)

    dt = T/nsteps;
    u = 1.0
    t = 0
    for it = 1:nsteps
        # Solve by secant method
        ut,utt = get_ut_utt(u,t,a)
        up0 = u+dt*ut
        up1 = u+dt*ut+0.5*dt^2*utt
        up0t,up0tt = get_ut_utt(up0,t+dt,a)
        f0 = up0 - u - 0.5*dt*(up0t+ut+dt/6*(utt-up0tt))
        iter = 1
        du = 1.0
        while du > 1e-12 && iter < 20
            up1t,up1tt = get_ut_utt(up1,t+dt,a)
            f1 = up1 - u - 0.5*dt*(up1t+ut + dt/6*(utt-up1tt))
            du = -f1*(up1-up0)/(f1-f0)
            up0 = up1
            f0 = f1
            up1 = up1 + du
            # println("Residual = ",du," iter = ",iter)
            iter = iter+1
        end
        up = up1
        u = up
        t = t+dt
        
    end
    return u
end

# Matrix based version
function PM(t,a,dt)
    b = (a*cos(t)-1)
    return b - dt/6*(b*b-a*sin(t))
end

function dPMda(t,a,dt)
    b = (a*cos(t)-1)
    return cos(t) - dt/6*(2*b*cos(t)-sin(t))
end

function MM(t,a,dt)
    b = (a*cos(t)-1)
    return b + dt/6*(b*b-a*sin(t))
end

function dMMda(t,a,dt)
    b = (a*cos(t)-1)
    return cos(t) + dt/6*(2*b*cos(t)-sin(t))
end

# Matrix based solver
function get_u(a,nsteps,T)
    # Hermite time discretization 
    dt = T/nsteps
    u = 1.0
    t = 0
    for it = 1:nsteps
        tnp = t+dt 
        up = 1/(1-0.5*dt*PM(tnp,a,dt))*(1+0.5*dt*MM(t,a,dt))*u
        u = up
        t = tnp
    end
    return u
end

function main(a,nsteps,T)

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
        t = t+dt
    end
    dLda *= -0.5*dt 
    return dLda
end

# Some tests against the exact gradient
# and aginst automagic differentiation 
function test()
    a = 1.1
    T = 1.2
    n = 100 

    uex = exp(a*sin(T)-T)
    nsteps = [10 20 40 80 160]
    get_err(n) = abs(get_u(a,n,T)-uex)
    err = get_err.(nsteps)
    println("Errors\n",err,"\n")
    println("Rates of convergence\n",log2.(err[1:4]./err[2:5]))

    cost(x)  = 0.5*get_u(x,n,T)^2
    cost_grad(x)  = sin(T)*get_u(x,n,T)^2 

    aa = LinRange(-2,2,100)
    J = cost.(aa)
    dJda_ex = cost_grad.(aa)

    dJda = ForwardDiff.derivative.(cost, aa)
    dJda_da = main.(aa,n,T)

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
    return pl1, pl2, pl3, pl4
end


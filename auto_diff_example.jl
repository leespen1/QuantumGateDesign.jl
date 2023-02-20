#using DifferentialEquations, Optimization, OptimizationPolyalgorithms, SciMLSensitivity
#using ForwardDiff, Plots
using ForwardDiff
using SciMLSensitivity
using DifferentialEquations
using Plots
using Zygote
using LinearAlgebra
include("2d_example.jl")


"""
Evolve state vector according to schrodinger's equation
"""
function schrodinger!(du, u, p, t)
   a = p

   du[1] = -a*cos(t)*u[4]
   du[2] = -a*cos(t)*u[3] - u[4]
   du[3] = a*cos(t)*u[2]
   du[4] = a*cos(t)*u[1] + u[2]
end

"""
Evolve state vector and its gradient wrt control parameter
"""
function gargamel!(du, u, p, t)
   a = p


   du[1] = -a*cos(t)*u[4]
   du[2] = -a*cos(t)*u[3] - u[4]
   du[3] = a*cos(t)*u[2]
   du[4] = a*cos(t)*u[1] + u[2]

   du[5] = -a*cos(t)*u[8] -cos(t)*u[4]
   du[6] = -a*cos(t)*u[7] - u[8] -cos(t)*u[3] 
   du[7] = a*cos(t)*u[6] + cos(t)*u[2]
   du[8] = a*cos(t)*u[5] + u[6] + cos(t)*u[1] 
end

"""
Loss function

Calculate loss (infidelity, or whatever function chose)
The parameter used to calculate the target Q is hard-coded (1.5)
"""
function loss_func(prob, target, newparam)
    R = target[:]
    T = vcat(R[3:4], -R[1:2])

    # Remake the original ODE problem, but with new parameters
    newprob = remake(prob, p=newparam)
    sol = solve(newprob, saveat=1, abstol=1e-10, reltol=1e-10)
    sol = Array(sol)
    Q = sol[:,end]

    #loss = sum(abs2, sol[:,end] .- data[:,end])
    #loss = sum(abs2, sol[:,end])
    loss = 1 - 0.25*((Q'*R)^2 + (Q'*T)^2) # Infidelity
    return loss
end


function grad_gargamel(prob, target, newparam)
    u0 = vcat(prob.u0, zeros(4)) # Make space for dQda
    R = target[:]
    T = vcat(R[3:4], -R[1:2])

    newprob = remake(prob, f=gargamel!, u0=u0, p=newparam)
    sol = solve(newprob, saveat=1, abstol=1e-10, reltol=1e-10)
    sol = Array(sol)
    Q = sol[1:4,end]
    dQda = sol[5:8,end]

    #grad_gargamel = 2*sum(Q .* dQda)
    grad_gargamel = -0.5*((Q'*R)*(dQda'*R) + (Q'*T)*(dQda'*T))
    return grad_gargamel
end

"""
Calculate gradient using "gargamel" method, but evolve the ODE using the
trapezoidal rule, not the ODESolver package.
"""
function grad_gargamel_trap(prob, target, newparam; N=100)
    u0 = vcat(prob.u0, zeros(4)) # Make space for dQda
    R = target[:]
    T = vcat(R[3:4], -R[1:2])

    fT = prob.tspan[end]
    dt = fT/N

    u = vcat(prob.u0, zeros(4))
    du = zeros(8)

    # Need to write out LHS matrix to do implicit solve, or else use
    # abstract arrays and interative solvers
    LHS(t, a) = [
    0.0 0.0 0.0 -a*cos(t)  0.0 0.0 0.0 0.0
    0.0 0.0 -a*cos(t) -1.0 0.0 0.0 0.0 0.0
    0.0 a*cos(t) 0.0 0.0   0.0 0.0 0.0 0.0
    a*cos(t) 1.0 0.0 0.0   0.0 0.0 0.0 0.0

    0.0 0.0 0.0 -cos(t)    0.0 0.0 0.0 -a*cos(t) 
    0.0 0.0 -cos(t) 0.0    0.0 0.0 -a*cos(t) -1.0
    0.0 cos(t) 0.0 0.0     0.0 a*cos(t) 0.0 0.0  
    cos(t) 0.0 0.0 0.0     a*cos(t) 1.0 0.0 0.0  
    ]
    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt
        gargamel!(du, u, newparam, tn)
        u = (I - 0.5*dt*LHS(tnp1, newparam)) \ (u + 0.5*dt*du)
    end

    Q = u[1:4,end]
    dQda = u[5:8,end]

    #grad_gargamel = 2*sum(Q .* dQda)
    grad_gargamel = -0.5*((Q'*R)*(dQda'*R) + (Q'*T)*(dQda'*T))
    return grad_gargamel
end

"""
To be done after every iteration of the optimization process
"""
function callback(p, loss, sol)
    display(loss)
    # Plot prediction with current parameters
    plt = plot(sol, ylim = (0,6), label="Current Prediction")
    # Also plot solution with correct parameters
    scatter!(plt, data_solution, label="Data")
    display(plt) # For some reason I am plotting everything twice?
    # Don't stop the optimization (this would be where I could insert an
    # accuracy stopping point)
    return false
end


function main(;n_timesteps=100)
    # Initial condition
    Q0_complex = [1.0+1.0im, 1.0+1.0im]
    Q0_complex = Q0_complex / norm(Q0_complex)
    Q0_real = complex_to_real(Q0_complex)
    # Simulation Interval
    tspan = (0.0, 10.0)
    # Set  Parameters
    # (we will pretend we don't know these and then try to optimize parameters to match these)
    p = 1.0
    # Set up ODE Problem (using DifferentialEquations package)
    prob = ODEProblem(schrodinger!, Q0_real, tspan, p)
    # Solve ODE Problem (saving solution at 0.0, 1.0, 2.0, etc)
    data_solution = solve(prob, saveat=1, abstol=1e-10, reltol=1e-10)
    # Convert solution to array (is currently type ODE Solution which also contains
    # a bunch more info)
    data = Array(data_solution)
    Q_target = data[:,end]
    Q_target_complex = real_to_complex(Q_target)

    N = 100
    a = LinRange(0.5, 1.5, N)
    losses = zeros(N)
    grads_fwd_dif = zeros(N)
    grads_garg = zeros(N)
    grads_garg_trap = zeros(N)
    grads_dis_adj = zeros(N)
    grads_fin_dif = zeros(N)
    for i = 1:N
        losses[i] = loss_func(prob, Q_target, a[i])
        grads_fwd_dif[i] = ForwardDiff.derivative(p -> loss_func(prob, Q_target, p), a[i])
        grads_garg[i] = grad_gargamel(prob, Q_target, a[i])
        grads_garg_trap[i] = grad_gargamel_trap(prob, Q_target, a[i], N=n_timesteps)
        grads_dis_adj[i] = disc_adj(a[i], Q0_complex, Q_target_complex, fT=tspan[end], N=n_timesteps)
        grads_fin_dif[i] = finite_diff_gradient(a[i], Q0_complex, Q_target_complex, fT=tspan[end], N=n_timesteps)
    end
    pl = plot(a, losses, label="Loss")
    plot!(pl, a, grads_fwd_dif, label="Gradient (ForwardDiff)")
    plot!(pl, a, grads_garg, label="Gradient (Gargamel)", linestyle=:dash)
    plot!(pl, a, grads_garg_trap, label="Gradient (Gargamel Trap)", linestyle=:dashdot)
    plot!(pl, a, grads_dis_adj, label="Gradient (Discrete Adjoint)")
    plot!(pl, a, grads_fin_dif, label="Gradient (Finite Diff)", linestyle=:dash)
    plot!(xlabel="α")
    plot!(title="Target using α=$p, T=$(tspan[end]), N=$(n_timesteps)")
    display(pl)

    #println("Loss: ", loss_func(Q0[:], 1.5))
    #println("Gradient ForwardDiff: ", ForwardDiff.derivative(p -> loss_func(Q0[:], p), 1.5))
    #println("Gradient Gargamel: ", grad_gargamel(1.5, Q0[:]))
    #println("Loss: ", loss_func(prob, Q_target, 1.5))
    #println("Gradient ForwardDiff: ", ForwardDiff.derivative(p -> loss_func(prob, Q_target, p), 1.5))
    #println("Gradient Gargamel: ", grad_gargamel(prob, Q_target, 1.5))

    return pl
end


#pl = main()
#=

## Set up optimization problem
# Set our optimization function to minimize
# (the loss function. 'p' is the parameters, which we don't include in our loss calculation)
adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((u, p) -> loss_func(u), adtype) # I'm confused. Doesn't the loss function take the parameters, not the solution? Is u being used to denote parameters? What is p?
# According to the documentation, u are the "state variables", and p are the "hyperparameters"

pguess = [1.0, 1.2, 2.5, 1.2]
optprob = Optimization.OptimizationProblem(optf, pguess)

## Solve Optimization problem (polyopt is the algorithm to use for the optimization)
result_ode = Optimization.solve(optprob, PolyOpt(), callback = callback, maxiters=200)
p_opt = result_ode.u

println("Prepared Parameters: $p")
println("Found    Parameters: $p_opt")
println("Objective Function: $(result_ode.objective)")

=#




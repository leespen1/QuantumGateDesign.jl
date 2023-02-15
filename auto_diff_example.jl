#using DifferentialEquations, Optimization, OptimizationPolyalgorithms, SciMLSensitivity
#using ForwardDiff, Plots
using ForwardDiff
using SciMLSensitivity
using DifferentialEquations
using Plots
using Zygote
using LinearAlgebra

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
function loss_func(u0, newp)
    # Simulation Interval
    tspan::Tuple{Float64, Float64} = (0.0, 1.0)
    # Set Parameters
    # (we will pretend we don't know these and then try to optimize parameters to match these)
    p::Float64 = 1.5
    ## Generate solution data
    # Set up ODE Problem (using DifferentialEquations package)
    prob = ODEProblem(schrodinger!, u0, tspan, p)
    # Solve ODE Problem (saving solution at 0.0, 1.0, 2.0, etc)
    data_solution = solve(prob, saveat=1, abstol=1e-10, reltol=1e-10)
    # Convert solution to array (is currently type ODE Solution which also contains
    # a bunch more info)
    data = Array(data_solution)
    R = data[:,end]
    T = vcat(R[3:4], -R[1:2])

    # Remake the original ODE problem, but with new parameters
    newprob = remake(prob, p=newp)
    sol = solve(newprob, saveat=1, abstol=1e-10, reltol=1e-10)
    sol = Array(sol)
    Q = sol[:,end]

    #loss = sum(abs2, sol[:,end] .- data[:,end])
    #loss = sum(abs2, sol[:,end])
    loss = 1 - 0.25*((Q'*R)^2 + (Q'*T)^2) # Infidelity
    return loss
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

function grad_gargamel(a, Q0_real)
    u0 = vcat(Q0_real, zeros(4)) # Make space for dQda

    tspan = (0.0, 1.0)

    # Evolve with default parameter, to get target gate
    p = 1.5
    prob = ODEProblem(schrodinger!, Q0_real, tspan, p)
    data_solution = solve(prob, saveat=1, abstol=1e-10, reltol=1e-10)
    data = Array(data_solution)
    R = data[:,end]
    T = vcat(R[3:4], -R[1:2])

    p = a
    prob = ODEProblem(gargamel!, u0, tspan, p)

    data_solution = solve(prob, saveat=1, abstol=1e-10, reltol=1e-10)
    data = Array(data_solution)
    Q = data[1:4,end]
    dQda = data[5:8,end]
    #grad_gargamel = 2*sum(data[1:4,end] .* data[5:8,end])
    grad_gargamel = -0.5*((Q'*R)*(dQda'*R) + (Q'*T)*(dQda'*T))
    return grad_gargamel
end

function grad_forward_diff(a, Q0_real)
    #u0 = complex_to_real(Q0_complex)
    tspan = (0.0, 1.0)
    p = a
    prob = ODEProblem(schrodinger!, Q0_real, tspan, p)
    data_solution = solve(prob, saveat=1, abstol=1e-10, reltol=1e-10)
    data = Array(data_solution)

    grad = ForwardDiff.derivative(p -> loss_func(u0, p), p)

    return grad
end


function main()
    # Initial condition
    Q0 = [1.0, 1.0, 1.0, 1.0]
    Q0 = Q0 / norm(Q0)
    # Simulation Interval
    tspan = (0.0, 1.0)
    # Set  Parameters
    # (we will pretend we don't know these and then try to optimize parameters to match these)
    p = 1.5
    #======
    ## Generate solution data
    # Set up ODE Problem (using DifferentialEquations package)
    prob = ODEProblem(gargamel!, u0, tspan, p)
    # Solve ODE Problem (saving solution at 0.0, 1.0, 2.0, etc)
    data_solution = solve(prob, saveat=1, abstol=1e-10, reltol=1e-10)
    # Convert solution to array (is currently type ODE Solution which also contains
    # a bunch more info)
    data = Array(data_solution)
    ======#


    N = 100
    a = LinRange(1, 2, N)
    L0 = zeros(N)
    grads = zeros(N)
    grads_garg = zeros(N)
    for i = 1:N
        loss = loss_func(Q0, a[i])
        L0[i] = loss
        grad = ForwardDiff.derivative(p -> loss_func(Q0, p), a[i])
        grads[i] = grad
        grads_garg[i] = grad_gargamel(a[i], Q0)
    end
    pl = plot(a, L0, label="Losses")
    plot!(pl, a, grads, label="Gradients (ForwardDiff)")
    plot!(pl, a, grads_garg, label="Gradients (Gargamel)")
    display(pl)

    println("Loss: ", loss_func(Q0, 1.5))
    println("Gradient ForwardDiff: ", ForwardDiff.derivative(p -> loss_func(Q0, p), 1.5))
    println("Gradient Gargamel: ", grad_gargamel(1.5, Q0))
end

pl = main()
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




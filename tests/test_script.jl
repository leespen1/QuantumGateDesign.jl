#using Plots; pythonplot()
include("bspline_prob.jl")
using LinearAlgebra

function main(α, nsteps, cost_type=:Infidelity)
    prob = gargamel_prob(tf=pi/2, nsteps=nsteps, β=0.3)
    history = eval_forward(prob, α, order=2)
    target = history[:,end]

    grad_da = discrete_adjoint(prob,target,α, cost_type=cost_type)
    grad_fd = eval_grad_finite_difference(prob,target,α, cost_type=cost_type)
    println("Discrete Adjoint: ", grad_da, "\nFinite Difference: ", grad_fd)
    return grad_da, grad_fd
end

function main2(;nsteps=100, cost_type=:Infidelity, order=2, β=0.3)
    prob = gargamel_prob(tf=pi/2, nsteps=nsteps, β=β)
    α = [0.0, 0.0 ,1.0, 1.0]
    history = eval_forward(prob, α, order=order)
    target = history[:,end]
    N = 10
    a = LinRange(0,2,N)

    errors = zeros(N,N)
    for i in 1:N
        for j in 1:N
            α[3] = a[i]
            α[4] = a[j]
            grad_da = discrete_adjoint(prob,target,α,cost_type=cost_type, order=order)
            grad_fd = eval_grad_finite_difference(prob,target,α,cost_type=cost_type, order=order)
            errors[i,j] = log10(norm(grad_da - grad_fd))
        end
    end
    return a, errors
    # return countour()
end


# Gargamel random pcof test
function main3(;nsteps=100, cost_type=:Infidelity, order=2, β=0.3)
    prob = gargamel_prob(tf=pi/2, nsteps=nsteps, β=β)
    α = [1.0, 1.0, 1.0, 1.0]
    history = eval_forward(prob, α, order=order)
    target = history[:,end]

    N = 20
    errors = zeros(N)
    for i in 1:N
        dα = rand()
        new_α = α .+ dα
        grad_da = discrete_adjoint(prob, target, new_α, cost_type=cost_type, order=order)
        grad_fd = eval_grad_finite_difference(prob, target, new_α, cost_type=cost_type, order=order)
        errors[i] = log10(norm(grad_da - grad_fd))
    end
    return errors
end


# Bspline prob random pcof test
function main4(;tf=1.0, nsteps=100, cost_type=:Infidelity, order=2)
    prob = bspline_prob(tf=tf, nsteps=nsteps)
    α = ones(8)
    history = eval_forward(prob, α, order=order)
    target = history[:,end]

    N = 20
    errors = zeros(N)
    for i in 1:N
        dα = rand()
        new_α = α .+ dα
        grad_da = discrete_adjoint(prob, target, new_α, cost_type=cost_type, order=order)
        grad_fd = eval_grad_finite_difference(prob, target, new_α, cost_type=cost_type, order=order)
        errors[i] = log10(norm(grad_da - grad_fd))
    end
    return errors
end

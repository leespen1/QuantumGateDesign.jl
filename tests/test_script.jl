function main(α, nsteps, cost_type=:Infidelity)
    prob = gargamel_prob(tf=pi/2, nsteps=nsteps, β=0.3)
    history = eval_forward(prob, α, order=2)
    target = history[:,end]

    grad_da = discrete_adjoint(prob,target,α, cost_type=cost_type)
    grad_fd = eval_grad_finite_difference(prob,target,α, cost_type=cost_type)
    println("Discrete Adjoint: ", grad_da, "\nFinite Difference: ", grad_fd)
    return grad_da, grad_fd
end

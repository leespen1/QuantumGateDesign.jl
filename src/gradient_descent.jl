function gradient_descent(prob, controls, pcof, target; order=4)
    pcof = copy(pcof)

    tf = prob.tf
    dt = tf / prob.nsteps
    W = prob.guard_subspace_projector
    N_ess = prob.N_ess_levels

    pcof_history = []
    pcof_grad_history = []
    obj_history = []


    learning_rate = 0.01
    maxIter = 100
    for iter in 1:maxIter

        history = eval_forward(prob, controls, pcof, order=order)
        obj_infidelity = infidelity(history[:,1,end,:], target, N_ess)
        obj_guard = guard_penalty(history, dt, tf, W)
        obj = obj_infidelity + obj_guard

        pcof_grad = discrete_adjoint(prob, controls, pcof, target, order=order)

        push!(pcof_history, copy(pcof))
        push!(pcof_grad_history, copy(pcof_grad))
        push!(obj_history, obj)

        @. pcof += learning_rate * pcof_grad
        if (iter % 10 == 1)
            println("Objective\tNorm(pcof_grad)\tNorm(pcof)")
        end
        println(obj, "\t", norm(pcof_grad), "\t", norm(pcof))
    end

    return pcof_history, pcof_grad_history, obj_history
end

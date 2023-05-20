using HermiteOptimalControl
using Distributions
using LinearAlgebra
using Plots

function random_pcof_test(prob, pcof_length, N=20; cost_type=:Infidelity)
    pcof = rand(Uniform(-1.0,1.0), pcof_length)
    new_pcofs = zeros(pcof_length, N)

    history2 = eval_forward(prob, pcof, order=2)
    target2 = history[:,end]
    history4 = eval_forward(prob, pcof, order=4)
    target4 = history4[:,end]

    gradient_errs = zeros(N,2)
    gradient_errs_forced = zeros(N,2)
    for i in 1:N
        pcof_new = rand(Uniform(0.0,2.0), pcof_length) .* pcof
        new_pcofs[:,i] .= pcof_new

        gradient_da_2 = discrete_adjoint(prob, target2, pcof_new,
                                       order=2, cost_type=cost_type)
        gradient_fd_2 = eval_grad_finite_difference(prob, target2, pcof_new,
                                                  order=2, cost_type=cost_type)
        gradient_forced_2 = eval_grad_forced(prob, target2, pcof_new,
                                         order=2, cost_type=cost_type)
        gradient_errs[i,1] = norm(gradient_da_2 - gradient_fd_2)
        gradient_errs_forced[i,1] = norm(gradient_da_2 - gradient_forced_2)

        gradient_da_4 = discrete_adjoint(prob, target4, pcof_new,
                                       order=4, cost_type=cost_type)
        gradient_fd_4 = eval_grad_finite_difference(prob, target4, pcof_new,
                                                  order=4, cost_type=cost_type)
        gradient_forced_4 = eval_grad_forced(prob, target4, pcof_new,
                                         order=4, cost_type=cost_type)

        gradient_errs[i,2] = norm(gradient_da_4 - gradient_fd_4)
        gradient_errs_forced[i,2] = norm(gradient_da_4 - gradient_forced_4)
    end

    #pl = plot(xlabel="", ylabel=L"|| \nabla_{\alpha,DA}\mathcal{J} - \nabla_{\alpha,FD}\mathcal{J}||")
    pl1 = plot(xlabel="Perturbation #", ylabel="DA Grad - FD Grad (L2 Norm)",
             title="Discrete Adjoint vs Finite Difference")
    plot!(pl1, 1:N, gradient_errs, labels=["2nd Order" "4th Order"], lw=2)
    pl2 = plot(xlabel="Perturbation #", ylabel="DA Grad - Forced Grad (L2 Norm)",
             title="Discrete Adjoint vs Forced Method")
    plot!(pl2, 1:N, gradient_errs_forced, labels=["2nd Order" "4th Order"], lw=2)

    t = LinRange(0, prob.tf, prob.nsteps+1)
    p = zeros(prob.nsteps+1)
    q = zeros(prob.nsteps+1)
    for i in 1:prob.nsteps+1
        p[i] = prob.p(t[i], pcof)
        q[i] = prob.q(t[i], pcof)
    end
    pl3 = plot(t, hcat(p,q), labels=["p(t)" "q(t)"], xlabel="t", ylabel="Control", lw=2)
    pl4 = scatter(pcof, xlabel="i", ylabel="pcof[i]", title="Base Control Vector (pcof)", label="")
    return (pl1, pl2, pl3, pl4), (pcof, new_pcofs, gradient_errs, gradient_errs_forced)
end

# Random Target
function random_pcof_test2(prob, pcof_length, N=20; cost_type=:Infidelity)
    pcofs = zeros(pcof_length, N)

    # Use random, normalized target
    target = rand(Uniform(-1.0,1.0), 2*prob.N_tot_levels)
    target ./= norm(target)

    gradient_errs = zeros(N,2)
    gradient_errs_forced = zeros(N,2)
    for i in 1:N
        pcof = rand(Uniform(-1.0,1.0), pcof_length)
        pcofs[:,i] .= pcof

        gradient_da_2 = discrete_adjoint(prob, target, pcof,
                                       order=2, cost_type=cost_type)
        gradient_fd_2 = eval_grad_finite_difference(prob, target, pcof,
                                                  order=2, cost_type=cost_type)
        gradient_forced_2 = eval_grad_forced(prob, target, pcof,
                                         order=2, cost_type=cost_type)
        gradient_errs[i,1] = norm(gradient_da_2 - gradient_fd_2)
        gradient_errs_forced[i,1] = norm(gradient_da_2 - gradient_forced_2)

        gradient_da_4 = discrete_adjoint(prob, target, pcof,
                                       order=4, cost_type=cost_type)
        gradient_fd_4 = eval_grad_finite_difference(prob, target, pcof,
                                                  order=4, cost_type=cost_type)
        gradient_forced_4 = eval_grad_forced(prob, target, pcof,
                                         order=4, cost_type=cost_type)

        gradient_errs[i,2] = norm(gradient_da_4 - gradient_fd_4)
        gradient_errs_forced[i,2] = norm(gradient_da_4 - gradient_forced_4)
    end

    #pl = plot(xlabel="", ylabel=L"|| \nabla_{\alpha,DA}\mathcal{J} - \nabla_{\alpha,FD}\mathcal{J}||")
    pl1 = plot(xlabel="Perturbation #", ylabel="DA Grad - FD Grad (L2 Norm)",
             title="Discrete Adjoint vs Finite Difference")
    plot!(pl1, 1:N, gradient_errs, labels=["2nd Order" "4th Order"], lw=2)
    pl2 = plot(xlabel="Perturbation #", ylabel="DA Grad - Forced Grad (L2 Norm)",
             title="Discrete Adjoint vs Forced Method")
    plot!(pl2, 1:N, gradient_errs_forced, labels=["2nd Order" "4th Order"], lw=2)

    #=
    t = LinRange(0, prob.tf, prob.nsteps+1)
    p = zeros(prob.nsteps+1)
    q = zeros(prob.nsteps+1)
    for i in 1:prob.nsteps+1
        p[i] = prob.p(t[i], pcof)
        q[i] = prob.q(t[i], pcof)
    end
    pl3 = plot(t, hcat(p,q), labels=["p(t)" "q(t)"], xlabel="t", ylabel="Control", lw=2)
    pl4 = scatter(pcof, xlabel="i", ylabel="pcof[i]", title="Base Control Vector (pcof)", label="")
    =#
    return (pl1, pl2), (target, pcofs, gradient_errs, gradient_errs_forced)
end

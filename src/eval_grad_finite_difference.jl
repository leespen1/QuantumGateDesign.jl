function eval_grad_finite_difference(
        prob::SchrodingerProb, controls,
        pcof::AbstractVector{Float64}, target::AbstractMatrix{Float64};
        dpcof=1e-5, order=2, cost_type=:Infidelity, kwargs...
    )

    N_coeff = length(pcof)
    grad = zeros(N_coeff)
    pcof_l = zeros(N_coeff)
    pcof_r = zeros(N_coeff)

    for i in 1:length(pcof)
        # Centered Difference Approximation
        pcof_r .= pcof
        pcof_r[i] += dpcof

        pcof_l .= pcof
        pcof_l[i] -= dpcof

        history_r = eval_forward(prob, controls, pcof_r; order=order, kwargs...)
        history_l = eval_forward(prob, controls, pcof_l; order=order, kwargs...)

        cost_r = 0.0
        cost_l = 0.0

        ψf_r = history_r[:,1,end,:]
        ψf_l = history_l[:,1,end,:]
        if cost_type == :Infidelity
            cost_r += infidelity(ψf_r, target, prob.N_ess_levels)
            cost_l += infidelity(ψf_l, target, prob.N_ess_levels)
        elseif cost_type == :Tracking
            cost_r += 0.5*norm(ψf_r - target)^2
            cost_l += 0.5*norm(ψf_l - target)^2
        elseif cost_type == :Norm
            cost_r += 0.5*norm(ψf_r)^2
            cost_l += 0.5*norm(ψf_l)^2
        else
            throw("Invalid cost type: $cost_type")
        end

        # Guard penalty
        dt = prob.tf/prob.nsteps
        cost_r += guard_penalty(history_r, dt, prob.tf, prob.guard_subspace_projector)
        cost_l += guard_penalty(history_l, dt, prob.tf, prob.guard_subspace_projector)


        # Calculate partial derivative by centered difference
        grad[i] = (cost_r - cost_l)/(2*dpcof)
    end

    return grad
end

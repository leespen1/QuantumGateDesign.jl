"""
Vector control version.

Evaluates gradient of the provided Schrodinger problem with the given target
gate and control parameter(s) pcof using a finite difference method, where a step
size of dpcof is used when perturbing the components of the control vector pcof.

Returns: gradient
"""
function eval_grad_finite_difference(
        prob::SchrodingerProb, controls,
        pcof::AbstractVector{Float64}, target::AbstractMatrix{Float64}, 
        dpcof=1e-5; order=2, cost_type=:Infidelity,
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

        history_r = eval_forward(prob, controls, pcof_r, order=order)
        history_l = eval_forward(prob, controls, pcof_l, order=order)
        ψf_r = history_r[:,end,:]
        ψf_l = history_l[:,end,:]
        if cost_type == :Infidelity
            cost_r = infidelity(ψf_r, target, prob.N_ess_levels)
            cost_l = infidelity(ψf_l, target, prob.N_ess_levels)
        elseif cost_type == :Tracking
            cost_r = 0.5*norm(ψf_r - target)^2
            cost_l = 0.5*norm(ψf_l - target)^2
        elseif cost_type == :Norm
            cost_r = 0.5*norm(ψf_r)^2
            cost_l = 0.5*norm(ψf_l)^2
        else
            throw("Invalid cost type: $cost_type")
        end
        grad[i] = (cost_r - cost_l)/(2*dpcof)
    end

    return grad
end

function eval_grad_finite_difference(
        prob::SchrodingerProb{M, V}, controls, 
        pcof::AbstractVector{Float64}, target::V; dpcof=1e-5, order=2, 
        cost_type=:Infidelity
    ) where {M<: AbstractMatrix{Float64}, V <: AbstractVector{Float64}}

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

        history_r = eval_forward(prob, controls, pcof_r, order=order)
        history_l = eval_forward(prob, controls, pcof_l, order=order)
        ψf_r = history_r[:, end]
        ψf_l = history_l[:, end]
        if cost_type == :Infidelity
            cost_r = infidelity(ψf_r, target, prob.N_ess_levels)
            cost_l = infidelity(ψf_l, target, prob.N_ess_levels)
        elseif cost_type == :Tracking
            cost_r = 0.5*norm(ψf_r - target)^2
            cost_l = 0.5*norm(ψf_l - target)^2
        elseif cost_type == :Norm
            cost_r = 0.5*norm(ψf_r)^2
            cost_l = 0.5*norm(ψf_l)^2
        else
            throw("Invalid cost type: $cost_type")
        end
        grad[i] = (cost_r - cost_l)/(2*dpcof)
    end

    return grad
end



function eval_grad_finite_difference_arbitrary_order(
        prob::SchrodingerProb, controls,
        pcof::AbstractVector{Float64}, target::AbstractMatrix{Float64}, 
        dpcof=1e-5; order=2, cost_type=:Infidelity,
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

        history_r = eval_forward_arbitrary_order(prob, controls, pcof_r, order=order)
        history_l = eval_forward_arbitrary_order(prob, controls, pcof_l, order=order)
        ψf_r = history_r[:,1,end,:]
        ψf_l = history_l[:,1,end,:]
        if cost_type == :Infidelity
            cost_r = infidelity(ψf_r, target, prob.N_ess_levels)
            cost_l = infidelity(ψf_l, target, prob.N_ess_levels)
        elseif cost_type == :Tracking
            cost_r = 0.5*norm(ψf_r - target)^2
            cost_l = 0.5*norm(ψf_l - target)^2
        elseif cost_type == :Norm
            cost_r = 0.5*norm(ψf_r)^2
            cost_l = 0.5*norm(ψf_l)^2
        else
            throw("Invalid cost type: $cost_type")
        end
        grad[i] = (cost_r - cost_l)/(2*dpcof)
    end

    return grad
end

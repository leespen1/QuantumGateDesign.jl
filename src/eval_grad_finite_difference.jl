"""
    eval_grad_finite_difference(prob, controls, pcof, target; [dpcof=1e-5, order=2, cost_type=:Infidelity, abstol=1e-10, reltol=1e-10])

Compute the gradient using centered difference for each control
parameter. Return the gradient.

# Arguments
- `prob::SchrodingerProb`: Object containing the Hamiltonians, number of timesteps, etc.
- `controls`: An `AstractControl` or vector of controls, where the i-th control corresponds to the i-th control Hamiltonian.
- `pcof::AbstractVector{<: Real}`: The control vector.
- `target::AbstractMatrix{Float64}`: The target gate, in 'stacked' real-valued format.
- `dpcof=1e-5`: The spacing to be used in the centered difference method.
- `cost_type=:Infidelity`: The cost function to use (ONLY USE INFIDELITY, OTHERS HAVE NOT BEEN TESTED RECENTLY)
- `order::Int64=2`: Which order of the method to use.
"""
function eval_grad_finite_difference(
        prob::SchrodingerProb,
        controls,
        pcof::AbstractVector{<: Real},
        target::AbstractVecOrMat{<: Number};
        dpcof::Real=1e-5,
        order::Integer=2,
        cost_type=:Infidelity
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

        history_r = eval_forward(prob, controls, pcof_r; order=order)
        history_l = eval_forward(prob, controls, pcof_l; order=order)


        ψf_r = history_r[:, end, :]
        ψf_l = history_l[:, end, :]

        cost_r = cost_function(ψf_r, target, prob.N_ess_levels, cost_type=cost_type)
        cost_l = cost_function(ψf_l, target, prob.N_ess_levels, cost_type=cost_type)

        # Guard penalty
        dt = prob.tf/prob.nsteps
        cost_r += guard_penalty(history_r, dt, prob.tf, prob.guard_subspace_projector)
        cost_l += guard_penalty(history_l, dt, prob.tf, prob.guard_subspace_projector)

        # Calculate partial derivative by centered difference
        grad[i] = (cost_r - cost_l)/(2*dpcof)
    end

    return grad
end

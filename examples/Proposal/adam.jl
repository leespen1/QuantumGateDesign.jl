using QuantumGateDesign

"""
Looks like it's actually pretty simple to do

α: stepsize
β₁, β₁ ∈ [0,1): Exponential decay rates for the moment estimates
ϵ : Not sure what this is

Good default setting for tested machine learning problems: 
    α = 0.001
    β1 = 0.9,
    β2 = 0.999,
    ϵ = 10^-8
"""
function adam(α, β1, β2, θ0, ϵ, prob::SchrodingerProb, controls, target)
    N_coeff = length(θ0)

    t = 0
    m_vec = [zeros(N_coeff)]
    v_vec = [zeros(N_coeff)]
    g_vec = [missing]
    θ_vec = [θ₀]
    while t < 10 # θₜ not converged
        t = t + 1

        #TODO modify target here to make it stochastic (but even non-stochastic could be used for comparison between gradient descent and L-BFGS)

        m_tminus1 = m_vec[end]
        v_tminus1 = v_vec[end]
        θ_tminus1 = θ_vec[end]

        gt = eval_grad_discrete_adjoint(prob, controls, θ_tminus1, target)
        mt = @. β₁ * m_tminus1 + (1-β₁)*gt # Update biased first moment estimate
        vt = @. β₂ * v_tminus1 + (1-β₂)*(gt .^ 2) # Update biased second raw moment estimate
        mt_bias_corrected = @. mt / (1-β1^t)
        vt_bias_corrected = @. vt / (1-β2^t)
        θt = @. θ_tminus1 - α * mt_bias_corrected / sqrt(vt_bias_corrected + ϵ)

        push!(g_vec, gt)
        push!(m_vec, mt)
        push!(v_vec, vt)
        push!(θ_vec, θ_t)
    end

    return θ_vec, g_vec
end


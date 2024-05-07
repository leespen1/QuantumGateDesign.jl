function eval_hessian(prob::SchrodingerProb, controls, pcof,
        target; dpcof=1e-5, kwargs...
    )

    N_coeff = length(pcof)
    hessian = zeros(N_coeff, N_coeff)
    dummy_pcof = copy(pcof)

    # TODO Check that formula is still 2nd order for i = j
    for i in 1:length(pcof)
      for j in 1:length(pcof)
        if i != j
            hessian_entry = 0.0

            dummy_pcof .= pcof
            dummy_pcof[i] += dpcof
            dummy_pcof[j] += dpcof
            hessian_entry += infidelity_plus_guard(prob, controls, dummy_pcof, target; kwargs...)

            dummy_pcof .= pcof
            dummy_pcof[i] -= dpcof
            dummy_pcof[j] -= dpcof
            hessian_entry += infidelity_plus_guard(prob, controls, dummy_pcof, target; kwargs...)

            dummy_pcof .= pcof
            dummy_pcof[i] += dpcof
            dummy_pcof[j] -= dpcof
            hessian_entry -= infidelity_plus_guard(prob, controls, dummy_pcof, target; kwargs...)

            dummy_pcof .= pcof
            dummy_pcof[i] -= dpcof
            dummy_pcof[j] += dpcof
            hessian_entry -= infidelity_plus_guard(prob, controls, dummy_pcof, target; kwargs...)

            hessian_entry /= (4*dpcof^2)

            hessian[i,j] = hessian_entry
        else
            hessian_entry = 0.0
            dummy_pcof .= pcof
            dummy_pcof[i] += dpcof
            hessian_entry += infidelity_plus_guard(prob, controls, dummy_pcof, target; kwargs...)

            dummy_pcof .= pcof
            hessian_entry -= 2*infidelity_plus_guard(prob, controls, dummy_pcof, target; kwargs...)
    
            dummy_pcof .= pcof
            dummy_pcof[i] -= dpcof
            hessian_entry += infidelity_plus_guard(prob, controls, dummy_pcof, target; kwargs...)

            hessian_entry /= dpcof^2
            hessian[i,j] = hessian_entry
        end
      end
    end

    return hessian
end


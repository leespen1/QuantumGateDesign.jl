"""
Calculates the infidelity for the given state vector 'ψ' and target state
'target.'

Returns: Infidelity
"""
function infidelity(ψ::AbstractVector{Float64}, target::AbstractVector{Float64}, N_ess::Int64)
    R = copy(target)
    N_tot = div(size(target,1), 2)
    T = vcat(R[1+N_tot:end], -R[1:N_tot])
    return 1 - (dot(ψ,R)^2 + dot(ψ,T)^2)/(N_ess^2)
end



"""
Calculates the infidelity for the given matrix of state vectors 'Q' and matrix
of target states 'target.'

Returns: Infidelity
"""
function infidelity(Q::AbstractMatrix{Float64}, target::AbstractMatrix{Float64}, N_ess::Int64)
    R = copy(target)
    N_tot = div(size(target,1), 2)
    T = vcat(R[1+N_tot:end,:], -R[1:N_tot,:])
    return 1 - (dot(Q, R)^2 + dot(Q, T)^2)/(N_ess^2)
end


function infidelity(prob, controls, pcof, target; order=2, forcing=missing, noBLAS=false)
    history = eval_forward_arbitrary_order(prob, controls, pcof, order=order, forcing=forcing, noBLAS=noBLAS)
    final_state = history[:,1,end,:]
    N_ess = prob.N_ess_levels

    return infidelity(final_state, target, N_ess)
end

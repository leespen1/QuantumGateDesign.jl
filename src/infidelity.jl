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

"""
dt is the timestep size.
T is the total time
W projects a state vector onto the guard subspace 
(should decide whether W should have dimensions of real or complex system
if complex, just take I₂×₂ * W.)

"""
function guard_penalty(history::AbstractArray{Float64, 3}, dt, T, W)
    penalty_value = 0.0

    N = size(history, 3)
    W_uv_placeholder = zeros(size(history, 1))
    for i in 1:N
        uv_vec = view(history, :, 1, i)
        mul!(W_uv_placeholder, W, uv_vec)
        if (i == 1 || i == N)
            penalty_value += 0.5*dot(uv_vec, W_uv_placeholder)
        else
            penalty_value += dot(uv_vec, W_uv_placeholder)
        end
    end
    penalty_value *= dt/T

    return penalty_value
end

function guard_penalty(history::AbstractArray{Float64, 4}, dt, T, W)
    penalty_value = 0.0

    for initial_condition_index in 1:size(history, 4)
        history_local = view(history, :, :, :, initial_condition_index)
        penalty_value += guard_penalty(history_local, dt, T, W)
    end

    return penalty_value
end

"""
Need the p's, q's, dpda's, dqda's, dpdt's, dqdt's, and the coss derivatives, to
higher order for higher order. But I think the derivative with respect to a
contorl parameter are always first order.

And the dpda's only need to go to first order

By u
"""
struct Control{N}
    p::Vector{Function}
    q::Vector{Function}
    grad_p::Vector{Function}
    grad_q::Vector{Function}
    #ncoeff::Int64
    function Control(p_vec, q_vec, grad_p_vec, grad_q_vec)
        N = length(p_vec)
        @assert N == length(q_vec) == length(grad_p_vec) == length(grad_q_vec)
        new{N}(p_vec, q_vec, grad_p_vec, grad_q_vec)
    end
end


"""
Alternative constructor. Use automatic differentiation to get derivatives of 
control functions.
"""
function AutoDiff_Control(p::Function, q::Function, N_derivatives::Int)
    p_vec = Vector{Function}(undef, N_derivatives)
    q_vec = Vector{Function}(undef, N_derivatives)
    grad_p_vec = Vector{Function}(undef, N_derivatives)
    grad_q_vec = Vector{Function}(undef, N_derivatives)

    p_vec[1] = p
    q_vec[1] = q

    # Compute time derivatives of control functions
    for i = 2:N_derivatives
        p_vec[i] = (t, pcof) -> ForwardDiff.derivative(t_dummy -> p_vec[i-1](t_dummy, pcof), t)
        q_vec[i] = (t, pcof) -> ForwardDiff.derivative(t_dummy -> q_vec[i-1](t_dummy, pcof), t)
    end
    # Compute gradients of control functions (and time derivatives) with
    # respect to control parameters
    for k = 1:N_derivatives
        grad_p_vec[k] = (t, pcof) -> ForwardDiff.gradient(pcof_dummy -> p_vec[k](t, pcof_dummy), pcof)
        grad_q_vec[k] = (t, pcof) -> ForwardDiff.gradient(pcof_dummy -> q_vec[k](t, pcof_dummy), pcof)
    end

    return Control(p_vec, q_vec, grad_p_vec, grad_q_vec)
end

function auto_increase_order(control_obj::Control{N}, N_derivatives) where N
    p_vec = Vector{Function}(undef, N_derivatives)
    q_vec = Vector{Function}(undef, N_derivatives)
    grad_p_vec = Vector{Function}(undef, N_derivatives)
    grad_q_vec = Vector{Function}(undef, N_derivatives)

    p_vec[1:N] .= control_obj.p
    q_vec[1:N] .= control_obj.q
    grad_p_vec[1:N] .= control_obj.grad_p
    grad_q_vec[1:N] .= control_obj.grad_q

    # Compute time derivatives of control functions
    for i = N+1:N_derivatives
        p_vec[i] = (t, pcof) -> ForwardDiff.derivative(t_dummy -> p_vec[i-1](t_dummy, pcof), t)
        q_vec[i] = (t, pcof) -> ForwardDiff.derivative(t_dummy -> q_vec[i-1](t_dummy, pcof), t)
    end
    # Compute gradients of control functions (and time derivatives) with
    # respect to control parameters
    for k = N+1:N_derivatives
        grad_p_vec[k] = (t, pcof) -> ForwardDiff.gradient(pcof_dummy -> p_vec[k](t, pcof_dummy), pcof)
        grad_q_vec[k] = (t, pcof) -> ForwardDiff.gradient(pcof_dummy -> q_vec[k](t, pcof_dummy), pcof)
    end

    return Control(p_vec, q_vec, grad_p_vec, grad_q_vec)
end

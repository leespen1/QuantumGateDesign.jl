"""
Need the p's, q's, dpda's, dqda's, dpdt's, dqdt's, and the coss derivatives, to
higher order for higher order. But I think the derivative with respect to a
contorl parameter are always first order.

And the dpda's only need to go to first order

Right now I have grad_p, grad_q return the gradients, but in the future I should
require them to be mutating. Or even have each function calculate the i-th
partial derivative, since I usually only need one at a time.

Will need to revise this as I move to systems with more controls (e.g. more qubits)
"""
struct Control{N}
    p::Vector{Function}
    q::Vector{Function}
    grad_p::Vector{Function}
    grad_q::Vector{Function}
    N_coeff::Int64
    function Control(p_vec, q_vec, grad_p_vec, grad_q_vec, N_coeff)
        N = length(p_vec)
        @assert N == length(q_vec) == length(grad_p_vec) == length(grad_q_vec)
        new{N}(p_vec, q_vec, grad_p_vec, grad_q_vec, N_coeff)
    end
end


"""
Alternative constructor. Use automatic differentiation to get derivatives of 
control functions.

Could do it to infinitely high order using lazy arrays.
"""
function AutoDiff_Control(p::Function, q::Function, N_coeff::Int64,
        N_derivatives::Int64)

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

    return Control(p_vec, q_vec, grad_p_vec, grad_q_vec, N_coeff)
end


"""
"""
function auto_increase_order(control_obj::Control{N}, N_derivatives) where N
    @assert N_derivatives >= N

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

    return Control(p_vec, q_vec, grad_p_vec, grad_q_vec, control_obj.N_coeff)
end

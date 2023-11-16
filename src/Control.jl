#=================================================
# 
# Abstract Control Supertype
#
=================================================#
"""
Abstract supertype for all controls.

Every concrete subtype must have the following methods defined:
    eval_p(control::AbstractControl, t::Float64, pcof::AbstractArray{Float64})
    eval_q(control::AbstractControl, t::Float64, pcof::AbstractArray{Float64})

The following methods can also be defined, but have defaults implemented using
automatic differentiation:
    partial_p(control::Control, t::Float64, pcof::AbstractVector{Float64}, 
              coefficient_index::Int, derivative_index::Int)
    partial_q
    eval_grad_p
    eval_grad_q


    eval_pt
    eval_qt
    eval_ptt
    eval_qtt
    ...
    ...


When I have multiple controls, I'm not sure if I should pass in a vector of
control objects, or just one control object which evaluate each of the controls.
I am leaning toward the former option, since it would be easier to implement.
"""
abstract type AbstractControl end


"""
I'm not sure if creating the lambda/anonymous function has a significant
negative impact on performance. If so, I could remedy this by having storing
pcof in the Control object. Then I could have a method eval_p(control, t) which
uses the pcof in the object, and eval_p(control, t, pcof) would mutate the pcof
in the control object.
"""
function eval_pt(control::AbstractControl, t::Float64, pcof::AbstractVector{Float64})
    return ForwardDiff.derivative(t_dummy -> eval_p(t_dummy, pcof), t)
end

function eval_qt(control::AbstractControl, t::Float64, pcof::AbstractVector{Float64})
    return ForwardDiff.derivative(t_dummy -> eval_q(t_dummy, pcof), t)
end

function eval_grad_p(control::AbstractControl, t::Float64, pcof::AbstractVector{Float64})
    return ForwardDiff.gradient(pcof_dummy -> eval_p(control, t, pcof_dummy), pcof)
end

function eval_grad_q(control::AbstractControl, t::Float64, pcof::AbstractVector{Float64})
    return ForwardDiff.gradient(pcof_dummy -> eval_q(control, t, pcof_dummy), pcof)
end


#=================================================
# 
# Bspline/Bcarrier 
#
=================================================#
struct BSplineControl <: AbstractControl
    N_coeff::Int64
    bcpar::bcparams
end

function bspline_control(T::Float64, D1::Int, omega::AbstractVector{Float64})
    pcof = zeros(2*D1*length(omega)) # For now, only doing one coupled pair of control
    omega_bcpar = [omega] # Need to wrap in another vector, since bcparams generally expects multiple controls (multiple frequencies != multiple controls)
    bcpar = bcparams(T, D1, omega_bcpar, pcof)
    return BSplineControl(bcpar.Ncoeff, bcpar)
end


function eval_p(control::BSplineControl, t::Float64, pcof::AbstractArray{Float64})
        return bcarrier2(t, control.bcpar, 0, pcof)
end

function eval_q(control::BSplineControl, t::Float64, pcof::AbstractArray{Float64})
    return bcarrier2(t, control.bcpar, 1, pcof)
end

function eval_pt(control::BSplineControl, t::Float64, pcof::AbstractArray{Float64})
    return bcarrier2_dt(t, control.bcpar, 0, pcof)
end

function eval_qt(control::BSplineControl, t::Float64, pcof::AbstractArray{Float64})
    return bcarrier2_dt(t, control.bcpar, 1, pcof)
end

function eval_grad_p(control::BSplineControl, t::Float64, pcof::AbstractArray{Float64})
    return gradbcarrier2(t, control.bcpar, 0)
end

function eval_grad_q(control::BSplineControl, t::Float64, pcof::AbstractArray{Float64})
    return gradbcarrier2_dt(t, control.bcpar, 1)
end



################################################################################
#
# Old Code, need to keep it around while refactoring
#
################################################################################


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
struct Control{N_derivatives} <: AbstractControl
    p::Vector{Function}
    q::Vector{Function}
    grad_p::Vector{Function}
    grad_q::Vector{Function}
    N_coeff::Int64
    function Control(p_vec, q_vec, grad_p_vec, grad_q_vec, N_coeff)
        N_derivatives = length(p_vec)
        @assert N_derivatives == length(q_vec) == length(grad_p_vec) == length(grad_q_vec)
        new{N_derivatives}(p_vec, q_vec, grad_p_vec, grad_q_vec, N_coeff)
    end
end



function Control(p_vec::Vector{Function}, q_vec::Vector{Function}, N_coeff::Int)
    N_derivatives = length(p_vec)
    @assert N_derivatives == length(q_vec)

    grad_p_vec = Vector{Function}(undef, N_derivatives)
    grad_q_vec = Vector{Function}(undef, N_derivatives)

    # Compute gradients of control functions (and time derivatives) with
    # respect to control parameters
    for k = 1:N_derivatives
        grad_p_vec[k] = (t, pcof) -> ForwardDiff.gradient(pcof_dummy -> p_vec[k](t, pcof_dummy), pcof)
        grad_q_vec[k] = (t, pcof) -> ForwardDiff.gradient(pcof_dummy -> q_vec[k](t, pcof_dummy), pcof)
    end

    return Control(p_vec, q_vec, grad_p_vec, grad_q_vec, N_coeff)
end



"""
Alternative constructor. Use automatic differentiation to get derivatives of 
control functions.

Could do it to implement infinitely high order using lazy arrays.
"""
function Control(p::Function, q::Function, N_coeff::Int, N_derivatives::Int)

    p_vec      = Vector{Function}(undef, N_derivatives)
    q_vec      = Vector{Function}(undef, N_derivatives)
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
Use automatic differentiation to increase the number of derivatives of the
control.
"""
function auto_increase_order(control_obj::Control{N_derivatives_old}, N_derivatives_new) where N_derivatives_old
    @assert N_derivatives >= N

    p_vec      = Vector{Function}(undef, N_derivatives_new)
    q_vec      = Vector{Function}(undef, N_derivatives_new)
    grad_p_vec = Vector{Function}(undef, N_derivatives_new)
    grad_q_vec = Vector{Function}(undef, N_derivatives_new)

    p_vec[1:N_derivatives_old]      .= control_obj.p
    q_vec[1:N_derivatives_old]      .= control_obj.q
    grad_p_vec[1:N_derivatives_old] .= control_obj.grad_p
    grad_q_vec[1:N_derivatives_old] .= control_obj.grad_q

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

"""
Return a control object where the controls are one order time derivative higher
than in the given control object. Thought this would be useful for the forced
gradient method, but it ended up not being relevant.

It might still come in handy as I rework utvt! and uttvtt!
"""
function differentiate_time(control_obj::Control{N_derivatives}) where N_derivatives

    p_vec      = Vector{Function}(undef, N_derivatives)
    q_vec      = Vector{Function}(undef, N_derivatives)
    grad_p_vec = Vector{Function}(undef, N_derivatives)
    grad_q_vec = Vector{Function}(undef, N_derivatives)

    for i in 2:N_derivatives
        p_vec[i-1]      = control_obj.p[i]
        q_vec[i-1]      = control_obj.q[i]
        grad_p_vec[i-1] = control_obj.grad_p[i]
        grad_q_vec[i-1] = control_obj.grad_q[i]
    end

    p_vec[N_derivatives] = (t, pcof) -> ForwardDiff.derivative(
        t_dummy -> p_vec[N_derivatives-1](t_dummy, pcof), t
    )
    q_vec[N_derivatives] = (t, pcof) -> ForwardDiff.derivative(
        t_dummy -> q_vec[N_derivatives-1](t_dummy, pcof), t
    )
    grad_p_vec[N_derivatives] = (t, pcof) -> ForwardDiff.gradient(
        pcof_dummy -> p_vec[N_derivatives](t, pcof_dummy), pcof
    )
    grad_q_vec[N_derivatives] = (t, pcof) -> ForwardDiff.gradient(
        pcof_dummy -> q_vec[N_derivatives](t, pcof_dummy), pcof
    )

    return Control(p_vec, q_vec, grad_p_vec, grad_q_vec, control_prob.N_coeff)
end

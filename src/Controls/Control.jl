#=================================================
# 
# Abstract Control Supertype
#
=================================================#
"""
Abstract supertype for all controls.

Every concrete subtype must have the following methods defined:
    eval_p(control::AbstractControl, t::Real, pcof::AbstractVector{<: Real})
    eval_q(control::AbstractControl, t::Real, pcof::AbstractVector{<: Real})

Every concrete subtype must have the following parameters
    N_coeff::Int
    tf::Float64

The following methods can also be defined, but have defaults implemented using
automatic differentiation:
    # For discrete adjoint / forced gradient calculation
    eval_grad_p
    eval_grad_q
    eval_grad_pt
    eval_grad_qt
    ...
    ...

    # For higher order forward evolution 
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
For compatibility between single and multiple controls.

Similar to how `1[1]` works in Julia

For multiple controls, should pass in a vector of controls. Each element should
have a control which corresponds to a control operator/matrix. For a single
qubit, there should be only one control object, because there is only one
control operator.

Although we might think about "multiple controls" in the
sense that a bcarrier control for a single qubit consists of multiple controls
with different frequencies, it should be considered as only one control.
"""
function Base.getindex(control::AbstractControl, index::Int64)
    if index != 1
        throw(BoundsError(control, index))
    end
    return control
end

function Base.length(control::AbstractControl)
    return 1
end


"""
Get the slice (view) of the control vector which corresponds to the given control index.

Does additions, but doesn't allocate memory.
"""
function get_control_vector_slice(pcof::AbstractVector{<: Real}, controls, control_index::Int64)
    start_index = 1
    for k in 1:(control_index-1)
        start_index += controls[k].N_coeff
    end
    end_index = start_index + controls[control_index].N_coeff - 1

    return view(pcof, start_index:end_index)
end

"""
For human readable display of control objects.
"""
function Base.show(io::IO, ::MIME"text/plain", control::AbstractControl)
    print(io, typeof(control), " with ", control.N_coeff, " control coefficients and final time tf=", control.tf)
end

"""
For iterating over a control. A control is length 1, so only the first
iteration returns something, and that something is the control itself.

Makes it so that functions expecting a vector of control objects will also work
for a single control.
"""
function Base.iterate(control::AbstractControl, state=missing)
    if ismissing(state)
        return (control, nothing)
    end
    return nothing
end

"""
I'm not sure if creating the lambda/anonymous function has a significant
negative impact on performance. If so, I could remedy this by having storing
pcof in the Control object. Then I could have a method eval_p(control, t) which
uses the pcof in the object, and eval_p(control, t, pcof) would mutate the pcof
in the control object.
"""
function eval_pt(control::AbstractControl, t::Real, pcof::AbstractVector{<: Real})
    return ForwardDiff.derivative(t_dummy -> eval_p(control, t_dummy, pcof), t)
end

function eval_qt(control::AbstractControl, t::Real, pcof::AbstractVector{<: Real})
    return ForwardDiff.derivative(t_dummy -> eval_q(control, t_dummy, pcof), t)
end

function eval_grad_p(control::AbstractControl, t::Real, pcof::AbstractVector{<: Real})
    return ForwardDiff.gradient(pcof_dummy -> eval_p(control, t, pcof_dummy), pcof)
end

function eval_grad_q(control::AbstractControl, t::Real, pcof::AbstractVector{<: Real})
    return ForwardDiff.gradient(pcof_dummy -> eval_q(control, t, pcof_dummy), pcof)
end

function eval_grad_pt(control::AbstractControl, t::Real, pcof::AbstractVector{<: Real})
    return ForwardDiff.gradient(pcof_dummy -> eval_pt(control, t, pcof_dummy), pcof)
end

function eval_grad_qt(control::AbstractControl, t::Real, pcof::AbstractVector{<: Real})
    return ForwardDiff.gradient(pcof_dummy -> eval_qt(control, t, pcof_dummy), pcof)
end

"""
Arbitrary order version, only ever uses automatic differentiation to get high
order derivatives.

For hermite-interpolant-envolpe, would override this with a lookup-table, which
throws an error when trying to do a higher order than that of the interpolant 
(or just return 0, I suppose would be more accurate).

Possibly type instability here, since ForwardDiff.derivative causes p_val to be a 
ForwardDiff.Dual{...} type.
"""
function eval_p_derivative(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64)

    p_val = NaN

    if (order == 0) 
        p_val = eval_p(control, t, pcof)
    elseif (order > 0)
        # This will work recursively until we get to order 0 (no derivative, where the function is implemented explicitly )
        # Not sure if the lambda functions will hurt performance significantly
        p_val = ForwardDiff.derivative(t_dummy -> eval_p_derivative(control, t_dummy, pcof, order-1), t)
    else
        throw(ArgumentError("Negative derivative order supplied."))
    end

    return p_val
end

"""
Arbitrary order version, only ever uses automatic differentiation to get high
order derivatives.
"""
function eval_q_derivative(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64)

    q_val = NaN

    if (order == 0) 
        q_val = eval_q(control, t, pcof)
    elseif (order > 0)
        # This will work recursively until we get to order 0 (no derivative, where the function is implemented explicitly )
        # Not sure if the lambda functions will hurt performance significantly
        q_val = ForwardDiff.derivative(t_dummy -> eval_q_derivative(control, t_dummy, pcof, order-1), t)
    else
        throw(ArgumentError("Negative derivative order supplied."))
    end

    return q_val
end

function eval_grad_p_derivative(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64)

    return ForwardDiff.gradient(pcof_dummy -> eval_p_derivative(control, t, pcof_dummy, order), pcof)
end


function eval_grad_q_derivative(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64)

    return ForwardDiff.gradient(pcof_dummy -> eval_q_derivative(control, t, pcof_dummy, order), pcof)
end
#===============================================================================
# 
# Helper types for use in forced gradient method.
# Allows for creation of Controls based on existing controls, but with the time
# derivative taken or a partial derivative taken with respect to one of the
# control parameters. 
#
# The latter method is currently inefficient, as for each partial derivative
# the entire gradient is computed, but it is onle used in the forced gradient
# computation. We only use that method for checking correctness of the discrete
# adjoint, so it is not that important this method efficient (at the moment).
#
===============================================================================#

"""
For use in forced gradient
"""
struct GradControl{T} <: AbstractControl
    N_coeff::Int64
    tf::Float64
    grad_index::Int64
    original_control::T
    function GradControl(original_control::T, grad_index::Int64) where T <: AbstractControl
        new{T}(original_control.N_coeff, original_control.tf, grad_index, original_control)
    end
end

function eval_p(grad_control::GradControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_grad_p(grad_control.original_control, t, pcof)[grad_control.grad_index]
end

function eval_pt(grad_control::GradControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_grad_pt(grad_control.original_control, t, pcof)[grad_control.grad_index]
end

function eval_q(grad_control::GradControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_grad_q(grad_control.original_control, t, pcof)[grad_control.grad_index]
end

function eval_qt(grad_control::GradControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_grad_qt(grad_control.original_control, t, pcof)[grad_control.grad_index]
end

function eval_p_derivative(grad_control::GradControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    return eval_grad_p_derivative(grad_control.original_control, t, pcof, order)[grad_control.grad_index]
end

function eval_q_derivative(grad_control::GradControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    return eval_grad_q_derivative(grad_control.original_control, t, pcof, order)[grad_control.grad_index]
end

struct TimeDerivativeControl{T} <: AbstractControl
    N_coeff::Int64
    tf::Float64
    original_control::T
    function TimeDerivativeControl(original_control::T) where T <: AbstractControl
        new{T}(original_control.N_coeff, original_control.tf, original_control)
    end
end

function eval_p(time_derivative_control::TimeDerivativeControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_p_derivative(time_derivative_control.original_control, t, pcof, 1)
end

function eval_q(time_derivative_control::TimeDerivativeControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_q_derivative(time_derivative_control.original_control, t, pcof, 1)
end


function eval_p_derivative(time_derivative_control::TimeDerivativeControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    return eval_p_derivative(time_derivative_control.original_control, t, pcof, order+1)
end

function eval_q_derivative(time_derivative_control::TimeDerivativeControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    return eval_q_derivative(time_derivative_control.original_control, t, pcof, order+1)
end

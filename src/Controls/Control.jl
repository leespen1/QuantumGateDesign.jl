#=================================================
# 
# Abstract Control Supertype
#
=================================================#
"""
Abstract supertype for all controls.

Every concrete subtype must have the following methods defined:
    eval_p(control::AbstractControl, t::Float64, pcof::AbstractVector{Float64})
    eval_q(control::AbstractControl, t::Float64, pcof::AbstractVector{Float64})

Every concrete subtype must have the following parameters
    N_coeff::Int

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
function get_control_vector_slice(pcof::AbstractVector{Float64}, controls, control_index::Int64)
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
    print(io, typeof(control), " with ", control.N_coeff, " control coefficients")
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

function eval_grad_pt(control::AbstractControl, t::Float64, pcof::AbstractVector{Float64})
    return ForwardDiff.gradient(pcof_dummy -> eval_pt(control, t, pcof_dummy), pcof)
end

function eval_grad_qt(control::AbstractControl, t::Float64, pcof::AbstractVector{Float64})
    return ForwardDiff.gradient(pcof_dummy -> eval_qt(control, t, pcof_dummy), pcof)
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
    original_control::T
    N_coeff::Int64
    grad_index::Int64
    function GradControl(original_control::T, grad_index::Int64) where T <: AbstractControl
        new{T}(original_control, original_control.N_coeff, grad_index)
    end
end

function eval_p(grad_control::GradControl, t::Float64, pcof::AbstractVector{Float64})
    return eval_grad_p(grad_control.original_control, t, pcof)[grad_control.grad_index]
end

function eval_pt(grad_control::GradControl, t::Float64, pcof::AbstractVector{Float64})
    return eval_grad_pt(grad_control.original_control, t, pcof)[grad_control.grad_index]
end

function eval_q(grad_control::GradControl, t::Float64, pcof::AbstractVector{Float64})
    return eval_grad_q(grad_control.original_control, t, pcof)[grad_control.grad_index]
end

function eval_qt(grad_control::GradControl, t::Float64, pcof::AbstractVector{Float64})
    return eval_grad_qt(grad_control.original_control, t, pcof)[grad_control.grad_index]
end


struct TimeDerivativeControl{T} <: AbstractControl
    original_control::T
    N_coeff::Int64
    function TimeDerivativeControl(original_control::T) where T <: AbstractControl
        new{T}(original_control, original_control.N_coeff)
    end
end

function eval_p(time_derivative_control::TimeDerivativeControl, t::Float64, pcof::AbstractVector{Float64})
    return eval_pt(time_derivative_control.original_control, t, pcof)
end

function eval_q(time_derivative_control::TimeDerivativeControl, t::Float64, pcof::AbstractVector{Float64})
    return eval_qt(time_derivative_control.original_control, t, pcof)
end

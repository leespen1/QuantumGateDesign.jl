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

function get_local_control_indices(controls, global_index)
    control_index = 1
    local_index = 1

    #TODO: Add exception here if global index is too large or < 1.
    while (global_index > controls[control_index].N_coeff)
        global_index -= controls[control_index].N_coeff
        control_index += 1
    end

    local_index = global_index

    return control_index, local_index
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

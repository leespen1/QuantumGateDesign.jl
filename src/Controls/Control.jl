#=================================================
# 
# Abstract Control Supertype
#
=================================================#
"""
Abstract supertype for all controls.

Every concrete subtype must have the following methods defined:
# Methods
- `eval_p(control::AbstractControl, t::Real, pcof::AbstractVector{<: Real})`
- `eval_q(control::AbstractControl, t::Real, pcof::AbstractVector{<: Real})`

Every concrete subtype must have the following parameters:
# Parameters
- `N_coeff::Int`
- `tf::Float64`

The following methods can also be handwritten for efficiency, but have defaults
implemented using automatic differentiation (currently broken):
# Optional Methods
- `eval_p_derivative`
- `eval_q_derivative`
- `eval_grad_p_derivative`
- `eval_grad_q_derivative`
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

function get_number_of_control_parameters(controls)
    return sum(control.N_coeff for control in controls)
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
function eval_p_derivative(control::AbstractControl, t::T,
        pcof::AbstractVector{<: Real},  order::Int64)::T where T <: Real
    # Note, the type annotation on t and the return type *REALLY* helps the
    # compiler out. If I don't do this, I get huge type-instability in the
    # 5-arg mul! calls where the return value from this function is used as one
    # of the scalar args. (I think compiler can't tell whether the return type
    # will be a Float64 or a Dual)

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
function eval_q_derivative(control::AbstractControl, t::T,
        pcof::AbstractVector{<: Real},  order::Int64)::T where T <: Real

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

"""
Evaluate gradient of p (or a derivative of p) with respect to the control parameters
"""
function eval_grad_p_derivative(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64)
    return ForwardDiff.gradient(pcof_dummy -> eval_p_derivative_untyped(control, t, pcof_dummy, order), pcof)
end


function eval_grad_q_derivative(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64)
    return ForwardDiff.gradient(pcof_dummy -> eval_q_derivative_untyped(control, t, pcof_dummy, order), pcof)
end

"""
Mutating version. The default is to call the allocating version and copy that.
(which makes mutation not very useful, but for an efficient implementation the
mutating version would be specified directly)
"""
function eval_grad_p_derivative!(grad::AbstractVector{Float64},
        control::AbstractControl, t::Real, pcof::AbstractVector{<: Real},
        order::Int64)

    copy!(grad, eval_grad_p_derivative(control, t, pcof, order))
end

function eval_grad_q_derivative!(grad::AbstractVector{Float64},
        control::AbstractControl, t::Real, pcof::AbstractVector{<: Real},
        order::Int64)

    copy!(grad, eval_grad_q_derivative(control, t, pcof, order))
end


"""
Version for just getting a single index
"""
function eval_grad_p_derivative(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64, pcof_index::Int)
    return eval_grad_p_derivative(control, t, pcof, order)[pcof_index]
end


function eval_grad_q_derivative(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64, pcof_index::Int)
    return eval_grad_q_derivative(control, t, pcof, order)[pcof_index]
end

"""
Version without the restriction on the return type. This version will be
horribly type-unstable in the forward evolution, but I am making this untyped
version until 
"""
function eval_p_derivative_untyped(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64)

    p_val = NaN

    if (order == 0) 
        p_val = eval_p(control, t, pcof)
    elseif (order > 0)
        # This will work recursively until we get to order 0 (no derivative, where the function is implemented explicitly )
        # Not sure if the lambda functions will hurt performance significantly
        p_val = ForwardDiff.derivative(t_dummy -> eval_p_derivative_untyped(control, t_dummy, pcof, order-1), t)
    else
        throw(ArgumentError("Negative derivative order supplied."))
    end

    return p_val
end

function eval_q_derivative_untyped(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64)

    q_val = NaN

    if (order == 0) 
        q_val = eval_q(control, t, pcof)
    elseif (order > 0)
        # This will work recursively until we get to order 0 (no derivative, where the function is implemented explicitly )
        # Not sure if the lambda functions will hurt performance significantly
        q_val = ForwardDiff.derivative(t_dummy -> eval_q_derivative_untyped(control, t_dummy, pcof, order-1), t)
    else
        throw(ArgumentError("Negative derivative order supplied."))
    end

    return q_val
end

"""
Given a time interval [0,tf], divided into `N_regions` regions, and a time `t`.
Return the index of the region to which `t` belongs.
"""
@inline function find_region_index(t, tf, N_regions)
    # Check that t is in the interval
    if (t < 0) || (t > tf*(1+eps()))
        throw(DomainError(t, "Value is outside the interval [0,tf]"))
    end
    # Calculate the width of each region
    region_width = tf / N_regions

    # Find the index of the region to which t belongs
    region_index = min(floor(Int, t / region_width) + 1, N_regions)

    return region_index
end

"""
Given a set of controls and the control vector for all of them, evaluate a 
single control on its portion of the control_vector
"""
function eval_p_single(controls, t, pcof, control_index)
    local_control = controls[control_index]
    local_pcof = get_control_vector_slice(pcof, controls, control_index)
    return eval_p(local_control, t, local_pcof)
end

function eval_q_single(controls, t, pcof, control_index)
    local_control = controls[control_index]
    local_pcof = get_control_vector_slice(pcof, controls, control_index)
    return eval_q(local_control, t, local_pcof)
end

function eval_grad_p_derivative_fin_diff(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64)
    pcof_copy = copy(pcof)
    grad = zeros(length(pcof))
    for i in 1:length(pcof)
        pcof_copy .= pcof
        pcof_copy[i] += 1e-5
        pval_r = eval_p_derivative(control, t, pcof_copy, order)
        pcof_copy[i] -= 2e-5
        pval_l = eval_p_derivative(control, t, pcof_copy, order)
        grad[i] = (pval_r - pval_l)/2e-5
    end
    return grad
end

function eval_grad_q_derivative_fin_diff(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64)
    pcof_copy = copy(pcof)
    grad = zeros(length(pcof))
    for i in 1:length(pcof)
        pcof_copy .= pcof
        pcof_copy[i] += 1e-5
        pval_r = eval_q_derivative(control, t, pcof_copy, order)
        pcof_copy[i] -= 2e-5
        pval_l = eval_q_derivative(control, t, pcof_copy, order)
        grad[i] = (pval_r - pval_l)/2e-5
    end
    return grad
end

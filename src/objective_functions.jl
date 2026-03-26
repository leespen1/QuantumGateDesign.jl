"""
This file is for holding structs and function definitions for objective functions.
The idea is to have a different struct for each objective type, with some
abstract class unifying them, so that users can make their own objective
functions. This should also make it easier to do non-final-time objective
functions.

Ideas:
- function eval(input, p)
- function eval_grad(input, p)
- p contains things like the current time (or final-time as a fraction of total time)
- Don't let the interface make things too complicated. Should still have the
old if-else interface available, since that is the simplest. But use multiple
dispatch or something to allow this interface as an option. Could use function
barriers, maybe, so that apart from the main entry point, internally we are always using this interface
- Could have have "vector/local" and "matrix/global" objectives.
- For an optimization, I think passing something like final_objective=[obj1,
obj2] and always_on_objective=[obj1, obj2] is good.
- `TimeScaledObj" objective which multiples everything by t/tf, for things like
a ramping up over time objective.
- `SumObj` for adding objective functions together (can use + operator to build them)
"""



# TODO Make this type stable
"""
Objective function used to implement infidelity.
"""
struct Infidelity{T <: AbstractMatrix{<: Real}} <: AbstractObjective
    target::T
    R::T
    T::T
end

# TODO Right now R and T are always matirces.
# what is the easiest way to make this use only vectors when the target is a vector?
# I would rather not write another constructor
# TODO right now it is assumed that the target is real-valued. Perhaps
# imaginary would be better?
function Infidelity(target::AbstractMatrix{<: Real})
    @assert iseven(size(target, 1)) "Target should be a real-valued version of a complex matrix. Therefore, size(target, 1) must be even."
    R = target
    N = div(size(R, 1), 2)
    T = vcat(R[1+N:end,:], -R[1:N,:])
    return Infidelity(target, R, T)
end


# TODO since I am doing this in-place, could probably get away with using
# views, or axpy, without copying to T and R
"""
Take the gradient of the infidelity w.r.t to each of the columns of `state`,
and put each gradient in each column of `grad`
"""
function state_gradient!(grad::AbstractMatrix{<: Real}, obj::Infidelity, state::AbstractMatrix{<: Real}, p=nothing)
    c = 2.0 / (size(obj.target, 2)^2) # 2/E²
    grad .= ((c * dot(state, obj.R)) .* obj.R) .+ ((c * dot(state, obj.T)) .* obj.T)
    # TODO check that this broadcast works well
    return grad
end

function value(obj::Infidelity, state::AbstractMatrix{<: Real}, p=nothing)
    return 1 - ((dot(state, obj.R)^2 + dot(state, obj.T)^2) / (size(obj.target, 2)^2))
end



struct GeneralizedInfidelity{T <: AbstractMatrix{<: Real}} <: AbstractObjective
    infidelity_obj::Infidelity{T}
end

function GeneralizedInfidelity(target)
    return GeneralizedInfidelity(Infidelity(target))
end

function state_gradient!(grad::AbstractMatrix{<: Real}, obj::GeneralizedInfidelity, state::AbstractMatrix{<: Real}, p=nothing)
    state_gradient!(grad, obj.infidelity_obj, state, p)
    grad .-= (2 / size(state, 2)) .* state  # TODO can I do a BLAS axpy here?
end

function value(obj::GeneralizedInfidelity, state::AbstractMatrix{<: Real}, p=nothing)
    infidelity = value(obj.infidelity_obj, state, p)
    state_complex = state[1:div(end,2)] + im .* state[div(end,2)+1:end]
    return infidelity - 1 + (1/size(state,2)) * norm(state_complex)^2
    # FIXME Is this correct when state is real instead of complex?
end

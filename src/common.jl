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
Abstract supertype for preconditioners used in the forward evolution and adjoint
evolution in the discrete adjoint method.

This will be used as the left preconditioner in GMRES as implemented by the
`IterativeSolvers` package. Consequently, for a concrete subtype P the
following operations must be defined:
- `ldiv!(y, P, x)`
- `ldiv!(P, x)`
- `P \\ x`.

By default, it is assumed that an `AbstractQGDPreconditioner` has a parameter
`P` which has these operations implemented (i.e. the type simply wraps another
type which can be used as a preconditioner by `IterativeSolvers`).

We must also define the constructor P(prob::SchrodingerProb, order, adjoint),
which will be called each time a forward simulation or gradient calculation is
performed to construct the preconditioners used for the forward and adjoint
linear solves.

This is done so that the preconditioner can easily be changed as the problem
parameters and order of the method change.
"""
abstract type AbstractQGDPreconditioner end

"""
WORK IN PROGRESS

Abstract type for objective functions.
The following operations must be defined
- `value(obj, state::AbstractMatrix{<: Real}, p=nothing)`
- `state_gradient!(grad::AbstractMatrix{<: Real}, obj, state::AbstractMatrix{<: Real}, p=nothing)`
"""
abstract type AbstractObjective end

"""
Currently unused. Idea is to allow distributed computing when the all-time
objectives don't require knowing the states from the other initial/terminal
conditions. Guard penalty should be this type.
"""
abstract type AbstractLocalObjective <: AbstractObjective end


multiples_type(a_type) = Union{a_type, Tuple{Vararg{<: a_type}}, Vector{<: a_type}}
const IntegersType = multiples_type(Integer)
const RealsType = multiples_type(Real)
const ControlsType = multiples_type(AbstractControl)

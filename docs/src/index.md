# HermiteOptimalControl.jl Documentation
```@contents
```

# Workflow
Right now the user can set up Schrodinger problems, and compute the gradients of
them with the control vector and target gate of their choosing. In a complete
package, there would also be an optimization procedure which uses this gradient
calculation.

The basic workflow is:
1. Set up a SchrodingerProblem, either yourself using the constructor
   [SchrodingerProb](@ref), or by
   using one of the example problems provided. [Schrodinger Problem Examples](@ref)
2. Choose a control vector and target.
3. Compute a gradient using one of the methods provided. [Gradient Evaluation](@ref)

# Functions
## Schrodinger Problem Definition
```
SchrodingerProb
```
## Schrodinger Problem Examples
```@docs
rabi_osc
gargamel_prob
bspline_prob
```
## Forward Evolution
Functions for evolving the state vector in a problem forward in time according
to Schrodinger's equation, with or without forcing.
```@docs
eval_forward
eval_forward_forced
```

## Gradient Evaluation
```@docs
discrete_adjoint
eval_grad_forced
eval_grad_finite_difference
infidelity
```

## Bsplines
```@docs
bcparams
bcarrier2
bcarrier2_dt
gradbcarrier2!
gradbcarrier2_dt!
```

# Gradient Evaluation
The gradient can be computed with a function call which is similar to that of 
[`eval_forward`](@ref), but with the additional `target` argument, which is the
target gate we wish to implement.

The intended way of computing the gradient is the discrete adjoint method, which
can be called using the [`discrete_adjoint`](@ref) function, but the functions
[`eval_grad_forced`](@ref) and [`eval_grad_finite_difference`](@ref) can also be
used, for example to check the correctness of the discrete adjoint method.

```@docs
discrete_adjoint
eval_grad_finite_difference
eval_grad_forced
```


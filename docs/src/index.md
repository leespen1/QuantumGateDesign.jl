# QuantumGateDesign.jl
# Table of Contents
```@contents
```
# Problem Description
QuantumGateDesign.jl solves optimization problems of the form
```math
\min_{\boldsymbol{\theta}} \mathcal{J}_1(U(t_f)) + \int_0^{t_f} \mathcal{J}_2
U(t) dt,
```
where the dynamics of $U$ are governed by Schrodinger's equation:
```math
\frac{dU}{dt} = -iH(t;\boldsymbol{\theta}) U,
\quad 0 \leq t \leq t_f,\quad U(0) = U_0 \in \mathbb{C}^{N_s},
```
and the Hamiltonian can be decomposed into a constant 'system'/'drift'
component, and several 'control' hamiltonians which are modulated by scalar
functions:
```math
H(t;\boldsymbol\theta) = H_d + \sum_{j=1}^{N_c} c_j(t; \boldsymbol\theta) \cdot H_j.
```


# Workflow
The basic workflow is:
1. Set up the physics of the [problem](problem_setup.md).
2. Set up the [control functions](control_functions.md).
3. Choose an initial guess for the control vector.
4. Choose a target gate.
5. [Optimize](optimization.md) the control vector using ipopt.

The user can also directly simulate the time evolution of the state vector and
compute gradients of the gate-infidelity, outside of the optimization loop.

# Real-Valued Representation
We perform all our computations using real-valued arithmetic. When the real and
imaginary part of a state vector are not handled in separate data structures, we
'stack' them into one vector. For example, the complex-valued state
$\boldsymbol\psi = [1+2i, 3+4i]^T$ becomes the real-valued state $[1, 2, 3, 4]$.

In several parts of the source code, we use `u` to indicate the real part of a
state, and `v` to indicate the imaginary part. 

For control pulses, the real part is indicated by `p`, and the imaginary part
by `q`.

Eventually we would like to replace the `u`/`v` and `p`/`q` notation with
something more descriptive, like `real`/`imag`.

# State Storage
Because we solve the initial value problem stated above using Hermite
interpolation, after solving the problem we have computed the state vector and
its derivatives at many points in time. For a problem with multiple initial
conditions, the history is stored as a four-dimensional array. The four indices
correspond respectively to
1. the component of the state vector or derivative,
2. the derivative order (with a $1/j!$ factor),
3. the timestep index,
4. and the initial condition.

Therefore, given a 4D array `history`, `history[i,j,k,l]` gives the  $i$-th
component of the real part (for $i \leq N_s$) of the $j$-th derivative (divided
by $j!$) of the state vector corresponding to the $l$-th initial condition after
$k-1$ timesteps.

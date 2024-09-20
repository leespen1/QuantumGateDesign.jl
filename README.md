[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://leespen1.github.io/QuantumGateDesign.jl/dev/)
# QuantumGateDesign.jl
Julia package for designing quantum gates via quantum optimal control using
high-order Hermite methods.

## Installation
`QuantumGateDesign` may be installed using the following commands:
```
julia> ]
pkg> add https://github.com/leespen1/QuantumGateDesign.jl.git
```

## Basic Workflow
The basic workflow consists of the steps
1. Set up a `SchrodingerProb`, which determines the Hamiltonian of the system,
   as well as some other properties involved in the numerical simulation of
   Schrodinger's equation.
2. Set up the controls, which determine the basis used for controlling the pulse
   amplitudes used manipulate the quantum system.
3. Set up the target matrix which defines the target gate to be implemented by
   this package.
4. Choose an initial guess for the control vector.
5. Optimize the control vector by calling `optimize_gate`.

For a more detailed understanding, please read the documentation.

## Current State and Future Work
QuantumGateDesign.jl is still in an early stage of development, so there are
bound to be bugs, and the API is subject to change in the future. If you have
any bug reports, feature suggestions, or questions about how this package works,
please contact Spencer Lee at leespen1@msu.edu.



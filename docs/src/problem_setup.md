# Problem Setup
The parameters that determine the physics of the problem are held in a
[`SchrodingerProb`](@ref).

For a state-transfer problem, please give the initial conditions as column
matrices. There are currently bugs when giving them as vectors.

```@docs
SchrodingerProb
```

```@docs
QuantumGateDesign.guard_projector
```

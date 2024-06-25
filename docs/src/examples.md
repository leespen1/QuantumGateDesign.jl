## Rabi Oscillator
In this example we set up a Rabi oscillator problem, and optimize for a Pauli-X
gate.

The Rabi oscillator consists of a single qubit in the rotating frame, with
the rotation frequency chosen perfectly so that the state vector is
constant-in-time unless a control pulse is applied.

We have one control pulse available in the lab frame (and therefore two in the
rotating frame), which will have a constant amplitude in the rotating frame.

### Setting up the Problem
First we construct the Hamiltonians and initial conditions, and put them
together in a [`SchrodingerProb`](@ref). For a Rabi oscillator the system/drift
Hamiltonian is zero, and the control Hamiltonian has real part $a+a^\dagger$ and
imaginary part $a-a^\dagger$, where $a$ is the lowering/annihilation operator
```math
a = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}.
```
Consequently, the dynamics of the problem are
```math
\frac{dU}{dt} = \left(p_0\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} +iq_0
\begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}\right)U
```
(NOTE: not sure if there should be a $-i$ on the right-hand side when we work in
the complex formulation).

```
using QuantumGateDesign

H0_sym = zeros(2,2)
H0_asym = zeros(2,2)
a = [0.0 1;
     0   0]
Hc_sym = a + a'
Hc_asym = a - a'

sym_ops = [Hc_sym]
asym_ops = [Hc_asym]

U0_complex = [1.0 0;
              0   1]

u0 = real(U0_complex)
v0 = imag(U0_complex)

Ω = 0.5 + 0.0im

# 5 Rabi Oscillations
tf = 10pi / (2*abs(Ω))
nsteps = 10
N_ess_levels = 2

prob = SchrodingerProb(
    H0_sym, H0_asym, sym_ops, asym_ops, u0, v0, tf, nsteps, N_ess_levels
)
```

### Setting up the Control
Next we establish the controls. We use a `GRAPEControl`, which implements a
piecewise constant control (like what is used in the GRAPE method, with one
control coefficient (per real/imaginary part) in order to achieve control pulses
that are constant throughout the duration of the gate.
```
N_control_coeff = 1
control = QuantumGateDesign.GRAPEControl(N_control_coeff, prob.tf)

pcof = [real(Ω), imag(Ω)]
```

### Performing the Optimization
Finally, we set up our target and interface to IPOPT (using
[`optimize_gate`](@ref))to perform the
optimization.
```
# Pauli-X gate, or 1-qubit SWAP
target_complex = [0.0 1;
                  1   0]
target = vcat(real(target_complex), imag(target_complex))

ret_dict = optimize_gate(prob, control, pcof, target, order=4)
```

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

### 2 Qudits in the Dispersive limit
QuantumGateDesign.jl provides convenience functions for setting up
several [`SchrodingerProb`](@ref)'s which arise frequently in quantum computing.
The code below uses the `DispersiveProblem` constructor to make a
`SchrodingerProb` representing two qubits in the dispersive limit.



```
# Set up the SchrodingerProb
subsystem_sizes = (4,4)
essential_subsystem_sizes = (2,2)
transition_freqs = [4.10595, 4.81526] .* 2pi
rotation_freqs = copy(transition_freqs)
# 0.1 for artificially large fast coupling. Actual value is 1e-6
kerr_coeffs = 2pi .* [2*0.1099   0.1
                      0.1        2*0.1126]
tf = 50.0
nsteps = 100
sparse_rep=true

prob = DispersiveProblem(
    subsystem_sizes, essential_subsystem_sizes,
    transition_freqs, rotation_freqs, kerr_coeffs,
    tf, nsteps;
    sparse_rep=sparse_rep
)

# Set up the Controls
carrier_wave_freqs = transition_freqs .- rotation_freqs
bspline_degree = 2
N_knots = 5

D1 = 10
fa = sum(transition_freqs) / length(transition_freqs)
om1 = 2pi .* [transition_freqs[1] - rotation_freqs[1],
              transition_freqs[1] - rotation_freqs[1] - kerr_coeffs[1,2],
             ]
om2 = 2pi .* [transition_freqs[2] - rotation_freqs[2],
              transition_freqs[2] - rotation_freqs[2] - kerr_coeffs[1,2],
             ]

control1 =  BSplineControl(tf, D1, om1)
control2 =  BSplineControl(tf, D1, om2)

controls = [control1, control2]

N_coeff = get_number_of_control_parameters(controls)

amax = 0.04 # Keeping pulse amplitudes below 40 MHz
pcof0 = rand(N_coeff) .* 0.04 / 10

# Set up target gate
CNOT_target = create_gate(
    subsystem_sizes, essential_subsystem_sizes,
    [(1,0) => (1,1),
     (1,1) => (1,0)]
)

# Transform target into rotating frame
rot_mat = rotation_matrix(subsystem_sizes, rotation_freqs, tf)
CNOT_target_complex = real_to_complex(CNOT_target)
CNOT_target_rotating_frame_complex = prod(rot_mat) * CNOT_target_complex
CNOT_target = complex_to_real(CNOT_target_rotating_frame_complex)

# Compare stepsize to relative error in solution history
run_convergence_test = false
if run_convergence_test
    convergence_dict = QuantumGateDesign.get_histories(prob, controls, pcof0, 15, base_nsteps=100, orders=(2,4))
    pl1 = QuantumGateDesign.plot_stepsize_convergence(convergence_dict)
    pl2 = QuantumGateDesign.plot_timing_convergence(convergence_dict)
end

# Use step sizes based on output of above tests
nsteps_order2 = trunc(Int, tf/10.0^(-3.5))
nsteps_order4 = trunc(Int, tf/10.0^(-0.5))

#prob.nsteps = nsteps_order2
prob.nsteps = 100
return_dict_order2 = optimize_gate(prob, controls, pcof0, CNOT_target, order=2, maxIter=50, pcof_L=-amax, pcof_U=amax)

prob.nsteps = nsteps_order4
return_dict_order4 = optimize_gate(prob, controls, pcof0, CNOT_target, order=4, maxIter=50, pcof_L=-amax, pcof_U=amax)
```

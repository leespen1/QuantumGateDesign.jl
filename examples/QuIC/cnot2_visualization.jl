#==========================================================
This routine visualizes the effect of the control parameters on a coupled
2-qubit system with 2 energy levels on each oscillator (with 1 guard state on
one and 2 guard states on the other). The drift Hamiltonian in the rotating
frame is
    H_0 = - 0.5*ξ_a(a^†a^†aa) 
          - 0.5*ξ_b(b^†b^†bb) 
          - ξ_{ab}(a^†ab^†b),
where a,b are the annihilation operators for each qubit.
Here the control Hamiltonian in the rotating frame
includes the usual symmetric and anti-symmetric terms 
H_{sym,1} = p_1(t)(a + a^†),  H_{asym,1} = q_1(t)(a - a^†),
H_{sym,2} = p_2(t)(b + b^†),  H_{asym,2} = q_2(t)(b - b^†).
The problem parameters for this example are,
            ω_a    =  2π × 4.10595   Grad/s,
            ξ_a    =  2π × 2(0.1099) Grad/s,
            ω_b    =  2π × 4.81526   Grad/s,
            ξ_b    =  2π × 2(0.1126) Grad/s,
            ξ_{ab} =  2π × 0.1       Grad/s,
==========================================================# 
using GLMakie
using QuantumGateDesign

Ne1 = 2 # essential energy levels per oscillator 
Ne2 = 2

N_ess_levels = 4 # Total number of essential energy levels in system

tf = 50.0 # Duration of gate
tf = 10.0 # Duration of gate

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
wa = 4.10595    # official
wb = 4.81526   # official
x1 = 2* 0.1099  # official
x2 = 2* 0.1126   # official
x12 = 0.1 # Artificially large to allow fast coupling. Actual value: 1e-6 

subsystem_sizes = [Ne1, Ne2]
transition_freqs = [0.0, 0.0] # Rotating frame with no detuning
kerr_coeffs = [x1  0.0; 
               x12 x2]

system_sym, system_asym = multi_qudit_hamiltonian(subsystem_sizes, transition_freqs, kerr_coeffs)
sym_ops, asym_ops = control_ops(subsystem_sizes)

u0 = zeros(4, 4)
v0 = zeros(4, 4)
for i in 1:size(u0, 2)
    u0[i,i] = 1
end

target = zeros(8,4)
target[1,1] = target[2,2] = target[3,4] = target[4,3] = 1

nsteps = 1000 # Number of timesteps to use in forward evolution

prob = SchrodingerProb(
    system_sym,
    system_asym,
    sym_ops,
    asym_ops,
    u0,
    v0,
    tf,
    nsteps,
    N_ess_levels
)
  
control = QuantumGateDesign.GRAPEControl(1, prob.tf)
controls = [control, control]

visualize_control(controls, prob=prob, target=target)

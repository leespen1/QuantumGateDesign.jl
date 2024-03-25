
using QuantumGateDesign
using Plots
# Single qubit in the rotating frame
system_sym = zeros(2,2)
system_sym[1,1] = 0.84
system_sym[2,2] = 1.78
system_asym = zeros(2,2)

a = [0.0 1; 0 0]
sym_ops = [a + a']
asym_ops = [a - a']

u0 = [1.0 0; 0 1]
v0 = [0.0 0; 0 0]

tf = 3.5 # Change this
nsteps = 500
N_ess_levels = 2

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

# Swap gate, analytic solution should be a pi/2-pulse
# If p = Re(Ω)cos(θ) + Im(Ω)sin(θ), then tf = pi/(2*|Ω|)
target = [0.0 1.0
          1   0
          0   0
          0   0]

control = QuantumGateDesign.GRAPEControl(1, prob.tf+0.1) # For now, it doesn't matter since amplitudes last for whole duration

N_samples = 31
var_range = LinRange(-2, 2, N_samples)
var1s = zeros(N_samples, N_samples)
var2s = zeros(N_samples, N_samples)
infidelities = zeros(N_samples, N_samples)



pl = heatmap(var_range, var_range, infidelities, color=:thermal,
            xlabel="θ₁", ylabel="θ₂", colorbar_title="Infidelity")

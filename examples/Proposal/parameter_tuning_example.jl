
using QuantumGateDesign
using LinearAlgebra
using Plots
# Single qubit in the rotating frame
system_sym = zeros(2,2)
# These entries make the problem more challenging (more like lab frame)
#system_sym[1,1] = 0.84 
#system_sym[2,2] = 1.78
system_asym = zeros(2,2)

a = [0.0 1; 0 0]
sym_ops = [a + a']
asym_ops = [a - a']

u0 = [1.0 0; 0 1]
v0 = [0.0 0; 0 0]

tf = 3.5 # Change this
nsteps = 100
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

N_samples = 11
var_range = LinRange(-2, 2, N_samples)
infidelities = zeros(N_samples, N_samples)

for (j, var1) in enumerate(var_range)
    println("j=", j)
    for (k, var2) in enumerate(var_range)
        pcof = [var1, var2]
        infidelities[j,k] = infidelity(prob, control, pcof, target, order=4)
        #infidelities[j,k] = 1
    end
end


pl1 = heatmap(var_range, var_range, infidelities, color=:thermal,
            xlabel="θ₁", ylabel="θ₂", colorbar_title="Infidelity")

"""
Analytic result for rabi oscillation in rotating frame
(assuming control is constant, could asl)
"""
function analytic_infidelity(complex_Ω, T)
    Ω = abs(complex_Ω)
    θ = angle(complex_Ω)

    U_evo = [cos(Ω*T)     (sin(θ)-im*cos(θ))*sin(Ω*T)
             -(sin(θ)+im*cos(θ))*sin(Ω*T)    cos(Ω*T)]

    U_init = [1 0; 0 1]
    U_final = U_evo*U_init
    U_targ = [0 1; 1 0]

    this_infidelity = 1 - 0.25*abs( dot(U_final, U_targ) )^2

    return this_infidelity
end

analytic_N_samples = 20001
analytic_var_range = LinRange(-2,2,analytic_N_samples)
analytic_infidelities = zeros(analytic_N_samples, analytic_N_samples)

for (j, var1) in enumerate(analytic_var_range)
    println("j=", j)
    for (k, var2) in enumerate(analytic_var_range)
        complex_Ω = var1+im*var2
        analytic_infidelities[j,k] = analytic_infidelity(complex_Ω, prob.tf)
        #infidelities[j,k] = 1
    end
end

pl2 = heatmap(analytic_var_range, analytic_var_range, analytic_infidelities, color=:thermal,
              xlabel="Re(Ω)", ylabel="Im(Ω)", colorbar_title="Infidelity")

#=
#
# transition frequency  ω₁/2pi = 4.8 GHz
# Self-Kerr coefficient ξ₁/2pi = 0.22 GHz
# Detuning              Δ₁ = 0
#
# Amplitude Restrictions: maxₜ|d(t;α)| ≤ c_∞, c_∞/2pi = 9 MHz (in rotating frame)
#
# For d-level SWAP gate, carrier wave frequencies are
# Ω₁,ₖ = (k-1)(-ξ₁), k = 1,2,…,N_f,  N_f = d
# (one frequency for each of the essential states)
#
# d T
# 3 140
# 4 215
# 5 265
# 6 425
#
# Number of spline segments D₁ per frequency 
# D₁ = 10 for d=3,4,5
# D₁ = 20 for d=6
#
=#

N_ess_levels = 2
N_guard_levels = 2
tf = 140.0 # If everything else is in GHz, then I think 140 should be 140 ns
nsteps = 14787

detuning_frequency = 0.0
self_kerr_coefficient =  2*pi*0.22 

prob = rotating_frame_qubit(
    N_ess_levels,
    N_guard_levels,
    tf=tf, 
    detuning_frequency=detuning_frequency,
    self_kerr_coefficient=self_kerr_coefficient
)

#control = bspline_control(tf, )

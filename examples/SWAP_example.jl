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

using HermiteOptimalControl

function main(;d=1, N_guard=2, D1=missing, tf=missing, nsteps=missing)

    N_ess_levels = d+1
    # Set up problem (for d = 1, 2, I just loosely followed pattern for nsteps and duration)

    # If everything else is in GHz, then I think times should be in ns
    duration_defaults = [50.0, 100.0, 140.0, 215.0, 265.0, 425.0]
    if ismissing(tf)
        tf = duration_defaults[d]
    end

    nsteps_defaults = [3500, 7000, 14787, 37843, 69962, 157082] # Defaults, as in Juqbox
    if ismissing(nsteps)
        nsteps = nsteps_defaults[d]
    end

    detuning_frequency = 0.0
    self_kerr_coefficient =  2*pi*0.22 


    prob = rotating_frame_qubit(
        N_ess_levels,
        N_guard,
        tf=tf, 
        detuning_frequency=detuning_frequency,
        self_kerr_coefficient=self_kerr_coefficient,
        nsteps = nsteps
    )

    # Set up control
    if ismissing(D1)
        if d == 6
            D1 = 20
        else
            D1 = 10
        end
    end



    carrier_wave_freqs = [(k-1)*(-self_kerr_coefficient) for k in 1:d]

    control = bspline_control(tf, D1, carrier_wave_freqs)

    pcof = rand(control.N_coeff)

    SWAP_target_complex = zeros(N_ess_levels,N_ess_levels)
    for i in 2:N_ess_levels-1
        SWAP_target_complex[i,i] = 1
    end
    SWAP_target_complex[1,N_ess_levels] = SWAP_target_complex[N_ess_levels,1] = 1

    SWAP_target_real = target_helper(SWAP_target_complex, N_guard)

    return prob, control, pcof, SWAP_target_real
end



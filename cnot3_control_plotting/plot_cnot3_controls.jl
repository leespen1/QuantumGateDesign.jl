using QuantumGateDesign, CairoMakie
import Makie
using QuantumGateDesign: setup_cnot3, get_controls
using Random: MersenneTwister
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

degree = 14
D1 = 15

cnot3ret = setup_cnot3(seed=0, atol=NaN, rtol=NaN, D1=D1, N_osc_levels=10)
controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

N_coeff = QuantumGateDesign.get_number_of_control_parameters(controls)

N_points = 1_001
t_range = LinRange(0, cnot3ret.tf, N_points)
#t_range = LinRange(0, 40.0, N_points)


inch = 96
fig = CairoMakie.Figure(size=(10inch, 5inch))
ax1 = CairoMakie.Axis(
    fig[1,1],
    ylabel="Amplitude",
)
ax2 = CairoMakie.Axis(
    fig[2,1],
    ylabel="Amplitude",
)
ax3 = CairoMakie.Axis(
    fig[3,1],
    xlabel="Time (ns)",
    ylabel="Amplitude",
)
axes = [ax1, ax2, ax3]

full_pcof = fill(NaN, N_coeff)

for seed in 0:4
    full_pcof .= cnot3ret.amax * 2* (0.5 .- rand(MersenneTwister(seed), N_coeff))

    for i in 1:3
        local this_control = controls[i]
        local this_control_pcof = QuantumGateDesign.get_control_vector_slice(full_pcof, controls, i)
        ys = [eval_p(this_control, t, this_control_pcof) for t in t_range]
        lines!(axes[i], t_range, ys, color=(Makie.wong_colors()[1+seed], 0.5))
        #lines!(axes[i], t_range, ys, color=(:dodgerblue, 0.5))
    end
end

fig

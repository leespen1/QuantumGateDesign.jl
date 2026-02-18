#=
# Script for testing that setting up cnot3 matches the results of juqbox.
# (which I assume to be correct)
=#

using QuantumGateDesign, Random, CairoMakie, LaTeXStrings
using Makie: wong_colors
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

recollect = true
if recollect
    println("Recollecting")
    σx = [0.0 1;1 0]
    Z = [0.0 0;0 0]
    σz = [1.0 0;0 -1] .* (0.1/2)
    U0 = [1 0;0 1]
    tf = 50.0
    nsteps = 1000
    N_ess_levels = 2

    degree = 4
    D1 = 8

    prob = SchrodingerProb(σz, [σx], [Z], U0, tf, nsteps, N_ess_levels)
    control = FortranBSplineControl(degree, D1, tf)
    #control = CarrierControl(control, [maximum(eigvals(σz))])
    hadamard_gate = [1 1;1 -1] ./ sqrt(2)

    pcof_l = -0.1
    pcof_u = 0.1
    pcof_l = ones(control.N_coeff)*pcof_l
    pcof_u = ones(control.N_coeff)*pcof_u
    pcof_l[begin] = pcof_l[end] = pcof_u[begin] = pcof_u[end] = 0
    pcof_l[div(end,2)+1] = pcof_l[div(end,2)] = pcof_u[div(end,2)+1] = pcof_u[div(end,2)] = 0

    #pcof0 = zeros(control.N_coeff)
    pcof0 = (0.5 .- rand(control.N_coeff)) .* pcof_u

    ret = optimize_gate(prob, control, pcof0, hadamard_gate, pcof_lbound=pcof_l, pcof_ubound=pcof_u, cost_type=:GeneralizedInfidelity, ipopt_options=["max_iter" => 100])
    pcof_f = ret.x
    println("Recollected")
end

## Plotting


inch = 96
fig = CairoMakie.Figure(size=(4.25inch, 4.0inch), fontsize=11, figure_padding=(0.05inch,0.00inch,0.05inch,0.000inch))
history = eval_forward(prob, control, pcof_f)
population_history = abs.(history) .^ 2
ts = LinRange(0, tf, 1_001)
control_ps = [QuantumGateDesign.eval_p_derivative(control, t, pcof_f, 0) for t in ts] .* 1_000
#control_qs = [QuantumGateDesign.eval_q_derivative(control, t, pcof_f, 0) for t in ts]

ax1 = CairoMakie.Axis(
    fig[1,1:2],
    title="Shaped Control Pulse",
    xlabel="Time (nanoseconds)",
    ylabel="Amplitude (MHz)",
    xticks=0:25:50,
    #limits = ((-0.2,50.2), nothing),
    limits = ((0,50), nothing),
    #limits = (nothing, nothing),
)
lines!(ax1, ts, control_ps, color=wong_colors()[3])

ax2 = CairoMakie.Axis(
    fig[3,1],
    title=L"|0\rangle \rightarrow \frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle",
    xlabel="Time (nanoseconds)",
    ylabel="Population",
    xticks=0:25:50,
    yticks=0:0.25:1,
    limits = ((0,50), nothing),
)

ax3 = CairoMakie.Axis(
    fig[3,2],
    title=L"|1\rangle \rightarrow \frac{1}{\sqrt{2}}|0\rangle - \frac{1}{\sqrt{2}}|1\rangle",
    xlabel="Time (nanoseconds)",
    #ylabel="Population",
    xticks=0:25:50,
    yticks=0:0.25:1,
    yticklabelsvisible = false,
    limits = ((0,50), nothing),
)

#ax3 = CairoMakie.Axis(
#    fig[1,3],
#    ylabel="Population",
#    xlabel="Time (nanoseconds)"
#)

lines!(ax2, ts, population_history[1,:,1], label=L"|0\rangle")
lines!(ax2, ts, population_history[2,:,1], label=L"|1\rangle")

lines!(ax3, ts, population_history[1,:,2], label=L"|0\rangle")
lines!(ax3, ts, population_history[2,:,2], label=L"|1\rangle")

#Label(fig[0, 1], "Control Pulse", halign = :center, valign = :bottom, fontsize = 11, font = Makie.theme(:fonts).bold)
Label(fig[2, 1:2], "Time Evolution of State Populations", halign = :center, valign = :bottom, fontsize = 11, font = Makie.theme(:fonts).bold)


#fig.layout.colsizes[1] = Relative(0.3)
#fig.layout.colsizes[2] = Relative(1/3)
#fig.layout.colsizes[3] = Relative(1/3)

Legend(fig[3,3], ax3, orientation = :vertical, tellwidth = true, framevisible=false)

rowgap!(fig.layout, 1, 0.15inch)
rowgap!(fig.layout, 2, 0.05inch)
#colgap!(fig.layout, 3, 0.05inch)
#colgap!(fig.layout, 2, 0.1inch)


fig

using QuantumGateDesign, DelimitedFiles, Plots, SparseArrays

order = 8
degree = 14
seed = 72
atol = 1e-15
rtol = 1e-15
D1 = 16
nsteps = 1974
cost_type = :GeneralizedInfidelity
Tmax = 550.0
N_osc_levels = 10

nthreads = Threads.nthreads()
cnot3ret = QuantumGateDesign.setup_cnot3(seed=seed, atol=atol, rtol=rtol, D1=D1, N_osc_levels=N_osc_levels, Tmax=Tmax)
prob = cnot3ret.qgd_prob

println("Schrodinger Problem:")
display(prob)


prob.nsteps = nsteps
controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

pcofs = readdlm("error1e-7_seed72_order8_pcofs.csv", ',')
last_pcof = pcofs[end,:]


mkpath("PaperPlots")
p_drift = spy(abs.(prob.system_sym .+ im.*prob.system_asym), title="Drift H (abs vals)", markersize=2)
p_controls = [spy(abs.(prob.sym_operators[i] .+ im.*prob.asym_operators[i]), title="Control $i H (abs vals)", markersize=2) for i in 1:3]
spy_plot = plot(
 p_drift, p_controls...
)
savefig(spy_plot, "PaperPlots/paper_hamiltonians.png")

history = eval_forward(prob, controls, last_pcof, order=order)
history = history[:,:,1] # Only use first initial condition
history_re = history |> real |> transpose
history_im = history |> imag |> transpose
history_pop = abs2.(history) |> transpose
t_grid = LinRange(0,Tmax,1+prob.nsteps)

labels = ["ψ$i" for i in 1:size(history, 1)] |> x -> reshape(x, 1, :)
l = @layout [a b; c]
xlabel="Time (1e-5 seconds)"
trajectory_plot = plot(
         plot(t_grid, history_re, xlabel=xlabel, title="Real", legend=false),
         plot(t_grid, history_im, xlabel=xlabel, title="Imag", legend=false),
         plot(t_grid, history_pop, labels=labels, xlabel=xlabel, title="Population", legend=:outerright),
         layout=l,
)
savefig(trajectory_plot, "PaperPlots/paper_trajectories.png")

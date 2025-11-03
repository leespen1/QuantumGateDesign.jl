using QuantumGateDesign, DelimitedFiles, SparseArrays, Random, LinearAlgebra, Plots

## Read data, set up problem

H = readdlm("H.dlm", '\t', ComplexF64)
Lx = readdlm("Lx.dlm", '\t', ComplexF64) |> real |> sparse
Ly = readdlm("Ly.dlm", '\t', ComplexF64) |> imag |> sparse
power_level = 2pi*50e3
Lx *= power_level
Ly *= power_level
rho_init = readdlm("rho_init.dlm", '\t', ComplexF64)
rho_targ = readdlm("rho_targ.dlm", '\t', ComplexF64)
tf = 1e-5 # Spinach value = 1e-5
nsteps = 10
N_ess_levels = 1 # legacy code, this should be number of initial conditions.
order = 10

prob = SchrodingerProb(
    H,
    [Lx],
    [Ly],
    rho_init,
    tf,
    nsteps,
    N_ess_levels,
    gmres_abstol=1e-15,
    gmres_reltol=1e-15,
)


N_amplitudes = 1 # Spinach value = 100 (but I need smoothness for convergence)
control = GRAPEControl(N_amplitudes, tf)
pcof = 2 .* (rand(MersenneTwister(0), 2*N_amplitudes) .- 0.5)

mkpath("SpinachPlots")

## Visualize hamiltonians

spy_plot = plot(
    spy(sparse(abs.(H)), title="Drift H (abs vals)", markersize=5),
    spy(abs.(Lx .+ (im .* Ly)), title="Control H (abs vals)", markersize=5),
)
savefig(spy_plot, "SpinachPlots/spinach_hamiltonians.png")


check_stepsize = false
    if check_stepsize
    # Check error at each number of timesteps, get a good stepsize before checking solution
    header = hcat("nsteps", "abs_err_L1", "abs_err_L2", "rel_err_L1", "rel_err_L2", "abs_err_Linf")
    writedlm(stdout, header)
    for nsteps in (2 .^ (1:10))
        prob.nsteps = nsteps
        history_h = eval_forward(prob, control, pcof, order=order)[:, 1:2:end, :] # Get every other timestep
        prob.nsteps = div(prob.nsteps, 2)
        history_2h = eval_forward(prob, control, pcof, order=order)
        R = RichardsonExtrapolation(history_h, history_2h, order)
        row = hcat(nsteps, R.abs_err_L1, R.abs_err_L2, R.rel_err_L1, R.rel_err_L2, R.abs_err_Linf)
        writedlm(stdout, row)
    end
end

## Plot trajectories

prob.nsteps = 1024
history = eval_forward(prob, control, pcof, order=order)
history = history[:,:,1] # Only use first initial condition
history_re = history |> real |> transpose
history_im = history |> imag |> transpose
history_pop = abs2.(history) |> transpose
t_grid = LinRange(0,tf,1+prob.nsteps) .* 1e5

labels = ["ψ$i" for i in 1:size(history, 1)] |> x -> reshape(x, 1, :)
l = @layout [a b; c]
xlabel="Time (1e-5 seconds)"
trajectory_plot = plot(
         plot(t_grid, history_re, xlabel=xlabel, title="Real", legend=false),
         plot(t_grid, history_im, xlabel=xlabel, title="Imag", legend=false),
         plot(t_grid, history_pop, labels=labels, xlabel=xlabel, title="Population", legend=:outerright),
         layout=l,
)
savefig(trajectory_plot, "SpinachPlots/spinach_trajectories.png")


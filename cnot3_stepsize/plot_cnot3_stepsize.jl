using CairoMakie, LaTeXStrings, Statistics
import Makie
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

include("processing_stepsize_data.jl")
    

data_directory = "./DataMar28/"
orders = [2,4,6,8,10,12, "2 (Stormer-Verlet)"]
#orders = [2,4,6,8,10,12]
gradient = true
colors = vcat(Makie.wong_colors()[1:6], :darkkhaki)

### Set up Makie Figures, Axes
ticks_10f(i) = L"10^{%$i}"
log10_ticks = (10.0 .^ (-15:15), ticks_10f.(-15:15))

unlabeled_timestep_ticks = (2 .^ (5:5:20), ["" for i in 5:5:20])
labeled_timestep_ticks = (2 .^ (5:5:20), [L"2^{%$i}" for i in 5:5:20])
minor_timestep_ticks = 2 .^ (0:20)

inch = 96 # Getting correct figure, font size: https://docs.makie.org/stable/how-to/match-figure-size-font-sizes-and-dpi
fig = CairoMakie.Figure(size=(6.25inch, 4.5inch), fontsize=11, figure_padding=(0.015inch,0.15inch,0.0inch,0.075inch))

nsteps_vs_error_grid = fig[1,1]
nsteps_vs_time_grid = fig[2,1]
error_vs_time_grid = fig[1:2,2]

ax_nsteps_vs_error = CairoMakie.Axis(
    nsteps_vs_error_grid,
    xscale=CairoMakie.log2,
    yscale=CairoMakie.log10,
    #xlabel="Number of Timesteps",
    ylabel="Mean Relative Error",
    xticks=unlabeled_timestep_ticks,
    xminorticks = minor_timestep_ticks,
    xminorticksvisible = true,
    xminorgridvisible = true,
    yticks=log10_ticks,
    #limits=((2^5, 2^20), (10.0^(-10.25), 10.0^0.25)),
    limits=((2^5, 2^20), (1e-10, 1e0)),
)

ax_nsteps_vs_time = CairoMakie.Axis(
    nsteps_vs_time_grid,
    xscale=CairoMakie.log2,
    yscale=CairoMakie.log10,
    xlabel="Number of Timesteps",
    ylabel="Mean Time to Compute Gradient (s)",
    xticks=labeled_timestep_ticks,
    xminorticks = minor_timestep_ticks,
    xminorticksvisible = true,
    xminorgridvisible = true,
    #xticks=(my_xticks, my_xticklabels),
    yticks=log10_ticks,
    #limits=((2^5, 2^20), (10.0^(-2.25), 10.0^3.25)),
    limits=((2^5, 2^20), (1e-2, 1e3)),
    #title="Elapsed Time Plot",
)

# Share the x-axis
linkxaxes!(ax_nsteps_vs_time, ax_nsteps_vs_error)

ax_error_vs_time = CairoMakie.Axis(
    error_vs_time_grid,
    xscale=CairoMakie.log10,
    yscale=CairoMakie.log10,
    xlabel="Mean Relative Error",
    ylabel="Mean Time to Compute Gradient (s)",
    #xminorticksvisible = true,
    #xminorgridvisible = true,
    xticks=log10_ticks,
    yticks=log10_ticks,
    yminorticks=IntervalsBetween(10),
    yminorticksvisible=true,
    yminorgridvisible=true,
    #title="Elapsed Time Plot",
    #limits=((10.0^(-10.25), 10.0^(-0.75)), (10.0^(-0.25),10.0^(5.25))),
    #limits=((1e-10, 1e-1), (10.0^(-0.25),10.0^(5.25))),
    limits=((1e-10, 1e-1), (1e0,1e5)),
)


for (k, order) in enumerate(orders)
    if order == "2 (Stormer-Verlet)"
        label = "Order 2 (St\u00f6rmer-Verlet)"
        juqbox = true
        order = 2
    else
        label = "Order $order"
        juqbox = false
    end
    ### Collect data
    target_labels = ("nsteps", "elapsed_time", "forward_time", "adjoint_time", "grad_accum_time",  "R_rel_err_L2")
    target_symbols = Symbol.(target_labels)
    entries = NamedTuple{target_symbols}(
        get_data(target_labels, order, data_directory=data_directory, juqbox=juqbox, gradient=gradient)
    )

    ### Process data, grabbing data columns from multiple files (possibly with
    # different lengths), combine them into a matrix
    full_nsteps_vec = combined_x_vec(entries.nsteps)
    elapsedtime_ymat = get_y_mat(entries.nsteps, entries.elapsed_time, full_nsteps_vec)
    elapsedtime_meanvec = mean(elapsedtime_ymat, dims=2) |> vec
    # Handle relative error from final states, which is calculated from a
    # separate file, and must be handeled differently
    relerr_nsteps_entries, relerr_err_entries = get_nsteps_errors_final_states(
        order, data_directory=data_directory, juqbox=juqbox, gradient=gradient
    )
    relerr_ymat = get_y_mat(relerr_nsteps_entries, relerr_err_entries, full_nsteps_vec)
    relerr_meanvec = mean(relerr_ymat, dims=2) |> vec

    ##### Make Plots
    ### Number of Timesteps vs Elapsed Time
    lines_obj = lines!(
        ax_nsteps_vs_time, full_nsteps_vec, elapsedtime_meanvec;
        color=colors[k], label=label
    )
    CairoMakie.translate!(lines_obj, 0, 0, -k) # Put the lines in the right z-order

    ### Number of Timesteps vs Relative Error in Final State
    lines_obj = lines!(
        ax_nsteps_vs_error, full_nsteps_vec, relerr_meanvec;
        color=colors[k], label=label
    )
    CairoMakie.translate!(lines_obj, 0, 0, -k) # Put the lines in the right z-order

    ### Relative Error vs Elapsed Time
    scatter_obj = lines!(
        ax_error_vs_time, relerr_meanvec, elapsedtime_meanvec;
        color=colors[k], label=label
    )
end

# Add a hard-coded line to approximately extrapolate 2nd order Hemite and Stormer-Verlet
dummy_relerr_mean = [10.0 ^(-i) for i in 4.25:11]
dummy_elapsedtime_mean = 1.1 .* (dummy_relerr_mean .^ -0.47)
lines!(
    ax_error_vs_time, dummy_relerr_mean, dummy_elapsedtime_mean;
    color=:black, linestyle=:dot, label="Order 2 (Approximate Extrapolation)"
)
    
# Add legend to very end
Legend(fig[end+1,:], ax_error_vs_time, orientation = :horizontal, tellwidth = false, nbanks=2, framevisible=false)
#rowsize!(fig.layout, 1, Relative(0.6))
rowgap!(fig.layout, 1, 0.125inch) # Minimize space between legend and plot area.
rowgap!(fig.layout, 2, 0.05inch) # Minimize space between legend and plot area.
colgap!(fig.layout, 1, 0.15inch) # Minimize space between legend and plot area.

fig

using CairoMakie, LaTeXStrings, Statistics
import Makie
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

include("processing_stepsize_data.jl")
    

#data_directory = "./49076519/49076519/"
data_directory = "./DataMar18/"
orders = [2,4,6,8,10,12, "2 (Stormer-Verlet)"]
colors = vcat(Makie.wong_colors()[1:6], :darkkhaki)

### Set up Makie Figures, Axes
ticks_10f(i) = L"10^{%$i}"
log10_ticks = (10.0 .^ (-15:15), ticks_10f.(-15:15))

unlabeled_timestep_ticks = (2 .^ (5:5:20), ["" for i in 5:5:20])
labeled_timestep_ticks = (2 .^ (5:5:20), [L"2^{%$i}" for i in 5:5:20])
minor_timestep_ticks = 2 .^ (0:20)

inch = 96 # Getting correct figure, font size: https://docs.makie.org/stable/how-to/match-figure-size-font-sizes-and-dpi
fig = CairoMakie.Figure(size=(6inch, 4inch), fontsize=12, figure_padding=5)
fig2 = CairoMakie.Figure(size=(4inch, 4inch), fontsize=12, figure_padding=5)

ax_nsteps_vs_error = CairoMakie.Axis(
    fig[1,1],
    xscale=CairoMakie.log2,
    yscale=CairoMakie.log10,
    xlabel="Number of Timesteps",
    ylabel="Mean Relative Error",
    xticks=labeled_timestep_ticks,
    xminorticks = minor_timestep_ticks,
    xminorticksvisible = true,
    xminorgridvisible = true,
    yticks=log10_ticks,
    limits=((2^5, 2^19), (10.0^(-10.25), 10.0^0.25)),
)

ax_nsteps_vs_time = CairoMakie.Axis(
    fig[1,2],
    xscale=CairoMakie.log2,
    yscale=CairoMakie.log10,
    xlabel="Number of Timesteps",
    ylabel="Mean Elapsed Wall Time (s)",
    xticks=labeled_timestep_ticks,
    xminorticks = minor_timestep_ticks,
    xminorticksvisible = true,
    xminorgridvisible = true,
    #xticks=(my_xticks, my_xticklabels),
    yticks=log10_ticks,
    limits=(nothing, (10.0^(-3.25), 10.0^3.25)),
    #limits=((2^5, 2^19), (10.0^(-3.25), 10.0^3.25)),
    #title="Elapsed Time Plot",
)

linkxaxes!(ax_nsteps_vs_time, ax_nsteps_vs_error)

ax_error_vs_time = CairoMakie.Axis(
    fig2[1,1],
    xscale=CairoMakie.log10,
    yscale=CairoMakie.log10,
    xlabel="Mean Relative Error",
    ylabel="Mean Elapsed Wall Time (s)",
    #xminorticksvisible = true,
    #xminorgridvisible = true,
    xticks=log10_ticks,
    yticks=log10_ticks,
    #title="Elapsed Time Plot",
    limits=((10.0^(-10.25), 10.0^(-0.75)), (1e-1,1e5)),
)


for (k, order) in enumerate(orders)
    if order == "2 (Stormer-Verlet)"
        label = "Order 2 (Stormer-Verlet)"
        juqbox = true
        order = 2
    else
        label = "Order $order"
        juqbox = false
    end
    ### Collect data
    target_labels = ("nsteps", "elapsed_time", "R_rel_err_L2")
    target_symbols = Symbol.(target_labels)
    entries = NamedTuple{target_symbols}(
        get_data(target_labels, order, data_directory=data_directory, juqbox=juqbox)
    )

    ### Process data, grabbing data columns from multiple files (possibly with
    # different lengths), combine them into a matrix
    full_nsteps_vec = combined_x_vec(entries.nsteps)
    elapsedtime_ymat = get_y_mat(entries.nsteps, entries.elapsed_time, full_nsteps_vec)
    elapsedtime_meanvec = mean(elapsedtime_ymat, dims=2) |> vec
    # Handle relative error from final states, which is calculated from a
    # separate file, and must be handeled differently
    relerr_nsteps_entries, relerr_err_entries = get_nsteps_errors_final_states(
        order, data_directory=data_directory, juqbox=juqbox
    )
    relerr_ymat = get_y_mat(relerr_nsteps_entries, relerr_err_entries, full_nsteps_vec)
    relerr_meanvec = mean(relerr_ymat, dims=2) |> vec
    

    ##### Make Plots
    ### Number of Timesteps vs Elapsed Time
    lines_obj = lines!(
        ax_nsteps_vs_time, full_nsteps_vec[5:end], elapsedtime_meanvec[5:end];
        color=colors[k], label=label
    )
    CairoMakie.translate!(lines_obj, 0, 0, -k) # Put the lines in the right z-order

    ### Number of Timesteps vs Relative Error in Final State
    lines_obj = lines!(
        ax_nsteps_vs_error, full_nsteps_vec[5:end], relerr_meanvec[5:end];
        color=colors[k], label=label
    )
    CairoMakie.translate!(lines_obj, 0, 0, -k) # Put the lines in the right z-order

    ### Relative Error vs Elapsed Time
    scatter_obj = lines!(
        ax_error_vs_time, relerr_meanvec[5:end], elapsedtime_meanvec[5:end];
        color=colors[k], label=label
    )
    # Add a hard-coded line to approximately extrapolate 2nd order Hemite and Stormer-Verlet
    dummy_relerr_mean = [10.0 ^(-i) for i in 5:10]
    dummy_elapsedtime_mean = 0.25 .* (dummy_relerr_mean .^ -0.47)
    scatter_obj = lines!(
        ax_error_vs_time, dummy_relerr_mean, dummy_elapsedtime_mean;
        color=colors[k], label=label, linestyle=:dot
    )
    CairoMakie.translate!(scatter_obj, 0, 0, -k) # Put the lines in the right z-order
end
    
Legend(fig[2,:], ax_nsteps_vs_error, orientation = :horizontal, tellwidth = false, nbanks=2, framevisible=false)
rowgap!(fig.layout, 1, 0)

fig2

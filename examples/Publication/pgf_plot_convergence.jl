#using PGFPLotsX

function pgf_plot_stepsize_convergence(dict_of_summary_dicts=Dict(); fontsize=16,
        true_history=missing)

    #xlabel = "Log₁₀(Step Size Δt)"
    #ylabel = "Log₁₀(Rel Err)"
    #yticks = -15:15 

    legend_entries = []
    coordinate_entries = []
    plots = []

    colors=["blue", "red", "magenta", "green", "orange"]

    i = 1
    for (key, summary_dict) in dict_of_summary_dicts

        step_sizes = summary_dict["step_sizes"]

        rel_err(x_approx, x_true) = norm(x_approx - x_true)/norm(x_true)

        if ismissing(true_history)
            errors = summary_dict["richardson_errors"]
        else
            errors = [rel_err(history, true_history) for history in summary_dict["histories"]]
        end

        # Convert to Log scale
        step_sizes = log10.(step_sizes)
        errors = log10.(errors)

        
        coords = Coordinates(step_sizes, errors)

        options = @pgf {color=colors[i], mark="o"}
        #options = @pgf {no_marks, color = "red", mark = "x"}

        this_plot = @pgf Plot(options, coords)
        #this_plot = @pgf Plot(coords) # No options

        #display(this_plot.options)

        push!(legend_entries, key)
        push!(coordinate_entries, coords)
        push!(plots, this_plot)
        i += 1
    end


    x = range(-1; stop = 1, length = 51) # so that it contains 1/0
    ret = @pgf Axis(
        {
            title="Sample Text",
            xlabel=raw"Log$_{10}$ Step Size (ns)",
            ylabel=raw"Log$_{10}$ Relative Error",
            legend_pos="outer north east",
            #enlargelimits = 0.15, # Make space for the legend
            #legend_style =
            #{
            #    at = Coordinate(0.5, -0.15),
            #    anchor = "north",
            #    legend_columns = -1
            #},
            xmajorgrids,
            ymajorgrids,
        },
        plots,
        Legend(legend_entries),
    )


    return ret
end

function pgf_plot_timing_convergence(dict_of_summary_dicts=Dict(); fontsize=16,
        true_history=missing)

    #xlabel = "Log₁₀(Step Size Δt)"
    #ylabel = "Log₁₀(Rel Err)"
    #yticks = -15:15 

    legend_entries = []
    coordinate_entries = []
    plots = []

    colors=["blue", "red", "magenta", "green", "orange"]

    i = 1
    for (key, summary_dict) in dict_of_summary_dicts

        elapsed_times = summary_dict["elapsed_times"]

        rel_err(x_approx, x_true) = norm(x_approx - x_true)/norm(x_true)

        if ismissing(true_history)
            errors = summary_dict["richardson_errors"]
        else
            errors = [rel_err(history, true_history) for history in summary_dict["histories"]]
        end

        # Convert to Log scale
        elapsed_times = log10.(elapsed_times)
        errors = log10.(errors)

        
        coords = Coordinates(elapsed_times, errors)

        options = @pgf {color=colors[i], mark="o"}
        #options = @pgf {no_marks, color = "red", mark = "x"}

        this_plot = @pgf Plot(options, coords)
        #this_plot = @pgf Plot(coords) # No options

        #display(this_plot.options)

        push!(legend_entries, key)
        push!(coordinate_entries, coords)
        push!(plots, this_plot)
        i += 1
    end


    x = range(-1; stop = 1, length = 51) # so that it contains 1/0
    ret = @pgf Axis(
        {
            title="Sample Text",
            xlabel=raw"Log$_{10}$ Elapsed Time (s)",
            ylabel=raw"Log$_{10}$ Relative Error",
            legend_pos="outer north east",
            #enlargelimits = 0.15, # Make space for the legend
            #legend_style =
            #{
            #    at = Coordinate(0.5, -0.15),
            #    anchor = "north",
            #    legend_columns = -1
            #},
            xmajorgrids,
            ymajorgrids,
        },
        plots,
        Legend(legend_entries),
    )


    return ret
end

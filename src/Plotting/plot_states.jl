"""
"""
function plot_states(history::AbstractArray{Float64, 4}; ts=missing,
        level_indices=missing, labels=missing)
    if ismissing(ts)
        ts = 0:size(history, 3)-1
        xlabel = "Timestep #"
    else
        xlabel = "Time (ns)"
    end

    if ismissing(level_indices)
        complex_system_size = div(size(history, 1), 2)
        level_indices = 1:complex_system_size
    end

    if ismissing(labels)
        re_labels = ["Level $i (real)" for i in level_indices]
        im_labels = ["Level $i (imag)" for i in level_indices]
    end


    ret = []
    # Iterate over initial conditions
    for initial_condition in 1:size(history, 4)
        title = "Initial Condition $initial_condition"
        pl = Plots.plot(xlabel=xlabel, ylabel="Population", 
                        title=title, legend=:outerright)
        # Iterate over essential states
        for (i, level_index) in enumerate(level_indices)
            Plots.plot!(pl, ts, history[level_index, 1, :, initial_condition],
                  label=re_labels[i], linecolor=i)
            Plots.plot!(pl, ts, history[level_index+complex_system_size, 1, :, initial_condition],
                  label=im_labels[i], linecolor=i, linestyle=:dash)
        end
        push!(ret, pl)
    end
    return ret
end

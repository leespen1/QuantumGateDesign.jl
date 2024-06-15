"""
    plot_populations(history; [ts=missing, level_indices=missing, labels=missing])

Given state vector history, plot population evolution (assuming single qubit for labels).
"""
function plot_populations(history::AbstractArray{Float64, 4}; ts=missing,
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
        labels = ["Level $i" for i in level_indices]
    end
    labels = reshape(labels, 1, :) # Labels must be a row matrix

    
    populations = get_populations(history)
    N_levels = size(populations, 1)


    ret = []
    # Iterate over initial conditions
    for initial_condition in 1:size(populations, 3)
        title = "Initial Condition $initial_condition"
        pl = Plots.plot(xlabel=xlabel, ylabel="Population", 
                        title=title, legend=:outerright)
        # Iterate over essential states
        for (i, level_index) in enumerate(level_indices)
            Plots.plot!(pl, ts, populations[level_index, :, initial_condition],
                  label=labels[i])
        end
        push!(ret, pl)
    end
    return ret
end

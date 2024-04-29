"""
Given state vector history, plot population evolution (assuming single qubit for labels).

Should make a version that takes a Schrodinger problem as an input, so I can
get the number of subsystems, etc.

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
        labels = [missing for i in level_indices]
    end

    
    populations = get_populations(history)
    N_levels = size(populations, 1)

    if !ismissing(labels)
        labels = reshape(labels, 1, :) # Labels must be a row matrix
    end

    ret = []
    # Iterate over initial conditions
    for initial_condition in 1:size(populations, 3)
        title = "Initial Condition $initial_condition"
        pl = Plots.plot(xlabel=xlabel, ylabel="Population", 
                        title=title, legend=false)
        # Iterate over essential states
        for (i, level_index) in enumerate(level_indices)
            Plots.plot!(pl, ts, populations[level_index, :, initial_condition],
                  label=labels[i])
        end
        push!(ret, pl)
    end
    return ret
end

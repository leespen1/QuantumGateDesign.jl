"""
Convert state vector history to population history.

Each column of history should be the stacked real and imaginary parts of the 
state vector: history[:,i] = vcat(u_i, v_i)
"""
function get_populations(history::AbstractMatrix{Float64})
    N_tot_levels = div(size(history, 1), 2) # 
    N_times = size(history, 2)

    populations = Matrix{Float64}(undef, N_tot_levels, N_times)
    for time in 1:N_times
        for level in 1:N_tot_levels
            populations[level, time] = history[level, time]^2 + history[level+N_tot_levels, time]^2
        end
    end

    return populations
end

"""
Convert possibly complex target to real-valued one, and pad with guard levels.
"""
function target_helper(target, N_guard_levels=0)
    N_essential_levels = size(target, 1)
    N_tot_levels = N_essential_levels + N_guard_levels
    N_initial_conditions = size(target, 2)

    target_real_valued = zeros(2*N_tot_levels, N_initial_conditions)
    target_real_valued[1:N_essential_levels, 1:N_initial_conditions] .= real.(target)
    target_real_valued[N_tot_levels+1:N_tot_levels+N_essential_levels, 1:N_initial_conditions] .= imag.(target)
        
    return target_real_valued
end


"""
Given state vector history, plot population evolution (assuming single qubit for labels).

Should make a version that takes a Schrodinger problem as an input, so I can
get the number of subsystems, etc.

"""
function plot_populations(history::AbstractMatrix{Float64})
    populations = get_populations(history)
    N_levels = size(populations, 1)

    labels = ["|$i⟩" for i in 1:N_levels]
    labels = reshape(labels, 1, length(labels)) # Labels must be a row matrix

    pl = Plots.plot(transpose(populations), labels=labels, ylabel="Population")
end

"""
Given state vector history for several initial conditions, plot population evolution
(assuming single qubit for labels).
"""
function plot_populations(history::AbstractArray{Float64, 3})
    N_initial_conditions = size(history, 3)
    population_graphs = [plot_populations(history[:,:,i]) for i in 1:N_initial_conditions]

    pl = Plots.plot(population_graphs..., layout = length(population_graphs))
    return pl
end


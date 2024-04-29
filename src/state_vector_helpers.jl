"""
Given a history array, with indices
    history[component, derivative_order, time_index, initial_condition_index]

Return state population array, with indices
    populations[component, time_index, initial_condition_index]

(drops derivative axis)
"""
function get_populations(history::AbstractArray{Float64, 4})
    real_system_size = size(history, 1)
    complex_system_size = div(real_system_size, 2)

    N_times = size(history, 3)
    N_initial_conditions = size(history, 4)

    populations = zeros(complex_system_size, N_times, N_initial_conditions)

    for i in 1:N_initial_conditions
        populations[:,:,i] .= get_populations(view(history, :, :, :, i))
    end

    return populations
end



"""
Given a history array, with indices
    history[component, derivative_order, time_index]

Return state population array, with indices
    populations[component, time_index]

(drops derivative axis)
"""
function get_populations(history::AbstractArray{Float64, 3})
    real_system_size = size(history, 1)
    complex_system_size = div(real_system_size, 2)

    N_times = size(history, 3)

    populations = zeros(complex_system_size, N_times)

    for n in 1:N_times
        for k in 1:complex_system_size
            populations[k, n] = history[k, 1, n]^2 + history[k+complex_system_size, 1, n]^2
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

    target_real_valued::Matrix{Float64} = zeros(2*N_tot_levels, N_initial_conditions)
    target_real_valued[1:N_essential_levels, 1:N_initial_conditions] .= real.(target)
    target_real_valued[N_tot_levels+1:N_tot_levels+N_essential_levels, 1:N_initial_conditions] .= imag.(target)
        
    return target_real_valued
end



function complex_to_real(x)
    return vcat(real(x), imag(x))
end



function real_to_complex(x)
    N = div(size(x, 1), 2)

    # Get "upper" and "lower" parts of vector, matrix, or 3D-array, 
    # corresponding to real and imaginary part (selectdim makes it easy to generalize)
    real_x = selectdim(x, 1, 1:N)
    imag_x = selectdim(x, 1, 1+N:2*N)

    return real_x .+ (im .* imag_x)
end



function real_to_complex(x_real, x_imag)
    return x_real .+ (im .* x_imag)
end



function initial_basis(N_ess, N_guard)
    N_tot = N_ess + N_guard
    u0 = zeros(N_tot, N_ess)
    v0 = zeros(N_tot, N_ess)
    for i in 1:N_ess
        u0[i,i] = 1
    end
    return u0, v0
end

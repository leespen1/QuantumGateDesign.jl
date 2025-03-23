using CairoMakie, DelimitedFiles, LaTeXStrings, LinearAlgebra, Statistics
using IterTools
#using Typst_jll
import Makie
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

function get_data(x_header, y_header, out_order, data_directory=missing, juqbox=false)
    file_pattern= r"""cnot3StepsizeTest
    _order=(\d+)
    _degree=(\d+)
    _seed=(\d+)
    _atol=(.*)
    _rtol=(.*)
    _D1=(\d+)
    _time=(.*)
    _nthreads=(\d+)
    (?:_usejuqbox=(true|false))? # Optionally match , doesn't appear in older files
    \.csv
    """x # 'x' tag ignores whitespace and comments

    x_data_entries = Any[]
    y_data_entries = Any[]

    if ismissing(data_directory)
        data_directory = dirname(@__FILE__) * "/Data/"
    end

    files_found = 0
    files = readdir(data_directory)
    for file in files
        if occursin(file_pattern, file)
            regex_match = match(file_pattern, file)
            order    = parse(Int,     regex_match[1])
            degree   = parse(Int,     regex_match[2])
            seed     = parse(Int,     regex_match[3])
            atol     = parse(Float64, regex_match[4])
            rtol     = parse(Float64, regex_match[5])
            D1       = parse(Int,     regex_match[6])
            time     = parse(Float64, regex_match[7])
            nthreads = parse(Int,     regex_match[8])
            usejuqbox = regex_match[9] == "true" ? true : false

            if ((order == out_order) && (juqbox == usejuqbox))
                files_found += 1
                filepath = data_directory * "/" * file

                data, header = readdlm(filepath, ',', header=true)
                header = vec(header)
               
                x_index = findfirst(x -> x == x_header, header)
                y_index = findfirst(x -> x == y_header, header)

                x_data = data[:, x_index]
                y_data = data[:, y_index]

                push!(x_data_entries, x_data)
                push!(y_data_entries, y_data)
            end
        end
    end

    if files_found == 0
        @warn "No files found matching conditions!"
    end

    return x_data_entries, y_data_entries

end

function get_x_vec_y_mat(x_data_entries, y_data_entries)
    if length(x_data_entries) == 0
        @warn "Length of data entries is zero, returning empty vectors and matrices"
        return zeros(0), zeros(0,0), zeros(0,0)
    end

    x_data_lengths = length.(x_data_entries)
    max_length = maximum(x_data_lengths)
    min_length = minimum(x_data_lengths)
    if max_length != min_length 
        @warn "Not all x_data entries are the same length. Max is $max_length, min is $min_length."
    end

    # Get the vector from x_data_entries with the longest length
    x_vec = argmax(length, x_data_entries)

    n_entries = length(x_data_entries)
    x_mat = fill(NaN, max_length, n_entries)
    y_mat = fill(NaN, max_length, n_entries)
    for i in 1:n_entries
        x_data = x_data_entries[i]
        y_data = y_data_entries[i]
        l = length(y_data)

        x_mat[1:l,i] .= x_data
        y_mat[1:l,i] .= y_data
    end

    return x_vec, x_mat, y_mat
end

function get_data_final_states(out_order, data_directory=missing, juqbox=false)
    file_pattern= r"""cnot3StepsizeTest
    _order=(\d+)
    _degree=(\d+)
    _seed=(\d+)
    _atol=(.*)
    _rtol=(.*)
    _D1=(\d+)
    _time=(.*)
    _nthreads=(\d+)
    (?:_usejuqbox=(true|false))? # Optionally match , doesn't appear in older files
    _finalStates
    \.csv
    """x # 'x' tag ignores whitespace and comments



    nsteps_vec_entries = Any[]
    relerr_vec_entries = Any[]

    if ismissing(data_directory)
        data_directory = dirname(@__FILE__) * "/Data/"
    end

    files_found = 0
    files = readdir(data_directory)
    for file in files
        if occursin(file_pattern, file)
            regex_match = match(file_pattern, file)
            order    = parse(Int,     regex_match[1])
            usejuqbox = regex_match[9] == "true" ? true : false

            if (order == out_order) && (usejuqbox == juqbox)
                files_found += 1
                filepath = data_directory * "/" * file

                # Read as String first, then convert. Otherwise NaN+NaN*im
                # won't be interpreted correctly
                final_states = readdlm(filepath, ',', String)
                final_states = map(x -> x == "NaN + NaN*im" ? NaN + NaN*im : parse(ComplexF64, x),
                                   final_states)

                true_final_state = final_states[end,:]
                true_final_state_size = norm(true_final_state)
                rel_err(x) = norm(x - true_final_state) / true_final_state_size

                n_runs = size(final_states, 1)

                nsteps_vec = [2^i for i in 1:n_runs-1]
                relerr_vec = [rel_err(final_states[i,:]) for i in 1:n_runs-1]
               
                push!(nsteps_vec_entries, nsteps_vec)
                push!(relerr_vec_entries, relerr_vec)
            end
        end
    end

    if files_found == 0
        @warn "No files found matching conditions!"
    end

    return nsteps_vec_entries, relerr_vec_entries
end

"""
Given points (x1,y1) and (x2,y2), find the value of x for which the line going
through the two points goes through y.

Note that I will want to pass the logarithm of the points, so that the linear
interpolation is valid.
"""
function find_x(y, x1, y1, x2, y2)
    m = (y2-y1) / (x2-x1)
    x = x1 + (y-y1)/m
    return x
end

function find_y(x, x1, y1, x2, y2)
    m = (y2-y1) / (x2-x1)
    y = y1 + (x-x1)*m
    return y
end
    

#data_directory = "Data"
#data_directory = "48854734"
#data_directory = "./49076519/49076519/"
data_directory = "./DataMar18/"
makie_stddev = Any[]
makie_spaghetti = Any[]

#ticks_10f(i) = (i < 0) ? L"10^{\text{-}%$(-i)}" : L"10^{%$i}"
ticks_10f(i) = L"10^{%$i}"

unlabeled_timestep_ticks = (2 .^ (5:5:20), ["" for i in 5:5:20])
labeled_timestep_ticks = (2 .^ (5:5:20), [L"2^{%$i}" for i in 5:5:20])
minor_timestep_ticks = 2 .^ (0:20)

log10_ticks = (10.0 .^ (-15:15), ticks_10f.(-15:15))
#my_minoryticks = (10.0 .^ (-15:5:15), ticks_10f.(-15:5:15))
# Getting correct figure, font size: https://docs.makie.org/stable/how-to/match-figure-size-font-sizes-and-dpi
inch = 96
#fig = CairoMakie.Figure(size=(3.25inch, 4.5inch), fontsize=12, figure_padding=5)
fig = CairoMakie.Figure(size=(6inch, 4inch), fontsize=12, figure_padding=5)
fig2 = CairoMakie.Figure(size=(4inch, 4inch), fontsize=12, figure_padding=5)


ax_error = CairoMakie.Axis(
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
    limits=(nothing, (10.0^(-11.25), 10.0^0.25)),
)

ax_timing = CairoMakie.Axis(
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
    #title="Elapsed Time Plot",
)

#linkxaxes!(ax_error, ax_timing)

ax_err_vs_timing = CairoMakie.Axis(
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





#orders = [2,4,6,8,10,12]
orders = [2,4,6,8,10,12, "2 (Stormer-Verlet)"]
#orders = [2,4,6,8,"2 (Stormer-Verlet)"]
colors = vcat(Makie.wong_colors()[1:6], :darkkhaki)

# TODO: merge the two for-loops into one, but this all into a function so I don't need to write `local` so many times
for (k, order) in enumerate(orders)

    ##=== Nsteps Vs Relative Error Plot ===##
    
    println("\nNsteps Vs Relative Eror Plot")
    if order == "2 (Stormer-Verlet)"
        local nsteps_vec_entries, relerr_vec_entries = get_data_final_states(2, data_directory, true)
    else
        local nsteps_vec_entries, relerr_vec_entries = get_data_final_states(order, data_directory)
    end
    @show maximum(length, nsteps_vec_entries)

    local full_nsteps_vec, nsteps_mat, relerr_mat = get_x_vec_y_mat(nsteps_vec_entries, relerr_vec_entries)

    local relerr_mean =  mean(relerr_mat, dims=2) |> vec
    local relerr_stddev = std(relerr_mat, dims=2) |> vec

    @show length(full_nsteps_vec) 
    @show full_nsteps_vec

    local lines_obj = lines!(ax_error, full_nsteps_vec[5:end], relerr_mean[5:end]; color=colors[k], label="Order $order")
    CairoMakie.translate!(lines_obj, 0, 0, -k) # Put the lines in the right z-order

    ##=== Nsteps Vs Wall Time Plot ===##
    #
    println("\nNsteps Vs Wall Time Plot")
    
    if order == "2 (Stormer-Verlet)"
        local nsteps_vec_entries, elapsedtime_vec_entries = get_data("nsteps", "elapsed_time", 2, data_directory, true)
    else
        local nsteps_vec_entries, elapsedtime_vec_entries = get_data("nsteps", "elapsed_time", order, data_directory)
    end
    @show maximum(length, nsteps_vec_entries)

    local full_nsteps_vec, nsteps_mat, elapsedtime_mat = get_x_vec_y_mat(nsteps_vec_entries, elapsedtime_vec_entries)
    local elapsedtime_mean =  mean(elapsedtime_mat, dims=2) |> vec
    local elapsedtime_stddev = std(elapsedtime_mat, dims=2) |> vec

    local lines_obj = lines!(ax_timing, full_nsteps_vec[5:end], elapsedtime_mean[5:end]; color=colors[k])
    CairoMakie.translate!(lines_obj, 0, 0, -k) # Draw the high-order methods in the back

    ##=== Relative Error Vs Wall Time Plot ===##

    @show length(full_nsteps_vec) 
    @show full_nsteps_vec
    @show length(relerr_mean)
    @show length(elapsedtime_mean)
    local scatter_obj = lines!(ax_err_vs_timing, relerr_mean[5:end], elapsedtime_mean[6:end]; color=colors[k], label="Order $order")
    local dummy_relerr_mean = [10.0 ^(-i) for i in 5:10]
    local dummy_elapsedtime_mean = 0.5 .* (dummy_relerr_mean .^ -0.47)
    local scatter_obj = lines!(ax_err_vs_timing, dummy_relerr_mean, dummy_elapsedtime_mean; color=colors[k], label="Order $order", linestyle=:dot)
    CairoMakie.translate!(scatter_obj, 0, 0, -k) # Put the lines in the right z-order

    #band!(ax_error, full_nsteps_vec, relerr_mean - relerr_stddev, relerr_mean + relerr_stddev; color=colors[k], 0.5))

    #= 
    #Spaghetti Style
    for (nsteps_vec, relerr_vec) in zip(nsteps_vec_entries, relerr_vec_entries)
        CairoMakie.lines!(ax_spaghetti, nsteps_vec[5:end], relerr_vec[5:end]; color=colors[k], 0.5))
    end
    =#
end


##=== Getting the target error vs nsteps relationship ===##
target_errors = collect(-1:-1:-7)
target_nsteps_mat = fill(NaN, length(target_errors), length(orders))
target_time_mat = fill(NaN, length(target_errors), length(orders))

for (k, order) in enumerate(orders)
    if order == "2 (Stormer-Verlet)"
        local nsteps_vec_entries, relerr_vec_entries = get_data_final_states(2, data_directory, true)
    else
        local nsteps_vec_entries, relerr_vec_entries = get_data_final_states(order, data_directory)
    end

    local full_nsteps_vec, nsteps_mat, relerr_mat = get_x_vec_y_mat(nsteps_vec_entries, relerr_vec_entries)

    if order == "2 (Stormer-Verlet)"
        @show full_nsteps_vec
        @show nsteps_mat[:,1]
        @show relerr_mat[:,1]
    end

    if order == "2 (Stormer-Verlet)"
        local nsteps_vec_entries2, elapsedtime_vec_entries = get_data("nsteps", "elapsed_time", 2, data_directory, true)
    else
        local nsteps_vec_entries2, elapsedtime_vec_entries = get_data("nsteps", "elapsed_time", order, data_directory)
    end
    local full_nsteps_vec2, nsteps_mat, elapsedtime_mat = get_x_vec_y_mat(nsteps_vec_entries2, elapsedtime_vec_entries)

    local elapsedtime_mean = mean(elapsedtime_mat, dims=2) |> vec
    local relerr_mean =  mean(relerr_mat, dims=2) |> vec

    local full_nsteps_vec = full_nsteps_vec[5:end]
    local relerr_mean = relerr_mean[5:end]
    # The richardson table has one more entrie than the data from the final states
    local elapsedtime_mean = elapsedtime_mean[5:end-1] 

    local log2_nsteps_vec = log2.(full_nsteps_vec)
    local log10_relerr_vec = log10.(relerr_mean)
    local log10_elapsedtime_vec = log10.(elapsedtime_mean)

    local zipped_vecs = zip(log2_nsteps_vec, log10_relerr_vec, log10_elapsedtime_vec)
    
    # This will handle all the target errors that can be done by 
    for (i, target_error) in enumerate(target_errors)
        for ((nsteps1, relerr1, time1), (nsteps2, relerr2, time2)) in partition(zipped_vecs, 2, 1)
            if (relerr1 > target_error > relerr2)
                target_nsteps = find_x(target_error, nsteps1, relerr1, nsteps2, relerr2)
                target_time = find_y(target_nsteps, nsteps1, time1, nsteps2, time2)
                target_nsteps_mat[i,k] = target_nsteps
                target_time_mat[i,k] = target_time
            end
        end
    end

    # Use Linear Least Squares to estimate remaining entries (which happen for 2nd order method)
    local asymptotic_nsteps_vec = log2_nsteps_vec[end-6:end] 
    local asymptotic_relerr_vec = log10_relerr_vec[end-6:end] 
    local asymptotic_elapsedtime_vec = log10_elapsedtime_vec[end-6:end]

    local A = [ones(7) asymptotic_nsteps_vec]

    # y = mx + b
    local y_intercept_relerr, slope_relerr = A \ asymptotic_relerr_vec 
    local y_intercept_time, slope_time = A \ asymptotic_elapsedtime_vec 

    for (i, target_error) in enumerate(target_errors)
        if (isnan(target_nsteps_mat[i,k])) 
            target_nsteps = (target_error - y_intercept_relerr) / slope_relerr
            target_time = y_intercept_time + slope_time*target_nsteps

            target_nsteps_mat[i,k] = target_nsteps
            target_time_mat[i,k] = target_time
        end
    end

end

target_nsteps_mat_nonlog = ceil.(Int64, 2 .^ target_nsteps_mat)
target_time_mat_nonlog = 10 .^ target_time_mat
target_time_ratios = copy(target_time_mat_nonlog)
for i in 1:size(target_time_ratios, 2)
    target_time_ratios[:,i] ./= target_time_mat_nonlog[:,1]
end
target_time_ratios = 1 ./ target_time_ratios


Legend(fig[2,:], ax_error, orientation = :horizontal, tellwidth = false, nbanks=2, framevisible=false)
rowgap!(fig.layout, 1, 0)
#rowsize!(fig.layout, 2, Relative(1/3))

fig


# Idea, scatter plot with x=desired error, y=expected elapsed time, and put
# number of timesteps required as an in-graph number annotation
# Actually, I think a dogged bar plot would be better

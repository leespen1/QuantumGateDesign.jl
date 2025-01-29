using CairoMakie, DelimitedFiles, LaTeXStrings, LinearAlgebra, Statistics
using IterTools
#using Typst_jll
import Makie
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

function get_data(x_header, y_header, out_order, data_directory=missing)
    file_pattern= r"""cnot3StepsizeTest
    _order=(\d+)
    _degree=(\d+)
    _seed=(\d+)
    _atol=(.*)
    _rtol=(.*)
    _D1=(\d+)
    _time=(.*)
    _nthreads=(\d+)
    .csv
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

            if order == out_order
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

    # Use the longest one
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

function get_data_final_states(out_order, data_directory=missing)
    file_pattern= r"""cnot3StepsizeTest
    _order=(\d+)
    _degree=(\d+)
    _seed=(\d+)
    _atol=(.*)
    _rtol=(.*)
    _D1=(\d+)
    _time=(.*)
    _nthreads=(\d+)
    _finalStates
    .csv
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

            if order == out_order
                files_found += 1
                filepath = data_directory * "/" * file

                final_states = readdlm(filepath, ',', ComplexF64)

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
data_directory = "./49076519/49076519/"
makie_stddev = Any[]
makie_spaghetti = Any[]

#ticks_10f(i) = (i < 0) ? L"10^{\text{-}%$(-i)}" : L"10^{%$i}"
ticks_10f(i) = L"10^{%$i}"

unlabeled_xticks = (2 .^ (5:5:20), ["" for i in 5:5:20])
labeled_xticks = (2 .^ (5:5:20), [L"2^{%$i}" for i in 5:5:20])
minor_xticks = 2 .^ (0:20)

my_yticks = (10.0 .^ (-15:15), ticks_10f.(-15:15))
#my_minoryticks = (10.0 .^ (-15:5:15), ticks_10f.(-15:5:15))
# Getting correct figure, font size: https://docs.makie.org/stable/how-to/match-figure-size-font-sizes-and-dpi
inch = 96
fig = CairoMakie.Figure(size=(3.25inch, 4.5inch), fontsize=12, figure_padding=5)
#fig = CairoMakie.Figure(size=(4inch, 4.5inch), fontsize=12, figure_padding=5)

ax_error = CairoMakie.Axis(
    fig[1,1],
    xscale=CairoMakie.log2,
    yscale=CairoMakie.log10,
    #xlabel="Number of Timesteps",
    ylabel="Mean Relative Error",
    xticks=unlabeled_xticks,
    xminorticks = minor_xticks,
    xminorticksvisible = true,
    xminorgridvisible = true,
    yticks=my_yticks,
)

ax_timing = CairoMakie.Axis(
    fig[2,1],
    xscale=CairoMakie.log2,
    yscale=CairoMakie.log10,
    xlabel="Number of Timesteps",
    ylabel="Mean Wall Time (s)",
    xticks=labeled_xticks,
    xminorticks = minor_xticks,
    xminorticksvisible = true,
    xminorgridvisible = true,
    #xticks=(my_xticks, my_xticklabels),
    yticks=my_yticks,
    #title="Elapsed Time Plot",
)

linkxaxes!(ax_error, ax_timing)




orders = [2,4,6,8,10,12]
# Timing Plot
for (k, order) in enumerate(orders)
    local nsteps_vec_entries, elapsedtime_vec_entries = get_data("nsteps", "elapsed_time", order, data_directory)

    local full_nsteps_vec, nsteps_mat, elapsedtime_mat = get_x_vec_y_mat(nsteps_vec_entries, elapsedtime_vec_entries)
    local elapsedtime_mean =  mean(elapsedtime_mat, dims=2) |> vec
    local elapsedtime_stddev = std(elapsedtime_mat, dims=2) |> vec

    local lines_obj = lines!(ax_timing, full_nsteps_vec[5:end], elapsedtime_mean[5:end]; color=Makie.wong_colors()[k], linewidth=0.95)
    CairoMakie.translate!(lines_obj, 0, 0, -k) # Draw the high-order methods in the back
    #= 
    #For Spaghetti Style
    for (nsteps_vec, elapsedtime_vec) in zip(nsteps_vec_entries, elapsedtime_vec_entries)
        CairoMakie.lines!(ax_timing, nsteps_vec[5:end], elapsedtime_vec[5:end]; color=(Makie.wong_colors()[k], 0.5))
    end
    =#
end

# Err Plot Plot
for (k, order) in enumerate(orders)
    local nsteps_vec_entries, relerr_vec_entries = get_data_final_states(order, data_directory)
    local full_nsteps_vec, nsteps_mat, relerr_mat = get_x_vec_y_mat(nsteps_vec_entries, relerr_vec_entries)

    local relerr_mean =  mean(relerr_mat, dims=2) |> vec
    local relerr_stddev = std(relerr_mat, dims=2) |> vec

    local full_nsteps_vec = full_nsteps_vec[5:end]
    local relerr_mean = relerr_mean[5:end]
    local relerr_stddev = relerr_stddev[5:end]

    local lines_obj = lines!(ax_error, full_nsteps_vec, relerr_mean; color=Makie.wong_colors()[k], label="Order $order")
    CairoMakie.translate!(lines_obj, 0, 0, -k)

    #band!(ax_error, full_nsteps_vec, relerr_mean - relerr_stddev, relerr_mean + relerr_stddev; color=(Makie.wong_colors()[k], 0.5))

    #= 
    #Spaghetti Style
    for (nsteps_vec, relerr_vec) in zip(nsteps_vec_entries, relerr_vec_entries)
        CairoMakie.lines!(ax_spaghetti, nsteps_vec[5:end], relerr_vec[5:end]; color=(Makie.wong_colors()[k], 0.5))
    end
    =#
end


# Table
# Err Plot Plot
target_errors = collect(-1:-1:-7)
target_nsteps_mat = fill(NaN, length(target_errors), length(orders))
target_time_mat = fill(NaN, length(target_errors), length(orders))

for (k, order) in enumerate(orders)

    local nsteps_vec_entries, relerr_vec_entries = get_data_final_states(order, data_directory)
    local full_nsteps_vec, nsteps_mat, relerr_mat = get_x_vec_y_mat(nsteps_vec_entries, relerr_vec_entries)

    local nsteps_vec_entries2, elapsedtime_vec_entries = get_data("nsteps", "elapsed_time", order, data_directory)
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


#categories = repeat(target_errors, inner=length(orders))
#heights = reshape(target_time_ratios', :)
#grp = repeat(orders, length(target_errors))

categories = repeat(target_errors, inner=length(orders)-1)
heights = reshape(target_time_ratios[:,2:end]', :)
grp = repeat(orders[2:end], length(target_errors))

@show categories
@show heights
@show length(categories)
@show length(heights)

fig2 = CairoMakie.Figure(size=(3.25inch, 4.5inch), fontsize=12, figure_padding=5)
ax_barplot = Axis(
    fig2[1,1],
    #fig[1:2,2],
    ylabel = "Target Error",
    xlabel = "Speedup over 2nd Order Method",
    yticks = (target_errors, [L"10^{%$i}" for i in target_errors]),
    xticks = 0:50:250,
    xminorticks = 0:10:250,
    xminorticksvisible = true,
    xminorgridvisible = true,
    #xlims = (0,100),
    #xscale = CairoMakie.log10,
)
# Set xlimits by hand so the labels don't get clipped
#xlims!(ax_barplot, (0,100))
xlims!(ax_barplot, (0,230))

barplot!(
    ax_barplot,
    categories, heights,
    dodge = grp,
    color = grp,
    colormap = [Makie.wong_colors()[k] for k in 2:length(orders)] ,
    #color_over_background=:red,
    #color_over_bar=:white,
    #flip_labels_at=0.85,
    direction=:x,
    gap=0.25,
    #width=20.0,
    bar_labels=:y,
    label_size=8,
    label_formatter = x -> round(x, digits=1),
    label_offset = [5 for i in 1:length(categories)],
    #flip_labels_at = 100,
)
#colsize!(fig.layout, 2, Relative(1/3))



Legend(fig[3,:], ax_error, orientation = :horizontal, tellwidth = false, nbanks=2, framevisible=false)
rowgap!(fig.layout, 2, 0)
rowsize!(fig.layout, 2, Relative(1/3))

fig


# Idea, scatter plot with x=desired error, y=expected elapsed time, and put
# number of timesteps required as an in-graph number annotation
# Actually, I think a dogged bar plot would be better

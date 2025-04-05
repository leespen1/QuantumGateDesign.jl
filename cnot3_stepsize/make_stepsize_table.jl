##=== Getting the target error vs nsteps relationship ===##
using Statistics, IterTools, Printf, Format
include("processing_stepsize_data.jl")

data_directory = "./DataMar28/"
#data_directory = "./DataApr3/"
#orders = [2,4,6,8,10,12, "2 (Stormer-Verlet)"]
orders = [2,4,6,8,10,12]
#orders = [2]
gradient = true

target_errors = collect(-1:-1:-7)
target_log2nsteps_mat = fill(NaN, length(target_errors), length(orders))
target_log10time_mat = fill(NaN, length(target_errors), length(orders))

for (k, order) in enumerate(orders)
    if order == "2 (Stormer-Verlet)"
        label = "Order 2 (St\u00f6rmer-Verlet)"
        juqbox = true
        order = 2
    else
        label = "Order $order"
        juqbox = false
    end

    target_labels = ("nsteps", "elapsed_time", "forward_time", "adjoint_time", "grad_accum_time",  "R_rel_err_L2")
    target_symbols = Symbol.(target_labels)
    entries = NamedTuple{target_symbols}(
        get_data(target_labels, order, data_directory=data_directory, juqbox=juqbox, gradient=gradient)
    )
    relerr_nsteps_entries, relerr_err_entries = get_nsteps_errors_final_states(
        order, data_directory=data_directory, juqbox=juqbox, gradient=gradient
    )

    # Need to remove last entry so NaN entries doesn't mess things up
    full_nsteps_vec = combined_x_vec(entries.nsteps)[1:end-1]
    relerr_ymat = get_y_mat(relerr_nsteps_entries, relerr_err_entries, full_nsteps_vec)
    relerr_mean = mean(relerr_ymat, dims=2) |> vec
    elapsedtime_ymat = get_y_mat(entries.nsteps, entries.elapsed_time, full_nsteps_vec)
    elapsedtime_mean = mean(elapsedtime_ymat, dims=2) |> vec



    full_nsteps_vec = full_nsteps_vec[5:end]
    relerr_mean = relerr_mean[5:end]
    # The richardson table has one more entry than the data from the final states
    elapsedtime_mean = elapsedtime_mean[5:end-1] 


    log2_nsteps_vec = log2.(full_nsteps_vec)
    log10_relerr_vec = log10.(relerr_mean)
    log10_elapsedtime_vec = log10.(elapsedtime_mean)

    zipped_vecs = zip(log2_nsteps_vec, log10_relerr_vec, log10_elapsedtime_vec)
    
    # This will handle all the target errors that can be done by interpolation
    for (i, target_error) in enumerate(target_errors)
        for ((nsteps1, relerr1, time1), (nsteps2, relerr2, time2)) in partition(zipped_vecs, 2, 1)
            if (relerr1 > target_error > relerr2)
                target_nsteps = find_x(target_error, nsteps1, relerr1, nsteps2, relerr2)
                target_time = find_y(target_nsteps, nsteps1, time1, nsteps2, time2)
                target_log2nsteps_mat[i,k] = target_nsteps
                target_log10time_mat[i,k] = target_time
            end
        end
    end

    # Use Linear Least Squares to estimate remaining entries (which happen for 2nd order method)
    asymptotic_nsteps_vec = log2_nsteps_vec[end-6:end] 
    asymptotic_relerr_vec = log10_relerr_vec[end-6:end] 
    asymptotic_elapsedtime_vec = log10_elapsedtime_vec[end-6:end]

    A = [ones(7) asymptotic_nsteps_vec]

    # y = mx + b
    y_intercept_relerr, slope_relerr = A \ asymptotic_relerr_vec 
    y_intercept_time, slope_time = A \ asymptotic_elapsedtime_vec 

    for (i, target_error) in enumerate(target_errors)
        if (isnan(target_log2nsteps_mat[i,k])) 
            target_nsteps = (target_error - y_intercept_relerr) / slope_relerr
            target_time = y_intercept_time + slope_time*target_nsteps

            target_log2nsteps_mat[i,k] = target_nsteps
            target_log10time_mat[i,k] = target_time
        end
    end

end
target_nsteps_mat = round.(Int64, 2 .^ target_log2nsteps_mat)
target_time_mat = 10.0 .^ target_log10time_mat

integer_format(x::Integer) = format(x, commas=true)
floatsci_format(x::Float64) = @sprintf("%.2e", x)
float_format(x::Float64) = @sprintf("%.2f", x)

function table(mat::Matrix{String})
    for row in eachrow(mat)
        println(reduce((x,y) -> x * " & " * y, row), " \\\\")
    end
end

header = hcat("Target Error", ["Order $order" for _ in 1:1, order in orders])
first_col = string.(target_errors)

# Speedup over 2nd order method
target_speedup_mat = target_time_mat[:,1] ./ target_time_mat 
 
target_nsteps_str_mat = hcat(first_col, integer_format.(target_nsteps_mat))
target_nsteps_str_mat = vcat(header, target_nsteps_str_mat)
target_time_str_mat = hcat(first_col, floatsci_format.(target_time_mat))
target_time_str_mat = vcat(header, target_time_str_mat)
target_speedup_str_mat = hcat(first_col, float_format.(target_speedup_mat))
target_speedup_str_mat = vcat(header, target_speedup_str_mat)

println("Nsteps Mat")
table(target_nsteps_str_mat)
println("\nTime Mat")
table(target_time_str_mat)
println("\nSpeedup Mat")
table(target_speedup_str_mat)

#latex_table(target_time_mat)

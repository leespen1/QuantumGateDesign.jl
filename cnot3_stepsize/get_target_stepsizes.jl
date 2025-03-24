using Statistics, IterTools
include("processing_stepsize_data.jl")

##=== Getting the target error vs nsteps relationship ===##
# I don't need this for the graphs, but I want to have precise numbers for when
# I do the pcof optimization.

data_directory = "./DataMar18/"
orders = [2,4,6,8,10,12, "2 (Stormer-Verlet)"]

target_errors = collect(-1:-1:-7)
target_nsteps_mat = fill(NaN, length(target_errors), length(orders))
target_time_mat = fill(NaN, length(target_errors), length(orders))

for (k, order) in enumerate(orders)
    if order == "2 (Stormer-Verlet)"
        label = "Order 2 (Stormer-Verlet)"
        juqbox = true
        order = 2
    else
        label = "Order $order"
        juqbox = false
    end

    nsteps_vec_entries, relerr_vec_entries = get_nsteps_errors_final_states(
        order, data_directory=data_directory, juqbox=juqbox
    )

    full_nsteps_vec, relerr_mat = get_x_vec_y_mat(nsteps_vec_entries, relerr_vec_entries)

    nsteps_vec_entries2, elapsedtime_vec_entries = get_data(
        ("nsteps", "elapsed_time"), order, data_directory=data_directory,
        juqbox=juqbox
    )
    elapsedtime_mat = get_y_mat(nsteps_vec_entries2, elapsedtime_vec_entries, full_nsteps_vec)

    elapsedtime_mean = mean(elapsedtime_mat, dims=2) |> vec
    relerr_mean =  mean(relerr_mat, dims=2) |> vec

    full_nsteps_vec = full_nsteps_vec[5:end-1]
    relerr_mean = relerr_mean[5:end-1]
    # The richardson table has one more entrie than the data from the final states
    elapsedtime_mean = elapsedtime_mean[5:end-1] 

    log2_nsteps_vec = log2.(full_nsteps_vec)
    log10_relerr_vec = log10.(relerr_mean)
    log10_elapsedtime_vec = log10.(elapsedtime_mean)

    zipped_vecs = zip(log2_nsteps_vec, log10_relerr_vec, log10_elapsedtime_vec)
    
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
    asymptotic_nsteps_vec = log2_nsteps_vec[end-6:end] 
    asymptotic_relerr_vec = log10_relerr_vec[end-6:end] 
    asymptotic_elapsedtime_vec = log10_elapsedtime_vec[end-6:end]

    A = [ones(7) asymptotic_nsteps_vec]

    # y = mx + b
    y_intercept_relerr, slope_relerr = A \ asymptotic_relerr_vec 
    y_intercept_time, slope_time = A \ asymptotic_elapsedtime_vec 

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

println("\nTarget Nsteps Mat (With rounding from undoing log)")
display(target_nsteps_mat_nonlog)
println("\nLog2 Target Nsteps Mat")
display(target_nsteps_mat)

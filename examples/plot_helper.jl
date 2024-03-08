interpolate(xs, ys) = xs[1] + (-7-ys[1])*(xs[2]-xs[1])/(ys[2]-ys[1])

function find_timing(timing_all, errors_all)
    optimal_times = zeros(size(errors_all, 2))
    optimal_times .= NaN

    for i in 1:size(errors_all, 2)
        for k in 1:size(errors_all, 1)-1

            x1 = log10(timing_all[k,i])
            x2 = log10(timing_all[k+1,i])
            y1 = log10(errors_all[k,i])
            y2 = log10(errors_all[k+1,i])

            println("i=$i, k=$k, y1=$y1, y2=$y2")
            # If we can interpolate to get 1e-7, do it and move on to next order
            if (y1 > -7 > y2)
                println("i=$i, k=$k")
                optimal_times[i] = interpolate([x1,x2], [y1,y2])
                break
            end
            # If we get to the end and reach NaN, use last two available points
            if isnan(y2)
                x1 = log10(timing_all[k-1,i])
                x2 = log10(timing_all[k,i])
                y1 = log10(errors_all[k-1,i])
                y2 = log10(errors_all[k,i])
                optimal_times[i] = interpolate([x1,x2], [y1,y2])
                break
            end
            # If we get to the bottom of the matrix and haven't reached NaN, extrapolate
            if k == size(errors_all, 1)-1
                optimal_times[i] = interpolate([x1,x2], [y1,y2])
                break
            end
        end

    end

    # Convert back into normal units
    optimal_times = 10.0 .^ optimal_times
    optimal_speedups = optimal_times[2:end] ./ optimal_times[1]

    return optimal_times, optimal_speedups
end

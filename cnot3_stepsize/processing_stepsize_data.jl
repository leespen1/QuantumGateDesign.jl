using DelimitedFiles, LinearAlgebra

function get_data(target_labels, out_order::Integer; data_directory=missing,
        juqbox::Bool=false)

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
    (?:_gradient=(true|false))? # Optionally match , doesn't appear in older files
    \.csv
    """x # 'x' tag ignores whitespace and comments

    data_entries_collection = ntuple(i -> Vector{Float64}[], length(target_labels))

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

            if ((order == out_order) && (juqbox == usejuqbox))
                files_found += 1
                filepath = data_directory * "/" * file

                @show filepath
                dlm_data, dlm_header = readdlm(filepath, ',', Float64, header=true)
                dlm_header = vec(dlm_header)

                for (data_entries, label) in zip(data_entries_collection, target_labels)
                    index = findfirst(x -> x == label, dlm_header)
                    if isnothing(index)
                        target_data = fill(NaN, size(dlm_data, 1))
                    else
                        target_data = dlm_data[:,index]
                    end
                    push!(data_entries, target_data)
                end
            end
        end
    end

    if files_found == 0
        @warn "No files found matching conditions!"
    end

    return data_entries_collection
end

function combined_x_vec(x_data_entries::Vector{<: Vector})
    combined_x_vec = sort(unique(vcat(x_data_entries...)))
    return combined_x_vec
end


"""
`x_data_entries` should be a vector of vectors, whose elements should be
ordered.
`y_data_entries` should be a vector of vectors, whose lengths match those of
`x_data_entries`.

Returns a vector and a matrix. The vector is the "combined" version of all the
entries in `x_data_entries`, and the matrix holds the y values corresponding to
the the x values in the vector. Each column holds the data from one of entries
in `y_data_entries`. If that entry does not have a y value for the
corresponding x value, then `NaN` is used.

If `full_x_vec` is provided, use that to get the desired x values, instead of
building it from x_data_entries, and don't return an x_vec, only the y_mat
"""
function get_y_mat(x_data_entries::Vector{<: Vector},
                   y_data_entries::Vector{<: Vector},
                   full_x_vec::Vector{<: Real})

    @assert length(x_data_entries) == length(y_data_entries)
    for (x_entry, y_entry) in zip(x_data_entries, y_data_entries)
        @assert length(x_entry) == length(y_entry)
    end

    max_length = maximum(length, x_data_entries)
    min_length = minimum(length, x_data_entries)
    if !allequal(x_data_entries) 
        @warn "Not all x_data entries are the same."
    end

    n_xs = length(full_x_vec)
    n_entries = length(x_data_entries)

    y_mat = fill(NaN, n_xs, n_entries)

    for k in 1:n_entries
        x_vec = x_data_entries[k]
        y_vec = y_data_entries[k]
        for (j, x) in enumerate(full_x_vec)
            let x = x
                l = findfirst(a -> a == x, x_vec)
                y = isnothing(l) ? NaN : y_vec[l]
                y_mat[j,k] = y
            end
        end
    end
    return y_mat
end

function get_x_vec_y_mat(x_data_entries::Vector{<: Vector},
                         y_data_entries::Vector{<: Vector})
    full_x_vec = combined_x_vec(x_data_entries)
    y_mat = get_y_mat(x_data_entries, y_data_entries, full_x_vec)
    return full_x_vec, y_mat
end

"""
Using the final state with the highest number of timesteps as the "true"
solution, return vector of nsteps vectors and a vector of relerr vectors, with
an inner vector for each file.
"""
function get_nsteps_errors_final_states(out_order; data_directory=missing, juqbox=false)
    file_pattern = r"""cnot3StepsizeTest
    _order=(\d+)
    _degree=(\d+)
    _seed=(\d+)
    _atol=(.*)
    _rtol=(.*)
    _D1=(\d+)
    _time=(.*)
    _nthreads=(\d+)
    (?:_usejuqbox=(true|false))? # Optionally match , doesn't appear in older files
    (?:_gradient=(true|false))? # Optionally match , doesn't appear in older files
    _finalStates
    \.csv
    """x # 'x' tag ignores whitespace and comments


    nsteps_vec_entries = Vector{Float64}[]
    relerr_vec_entries = Vector{Float64}[]

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

                nsteps_vec = [2^i for i in 1:n_runs]
                relerr_vec = vcat([rel_err(final_states[i,:]) for i in 1:n_runs-1], NaN) # "True" final state has no comparison point, so use NaN as error
               
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

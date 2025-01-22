using Plots, DelimitedFiles

function main(x_header, y_header, out_order=2)
    file_pattern= r"""cnot3StepsizeTest_seed=(\d+)
    _order=(\d+)
    _rtol=(.*)
    _D1=(\d+)
    _time=(.*)
    _nthreads=(\d+)
    .csv
    """x # 'x' tag ignores whitespace and comments


    top_dict = Dict(2 => Dict(), 4 => Dict(), 6 => Dict(), 8 => Dict(), 10 => Dict(), 12 => Dict())

    x_data_entries = Any[]
    y_data_entries = Any[]

    data_directory = dirname(@__FILE__) * "/Data/"
    files = readdir(data_directory)
    for file in files
        if occursin(file_pattern, file)
            regex_match = match(file_pattern, file)
            seed     = parse(Int,     regex_match[1])
            order    = parse(Int,     regex_match[2])
            rtol     = parse(Float64, regex_match[3])
            D1       = parse(Int,     regex_match[4])
            time     = parse(Float64, regex_match[5])
            nthreads = parse(Int,     regex_match[6])

            if order == out_order
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
    x_data_lengths = length.(x_data_entries)
    max_length = maximum(x_data_lengths)
    min_length = minimum(x_data_lengths)
    if max_length != min_length 
        @warn "Not all x_data entries are the same length. Max is $max_length, min is $min_length."
    end

    x_data_entries = [x_data[1:min_length] for x_data in x_data_entries]
    y_data_entries = [y_data[1:min_length] for y_data in y_data_entries]
    
    @assert allequal(x_data_entries)
    x_vec = x_data_entries[1]

    y_mat = hcat(y_data_entries...)

    return x_vec, y_mat
end


x_vecs = Any[]
y_mats = Any[]
pls = Any[]
for order in (2,4,6,8,10,12)
    local x_vec, y_mat = main("stepsize", "R_rel_err_L2", order)
    local y_mean = Plots.mean(y_mat, dims=2) |> vec
    local y_std = Plots.std(y_mat, dims=2) |> vec

    push!(x_vecs, x_vec)
    push!(y_mats, y_mat)

    local pl = plot(
        x_vec, y_mean,
        ribbon=y_std,  # Adds error bars as ribbons
        label="Mean ± Std Dev",
        xlabel="Stepsize",
        ylabel="Relative Error",
        title="Mean and Standard Deviation Plot, Order=$order",
        color=:blue,
        yscale=:log10,
        xscale=:log10
    )

    push!(pls, pl)
end

function index_to_state(index::Integer, levels::IntegersType)
    @assert 0 < index <= prod(levels) "Index out of bounds."
    index -= 1 # For 1-based indexing
    state = fill(-1,length(levels))
    for (i, n) in enumerate(reverse(levels))
        state[i] = index % n
        index = div(index, n)
    end
    return reverse(state)
end

function state_to_index(bitstring::IntegersType, subsys_sizes::IntegersType)
    @assert length(bitstring) == length(subsys_sizes) "bitstring and subsys_sizes must be the same length."
    @assert all(bitstring .< subsys_sizes) "All bitstring entries must be below their respective subsystem sizes."
    R = reverse(cumprod(vcat(1, reverse(subsys_sizes[2:end]))))
    return sum(bitstring .* R) + 1 # Plus 1 for 1-based indexing
end

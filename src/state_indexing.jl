function index_to_basis_state(index::Integer, levels::IntegersType)
    @assert 0 < index <= prod(levels) "Index out of bounds."
    index -= 1 # For 1-based indexing
    state = fill(-1,length(levels))
    for (i, n) in enumerate(reverse(levels))
        state[i] = index % n
        index = div(index, n)
    end
    return reverse(state)
end

function basis_state_to_index(basis_state::IntegersType, subsys_sizes::IntegersType)
    @assert length(basis_state) == length(subsys_sizes) "bitstring and subsys_sizes must be the same length."
    @assert all(basis_state .< subsys_sizes) "All bitstring entries must be below their respective subsystem sizes."
    R = reverse(cumprod(vcat(1, reverse(subsys_sizes[2:end]))))
    return sum(basis_state.* R) + 1 # Plus 1 for 1-based indexing
end

function basis_state_to_index(basis_state, subsys_sizes::IntegersType)
    basis_state_intvec = convert(Vector{Int}, basis_state)
    return basis_state_to_index(basis_state_intvec, subsys_sizes)
end


function basis_state_to_string(basis_state) 
    basis_state_intvec = convert(Vector{Int}, basis_state)
    return "|" * string(basis_state_intvec...) * "⟩"
end

function basis_states(subsys_sizes::IntegersType)
    return [index_to_basis_state(i, subsys_sizes) for i in 1:prod(subsys_sizes)]
end

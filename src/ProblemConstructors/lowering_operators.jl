"""
Construct the lowering operator for a single qudit with number of states equal
to N_levels.
"""
function lowering_operator(N_levels::Int)
    lowering_operator = zeros(N_levels, N_levels)
    for i = 2:N_levels
        lowering_operator[i-1,i] = sqrt(i-1)
    end
    return lowering_operator
end


function raising_operator(N_levels::Int)
    return transpose(lowering_operator(N_levels))
end


"""
Get the lowering operator for a particular subsystem.

Assumes all subsystems are the same size (E.g. multiple qudits with the same
number of levels).
"""
function subsytem_lowering_operator(N_levels, N_subsystems, subsystem_index)
    @assert (subsystem_index > 0) && (subsystem_index <= N_subsystems)
    one_qubit_lowering_operator = zeros(N_levels,N_levels)
    for i = 1:(N_levels-1)
        one_qubit_lowering_operator[i,i+1] = sqrt(i)
    end
    one_qubit_identity = zeros(N_levels, N_levels)
    for i = 1:n_levels
        one_qubit_identity[i,i] = 1
    end
    
    constructors = Vector{Matrix{Float64}}(undef, N_subsystems)
    for k = 1:n_subsystems
        if k != subsystem_index
            constructors[k] = one_qubit_identity
        else
            constructors[k] = one_qubit_lowering_operator
        end
    end
    lowering_operator = reduce(kron, constructors)
    @assert size(lowering_operator, 1) == size(lowering_operator, 1) == N_levels^N_subsystems
    return lowering_operator
end

"""
Get a vector of the lowering operators for all subsystems of a composite system 
(E.g. multiple qudits).

Assumes all subsystems are the same size (E.g. multiple qudits with the same
number of levels).
"""
function composite_system_lowering_operators(N_subsystems, N_levels)
    lowering_ops = Vector{Matrix{Float64}}(undef, N_subsystems)
    for subsystem_index = 1:N_subsystems
        lowering_ops[subsystem_index] = get_lowering_operator(N_subsystems, N_levels, subsystem_index)
    end
    return lowering_ops
end


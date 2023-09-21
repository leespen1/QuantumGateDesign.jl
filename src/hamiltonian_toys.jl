
function get_lowering_operator(n_subsystems, n_levels, subsystem_index)
    @assert (subsystem_index > 0) && (subsystem_index <= n_subsystems)
    one_qubit_lowering_operator = zeros(n_levels,n_levels)
    for i = 1:(n_levels-1)
        one_qubit_lowering_operator[i,i+1] = sqrt(i)
    end
    one_qubit_identity = zeros(n_levels, n_levels)
    for i = 1:n_levels
        one_qubit_identity[i,i] = 1
    end
    
    constructors = Vector{Matrix{Float64}}(undef, n_subsystems)
    for k = 1:n_subsystems
        if k != subsystem_index
            constructors[k] = one_qubit_identity
        else
            constructors[k] = one_qubit_lowering_operator
        end
    end
    lowering_operator = reduce(kron, constructors)
    @assert size(lowering_operator, 1) == size(lowering_operator, 1) == n_levels^n_subsystems
    return lowering_operator
end

function get_lowering_ops(n_subsystems, n_levels)
    lowering_ops = Vector{Matrix{Float64}}(undef, n_subsystems)
    for subsystem_index = 1:n_subsystems
        lowering_ops[subsystem_index] = get_lowering_operator(n_subsystems, n_levels, subsystem_index)
    end
    return lowering_ops
end

function get_system_hamiltonian(n_subsystems, n_levels, ground_transition_freqs, self_kerr_coeffs, cross_kerr_coeffs)
    N = n_levels^n_subsystems
    system_hamiltonian = zeros(N,N)

    lower_ops = get_lowering_ops(n_subsystems, n_levels)

    for subsystem_index = 1:n_subsystems
        lower_op_q = lower_ops[subsystem_index]
        system_hamiltonian .+= lower_op_q'*lower_op_q .* ground_transition_freqs[subsystem_index]
        system_hamiltonian .-= lower_op_q'*lower_op_q'*lower_op_q*lower_op_q .* (self_kerr_coeffs[subsystem_index] / 2)

        for subsystem2_index = subsystem_index+1:n_subsystems
            lower_op_p = lower_ops[subsystem2_index]
            system_hamiltonian .-= lower_op_p'*lower_op_p*lower_op_q'*lower_op_q .* cross_kerr_coeffs[subsystem2_index, subsystem_index]
        end
    end
    return system_hamiltonian
end

"""
Just use 1 for all the frequencies, for analysis
"""
function get_system_hamiltonian(n_subsystems, n_levels)
    ground_transition_freqs = ones(n_subsystems)
    self_kerr_coeffs = ones(n_subsystems)
    cross_kerr_coeffs = ones(n_subsystems, n_subsystems)
    return get_system_hamiltonian(n_subsystems, n_levels, ground_transition_freqs, self_kerr_coeffs, cross_kerr_coeffs)
end

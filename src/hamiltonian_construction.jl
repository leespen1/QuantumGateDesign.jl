"""
Get the lowering operator for a particular subsystem.
"""
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

"""
Get a vector of the lowering operators for all subsystems.
"""
function get_lowering_ops(n_subsystems, n_levels)
    lowering_ops = Vector{Matrix{Float64}}(undef, n_subsystems)
    for subsystem_index = 1:n_subsystems
        lowering_ops[subsystem_index] = get_lowering_operator(n_subsystems, n_levels, subsystem_index)
    end
    return lowering_ops
end

"""
Construct system hamiltonian for transmon qubits coupled to resonators, in the
dispersive limit. (Equation 2.2 in Juqbox paper)
"""
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
Construct system hamiltonian, using 1 for all ground state transition
frequencies and self kerr coefficients (just for testing). 
"""
function get_system_hamiltonian_nofreq(n_subsystems, n_levels)
    ground_transition_freqs = ones(n_subsystems)
    self_kerr_coeffs = ones(n_subsystems)
    cross_kerr_coeffs = ones(n_subsystems, n_subsystems)
    return get_system_hamiltonian(n_subsystems, n_levels, ground_transition_freqs, self_kerr_coeffs, cross_kerr_coeffs)
end


function single_transmon_qubit_rwa(ground_state_transition_frequency,
        self_kerr_coefficient, N_ess_levels, N_guard_levels, tf, nsteps)

    N_tot_levels = N_ess_levels + N_guard_levels
    n_subsystems = 1
    subsystem_index = 1
    cross_kerr_coeffs = []

    system_hamiltonian = get_system_hamiltonian(
        n_subsystems, N_tot_levels, ground_state_transition_frequency,
        self_kerr_coefficient, cross_kerr_coeffs
    )

    Ks::Matrix{Float64} = system_hamiltonian
    Ss::Matrix{Float64} = zeros(size(Ks)...)

    u0::Matrix{Float64} = zeros(N_tot_levels, N_ess_levels)
    for i in 1:N_ess_levels
        u0[i,i] = 1
    end
    v0::Matrix{Float64} = zeros(N_ess_levels+N_guard_levels, N_ess_levels)

    lower_op = get_lowering_operator(n_subsystems, N_tot_levels, subsystem_index)

    p_operator::Matrix{Float64} = lower_op + lower_op'
    q_operator::Matrix{Float64} = lower_op - lower_op'

    return SchrodingerProb(
        Ks, Ss, p_operator, q_operator, u0, v0, tf, nsteps, N_ess_levels,
        N_guard_levels
    )
end

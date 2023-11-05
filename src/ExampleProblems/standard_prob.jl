"""
A single qubit in the dispersive limit (e.g. section 2 of JuqBox paper), in the
rotating frame.

Frequencies should be in GHz, and will be multiplied by 2pi to get angular
frequencies in the Hamiltonian.
"""
function rotating_frame_qubit(N_ess_levels::Int, N_guard_levels::Int;
        tf::Float64=1.0, nsteps::Int64=10, detuning_frequency::Float64=1.0,
        self_kerr_coefficient::Float64=1.0
    )

    N_tot_levels = N_ess_levels + N_guard_levels

    a = lowering_operator(N_tot_levels)

    Ks = zeros(N_tot_levels, N_tot_levels)
    Ss = zeros(N_tot_levels, N_tot_levels)

    Ks .+= 2*pi*detuning_frequency .* (a'*a)
    Ks .-= (0.5*2*pi*self_kerr_coefficient) .* (a'*a'*a*a)

    p_operator = a + a'
    q_operator = a - a'

    u0, v0 = initial_basis(N_ess_levels, N_guard_levels)

    return SchrodingerProb(
        Ks, Ss, p_operator, q_operator,
        u0, v0, 
        tf, nsteps,
        N_ess_levels, N_guard_levels
    )
end

"""
Construct the lowering operator for a single qubit.

Later, should make version of this for multiple qubits.
"""
function lowering_operator(N_tot_levels::Int)
    lowering_operator = zeros(N_tot_levels, N_tot_levels)
    for i = 2:N_tot_levels
        lowering_operator[i-1,i] = sqrt(i-1)
    end
    return lowering_operator
end

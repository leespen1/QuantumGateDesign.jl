using LinearAlgebra

"""
It should be assumed that the hamiltonian is *complex-valued*, and hence should
be converted when passing to Schrodinger Prob.

kerr_coeffs should be a lower triangular matrix. The q-th diagonal entry should
give ω_q, and then entry kerr_coeff[p,q] (p > q) should
give ξ_pq. 

ω_q is the ground state transition frequency of subsystem q
ξ_q is the self-kerr coefficient of subsystem q
ξ_pq is the cross-kerr coefficient between subsystems p and q
g_pq is the Jaynes-Cummings coupling coefficient between subsystems p and q

Examples of frequencies/coefficients:
ω_q/2π = 4.416 GHz (seen as high as 6 GHz, I think for the ground state transition frequency of the cavity.)
ξ_q/2π = 230 MHz
T₁ = 93.79 μs
T₂ = 102.52 μs, 25 μs (for cavity)

A cavity appears to be treated the same as a qudit, just with many levels.

Maybe I should add a separate thing for rotating frequencies?
"""
function hamiltonian(subsystem_sizes::AbstractVector{Int}, transition_freqs::AbstractVector{<: Real},
        kerr_coeffs::AbstractMatrix{<: Real})

    @assert length(transition_freqs) == size(kerr_coeffs, 1) == size(kerr_coeffs, 2)

    Q = length(subsystem_sizes)
    full_system_size = prod(subsystem_sizes)
    H = zeros(full_system_size, full_system_size)

    lowering_ops = lowering_operators(subsystem_sizes)

    for q in 1:Q
        a_q = lowering_ops[q]
        H .+= transition_freqs[q] .* (a_q' * a_q)
        H .-= 0.5*kerr_coeffs[q,q] .* (a_q' * a_q' * a_q * a_q)
        for p in (q+1):Q
            a_p = lowering_ops[p]
            H .-= kerr_coeffs[p,q] .* (a_p' * a_p* a_q' * a_q)
            # Ignore Jaynes-Cummings coupling for now
        end
    end

    return H
end

function control_ops(subsystem_sizes::AbstractVector{Int})
    lower_ops = lowering_operators(subsystem_sizes)
    sym_ops = [a + a' for a in lower_ops]
    asym_ops = [a - a' for a in lower_ops]
    return sym_ops, asym_ops
end


"""
Given a base qubit transition frequency, and base self/cross-kerr coefficients,
perturb them to create a problem for an ensemble of semi-random qubits.
"""
function multi_qubit_system(subsystem_sizes::AbstractVector{Int},
        base_transition_frequency, base_self_kerr, base_cross_kerr; perturbation_ratio=0.0)
    N = length(subsystem_sizes)

    # Function for 'perturbing' a value (uniform distribution)
    perturb(val) = val + val*rand()*rand([-1,1])*perturbation_ratio

    transition_freqs = ones(N) .* base_transition_frequency
    transition_freqs = perturb.(transition_freqs)

    kerr_coeffs = zeros(N,N)
    for i in 1:N
        kerr_coeffs[i,i] = perturb(base_self_kerr)
        for j in (i+1):N
            kerr_coeffs[j,i] = perturb(base_cross_kerr)
        end
    end

    # Convert to angular
    transition_freqs .*= 2pi
    kerr_coeffs .*= 2pi

    H = hamiltonian(subsystem_sizes, transition_freqs, kerr_coeffs)
    sym_ops, asym_ops = control_ops(subsystem_sizes)

    tf = 50.0
    nsteps = 10

    complex_system_size = size(H, 1)

    U0 = diagm(ones(complex_system_size))

    # Assume all states are essential
    N_ess = complex_system_size
    N_guard = 0

    prob = SchrodingerProb(real(H), imag(H), sym_ops, asym_ops, real(U0), imag(U0),
        tf, nsteps, N_ess)

    return prob
end

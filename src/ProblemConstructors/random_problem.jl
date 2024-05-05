function random_sym_matrix(N)
    rand_mat = rand(N, N)
    return rand_mat + rand_mat'
end

function random_asym_matrix(N)
    rand_mat = rand(N, N)
    return rand_mat - rand_mat'
end

function construct_rand_prob(complex_system_size, N_operators; tf=2.0, nsteps=100)
    system_sym = random_sym_matrix(complex_system_size)
    system_asym = random_asym_matrix(complex_system_size)

    sym_operators = [random_sym_matrix(complex_system_size) for i in 1:N_operators]
    asym_operators = [random_asym_matrix(complex_system_size) for i in 1:N_operators]

    u0 = rand(complex_system_size, complex_system_size)
    v0 = rand(complex_system_size, complex_system_size)

    N_ess_levels = complex_system_size
    N_guard_levels = 0

    prob = SchrodingerProb(
        system_sym, system_asym, sym_operators, asym_operators, u0, v0,
        tf, nsteps, N_ess_levels
    )
    return prob
end

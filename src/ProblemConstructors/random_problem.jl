function random_sym_matrix(rng, N)
    rand_mat = rand(rng, N, N)
    return rand_mat + rand_mat'
end

function random_asym_matrix(rng, N)
    rand_mat = rand(rng, N, N)
    return rand_mat - rand_mat'
end

"""
Construct a problem where all the matrices are initialized randomly (but seeded
so that results are reporducible).
"""
function construct_rand_prob(complex_system_size, N_operators; tf=2.0, nsteps=100)

    u0 = rand(MersenneTwister(0), complex_system_size, complex_system_size)
    v0 = rand(MersenneTwister(1), complex_system_size, complex_system_size)

    system_sym = random_sym_matrix(MersenneTwister(2), complex_system_size)
    system_asym = random_asym_matrix(MersenneTwister(3), complex_system_size)

    sym_operators = [random_sym_matrix(MersenneTwister(100+i), complex_system_size) 
                     for i in 1:N_operators]
    asym_operators = [random_asym_matrix(MersenneTwister(200+i), complex_system_size)
                      for i in 1:N_operators]


    N_ess_levels = complex_system_size
    N_guard_levels = 0

    prob = SchrodingerProb(
        system_sym, system_asym, sym_operators, asym_operators, u0, v0,
        tf, nsteps, N_ess_levels
    )
    return prob
end


"""
A problem meant to be similar to the SWAP problem, but with only two levels and
with simpler hamiltonians, and with no skew-symmetric part.
"""
function main()
    system_sym = [1.0 0.0
                  0.0 0.0]

    system_asym = zeros(2,2) 

    sym_operators = [
        [0.0 1.0
         1.0 0.0]
    ]
    asym_operators = [zeros(2,2)]

    u0 = [0.0 1.0
          1.0 0.0]
    v0 = zeros(2,2)

    tf = 1.0
    nsteps = 1
    N_ess_levels = 2
    N_guard_levels = 0

    target = [0.0 1.0
              1.0 0.0
              0.0 0.0
              0.0 0.0]

    return SchrodingerProb(system_sym, system_asym, sym_operators, asym_operators, u0, v0, tf, nsteps, N_ess_levels, N_guard_levels), target
end



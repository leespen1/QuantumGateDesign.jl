"""
Constructs a rabi oscillator problem with a duration time of pi.

For this duration, analytically a pulse with amplitude |Ω|=0.5 will produce a 
SWAP gate.
"""
function construct_rabi_prob(;tf=pi)
    system_sym = zeros(2,2)
    system_asym = zeros(2,2)
    a = [0.0 1;
         0   0]
    sym_ops = [(a + a')]
    asym_ops = [(a - a')]
    u0 = [1.0 0;
          0   1]
    v0 = zeros(2,2)
    nsteps = 100
    N_ess_levels = 2

    return SchrodingerProb(
        system_sym, system_asym, sym_ops, asym_ops, u0, v0,
        tf, nsteps, N_ess_levels
    )
end

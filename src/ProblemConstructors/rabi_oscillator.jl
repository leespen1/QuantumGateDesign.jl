"""
Constructs a rabi oscillator problem with a duration time of pi.

For this duration, analytically a pulse with amplitude |Ω|=0.5 will produce a 
SWAP gate.
"""
function construct_rabi_prob(;tf=pi, gmres_abstol=1e-10, gmres_reltol=1e-10, nsteps=100)
    system_hamiltonian = zeros(2,2)
    a = [0.0 1;
         0   0]
    sym_ops = [(a + a')]
    asym_ops = [(a - a')]
    U0 = [1   0;
          0   1]
    N_ess_levels = 2

    return SchrodingerProb(
        system_hamiltonian, sym_ops, asym_ops, U0,
        tf, nsteps, N_ess_levels,
        gmres_abstol=gmres_abstol, gmres_reltol=gmres_reltol
    )
end

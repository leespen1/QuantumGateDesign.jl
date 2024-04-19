using QuantumGateDesign
using Test

# Test 1 
# See if we can converge on a SWAP gate for which we know the analytic solution
@testset "SWAP Gate Analytic Minimum" begin

    prob = QuantumGateDesign.construct_rabi_prob(tf=pi)
    prob.nsteps = 100

    # Analytically, we know p0 = 0.5, q0 = 0 is a minimum
    SWAP_target_complex = [0.0 1;
                           1   0]
    SWAP_target = complex_to_real(SWAP_target_complex)

    N_amplitudes = 1
    control = QuantumGateDesign.GRAPEControl(N_amplitudes, prob.tf)

    pcof_init = [0.4, 0.0]

    # Analytically, there are minima at pcof=[±0.5, 0], [±1.5, 0], [±2.5, 0], ...
    pcof_optimal = [0.5, 0.0]

    opt_ret = optimize_gate(prob, control, pcof_init, SWAP_target, order=4)
    pcof_ipopt = opt_ret["ipopt_prob"].x

    @test isapprox(pcof_optimal, pcof_ipopt, rtol=1e-3)
end

# Test 2
# Take a random problem and control vector, get the realized unitary, set that
# as the target, perturb the unitary, and see if we can converge back to the
# original unitary




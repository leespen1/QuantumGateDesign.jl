using QuantumGateDesign
using Test

#===============================================================================
# Check that optimizer can converge to analytic minimum for rabi oscillator
===============================================================================#

#@testset "Rabi Oscillator SWAP Optimization"
prob = QuantumGateDesign.construct_rabi_prob(tf=pi)
control = QuantumGateDesign.GRAPEControl(1, prob.tf)

SWAP_target_complex = [0.0 1;1 0]
target = complex_to_real(SWAP_target_complex)

pcof_init_saves = []
pcof_final_saves = []

@testset "Convergence to p0=0.5, q0=0.0 for nearby starting values" begin
pcof_optimal = [0.5, 0.0]
    for p0 in LinRange(0.4, 0.6, 11)
        for q0 in LinRange(-0.1, 0.1, 11)

            pcof_init  = [p0, q0]
            ret = optimize_gate(prob, control, pcof_init, target, order=4, 
                                ridge_penalty_strength=0, print_level=0)
            pcof_final = ret["ipopt_prob"].x

            @test isapprox(pcof_final, pcof_optimal, rtol=1e-5)

            push!(pcof_init_saves, copy(pcof_init))
            push!(pcof_final_saves, copy(pcof_final))
        end
    end
end


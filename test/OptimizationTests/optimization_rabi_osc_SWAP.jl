#===============================================================================
# Test that optimizer converges to nearest analytic minimum for rabi oscillator
# problem with a SWAP gate target.
===============================================================================#
using QuantumGateDesign
using Test


#@testset "Rabi Oscillator SWAP Optimization"
prob = QuantumGateDesign.construct_rabi_prob(tf=pi)
control = QuantumGateDesign.GRAPEControl(1, prob.tf)

SWAP_target_complex = [0.0 1;1 0]
target = complex_to_real(SWAP_target_complex)

pcof_init_saves = []
pcof_final_saves = []

@testset "Optimization Convergence to Analytic Minimum Test: Rabi Oscillator SWAP Gate" begin
    @info "Checking that analytic minimum of p0=0.5, q0=0.0 is found for nearby starting values."        
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


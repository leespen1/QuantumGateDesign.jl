#===============================================================================
#
# Test that optimizer converges to nearest analytic minimum for rabi oscillator
# problem with a SWAP gate target.
#
# #TODO it sometimes seems to converge to a value nearby the local minimum. Is
# there a whole line of optimal solutions near [0.5, 0]? If so, that might 
# explain the discrepancy. I should compare the final objective value with the
# analytic value.
===============================================================================#
using QuantumGateDesign
using Test: @test, @testset


@testset "Optimization Convergence to Analytic Minimum" begin
    @info "Checking that analytic minimum of p0=0.5, q0=0.0 is found for nearby starting values. Problem is Rabi Oscillator SWAP gate"

    prob = QuantumGateDesign.construct_rabi_prob(
        tf=pi, nsteps=20, gmres_abstol=1e-15, gmres_reltol=1e-15
    )
    control = QuantumGateDesign.GRAPEControl(1, prob.tf)

    SWAP_target_complex = [0.0 1;1 0]

    pcof_init_saves = []
    pcof_final_saves = []

    pcof_optimal = [0.5, 0.0]
    for p0 in LinRange(0.4, 0.6, 5)
        for q0 in LinRange(-0.1, 0.1, 5)

            pcof_init  = [p0, q0]
            opt_hist = optimize_gate(prob, control, pcof_init, SWAP_target_complex, order=8, 
                                ridge_penalty_strength=0, print_level=0)
            final_obj_val = opt_hist.analytic_obj_value[end]
            pcof_final = opt_hist.pcof[end]
            println("For initial pcof $pcof_init final objective value was $final_obj_val.")

            @test isapprox(pcof_final, pcof_optimal, rtol=5e-4)

            push!(pcof_init_saves, copy(pcof_init))
            push!(pcof_final_saves, copy(pcof_final))
        end
    end
end


#==============================================================================
#
# Test that the discrete adjoint and forced methods for computing the gradient
# (which are mathematically exact with respect to the discretized problem) 
# agree to nearly machine precision, and that those methods agree with a
# finite difference approximation of the gradient to a lower level of
# precision.
#
# The stepsize is not very important here because the gradient calculation is
# exact for the discrete problem. So even for a very innacurate solution of
# Schrodinger's equation, the discrete adjoint and forced methods should still
# agree. In fact, an inaccurate solution may be more ideal for testing the
# correctness of the algorithm.
#
==============================================================================#
using QuantumGateDesign
import QuantumGateDesign as QGD
using Test: @test, @testset
using Random: rand, MersenneTwister

# NOTE: Use 1e-15 gmres tolerance so that gradients don't differ due to different
# gmres results.
function test_gradient_agreement(prob, control, pcof, target; 
        orders=(2,4,6,8,10), cost_type=:Infidelity,
        print_results=true
    )

    for order in orders
        @testset "Order $order" begin
            # Check that gradients calculated using discrete adjoint and finite difference
            # methods agree to reasonable precision
            grad_disc_adj = discrete_adjoint(
                prob, control, pcof, target, order=order,
                cost_type=cost_type
            )

            grad_forced = eval_grad_forced(
                prob, control, pcof, target, order=order,
                cost_type=cost_type
            )

            grad_fin_diff = eval_grad_finite_difference(
                prob, control, pcof, target, order=order,
                cost_type=cost_type
            )

            forced_atol = 1e-14
            fin_diff_atol = 1e-9

            @testset "Discrete Adjoint vs Forced Method" begin
                for k in 1:length(grad_disc_adj)
                    @test isapprox(grad_disc_adj[k], grad_forced[k], atol=forced_atol, rtol=forced_atol)
                end
            end

            @testset "Discrete Adjoint vs Finite Difference" begin
                for k in 1:length(grad_disc_adj)
                    @test isapprox(grad_disc_adj[k], grad_fin_diff[k], atol=fin_diff_atol, rtol=fin_diff_atol)
                end
            end

            @testset "Forced Method vs Finite Difference" begin
                for k in 1:length(grad_disc_adj)
                    @test isapprox(grad_forced[k], grad_fin_diff[k], atol=fin_diff_atol, rtol=fin_diff_atol)
                end
            end


            disc_adj_minus_forced = grad_disc_adj - grad_forced
            disc_adj_minus_fin_diff = grad_disc_adj - grad_fin_diff
            forced_minus_fin_diff = grad_forced - grad_fin_diff

            
            maximum(abs.(disc_adj_minus_fin_diff))


            if print_results
                println("Order=$order, cost_type=$cost_type")
                println("--------------------------------------------------")
                #println("Discrete Adjoint Gradient:\n", grad_disc_adj)
                #println("\nForced Method Gradient:\n", grad_forced)
                #println("\nFinite Difference Gradient:\n", grad_fin_diff)
                #println("\n\nDiscrete Adjoint - Forced Method:\n", disc_adj_minus_forced)
                #println("Discrete Adjoint - Finite Difference:\n", disc_adj_minus_fin_diff)
                println("||Discrete Adjoint - Forced Method||∞: ", maximum(abs.(disc_adj_minus_forced)))
                println("||Discrete Adjoint - Finite Difference||∞: ", maximum(abs.(disc_adj_minus_fin_diff)))
                println("||Forced Method - Finite Difference||∞: ", maximum(abs.(forced_minus_fin_diff)))
                println("\n")
            end
        end
    end


    return nothing
end



println("#"^40, "\n")
println("Comparing Gradients for several problems\n")
println("#"^40, "\n")


@testset "Checking That 3 Gradient Computation Methods Agree" begin
  @testset "Piecewise Constant Control" begin
    @testset "Rabi Oscillator" begin
        println("-"^40, "\n")
        println("Problem: Rabi Oscillator\n")
        println("-"^40, "\n")

        prob = QuantumGateDesign.construct_rabi_prob(
            tf=pi,
            gmres_abstol=1e-15,
            gmres_reltol=1e-15
        )
        prob.nsteps = 10

        N_amplitudes = 5
        control = QuantumGateDesign.GRAPEControl(N_amplitudes, prob.tf)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        target = rand(MersenneTwister(0), ComplexF64, prob.N_tot_levels, prob.N_initial_conditions)
        test_gradient_agreement(prob, control, pcof, target)
    end
        
    @testset "Random Problem" begin
        println("-"^40, "\n")
        println("Problem: Random\n")
        println("-"^40, "\n")
        complex_system_size = 4
        N_operators = 1

        prob = QuantumGateDesign.construct_rand_prob(
            complex_system_size,
            N_operators,
            tf = 1.0,
            nsteps = 10,
            gmres_abstol=1e-15, gmres_reltol=1e-15
        )

        N_amplitudes = 5
        control = QuantumGateDesign.GRAPEControl(N_amplitudes, prob.tf)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        target = rand(MersenneTwister(0), ComplexF64, prob.N_tot_levels, prob.N_initial_conditions)
        test_gradient_agreement(prob, control, pcof, target)
    end
  end

  @testset "B-Spline Control" begin
    @testset "Rabi Oscillator" begin
        println("-"^40, "\n")
        println("Problem: Rabi Oscillator\n")
        println("-"^40, "\n")

        prob = QuantumGateDesign.construct_rabi_prob(
            tf=pi,
            gmres_abstol=1e-15,
            gmres_reltol=1e-15,
            nsteps=10
        )

        # High degree, because we want a smooth control
        degree = 16
        N_basis_functions = 20
        control = QGD.FortranBSplineControl(degree, N_basis_functions, prob.tf)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        target = rand(MersenneTwister(0), ComplexF64, prob.N_tot_levels, prob.N_initial_conditions)
        test_gradient_agreement(prob, control, pcof, target)
    end
        
    @testset "Random Problem" begin
        println("-"^40, "\n")
        println("Problem: Random\n")
        println("-"^40, "\n")
        complex_system_size = 4
        N_operators = 1

        prob = QuantumGateDesign.construct_rand_prob(
            complex_system_size,
            N_operators,
            tf = 0.1,
            nsteps = 10,
            gmres_abstol=1e-15, gmres_reltol=1e-15
        )

        # High degree, because we want a smooth control
        degree = 16
        N_basis_functions = 20
        control = QGD.FortranBSplineControl(degree, N_basis_functions, prob.tf)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        target = rand(MersenneTwister(0), ComplexF64, prob.N_tot_levels, prob.N_initial_conditions)
        test_gradient_agreement(prob, control, pcof, target)
    end
  end

  @testset "B-Spline with Carrier Control" begin
    @testset "Rabi Oscillator" begin
        println("-"^40, "\n")
        println("Problem: Rabi Oscillator\n")
        println("-"^40, "\n")

        prob = QuantumGateDesign.construct_rabi_prob(
            tf=pi,
            gmres_abstol=1e-15,
            gmres_reltol=1e-15
        )
        prob.nsteps = 10

        # High degree, because we want a smooth control
        degree = 16
        N_basis_functions = 20
        bspline_control = QGD.FortranBSplineControl(degree, N_basis_functions, prob.tf)
        carrier_frequencies = [-10, -1, 0, 1, 10]
        control = QGD.CarrierControl(bspline_control, carrier_frequencies)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        target = rand(MersenneTwister(0), ComplexF64, prob.N_tot_levels, prob.N_initial_conditions)
        test_gradient_agreement(prob, control, pcof, target)
    end
        
    @testset "Random Problem" begin
        println("-"^40, "\n")
        println("Problem: Random\n")
        println("-"^40, "\n")
        complex_system_size = 4
        N_operators = 1

        prob = QuantumGateDesign.construct_rand_prob(
            complex_system_size,
            N_operators,
            tf = 1.0,
            nsteps = 10,
            gmres_abstol=1e-15, gmres_reltol=1e-15
        )

        # High degree, because we want a smooth control
        degree = 16
        N_basis_functions = 20
        bspline_control = QGD.FortranBSplineControl(degree, N_basis_functions, prob.tf)
        carrier_frequencies = [-10, -1, 0, 1, 10]
        control = QGD.CarrierControl(bspline_control, carrier_frequencies)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        target = rand(MersenneTwister(0), ComplexF64, prob.N_tot_levels, prob.N_initial_conditions)
        test_gradient_agreement(prob, control, pcof, target)
    end
  end

end

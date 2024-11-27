#==============================================================================
#
# Test the ability of the forward timestepping method to converge (with the
# correct order of convergence) to a numerical solution.
#
# This is done for a "Rabi Oscillator" problem, where the system hamiltonian is
# zero, and the control hamiltonians are a^†+a^† and a^†-a^†, and a "Random"
# problem, where the system and control hamiltonians are pseudo-randomly
# generated.
#
# This is done for a series of increasingly complicated controls. First, a
# constant amplitude control, then a B-Spline control, then a B-Spline with
# carrier waves control.
#
==============================================================================#
using QuantumGateDesign
import QuantumGateDesign as QGD
using LinearAlgebra: norm
using Random: rand, MersenneTwister
using UnicodePlots
using Test

function test_order(prob::SchrodingerProb, controls, pcof; N_iterations=6,
        orders=(2,4,6,8,10,12)
    )
    histories_dict = QuantumGateDesign.get_histories(
        prob, controls, pcof, N_iterations, orders=orders,
        min_error_limit=1e-14, quiet=true
    )

    true_history = histories_dict["Order $(maximum(orders)) (QGD)"]["histories"][end]

    for order in orders
      @testset "Order $order" begin
        dict_key = "Order $order (QGD)"
        run_dict = histories_dict[dict_key]
        step_sizes = run_dict["step_sizes"]
        histories = run_dict["histories"]


        # Make sure we have enough data to check order of convergence, skip if not
        if length(histories) <= 2
            @warn "Only $(length(histories)) histories generated, not enough to check order of convergence. Method may be too accurate and a larger initial stepsize is needed.\nSkipping tests for this order..."
            continue
        end

        # Method 1 : Check that the point-to-point slope is roughly the order of the method
        log_relative_errors = [log10(norm(true_history - history)/norm(true_history))
                           for history in histories[1:end-1]]
        log_step_sizes = log10.(step_sizes[1:end-1])
        error_differences = diff(log_relative_errors) ./ diff(log_step_sizes)


        
        for error_diff in error_differences
            @test isapprox(error_diff, order, atol=0.5)
        end

        # Method 2 : Do a least squares fit of the entire line, check that
        # slope is roughly the order of the method
        A = [ones(length(log_step_sizes)) log_step_sizes]
        b = log_relative_errors
        x = A \ b
        slope = x[2]
        @test isapprox(slope, order, atol=0.5)

        pl = scatterplot(log_step_sizes, log_relative_errors,
            title="Order $order, Rel Err vs Step Size",
            xlabel="Log10(step_size)", ylabel="Log10(Rel Err)",
            marker=:circle
        )
        display(pl)
        println("[Order $order] Least-Squares Slope (log scale) of entire line = ", slope)
        println("[Order $order] Slopes (log scale) between adjacent points:")
        println("\t", error_differences)

      end
    end

    return nothing
end

@testset "Checking Forward Evolution Convergence Order (Using Richardson Extrapolation)" begin
  @testset "Constant Control" begin
    println("\n\n", "="^40, "\nConstant Control Convergence Tests\n", "="^40, "\n\n\n")
    @testset "Rabi Oscillator" begin

        # Do rabi oscillation for one (two?) period(s) for lower-order methods
        prob = QuantumGateDesign.construct_rabi_prob(
            tf=2*pi, gmres_abstol=1e-15, gmres_reltol=1e-15
        )

        control = QGD.GRAPEControl(1, prob.tf)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        println("="^40, "\nProblem: Rabi Oscillator (order 2, 4, 6)\n", "="^40, "\n")
        test_order(prob, control, pcof, orders=(2,4,6))

        # Do rabi oscillation for 20 (40?) period for higher-order methods
        # (need a more challenging problem for the high-order methods to not
        # just reach machine precision immediately) 
        prob = QuantumGateDesign.construct_rabi_prob(
            tf=40*pi, gmres_abstol=1e-15, gmres_reltol=1e-15
        )
        control = QGD.GRAPEControl(1, prob.tf)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        println("="^40, "\nProblem: Rabi Oscillator (order 8, 10, 12)\n", "="^40, "\n")
        test_order(prob, control, pcof, orders=(8, 10, 12))
    end

    @testset "Random Schrodinger Problem" begin
        complex_system_size = 4
        N_operators = 1

        prob = QuantumGateDesign.construct_rand_prob(
            complex_system_size,
            N_operators,
            tf = 1.75,
            nsteps = 10,
            gmres_abstol=1e-15,
            gmres_reltol=1e-15
        )

        control = QGD.GRAPEControl(1, prob.tf)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        println("="^40, "\nProblem: Random\n", "="^40, "\n")
        test_order(prob, control, pcof)
    end
  end 

  @testset "B-Spline Control" begin
    println("\n\n", "="^40, "\nB-Spline Control Convergence Tests\n", "="^40, "\n\n\n")
    @testset "Rabi Oscillator" begin

        # Do rabi oscillation for one (two?) period(s) for lower-order methods
        prob = QuantumGateDesign.construct_rabi_prob(
            tf=2*pi, gmres_abstol=1e-15, gmres_reltol=1e-15
        )

        # High degree, because we want a smooth control
        degree = 16
        N_basis_functions = 20
        control = QGD.FortranBSplineControl(degree, N_basis_functions, prob.tf)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        println("="^40, "\nProblem: Rabi Oscillator (order 2, 4, 6)\n", "="^40, "\n")
        test_order(prob, control, pcof, orders=(2,4,6))

        # Do rabi oscillation for 20 (40?) period for higher-order methods
        # (need a more challenging problem for the high-order methods to not
        # just reach machine precision immediately) 
        prob = QuantumGateDesign.construct_rabi_prob(
            tf=40*pi, gmres_abstol=1e-15, gmres_reltol=1e-15
        )
        control = QGD.GRAPEControl(1, prob.tf)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        println("="^40, "\nProblem: Rabi Oscillator (orders 8, 10, 12)\n", "="^40, "\n")
        test_order(prob, control, pcof, orders=(8, 10, 12))
    end

    @testset "Random Schrodinger Problem" begin
        complex_system_size = 4
        N_operators = 1

        prob = QuantumGateDesign.construct_rand_prob(
            complex_system_size,
            N_operators,
            tf = 1.0,
            nsteps = 20,
            gmres_abstol=1e-15,
            gmres_reltol=1e-15
        )

        # High degree, because we want a smooth control
        degree = 16
        N_basis_functions = 20
        control = QGD.FortranBSplineControl(degree, N_basis_functions, prob.tf)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        println("="^40, "\nProblem: Random (orders 2,4,6)\n", "="^40, "\n")
        test_order(prob, control, pcof, orders=(2,4,6))
    
        prob = QuantumGateDesign.construct_rand_prob(
            complex_system_size,
            N_operators,
            tf = 1.0,
            nsteps = 60,
            gmres_abstol=1e-15,
            gmres_reltol=1e-15
        )

        # High degree, because we want a smooth control
        degree = 16
        N_basis_functions = 20
        control = QGD.FortranBSplineControl(degree, N_basis_functions, prob.tf)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        println("="^40, "\nProblem: Random (orders 8, 10)\n", "="^40, "\n")
        test_order(prob, control, pcof, orders=(8,10))

        prob = QuantumGateDesign.construct_rand_prob(
            complex_system_size,
            N_operators,
            tf = 10.0,
            nsteps = 40,
            gmres_abstol=1e-15,
            gmres_reltol=1e-15
        )

        # High degree, because we want a smooth control
        degree = 16
        N_basis_functions = 20
        control = QGD.FortranBSplineControl(degree, N_basis_functions, prob.tf)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        println("="^40, "\nProblem: Random (order 12)\n", "="^40, "\n")
        test_order(prob, control, pcof, orders=12)
    end
  end 
  @testset "B-Spline with Carrier Wave Control" begin
    @testset "Rabi Oscillator" begin
        # Do rabi oscillation for one (two?) period(s) for lower-order methods
        prob = QuantumGateDesign.construct_rabi_prob(
            tf=2*pi, gmres_abstol=1e-15, gmres_reltol=1e-15,
            nsteps=50
        )

        # High degree, because we want a smooth control
        degree = 16
        N_basis_functions = 20
        bspline_control = QGD.FortranBSplineControl(degree, N_basis_functions, prob.tf)
        carrier_frequencies = [-10, -1, 0, 1, 10]
        control = QGD.CarrierControl(bspline_control, carrier_frequencies)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        println("="^40, "\nProblem: Rabi Oscillator (order 2, 4, 6, 8, 10, 12)\n", "="^40, "\n")
        test_order(prob, control, pcof, orders=(2,4,6,8,10,12))
    end

    @testset "Random Schrodinger Problem" begin
        complex_system_size = 4
        N_operators = 1

        prob = QuantumGateDesign.construct_rand_prob(
            complex_system_size,
            N_operators,
            tf = 1.0,
            nsteps = 40,
            gmres_abstol=1e-15,
            gmres_reltol=1e-15
        )

        # High degree, because we want a smooth control
        degree = 16
        N_basis_functions = 20
        bspline_control = QGD.FortranBSplineControl(degree, N_basis_functions, prob.tf)
        carrier_frequencies = [-10, -1, 0, 1, 10]
        control = QGD.CarrierControl(bspline_control, carrier_frequencies)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        println("="^40, "\nProblem: Random (order 2,4,6,8)\n", "="^40, "\n")
        test_order(prob, control, pcof, orders=(2,4,6,8))

        prob = QuantumGateDesign.construct_rand_prob(
            complex_system_size,
            N_operators,
            tf = 10.0,
            nsteps = 100,
            gmres_abstol=1e-15,
            gmres_reltol=1e-15
        )

        # High degree, because we want a smooth control
        degree = 16
        N_basis_functions = 20
        bspline_control = QGD.FortranBSplineControl(degree, N_basis_functions, prob.tf)
        carrier_frequencies = [-10, -1, 0, 1, 10]
        control = QGD.CarrierControl(bspline_control, carrier_frequencies)
        pcof = rand(MersenneTwister(0), control.N_coeff) 

        println("="^40, "\nProblem: Random (order 2,4,6,8)\n", "="^40, "\n")
        test_order(prob, control, pcof, orders=(10,12))
    end
  end 
end

using QuantumGateDesign
using LinearAlgebra: norm
using Random
using UnicodePlots
using Test

function test_order(prob::SchrodingerProb, controls, pcof; N_iterations=6,
        orders=(2,4,6,8,10,12)
    )
    histories_dict = QuantumGateDesign.get_histories(
        prob, controls, pcof, N_iterations, orders=orders,
        min_error_limit=1e-14
    )

    for order in orders
      @testset "Order $order" begin
        dict_key = "Order $order (QGD)"
        run_dict = histories_dict[dict_key]
        step_sizes = run_dict["step_sizes"]
        histories = run_dict["histories"]

        true_history = histories[end]

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


        println("Slopes (log scale) between adjacent points")
        println(error_differences)
        
        for error_diff in error_differences
            @test isapprox(error_diff, order, atol=0.5)
        end

        pl = scatterplot(log_step_sizes, log_relative_errors,
            title="Order $order, Rel Err vs Step Size",
            xlabel="Log10(step_size)", ylabel="Log10(Rel Err)",
            marker=:circle
        )
        display(pl)

        # Method 2 : Do a least squares fit of the entire line, check that
        # slope is roughly the order of the method
        A = [ones(length(log_step_sizes)) log_step_sizes]
        b = log_relative_errors
        x = A \ b
        slope = x[2]
        println("Slope (log scale) of entire line = ", slope)
        @test isapprox(slope, order, atol=0.5)
      end
    end

    return nothing
end

@testset "Checking Forward Evolution Convergence Order" begin
Random.seed!(42)
@testset "Rabi Oscillator" begin
    println("-"^40, "\n")
    println("Problem: Rabi Oscillator\n")
    println("-"^40, "\n")

    prob = QuantumGateDesign.construct_rabi_prob(tf=pi)
    prob.nsteps = 10
    control = QuantumGateDesign.HermiteControl(2, prob.tf, 12, :Taylor)
    pcof = rand(control.N_coeff) 

    test_order(prob, control, pcof)
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
        nsteps = 10
    )
    control = QuantumGateDesign.HermiteControl(2, prob.tf, 12, :Taylor)
    pcof = rand(control.N_coeff) 

    test_order(prob, control, pcof)
end
end

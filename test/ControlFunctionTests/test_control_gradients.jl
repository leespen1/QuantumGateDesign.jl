using QuantumGateDesign
import QuantumGateDesign as QGD
using LinearAlgebra: norm
using Test: @test, @testset
using Random: rand, MersenneTwister

function test_control_gradient(control::AbstractControl, pcof; upto_order=0)
    t_start = upto_order
    ts = LinRange(0, control.tf, 1001)
    grad_analytic = zeros(control.N_coeff)

    tol = 50*(1e-15)^(2/3)
    acceptable_pass_ratio = 0.95
    data = ["Order" "% Passed" "Avg Error" "Max Error" "t (Max Error)"]

    for order in 0:upto_order
        maximum_err = 0.0
        avg_err = 0.0
        maximum_err_t = NaN
        N_passed = 0
        N_tests = length(ts)

        @testset "Gradient of p/q derivative order $order" begin
            for t in ts
                QGD.eval_grad_p_derivative!(grad_analytic, control, t, pcof, order)
                grad_fin_diff = QGD.eval_grad_p_derivative_fin_diff(control, t, pcof, order)

                if norm(grad_analytic) != 0
                    err = norm(grad_analytic - grad_fin_diff)/norm(grad_analytic)
                else 
                    err = norm(grad_analytic - grad_fin_diff)
                end
                avg_err += err
                maximum_err = max(err, maximum_err)
                if maximum_err == err 
                    maximum_err_t = t
                end
                if (err < tol)
                    N_passed += 1
                end

                #println(all(isapprox.(grad_analytic, grad_fin_diff, rtol=1e-10)) )
                #println(norm(grad_analytic - grad_fin_diff))
            end

            percent_passed = N_passed / N_tests
            avg_err /= N_tests

            data = vcat(data, [order percent_passed avg_err maximum_err maximum_err_t])
            println("Order $order")
            @printf("%d finite difference evaluations agreed with analytic result with tolerance of %.2e.\n", N_passed, tol)
            @printf("Maximum relative error was %.2e, at time t=%.4e. (Absolute error is used if analytic value is zero).\n\n", maximum_err, maximum_err_t)

            @test N_passed > acceptable_pass_ratio*N_tests
        end #@testset

    end
    hl_passed = Highlighter((data, i, j) -> (j == 2) && (data[i,j] < acceptable_pass_ratio),
                     crayon"fg:red bold bg:dark_gray")
    hl_avg_err = Highlighter((data, i, j) -> (j == 3) && (data[i,j] > tol),
                     crayon"fg:yellow bg:dark_gray")
    hl_max_err = Highlighter((data, i, j) -> (j == 4) && (data[i,j] > 100*tol),
                     crayon"fg:yellow bg:dark_gray")
    results_table = pretty_table(
        data[2:end,:];
        header=data[1,:],
        header_crayon = crayon"yellow bold",
        highlighters = (hl_passed, hl_avg_err, hl_max_err),
    )
    return nothing
end



@testset "Testing Control Dervatives" begin
    @testset "GRAPE Control" begin
    end
    @testset "Bspline Control" begin
        tf = 5.0
        D1 = 10
        omega = [0.0, 1.0, 2.0]
        bspline_control = QGD.BsplineControl(tf, D1, omega)
        bspline_control_autodiff = QGD.BSplineControlAutodiff(tf, D1, omega)

        pcof = rand(MersenneTwister(0), bspline_control.N_coeff)

        @testset "Hard Coded Derivative" begin
            test_control_gradient(bspline_control, pcof, upto_order=1)
        end
        #@testset "Automatic Differentiation" begin
        #    test_control_gradient(bspline_control_autodiff, pcof, upto_order=4)
        #end
    end
    @testset "Hermite Control" begin
    end
end

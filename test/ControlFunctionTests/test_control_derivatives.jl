#==============================================================================
#
# Test correctness of derivatives of controls, comparing the derivative values
# with finite difference approximations of the derivatives.
#
=============================================================================#
using QuantumGateDesign
import QuantumGateDesign as QGD
using Test: @test, @testset
using Random: rand, MersenneTwister
using Printf: @printf
using PrettyTables

function test_control_derivatives(control::AbstractControl,
        pcof::AbstractVector{<: Real}; upto_order::Integer=1, ts=missing
    )

    if ismissing(ts)
        # Add cushion to start and end ts so we have enough space to do centered difference methods
        max_cd_stepsize = 1e-15 ^ (1/(2*upto_order+1))
        min_t = 0.0 + max_cd_stepsize*upto_order
        max_t = control.tf - max_cd_stepsize*upto_order
        ts = LinRange(min_t, max_t, 1000)
    end

    tol = 50*(1e-15)^(2/3)
    acceptable_pass_ratio = 0.95

    data = ["Order" "% Passed" "Avg Error" "Max Error" "t (Max Error)"]

    for order in 1:upto_order
        @testset "Value p/q derivative order $order" begin
            # Once we believe eval_p_derivative gives analytically correct results,
            # fine to use it central difference to compute first derivative of that
            p(x) = eval_p_derivative(control, x, pcof, order-1)
            q(x) = eval_q_derivative(control, x, pcof, order-1)

            maximum_err = 0.0
            avg_err = 0.0
            maximum_err_t = NaN
            N_passed = 0
            N_tests = 2*length(ts)
            for t in ts function_val = eval_p_derivative(control, t, pcof, order-1)
                dval_analytic = eval_p_derivative(control, t, pcof, order)
                dval_fin_diff = central_difference(p, t, 1)

                if dval_analytic != 0
                    err = abs((dval_analytic - dval_fin_diff) / dval_analytic)
                else # If analytic value is zero, switch to absolute tolerance to avoid NaN
                    err = abs(dval_analytic - dval_fin_diff)
                end
                avg_err += err
                maximum_err = max(err, maximum_err)
                if maximum_err == err 
                    maximum_err_t = t
                end
                if (err < tol)
                    N_passed += 1
                end
                


                function_val = eval_q_derivative(control, t, pcof, order-1)
                dval_analytic = eval_q_derivative(control, t, pcof, order)
                dval_fin_diff = central_difference(q, t, 1)

                if dval_analytic != 0
                    err = abs((dval_analytic - dval_fin_diff) / dval_analytic)
                else # If analytic value is zero, switch to absolute tolerance to avoid NaN
                    err = abs(dval_analytic - dval_fin_diff)
                end
                avg_err += err
                maximum_err = max(err, maximum_err)
                if maximum_err == err 
                    maximum_err_t = t
                end
                if (err < tol)
                    N_passed += 1
                end
            end #for

            percent_passed = N_passed / N_tests
            avg_err /= N_tests

            data = vcat(data, [order percent_passed avg_err maximum_err maximum_err_t])
            println("Order $order")
            @printf("%d finite difference evaluations agreed with analytic result with tolerance of %.2e.\n", N_passed, tol)
            @printf("Maximum relative error was %.2e, at time t=%.4e. (Absolute error is used if analytic value is zero).\n\n", maximum_err, maximum_err_t)

            @test N_passed > acceptable_pass_ratio*N_tests
        end #@testset
    end #for

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



function central_difference(f, x, n, h=missing)
    if ismissing(h)
        # Balance impact of roundoff error and taylor expansion error
        h = 1e-15 ^ (1/(2n+1)) # Should maybe change to minimize relative error, not absolute
        # Expected (abs) error will be 10.0 ^ (-15*(n+1) / (2n+1))
    end

    derivative_val = 0.0
    for i in 0:n
        derivative_val += (-1)^i * binomial(n,i) * f(x + (0.5*n-i)*h)
    end
    derivative_val /= h^n
    return derivative_val
end



@testset "Testing Control Dervatives" begin
    @testset "GRAPE Control" begin
        N_amplitudes = 10
        tf = 5.0
        control = QGD.GRAPEControl(N_amplitudes, tf)
        pcof = rand(MersenneTwister(0), control.N_coeff)

        ts = Float64[]
        println("="^40, "\nGRAPEControl\n", "="^40, "\n")
        test_control_derivatives(control, pcof, upto_order=4) 
    end

    @testset "Hard-Coded degree 2 B-Spline Control" begin
        tf = 5.0
        N_basis_functions = 10

        control = QGD.BSpline2Control(N_basis_functions, tf)
        pcof = rand(MersenneTwister(0), control.N_coeff)

        println("="^40, "\nBSpline2Control\n", "="^40, "\n")
        test_control_derivatives(control, pcof, upto_order=4)
    end

    @testset "PPPACK B-Spline Control" begin
        tf = 5.0
        N_basis_functions = 10

        for degree = (2,4,6, 8)
          @testset "degree $degree" begin
            control = QGD.FortranBSplineControl(degree, N_basis_functions, tf)
            pcof = rand(MersenneTwister(0), control.N_coeff)

            println("="^40, "\nFortranBSplineControl, degree $degree\n", "="^40, "\n")
            test_control_derivatives(control, pcof, upto_order=4)
          end
        end
    end

    @testset "PPPACK B-Spline Control with Carrier" begin
        tf = 5.0
        N_basis_functions = 10
        degree = 8
        bspline_control = QGD.FortranBSplineControl(degree, N_basis_functions, tf)

        carrier_frequencies = [-10.0, -1.0, 0.0, 1.0, 10.0]
        carrier_control = CarrierControl(bspline_control, carrier_frequencies)
        pcof = rand(MersenneTwister(0), carrier_control.N_coeff)

        println("="^40, "\nCarrier FortranBSplineControl, degree $degree\n", "="^40, "\n")
        test_control_derivatives(carrier_control, pcof, upto_order=4)
    end

    #=
    @testset "Hermite Control" begin
        tf = 5.0
        N_points = 3
        N_derivatives = 4
        scaling_type = :Derivative

        hermite_control = QGD.HermiteControl(N_points, tf, N_derivatives, scaling_type)
        pcof = rand(MersenneTwister(0), hermite_control.N_coeff)

        test_control_derivatives(hermite_control, pcof, upto_order=2*(1+N_derivatives))
    end
    =#
end


using QuantumGateDesign
using Polynomials
using Test: @test, @testset
using Random: rand, MersenneTwister
using LinearAlgebra: norm

#===============================================================================
# Take a polynomial, sample it and it's derivatives (evaluated
# analytically using the Derivatives.jl package), and construct a HermiteControl
# based on that.
#
# Test that the original polynomial is reproduced (with enough derivatives taken
# such that the polynomial should be unique).
===============================================================================#

function test_hermite_poly_agreement(poly_coeffs; rtol=1e-14)
    # Choose number of derivatives so that interpolation is exact
    N_derivatives = div(length(poly_coeffs), 2) - 1
    inf_norm(x) = norm(x, Inf)

    N_control_points = 5

    # Set up polynomials and their derivatives (technic)
    p_poly = Polynomial(poly_coeffs)
    q_poly = Polynomial(poly_coeffs)
    p_poly_derivatives = [derivative(p_poly, i) for i in 0:N_derivatives]
    q_poly_derivatives = [derivative(q_poly, i) for i in 0:N_derivatives]

    tf = 1.0
    ts_control = LinRange(0, tf, N_control_points)
    
    # Get control vector by evaluating polynomials at control points
    pcof_array = zeros(1+N_derivatives, N_control_points, 2)
    for (j, t) in enumerate(ts_control)
        for k in 0:N_derivatives
            pcof_array[1+k, j, 1]  = p_poly_derivatives[1+k](t)
            pcof_array[1+k, j, 2]  = q_poly_derivatives[1+k](t)
        end
    end
    pcof_vec = reshape(pcof_array, :)

    hermite_control = QuantumGateDesign.HermiteControl(
        N_control_points, tf, N_derivatives, :Derivative
    )

    data = ["Order" "Avg Error" "Max Error" "Max Function Value"]

    # Test that HermiteControl agrees with analytic polynomial derivative values when constructed 
    ts = LinRange(0, tf, 101)
    for derivative_order in 0:N_derivatives
        @testset "Derivative Order $derivative_order" begin
            pvals = [eval_p_derivative(hermite_control, t, pcof_vec, derivative_order)
                     for t in ts]
            qvals = [eval_q_derivative(hermite_control, t, pcof_vec, derivative_order)
                     for t in ts]

            pvals_analytic = [p_poly_derivatives[1+derivative_order](t)
                             for t in ts]
            qvals_analytic = [q_poly_derivatives[1+derivative_order](t)
                             for t in ts]

            vals = vcat(pvals, qvals)
            vals_analytic = vcat(pvals_analytic, qvals_analytic)

            println("Maxval = ", norm(vals_analytic, Inf))
            println("Order $derivative_order rel errors: ", norm((vals - vals_analytic)) / norm(vals_analytic))
            @test isapprox(vals, vals_analytic, rtol=rtol)

            errors = abs.(vals - vals_analytic)
            rel_errors = abs.(errors ./ vals_analytic)
            avg_error = sum(rel_errors) / length(rel_errors)
            max_error = norm(rel_errors, Inf)
            max_val = norm(vals_analytic, Inf)

            data = vcat(data, [derivative_order avg_error max_error max_val])
        end
    end

    results_table = pretty_table(
        data[2:end,:];
        header=data[1,:],
        header_crayon = crayon"yellow bold",
    )
end



@testset "HermiteControl Interpolation of Analytic Polynomial" begin
    @info "Testing HermiteControl Interpolation of a Polynomial"
    @testset "pcof = ones(...) tests" begin
        println("#"^40, "\npcof = ones(...) tests\n", "#"^40)
        @testset "Degree 5 polynomial" begin
            println("-"^40, "\nTest: Degree 5 Polynomial\n", "-"^40)
            poly_coeffs = ones(6)
            test_hermite_poly_agreement(poly_coeffs, rtol=1e-12)
        end
        @testset "Degree 11 polynomial" begin
            println("-"^40, "\nTest: Degree 10 Polynomial\n", "-"^40)
            poly_coeffs = ones(12)
            test_hermite_poly_agreement(poly_coeffs, rtol=1e-12)
        end

        @testset "Degree 15 polynomial" begin
            println("-"^40, "\nTest: Degree 15 Polynomial\n", "-"^40)
            poly_coeffs = ones(16)
            test_hermite_poly_agreement(poly_coeffs, rtol=1e-12)
        end
        @testset "Degree 21 polynomial" begin
            println("-"^40, "\nTest: Degree 20 Polynomial\n", "-"^40)
            poly_coeffs = ones(22)
            test_hermite_poly_agreement(poly_coeffs, rtol=1e-12)
        end
    end
    @testset "pcof = rand(...) tests" begin
        println("#"^40, "\npcof = rand(...) tests\n", "#"^40)
        @testset "Degree 5 polynomial" begin
            println("-"^40, "\nTest: Degree 5 Polynomial\n", "-"^40)
            poly_coeffs = rand(MersenneTwister(0), 6)
            test_hermite_poly_agreement(poly_coeffs, rtol=1e-12)
        end
        @testset "Degree 11 polynomial" begin
            println("-"^40, "\nTest: Degree 10 Polynomial\n", "-"^40)
            poly_coeffs = rand(MersenneTwister(0), 12)
            test_hermite_poly_agreement(poly_coeffs, rtol=1e-12)
        end

        @testset "Degree 15 polynomial" begin
            println("-"^40, "\nTest: Degree 15 Polynomial\n", "-"^40)
            poly_coeffs = rand(MersenneTwister(0), 16)
            test_hermite_poly_agreement(poly_coeffs, rtol=1e-12)
        end
        @testset "Degree 21 polynomial" begin
            println("-"^40, "\nTest: Degree 20 Polynomial\n", "-"^40)
            poly_coeffs = rand(MersenneTwister(0), 22)
            test_hermite_poly_agreement(poly_coeffs, rtol=1e-12)
        end
    end
end

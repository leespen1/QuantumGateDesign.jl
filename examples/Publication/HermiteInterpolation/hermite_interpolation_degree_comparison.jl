using QuantumGateDesign
using Polynomials
using Test: @test, @testset
using Random: rand, MersenneTwister
using LinearAlgebra: norm
using PrettyTables
using Printf

using LaTeXTabulars
using LaTeXStrings               # not a dependency, but works nicely

#===============================================================================
# Take a polynomial, sample it and it's derivatives (evaluated
# analytically using the Derivatives.jl package), and construct a HermiteControl
# based on that.
#
# Test that the original polynomial is reproduced (with enough derivatives taken
# such that the polynomial should be unique).
===============================================================================#

function test_hermite_degree(poly_coeffs; table_name="/tmp/table.tex")
    poly_coeffs_original = copy(poly_coeffs)

    N_control_points = 2
    tf = 1.0
    ts_control = LinRange(0, tf, N_control_points)

    inf_norm(x) = norm(x, Inf)

    header = ["Degree", "1", "2", "3", "4"]
    degrees = collect(1:4:length(poly_coeffs_original)-1)
    data = Matrix{Any}(undef, length(degrees), length(header))

    for (i, degree) in enumerate(degrees)
        # Choose number of derivatives so that interpolation is exact
        N_derivatives = div(degree-1, 2)

        # Set up polynomial and derivatives
        poly_coeffs = poly_coeffs_original[1:degree+1]
        p_poly = Polynomial(poly_coeffs)
        p_poly_derivatives = [derivative(p_poly, i) for i in 0:N_derivatives]


        pcof_array = zeros(1+N_derivatives, N_control_points, 2)
        for (j, t) in enumerate(ts_control)
            for k in 0:N_derivatives
                pcof_array[1+k, j, 1]  = p_poly_derivatives[1+k](t)
            end
        end
        # Get control vector by evaluating polynomials at control points
        pcof_vec = reshape(pcof_array, :)

        hermite_control = QuantumGateDesign.HermiteControl(
            N_control_points, tf, N_derivatives, :Derivative
        )


        max_ℓ2_error = 0.0
        max_ℓ∞_error = 0.0
        max_ℓ2_vals = 0.0
        max_ℓ∞_vals = 0.0
        ts = LinRange(0, tf, 101)
        # Only go up to the derivative order we would use when using
        # timestepping method with order of accuracy equal to smoothness of
        # polynomial.
        for derivative_order in 0:div(N_derivatives, 2)
            vals = [eval_p_derivative(hermite_control, t, pcof_vec, derivative_order)
                     for t in ts]

            vals_analytic = [p_poly_derivatives[1+derivative_order](t)
                             for t in ts]


            println("Maxval = ", norm(vals_analytic, Inf))
            println("Order $derivative_order rel errors: ", norm((vals - vals_analytic)) / norm(vals_analytic))

            errors = vals - vals_analytic
            

            max_ℓ2_error = max(max_ℓ2_error, norm(errors, 2))
            max_ℓ∞_error = max(max_ℓ∞_error, norm(errors, Inf))
            max_ℓ2_vals  = max(max_ℓ2_vals,  norm(vals_analytic, 2))
            max_ℓ∞_vals  = max(max_ℓ∞_vals,  norm(vals_analytic, Inf))
        end

        data[i,1] = degree
        data[i,2] = max_ℓ2_error
        data[i,3] = max_ℓ∞_error
        data[i,4] = max_ℓ2_vals
        data[i,5] = max_ℓ∞_vals
    end


    results_table = pretty_table(
        data;
        header=header,
        header_crayon = crayon"yellow bold",
    )

    # Format Int and Float arguments correctly as strings
    formatter(x::Int) = @sprintf("%d", x)
    formatter(x::Float64) = @sprintf("%.1e", x)
    formatted_data = formatter.(data)

    latex_tabular(
        table_name,
        Tabular("ccccc"), # Table format, 5 centered entries in each row
        [  # Table lines
            Rule(:top), # Top rule
            # Top row, 2nd and 3rd entries each span two columns
            ["", MultiColumn(4, :c, L"\max_{k\ \leq\ (\textrm{Degree}-1)/4}")],
            CMidRule("lr", 2, 5),
            ["Degree",
                 L"\| \frac{d^k\tilde{p}}{dx^k} - \frac{d^k p}{dx^k}\|_2", 
                 L"\| \frac{d^k\tilde{p}}{dx^k} - \frac{d^k p}{dx^k}\|_\infty",
                 L"\| \frac{d^k p}{dx^k}\|_2",
                 L"\| \frac{d^k p}{dx^k}\|_\infty",
            ],
            Rule(:mid), # Full horizontal rule
            formatted_data, # The data
            Rule(:bottom), # Bottom rule
        ]
    )

    return data
end

function rescale_poly_coeffs(poly_coeffs)
    rescaled_poly_coeffs = poly_coeffs .- 0.5
    for i in 1:length(rescaled_poly_coeffs)
        rescaled_poly_coeffs[i] /= factorial(big(i-1))
    end
    return rescaled_poly_coeffs
end




@info "Testing HermiteControl Interpolation of a Polynomial"
println("#"^40, "\npcof = rand(...) tests\n", "#"^40)

#=
println("-"^40, "\nTest: Degree 5 Polynomial\n", "-"^40)
poly_coeffs = rand(MersenneTwister(0), 6)
poly_coeffs = rescale_poly_coeffs(poly_coeffs)
test_hermite_poly_agreement(poly_coeffs)

println("-"^40, "\nTest: Degree 10 Polynomial\n", "-"^40)
poly_coeffs = rand(MersenneTwister(0), 12)
poly_coeffs = rescale_poly_coeffs(poly_coeffs)
test_hermite_poly_agreement(poly_coeffs)

println("-"^40, "\nTest: Degree 15 Polynomial\n", "-"^40)
poly_coeffs = rand(MersenneTwister(0), 16)
poly_coeffs = rescale_poly_coeffs(poly_coeffs)
test_hermite_poly_agreement(poly_coeffs)
=#

println("-"^40, "\nTest: Degree 29 Polynomial\n", "-"^40)
poly_coeffs = rand(MersenneTwister(0), 38)
poly_coeffs = rescale_poly_coeffs(poly_coeffs)
test_hermite_degree(poly_coeffs)
#=
=#

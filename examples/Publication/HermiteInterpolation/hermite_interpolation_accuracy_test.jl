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

function test_hermite_poly_agreement(poly_coeffs; table_name="/tmp/table.tex")
    # Choose number of derivatives so that interpolation is exact
    N_derivatives = div(length(poly_coeffs), 2) - 1
    inf_norm(x) = norm(x, Inf)

    N_control_points = 2

    # Set up polynomials and their derivatives (technic)
    p_poly = Polynomial(poly_coeffs)
    p_poly_derivatives = [derivative(p_poly, i) for i in 0:N_derivatives]

    tf = 1.0
    ts_control = LinRange(0, tf, N_control_points)
    
    # Get control vector by evaluating polynomials at control points
    pcof_array = zeros(1+N_derivatives, N_control_points, 2)
    for (j, t) in enumerate(ts_control)
        for k in 0:N_derivatives
            pcof_array[1+k, j, 1]  = p_poly_derivatives[1+k](t)
        end
    end
    pcof_vec = reshape(pcof_array, :)

    hermite_control = QuantumGateDesign.HermiteControl(
        N_control_points, tf, N_derivatives, :Derivative
    )



    header = ["Order", "Error ℓ₂", "Error ℓ∞", "Func Val ℓ₂", "Func Val ℓ∞"]
    data = Matrix{Any}(undef, 1+N_derivatives, length(header))

    # Test that HermiteControl agrees with analytic polynomial derivative values when constructed 
    ts = LinRange(0, tf, 101)
    for derivative_order in 0:N_derivatives
        vals = [eval_p_derivative(hermite_control, t, pcof_vec, derivative_order)
                 for t in ts]

        vals_analytic = [p_poly_derivatives[1+derivative_order](t)
                         for t in ts]


        println("Maxval = ", norm(vals_analytic, Inf))
        println("Order $derivative_order rel errors: ", norm((vals - vals_analytic)) / norm(vals_analytic))

        errors = vals - vals_analytic
        

        ℓ2_error = norm(errors, 2)
        ℓ∞_error = norm(errors, Inf)
        
        ℓ2_vals_analytic = norm(vals_analytic)
        ℓ∞_vals_analytic = norm(vals_analytic, Inf)



        data[1+derivative_order, 1] = derivative_order # Handle ints and floats separately
        data[1+derivative_order, 2:5] .= [ℓ2_error, ℓ∞_error, ℓ2_vals_analytic, ℓ∞_vals_analytic]
    end

    display(data)

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
            [L"k", L"\| \frac{d^k\tilde{p}}{dx^k} - \frac{d^k p}{dx^k}\|_2", 
                 L"\| \frac{d^k\tilde{p}}{dx^k} - \frac{d^k p}{dx^k}\|_\infty",
                 L"\| \frac{d^k p}{dx^k}\|_2",
                 L"\| \frac{d^k p}{dx^k}\|_\infty",
            ],
            Rule(:mid), # Full horizontal rule
            formatted_data, # The data
            Rule(:bottom), # Bottom rule
        ]
    )
    #=
    # Old version
    latex_tabular(
        table_name,
        Tabular("ccccc"), # Table format, 5 centered entries in each row
        [  # Table lines
            Rule(:top), # Top rule
            # Top row, 2nd and 3rd entries each span two columns
            ["", MultiColumn(2, :c, "Error Norms"), MultiColumn(2, :c, "Value Norms")], 
            # Put horizontal rule underneath the multi-column entries
            # "lr" trims the left and right sides of the rule, so that adjacent midrules don't connect
            CMidRule("lr", 2, 3), 
            CMidRule("lr", 4, 5),
            ["Order", L"\ell_2", L"\ell_\infty", L"\ell_2", L"\ell_\infty"],
            Rule(:mid), # Full horizontal rule
            formatted_data, # The data
            Rule(:bottom), # Bottom rule
        ]
    )
    =# 
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

println("-"^40, "\nTest: Degree 21 Polynomial\n", "-"^40)
poly_coeffs = rand(MersenneTwister(0), 22)
poly_coeffs = rescale_poly_coeffs(poly_coeffs)
test_hermite_poly_agreement(poly_coeffs)
#=
=#

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
#using PrettyTables

function mean(collection)
    return sum(collection) / length(collection)
end

"""
Approximate the derivative of the function f at point x using the central
difference formula

    f'(x) ≈ ( f(x+h) - f(x-h) ) / 2h.

This approximation is O(h^2). The default value of h is chosen to balance the
truncation error with the roundoff error due to the effects of machine precision.
"""
function approximate_derivative_using_central_difference(f, x, h=eps()^(1/3))
    return ( f(x+h) - f(x-h) ) / (2*h)
end


function test_control_derivative_errors(control::AbstractControl,
    pcof::AbstractVector{<: Real}, deriv_order::Integer, t_grid
)

    values_agree(x,y) = isapprox(x,y, atol=1e-12, rtol=1e-7)

    real_deriv_val(t) = eval_p_derivative(control, t, pcof, deriv_order)
    real_deriv_val_fin_diff(t) = approximate_derivative_using_central_difference(
        x -> eval_p_derivative(control, x, pcof, deriv_order-1), t
    )


    real_vals = real_deriv_val.(t_grid)
    real_vals_fin_diff = real_deriv_val_fin_diff.(t_grid)
    
    @test all(values_agree.(real_vals, real_vals_fin_diff)) 


    imag_deriv_val(t) = eval_p_derivative(control, t, pcof, order_n)
    imag_deriv_val_fin_diff(t) = approximate_derivative_using_central_difference(
        x -> eval_q_derivative(control, x, pcof, deriv_order-1), t
    )


    imag_vals = imag_deriv_val.(t_grid)
    imag_vals_fin_diff = imag_deriv_val_fin_diff.(t_grid)

    real_deriv_errors = abs.(real_deriv_vals - real_deriv_val_fin_diff)
    imag_deriv_errors = abs.(imag_deriv_vals - imag_deriv_val_fin_diff)


    return real_deriv_errors, imag_deriv_errors
end




@testset "B-Spline Controls (Hard-Coded degree 2)" begin
    tf = 10.0
    N_basis_functions = 10
    control = QGD.BSpline2Control(N_basis_functions, tf)
    pcof = rand(MersenneTwister(0), control.N_coeff)
    h = eps()^(1/3)
    N_points = 1_000
    t_grid = LinRange(2h, tf-2h, N_points)
    real_errors, imag_errors = test_control_derivative_errors(
        control, pcof, deriv_order=1
    )
    @info "Degree 2 B-Spline Derivative Errors" maximum(real_errors) mean(real_errors) maximum(imag_errors) mean(imag_errors) 
end

#=
@testset "B-Spline Controls (PPPACK Implementation)" begin
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

=#

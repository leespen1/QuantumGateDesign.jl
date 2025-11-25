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
using Statistics
#using PrettyTables


function scaled_relative_error(x, y; atol=1e-12, rtol=1e-7)
    denom = atol + rtol * max(abs(x), abs(y))
    return denom == 0 ? 0.0 : abs(x - y) / denom
end


"""
    pretty_nt(nt; digits=3)

Return a NamedTuple where each value of `nt` is converted to a pretty
string. Nested NamedTuples are processed recursively.
"""
function pretty_named_tuple(nt::NamedTuple; digits=3)
    return NamedTuple{keys(nt)}(pretty_value.(values(nt)))
end

"""
    pretty_value(x; digits=3)
    
Return a string representing `x` in scientific notation.
"""
function pretty_value(x; digits=3)
    if x isa Real
        return @sprintf("%.*g", digits, float(x))
    else
        return string(x)
    end
end

function stats(collection)
    return (
        max = maximum(collection),
        root_mean_square = sqrt(mean(abs2, collection)),
        mean = mean(collection),
        median = median(collection),
        q95 = quantile(collection, 0.95),
    )
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


"""
Test that the given control has accurate derivatives of order `deriv_order`, as
compared to their finite difference approximation using the values of the order
`deriv_order-1` derivatives of the control.
"""
function test_control_derivative_accuracy(control::AbstractControl,
    pcof::AbstractVector{<: Real}, t_grid, deriv_order::Integer, h::Real
)


    real_deriv_val(t) = eval_p_derivative(control, t, pcof, deriv_order)
    real_deriv_val_fin_diff(t) = approximate_derivative_using_central_difference(
        x -> eval_p_derivative(control, x, pcof, deriv_order-1), t, h
    )

    real_vals = real_deriv_val.(t_grid)
    real_vals_fin_diff = real_deriv_val_fin_diff.(t_grid)
    
    @test all(isapprox.(real_vals, real_vals_fin_diff, atol=1e-10, rtol=1e-7)) 

    imag_deriv_val(t) = eval_q_derivative(control, t, pcof, deriv_order)
    imag_deriv_val_fin_diff(t) = approximate_derivative_using_central_difference(
        x -> eval_q_derivative(control, x, pcof, deriv_order-1), t, h
    )

    imag_vals = imag_deriv_val.(t_grid)
    imag_vals_fin_diff = imag_deriv_val_fin_diff.(t_grid)


    @test all(isapprox.(imag_vals, imag_vals_fin_diff, atol=1e-10, rtol=1e-7)) 

    all_vals = vcat(real_vals, imag_vals)
    all_vals_fin_diff = vcat(real_vals_fin_diff, imag_vals_fin_diff)
    if !all(isapprox.(all_vals, all_vals_fin_diff, atol=1e-10, rtol=1e-7))
        max_sre, max_sre_i = findmax(scaled_relative_error.(all_vals,all_vals_fin_diff))
        deriv_val = all_vals[max_sre_i]
        fin_diff_val = all_vals_fin_diff[max_sre_i]
        @warn "Test did not pass. Data for worst error:" (
            control_type=typeof(control),
            derivative_order=deriv_order,
            max_scaled_relative_error=max_sre,
            time=t_grid[max_sre_i],
            derivative_value=deriv_val,
            finite_difference_value=fin_diff_val,
            absolute_error=abs(deriv_val-fin_diff_val),
            relative_error=abs(deriv_val-fin_diff_val) / max(abs(deriv_val), abs(fin_diff_val)),
        )...

    end

    return all_vals, all_vals_fin_diff
end

@testset "B-Spline Controls (Hard-Coded degree 2)" begin
    tf = 10.0
    N_basis_functions = 10

    control = Degree2BSplineControl(N_basis_functions, tf)
    pcof = rand(MersenneTwister(0), control.N_coeff)

    h = eps()^(1/3)
    N_points = 1_000
    t_grid = LinRange(2h, tf-2h, N_points)


    @testset "Derivative Order 1" begin
        deriv_order=1
        test_control_derivative_accuracy(
            control, pcof, t_grid, deriv_order, h
        )
    end
end

@testset "PPPACK B-Spline Control" begin
    tf = 10.0
    N_basis_functions = 10

    control = Degree2BSplineControl(N_basis_functions, tf)
    pcof = rand(MersenneTwister(0), control.N_coeff)

    h = eps()^(1/3)
    N_points = 1_000
    t_grid = LinRange(2h, tf-2h, N_points)

    for degree in (2, 4, 8, 16)
        @testset "Degree $degree" begin
        control = FortranBSplineControl(degree, N_basis_functions, tf)
            for deriv_order in 1:degree-1
                @testset "Derivative Order $deriv_order" begin
                    test_control_derivative_accuracy(
                        control, pcof, t_grid, deriv_order, h
                    )
                end
            end
        end
    end
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

using QuantumGateDesign
using Polynomials
using Test

#===============================================================================
# Take a random polynomial, sample it and its derivatives at control points,
# and check that a HermiteControl using those values as control parameters
# produces the correct analytic result, as given by the Polynomials.jl package
# (which takes derivatives of polynomials analytically).
===============================================================================#

#=
# A hardcoded example
p(x) = x^7 + 1
D1p(x) = 7*x^6
D2p(x) = 7*6*x^5
D3p(x) = 7*6*5*x^4

tf = 1.0
N_control_points = 3
dt = tf / (N_control_points-1)
pcof_mat1 = zeros(4, N_control_points, 2)
for n in 1:N_control_points
    t = dt*(n-1)
    pcof_mat1[1,n,1] = p(t)
    pcof_mat1[2,n,1] = D1p(t)
    pcof_mat1[3,n,1] = D2p(t)
    pcof_mat1[4,n,1] = D3p(t)
end
=#

function test_hermite_poly_agreement(;N_derivatives=10, randseed=42, rtol=1e-14)
    Random.seed!(randseed)
    # Set up polynomials and their derivatives (technic)
    p_poly = Polynomial(2*N_derivatives)
    q_poly = Polynomial(2*N_derivatives)
    p_poly_derivatives = [derivative(p_poly, i) for i in 0:N_derivatives]
    q_poly_derivatives = [derivative(q_poly, i) for i in 0:N_derivatives]

    tf = 100.0
    N_control_points = 5
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

    # Test that HermiteControl agrees with analytic polynomial derivative values when constructed 
    ts_test = LinRange(0, tf, 1001)
    for derivative_order in 0:N_derivatives
        @testset "p | Derivative Order $derivative_order" begin
            for t in ts_test
                pval = eval_p_derivative(hermite_control, t, pcof_vec, derivative_order)
                pval_analytic = p_poly_derivatives[1+derivative_order](t)
                @test isapprox(pval, pval_analytic, rtol=rtol)
            end
        end
        @testset "q | Derivative Order $derivative_order" begin
            for t in ts_test
                qval = eval_p_derivative(hermite_control, t, pcof_vec, derivative_order)
                qval_analytic = q_poly_derivatives[1+derivative_order](t)
                @test isapprox(qval, qval_analytic, rtol=rtol)
            end
        end
    end
end



@testset "HermiteControl values and derivatives agree with analytic polynomial results" begin
    rtols = [1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
    for rtol in rtols
        @testset "rtol=$rtol" begin
            test_hermite_poly_agreement(rtol=rtol)
        end
    end
end

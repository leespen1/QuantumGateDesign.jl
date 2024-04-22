#===============================================================================
#  Compare function values of a HermiteControl with 2 derivatives specified per
#  control point with those of a hard-coded quntic hermite interpolation polynomial
#  using the same control points and derivatives.
===============================================================================#

using QuantumGateDesign
using Random
using Test

function quintic_hermite(x0, x1, f_x0, f_x1, x)
    # Compute the term which is 0th order in x
    order0 = f_x0[1]
    # Compute the term which is 1st order in x
    order1 = f_x0[2]*(x-x0)
    # Compute the term which is 2nd order in x ...
    order2 = 0.5*f_x0[3]*(x-x0)^2
    order3 = begin 
        term0 = f_x1[1] - f_x0[1]
        term1 = -f_x0[2]*(x1-x0)
        term2 = -0.5*f_x0[3]*(x1-x0)^2
        (term0 + term1 + term2)*(x-x0)^3 / (x1-x0)^3
    end
    order4 = begin
        term0 = 3*(f_x0[1] - f_x1[1])
        term1 = 2*(f_x0[2] + 0.5*f_x1[2])*(x1-x0)
        term2 = 0.5*f_x0[3]*(x1-x0)^2
        (term0 + term1 + term2) * (x-x0)^3 * (x-x1) / (x1-x0)^4
    end
    order5 = begin
        term0 = 6*(f_x1[1] - f_x0[1])
        term1 = -3*(f_x0[2] + f_x1[2])*(x1-x0)
        term2 = 0.5*(f_x1[3]-f_x0[3])*(x1-x0)^2
        (term0 + term1 + term2) * (x-x0)^3 * (x-x1)^2 / (x1-x0)^5
    end

    return order0 + order1 + order2 + order3 + order4 + order5 
end



function test_quintic_hermite(;randseed=42, rtol=1e-15, pcof_array=missing)

    N_control_points = 3
    tf = 100.0
    N_derivatives = 2

    # Default to random pcof_array, but can specify one if desired
    if ismissing(pcof_array)
        Random.seed!(randseed)
        pcof_array = rand(1+N_derivatives, N_control_points, 2)
    end

    @assert size(pcof_array) == (1+N_derivatives, N_control_points, 2)
    
    pcof_vec = reshape(pcof_array, :)

    quintic_hermite_control = QuantumGateDesign.HermiteControl(
        N_control_points, tf, N_derivatives, :Derivative
    )

    t_mid = tf/2

    region1_ts = LinRange(0, t_mid, 101)
    @testset "Region 1 p" begin
        for t in region1_ts
            pval = eval_p(quintic_hermite_control, t, pcof_vec)

            x0 = 0.0
            x1 = t_mid
            f_x0 = pcof_array[:,1,1]
            f_x1 = pcof_array[:,2,1]
            pval_analytic = quintic_hermite(x0, x1, f_x0, f_x1, t)
            @test isapprox(pval, pval_analytic, rtol=rtol)
        end
    end
    @testset "Region 1 q" begin
        for t in region1_ts
            qval = eval_q(quintic_hermite_control, t, pcof_vec)

            x0 = 0.0
            x1 = t_mid
            f_x0 = pcof_array[:,1,2]
            f_x1 = pcof_array[:,2,2]
            qval_analytic = quintic_hermite(x0, x1, f_x0, f_x1, t)
            @test isapprox(qval, qval_analytic, rtol=rtol)
        end
    end
    region2_ts = LinRange(t_mid, tf, 101)
    @testset "Region 2 p" begin
        for t in region2_ts
            pval = eval_p(quintic_hermite_control, t, pcof_vec)

            x0 = t_mid
            x1 = tf
            f_x0 = pcof_array[:,2,1]
            f_x1 = pcof_array[:,3,1]
            pval_analytic = quintic_hermite(x0, x1, f_x0, f_x1, t)
            @test isapprox(pval, pval_analytic, rtol=rtol)
        end
    end
    @testset "Region 2 q" begin
        for t in region2_ts
            qval = eval_q(quintic_hermite_control, t, pcof_vec)

            x0 = t_mid
            x1 = tf
            f_x0 = pcof_array[:,2,2]
            f_x1 = pcof_array[:,3,2]
            qval_analytic = quintic_hermite(x0, x1, f_x0, f_x1, t)
            @test isapprox(qval, qval_analytic, rtol=rtol)
        end
    end
end



@testset "HermiteControl Agrees with Quintic Hermite Interpolation" begin

    N_control_points = 3
    N_derivatives = 2

    # pcof = [1, 2, 3, ...]
    pcof_vec = collect(1:N_control_points*(1+N_derivatives)*2)
    pcof_array = reshape(pcof_vec, 1+N_derivatives, N_control_points, 2)

    #rtols = [1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
    rtols = [1e-10, 1e-11, 1e-12]
    for rtol in rtols
        @testset "pcof = [1, 2, 3, ...], rtol=$rtol" begin
            test_quintic_hermite(pcof_array=pcof_array, rtol=rtol)
        end
    end

    for rtol in rtols
        @testset "random pcof, rtol=$rtol" begin
            test_quintic_hermite(rtol=rtol)
        end
    end

end

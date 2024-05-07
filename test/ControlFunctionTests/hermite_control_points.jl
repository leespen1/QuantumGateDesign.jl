#===============================================================================
# Test that when a HermiteControl is evaluated at a control point, the returned
# value is the corresponding control coefficient.
===============================================================================#

using QuantumGateDesign: HermiteControl
using Test: @test, @testset
using LinearAlgebra: norm
using PrettyTables


function test_hermite_control_points(control::HermiteControl,
        pcof::AbstractVector{Float64}, rtol=1e-12)

    q_offset = div(control.N_coeff, 2)
    inf_norm(x) = norm(x, Inf)
    #inf_norm(x) = norm(x)

    evaluations_p = zeros(q_offset)
    evaluations_q = zeros(q_offset)

    @assert control.scaling_type == :Derivative
    @testset "Checking Hermite Interpolant at control points" begin
        for derivative_order in 0:control.N_derivatives
            for i in 0:control.N_points-1
                t = control.dt*i
                p_val = eval_p_derivative(control, t, pcof, derivative_order)
                q_val = eval_q_derivative(control, t, pcof, derivative_order)

                evaluations_p[1 + i*(1+control.N_derivatives) + derivative_order] = p_val
                evaluations_q[1 + i*(1+control.N_derivatives) + derivative_order] = q_val
            end
        end

        evaluations_p_array = reshape(evaluations_p, 1+control.N_derivatives, control.N_points)
        evaluations_q_array = reshape(evaluations_q, 1+control.N_derivatives, control.N_points)
        evaluations_array = cat(evaluations_p_array, evaluations_q_array, dims=3)
        pcof_array = reshape(pcof, 1+control.N_derivatives, control.N_points, 2)



        errors = abs.(evaluations_array - pcof_array)
        rel_errors = abs.(errors ./ pcof_array)

        data = ["Order" "Avg Error" "Max Error"]
        for derivative_order in 0:control.N_derivatives
            @testset "Derivative Order $derivative_order" begin
                @test isapprox(evaluations_array[1+derivative_order,:,:], pcof_array[1+derivative_order,:,:], rtol=rtol, norm=inf_norm)
                @test isapprox(evaluations_array[1+derivative_order,:,:], pcof_array[1+derivative_order,:,:], rtol=rtol, norm=inf_norm)
            end

            this_order_rel_errors = rel_errors[1+derivative_order, :, :]
            avg_err = sum(this_order_rel_errors)  / length(this_order_rel_errors)
            max_err = norm(this_order_rel_errors, Inf)
            data = vcat(data, [derivative_order avg_err max_err])
        end

        results_table = pretty_table(
            data[2:end,:];
            header=data[1,:],
            header_crayon = crayon"yellow bold",
        )
    end # @testset
end
@testset "Testing HermiteControl Control Point Evaluation" begin
    @info "Testing HermiteControl Control Point Evaluation"
    tf = 5.0
    N_points = 3
    N_derivatives = 4
    scaling_type = :Derivative

    hermite_control = HermiteControl(N_points, tf, N_derivatives, scaling_type)

    @testset "pcof = ones(N_coeff)" begin
        println("-"^40, "\nTest: pcof = ones(N_coeff)\n", "-"^40)
        pcof = ones(hermite_control.N_coeff)
        test_hermite_control_points(hermite_control, pcof)
    end
    @testset "pcof = rand(N_coeff)" begin
        println("-"^40, "\nTest: pcof = rand(N_coeff)\n", "-"^40)
        pcof = rand(hermite_control.N_coeff)
        test_hermite_control_points(hermite_control, pcof)
    end
end

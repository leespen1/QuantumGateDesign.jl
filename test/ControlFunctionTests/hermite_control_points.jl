using QuantumGateDesign: HermiteControl
using Test: @test, @testset

function test_hermite_interpolant(control::HermiteControl, pcof::AbstractVector{Float64})
    q_offset = div(control.N_coeff, 2)
    rtol = 1e-14

    @test control.scaling_type == :Derivative
    @testset "Checking Hermite Interpolant at control points" begin
    for derivative_order in 0:control.N_derivatives
        @testset "Order $derivative_order values" begin
        for i in 0:control.N_points-1
            t = control.dt*i
            p_val = eval_p_derivative(control, t, pcof, derivative_order)
            q_val = eval_q_derivative(control, t, pcof, derivative_order)

            @test isapprox(p_val, pcof[1 + i*(1+control.N_derivatives) + derivative_order], rtol=rtol)
            @test isapprox(q_val, pcof[q_offset + 1 + i*(1+control.N_derivatives) + derivative_order], rtol=rtol)
        end
        end  # @testset
    end
    end # @testset
end

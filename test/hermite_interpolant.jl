using HermiteOptimalControl: HermiteControl
using Test: @test, @testset

function test_hermite_interpolant(control::HermiteControl, pcof::AbstractVector{Float64})
    q_offset = control.N_points*(1+control.N_derivatives)
    rtol = 1e-14

    @testset "Checking Hermite Interpolant at pcof points" begin
    for i in 0:control.N_points-1
        t = control.dt*i
        for derivative_order in 0:control.N_derivatives
            p_val = eval_p_derivative(control, t, pcof, derivative_order)
            q_val = eval_p_derivative(control, t, pcof, derivative_order)

            @test isapprox(p_val, pcof[1 + i*(1+control.N_derivatives) + derivative_order], rtol=rtol)
            @test isapprox(q_val, pcof[q_offset + 1 + i*(1+control.N_derivatives) + derivative_order], rtol=rtol)
        end
    end
end
end

using QuantumGateDesign
import QuantumGateDesign as QGD
using Test

function test_control_gradient(control::AbstractControl, upto_order=0)
    pcof = rand(control.N_coeff)
    t_start = upto_order
    ts = LinRange(0, control.tf, 1001)
    grad_analytic = zeros(control.N_coeff)
    for order in 0:upto_order
        @testset "Gradient of p/q derivative order $order" begin
            for t in ts
                QGD.eval_grad_p_derivative!(grad_analytic, control, t, pcof, order)
                grad_fin_diff = QGD.eval_grad_p_derivative_fin_diff(control, t, pcof, order)

                #println(all(isapprox.(grad_analytic, grad_fin_diff, rtol=1e-10)) )
                @test isapprox(grad_analytic, grad_fin_diff, rtol=1e-9)
                #println(norm(grad_analytic - grad_fin_diff))
            end
        end #@testset
    end
    return nothing
end

function test_control_derivatives(control::AbstractControl, upto_order=1, pcof=missing)
    if ismissing(pcof)
        pcof = rand(control.N_coeff)
    end

    # Add cushion to start and end ts so we have enough space to do centered difference methods
    max_cd_stepsize = 1e-15 ^ (1/(2*upto_order+1))
    println(max_cd_stepsize)
    min_t = 0.0 + max_cd_stepsize*upto_order
    max_t = control.tf - max_cd_stepsize*upto_order
    ts = LinRange(min_t, max_t, 1000)
    grad_analytic = zeros(control.N_coeff)

    for order in 1:upto_order
        @testset "Value p/q derivative order $order" begin
        # Once we believe eval_p_derivative gives analytically correct results,
        # fine to use it central difference to compute first derivative of that
        p(x) = eval_p_derivative(control, x, pcof, order-1)
        q(x) = eval_q_derivative(control, x, pcof, order-1)
            for t in ts

                function_val = eval_p_derivative(control, t, pcof, order-1)
                dval_analytic = eval_p_derivative(control, t, pcof, order)
                dval_fin_diff = central_difference(p, t, 1)

                atol = 10*abs(1e-15*function_val)^(2/3)
                @test isapprox(dval_analytic, dval_fin_diff, atol=atol)
                # Expected error is ~1e-10

                function_val = eval_q_derivative(control, t, pcof, order-1)
                dval_analytic = eval_q_derivative(control, t, pcof, order)
                dval_fin_diff = central_difference(q, t, 1)

                atol = 10*abs(1e-15*function_val)^(2/3)
                @test isapprox(dval_analytic, dval_fin_diff, atol=atol)

                # Should add a relative tolerance check here as well
            end
        end #@testset
    end
    return nothing
end



function central_difference(f, x, n, h=missing)
    if ismissing(h)
        # Balance impact of roundoff error and taylor expansion error
        h = 1e-15 ^ (1/(2n+1)) # Should maybe change to minimize relative error, not absolute
        # Expected (abs) error will be 10.0 ^ (-15*(n+1) / (2n+1))
    end

    derivative_val = 0.0
    for i in 0:n
        derivative_val += (-1)^i * binomial(n,i) * f(x + (0.5*n-i)*h)
    end
    derivative_val /= h^n
    return derivative_val
end





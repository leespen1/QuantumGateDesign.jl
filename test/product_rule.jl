#using QuantumGateDesign
using Test: @testset, @test

"""
Compute derivatives of cos(ωt)
"""
function coswt_derivatives(w, t, N_derivatives)
    ret = zeros(N_derivatives+1)
    for derivative_order in 0:N_derivatives
        if iseven(derivative_order)
            ret[1+derivative_order] = cos(w*t)
        else
            ret[1+derivative_order] = sin(w*t)
        end
        ret[1+derivative_order] *= w^derivative_order
        if (derivative_order % 4) in (1,2)
            ret[1+derivative_order] *= -1
        end
        ret[1+derivative_order] /= factorial(derivative_order)
    end
    return ret
end

"""
Compute derivatives of sin(ωt)
"""
function sinwt_derivatives(w, t, N_derivatives)
    ret = zeros(N_derivatives+1)
    for derivative_order in 0:N_derivatives
        if iseven(derivative_order)
            ret[1+derivative_order] = sin(w*t)
        else
            ret[1+derivative_order] = cos(w*t)
        end
        ret[1+derivative_order] *= w^derivative_order
        if (derivative_order % 4) in (2,3)
            ret[1+derivative_order] *= -1
        end
        ret[1+derivative_order] /= factorial(derivative_order)
    end
    return ret
end

@testset "Product Rule Computation for cos(wt)*sin(wt)" begin

w = 2
t = pi/6
N_derivatives = 3

x_derivatives = coswt_derivatives(w, t, N_derivatives)
y_derivatives = sinwt_derivatives(w, t, N_derivatives)

prod_derivatives = zeros(1+N_derivatives)
QuantumGateDesign.product_rule!(x_derivatives, y_derivatives, prod_derivatives)

coswt = cos(w*t)
sinwt = sin(w*t)

d0 = (coswt*sinwt) / factorial(0)
d1 = w*( -(sinwt^2) + (coswt^2) ) / factorial(1)
d2 = -4*(w^2)*coswt*sinwt / factorial(2)
d3 = -4*(w^3) * (-sinwt^2 + coswt^2) / factorial(3)

analytic_derivatives = [d0, d1, d2, d3]
# Check that each computed derivatives is equal to the analytic result (to machine precision)
for i in 1:length(analytic_derivatives)
    @test isapprox(prod_derivatives[i], analytic_derivatives[i], rtol=1e-15)
end

end #testset

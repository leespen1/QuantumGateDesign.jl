using QuantumGateDesign
using BenchmarkTools
using BSplines
using BasicBSpline

degree = 2
N_knots = 10
tf = 1.0

package_bspline = GeneralBSplineControl(degree, N_knots, tf)
hardcoded_bspline = MySplineControl(tf, N_knots)

k = KnotVector(knots(package_bspline.bspline_basis))
P = BSplineSpace{degree}(k)
P1 = BSplineDerivativeSpace{1}(P)
P2 = BSplineDerivativeSpace{2}(P)
P3 = BSplineDerivativeSpace{3}(P)

t = 0.2


pcof1 = ones(package_bspline.N_coeff)
pcof2 = ones(hardcoded_bspline.N_coeff)
t_range = LinRange(0, tf, 1001)



function sum_control(control, pcof, t_range)
    result = 0.0
    for t in t_range
        for derivative_order in (0,1,2,3)
            result += eval_p_derivative(control, t, pcof, derivative_order)
            result += eval_q_derivative(control, t, pcof, derivative_order)
        end
    end
    return result
end

function sum_BasicBS(P,P1,P2,P3,t_range)
    sm = 0.0
    for t in t_range
        i = BasicBSpline.intervalindex(P,t)
        sm += sum(bsplinebasisall(P,i,t))
        sm += sum(bsplinebasisall(P1,i,t))
        sm += sum(bsplinebasisall(P2,i,t))
        sm += sum(bsplinebasisall(P3,i,t))
    end
    return sm
end

@btime sum_control($package_bspline, $pcof1, $t_range)
println("Hardcoded")
@btime sum_control($hardcoded_bspline, $pcof1, $t_range)
println("BasicBSplines")
@btime sum_BasicBS($P,$P1,$P2,$P3,$t_range)

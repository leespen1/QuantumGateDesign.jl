using QuantumGateDesign
using BenchmarkTools
degree = 2
N_knots = 10
tf = 1.0

package_bspline = GeneralBSplineControl(degree, N_knots, tf)
hardcoded_bspline = MySplineControl(tf, N_knots)

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

@btime sum_control(package_bspline, pcof1, t_range)
@btime sum_control(hardcoded_bspline, pcof1, t_range)

using QuantumGateDesign
using BenchmarkTools
using BSplines
using BasicBSpline

degree = 2
N_knots = 10
tf = 1.0

package_bspline = GeneralBSplineControl(degree, N_knots, tf)
hardcoded_bspline = MySplineControl(tf, N_knots)
package_basic_bspline = QuantumGateDesign.BasicBSplineControl(degree, tf, N_knots)

k = KnotVector(knots(package_bspline.bspline_basis))
P = BSplineSpace{degree}(k)
P1 = BSplineDerivativeSpace{1}(P)
P2 = BSplineDerivativeSpace{2}(P)
P3 = BSplineDerivativeSpace{3}(P)

t = 0.2


pcof1 = ones(package_bspline.N_coeff)
pcof2 = ones(hardcoded_bspline.N_coeff)
pcof3 = ones(package_basic_bspline.N_coeff)
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

function sum_BasicBS2(P,P1,P2,P3,t_range)
    sm = 0.0
    for t in t_range
        i = BasicBSpline.intervalindex(P,t)
        vals = bsplinebasisall(P,i,t)
        for val in vals
            sm += val
        end
        sm += sum(bsplinebasisall(P1,i,t))
        sm += sum(bsplinebasisall(P2,i,t))
        sm += sum(bsplinebasisall(P3,i,t))
    end
    return sm
end

#=
#println("Package BSplines")
#@btime sum_control($package_bspline, $pcof1, $t_range)
println("Hardcoded")
@btime sum_control($hardcoded_bspline, $pcof2, $t_range)
println("BasicBSplines")
@btime sum_BasicBS($P,$P1,$P2,$P3,$t_range)
println("BasicBSplines2")
@btime sum_BasicBS2($P,$P1,$P2,$P3,$t_range)
println("QuantumGateDesign BasicBSplines")
@btime sum_control($package_basic_bspline, $pcof3, $t_range)
=#

println("Hardcoded")
sum_control(hardcoded_bspline, pcof2, t_range)
println("BasicBSplines")
sum_BasicBS(P,P1,P2,P3,t_range)
println("BasicBSplines2")
sum_BasicBS2(P,P1,P2,P3,t_range)
println("QuantumGateDesign BasicBSplines")
sum_control(package_basic_bspline, pcof3, t_range)

# There are allocations caused by the compiler not being smart enough to
# realize that although whether P1, P2, P3, ... is used, the SVector is the same
# type/size. So it has to do weird workarounds and it gets just as bad as in the
# regular BSplines.jl package. I think the only way around this is to dispatch
# over a Derivative{N} stype like in BSplines.jl. This will also require dispatching
# over an Order{N} type for eval_forward and discrete_adjoint, etc. That means
# a lot of boilerplate rewriting. First, see if doing this just for the control
# fixes the speed issues.

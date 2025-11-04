using GeometricIntegrators
include("high_order_composition.jl")


function dx!(v, t, x, params)
    v .= x
    return nothing
end

x0 = [1.0]

tspan = (0.0, 10.0)
tstep = 1.0

ode = ODEProblem(dx!, tspan, tstep, x0)

comp4 = Composition(ImplicitMidpoint(), SuzukiFractal())
comp6 = Composition(ImplicitMidpoint(), Order6Stages9())
comp8 = Composition(ImplicitMidpoint(), Order8Stages17())
comp10 = Composition(ImplicitMidpoint(), Order10Stages35())

int2 = GeometricIntegrator(ode, ImplicitMidpoint())
int4 = GeometricIntegrator(ode, comp4)
int6 = GeometricIntegrator(ode, comp6)
int8 = GeometricIntegrator(ode, comp8)
int10 = GeometricIntegrator(ode, comp10)

sol2 = integrate(int2)
sol4 = integrate(int4)
sol6 = integrate(int6)
sol8 = integrate(int8)
sol10 = integrate(int10)

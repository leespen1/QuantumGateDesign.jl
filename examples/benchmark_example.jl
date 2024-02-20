using HermiteOptimalControl
using OrdinaryDiffEq
using ProfileView
using BenchmarkTools

include("SWAP_example.jl")

prob, control, pcof, target, pcof_u, pcof_l= main(d=1, N_guard=1, tf=1.0, D1=10)
control = HermiteOptimalControl.SinCosControl(prob.tf)
pcof = [1.0, 1.0]

@btime history = eval_forward(prob, control, pcof, order=2);

ode_prob = HermiteOptimalControl.construct_ODEProb(prob, control, pcof)
@btime ode_ret = solve(ode_prob, Tsit5(), reltol=1e-14, abstol=1e-14);

# For 4th order, 400 timesteps gets 1e-12 norm agreement, 1e-13 entry agreement.
# Took like 30 ms for my code, 5 ms for theirs. That's actually not so bad. Also
# not that theirs is 5-th order (adaptive Tsit5) and took 247 timesteps. 
#
# I think my method is actually competitive. If I can get a 50% runtime
# reduction, which I think is plausible (I estimated 30% of runtime was
# allocation, which I should be able to nearly eliminate), then it would be
# 15ms compared to 5ms. I think that's competitive, epsecially since we can do
# discrete adjoint.
#
# In fact, when setting nsteps to 250, 2nd order is 8ms and 4th order is 21 ms.
# I think it's viable that I can get the cost of taking a single step to be nearly equal.

# Moreover, when using RK4 it still took like 250 timesteps, but the runtime
# was 45 ms.

@profview ode_ret = solve(ode_prob, Tsit5(), reltol=1e-14, abstol=1e-14);
@profview history = eval_forward(prob, control, pcof, order=2);
@profview history = eval_forward(prob, control, pcof, order=2);

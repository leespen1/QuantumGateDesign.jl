using ForwardDiff
using SciMLSensitivity
using DifferentialEquations
using Zygote
using LinearAlgebra
using Plots
include("auto_diff_example.jl")
#=
# Keep everything real-valued
# Assume everything is 2-level
=#
"""
Parameters needed to evolve a schrodinger problem.
u0 must be a matrix, even if an Nx1 Matrix, to keep syntax correct between
single vector and entire basis.
"""
struct DAProb
    tspan::Tuple{Float64, Float64}
    u0::Matrix{Float64} # Even for vector u0, represent as a 4x1 matrix!
    a::Float64
    M::Function
    dMda::Function
end

"""
Alternative constructor, to allow giving u0 as a vector.
Also copy u0.
"""
function DAProb(tspan::Tuple{Float64, Float64}, u0::Vector{Float64},
        a::Float64, M::Function, dMda::Function)
    return DAProb(tspan, reshape(copy(u0), length(u0), 1), a, M, dMda)
end

function complex_to_real(A)
    return vcat(real(A), imag(A))
end

"""
Objective Type:
- 0 for infidelity
- 1 for |u_T|^2
- 2 for "tracking"
"""
function disc_adj(prob::DAProb, target::Matrix{Float64}, newparam::Float64; N::Int64=100, obj_type::Int64=0)
    t0 = prob.tspan[1]
    tf = prob.tspan[2]
    a = newparam
    M    = prob.M
    dMda = prob.dMda

    E = 2
    dt = (tf-t0)/N # Timestep size

    Qs = zeros(4, size(prob.u0, 2), N+1)
    Qs[:,:,1+0] = copy(prob.u0)

    # Forward eval, saving all points
    for n in 0:N-1
        tn = t0 + n*dt
        tnp1 = t0 + (n+1)*dt
        Qs[:,:,1+n+1] = (I - 0.5*dt*M(tnp1, a)) \ ((I + 0.5*dt*M(tn, a))*Qs[:,:,1+n])
    end

    # Get Λ_N using terminal condition
    R = copy(target)
    T = vcat(target[3:4,:], -target[1:2,:])

    lambdas = zeros(4, size(prob.u0, 2), N+1)
    if obj_type == 2
        lambdas[:,:,1+N] = (I - 0.5*dt*M(tf, a)') \ (R-Qs[:,:,1+N])
    elseif obj_type == 1
        lambdas[:,:,1+N] = (I - 0.5*dt*M(tf, a)') \ (2*Qs[:,:,1+N])
    else
        lambdas[:,:,1+N] = (I - 0.5*dt*M(tf, a)') \ ((2/(E^2))*(tr(Qs[:,:,1+N]'*R)*R + tr(Qs[:,:,1+N]'*T)*T)) # No guard-level term yet
    end

    # Backwards evolution of Λ
    for n in N-1:-1:1
        tn = t0 + n*dt
        lambdas[:,:,1+n] = (I - 0.5*dt*M(tn, a)') \ ((I + 0.5*dt*M(tn, a)')*lambdas[:,:,1+n+1]) # No guard-level forcing term yet
    end

    # Compute gradient
    grad_disc_adj = 0.0
    for n in 0:N-1
        tn = t0 + n*dt
        tnp1 = t0 + (n+1)*dt
        grad_disc_adj += tr((dMda(tn, a)*Qs[:,:,1+n] + dMda(tnp1, a)*Qs[:,:,1+n+1])'*lambdas[:,:,1+n+1])
    end
    grad_disc_adj *= -0.5*dt

    #= Computing value of lagrangian
    lagrangian = 1 - 0.25*((tr(Qs[:,:,1+N]'*R))^2 + (tr(Qs[:,:,1+N]'*T))^2) # infidelity
    for n in 0:N-1 # lambda sum contributions (each one should be zero if ODE is satisfied)
        tn = t0 + n*dt
        tnp1 = t0 + (n+1)*dt
        Qn = Qs[:,:,1+n]
        Qnp1 = Qs[:,:,1+n+1]
        lambda_np1 = lambdas[:,:,1+n+1]
        lagrangian += tr(( (I-0.5*dt*M(tnp1, a))*Qnp1 - (I+0.5*dt*M(tn,a))*Qn)'*(lambda_np1))
    end
    =#

    return grad_disc_adj
end


function eval_forward(prob::DAProb, newparam::Float64; N=100)
    t0 = prob.tspan[1]
    tf = prob.tspan[2]
    a = newparam
    M = prob.M
    dt = (tf-t0)/N # Timestep size

    Q = copy(prob.u0) 
    for n in 0:N-1
        tn = t0 + n*dt
        tnp1 = t0 + (n+1)*dt
        Q = (I - 0.5*dt*M(tnp1, a)) \ ((I + 0.5*dt*M(tn, a))*Q)
        #println("$tn, $tnp1")
    end

    return Q
end


function infidelity(Q, target; obj_type=0)
    R = copy(target)
    T = vcat(target[3:4,:], -target[1:2,:])
    E = 2
    if obj_type == 2
        return 0.5*norm(R-Q)^2
    elseif obj_type == 1
        return Q'*Q
    end
    return 1 - (1/E^2)*(tr(Q'*R)^2 + tr(Q'*T)^2)
end


function finite_diff_gradient(prob::DAProb, target::Matrix{Float64}, newparam::Float64; N=100, da=1e-5, obj_type=0, fT=1.0)
    a = newparam 
    Q_r = eval_forward(prob, a+da, N=N)
    Q_l = eval_forward(prob, a-da, N=N)
    infidelity_r = infidelity(Q_r, target; obj_type=obj_type)
    infidelity_l = infidelity(Q_l, target; obj_type=obj_type)
    fin_dif_grad = (infidelity_r - infidelity_l)/(2*da)
    return fin_dif_grad
end

function main(;Nsamples=50, Ntimesteps=100)
    Q0 = complex_to_real([1.0+1.0im, 1.0+1.0im])
    Q0 = Q0 / norm(Q0)
    tspan = (0.0, 10.0)
    # Parameter
    p = 1.0

    ## Get target/solution
    # Set up ODE Problem (using DifferentialEquations package)
    ODE_prob = ODEProblem{true, SciMLBase.FullSpecialize}(schrodinger!, Q0, tspan, p) # Need this option for debug to work
    data_solution = solve(ODE_prob, saveat=1, abstol=1e-10, reltol=1e-10)
    # Convert solution to array
    data = Array(data_solution)
    Q_target = reshape(data[:,end], (4,1))
    #Q_target = reshape([1.0, 1.0, 1.0, 1.0], (4,1))

    S(t, a) = [0.0 0.0;
               0.0 0.0]
    dSda(t, a) = [0.0 0.0;
                  0.0 0.0]
    K(t, a) = [0.0 a*cos(t);
               a*cos(t) 1.0]
    dKda(t, a) = [0.0 cos(t);
                  cos(t) 0.0]
    M(t, a) = [S(t, a) -K(t,a);
               K(t, a) S(t,a)]
    dMda(t, a) = [dSda(t, a) -dKda(t,a);
                  dKda(t, a) dSda(t,a)]
    DA_prob = DAProb(tspan, Q0, p, M, dMda)

    dSdt(t,a) = [0.0 0.0;
                 0.0 0.0]
    dKdt(t,a) = [0.0 -a*sin(t);
                 -a*sin(t) 0.0]
    dMdt(t,a) = [dSdt(t,a) -dKdt(t,a);
                 dKdt(t,a) dSdt(t,a)]
    w0(t) = missing # Need to get the hermite interpolant weights
    w1(t) = missing # Need to get the hermite interpolant weights
    M_tilde(t,a) = w1(t)*dMdt(t,a) + (w0(t)*I + w1(t)*M(t,a))*M(t,a)

    a = LinRange(0.5, 1.5, Nsamples)
    losses = zeros(Nsamples)
    lagrangians = zeros(Nsamples)
    grads_fwd_dif = zeros(Nsamples)
    grads_garg = zeros(Nsamples)
    grads_garg_trap = zeros(Nsamples)
    grads_dis_adj = zeros(Nsamples)
    grads_fin_dif = zeros(Nsamples)
    for i = 1:Nsamples
        losses[i] = loss_func(ODE_prob, Q_target, a[i])
        grads_fwd_dif[i] = ForwardDiff.derivative(p -> loss_func(ODE_prob, Q_target, p), a[i])
        grads_garg[i] = grad_gargamel(ODE_prob, Q_target, a[i])
        grads_garg_trap[i] = grad_gargamel_trap(ODE_prob, Q_target, a[i], N=Ntimesteps)
        grads_dis_adj[i] = disc_adj(DA_prob, Q_target, a[i], N=Ntimesteps)
        grads_fin_dif[i] = finite_diff_gradient(DA_prob, Q_target, a[i], N=Ntimesteps)
    end
    pl = plot()
    plot!(pl, a, losses, label="Infidelity")
    plot!(pl, a, grads_fwd_dif, label="Gradient (ForwardDiff)")
    plot!(pl, a, grads_garg, label="Gradient (Gargamel)", linestyle=:dash)
    plot!(pl, a, grads_garg_trap, label="Gradient (Gargamel Trap)", linestyle=:dashdot)
    plot!(pl, a, grads_dis_adj, label="Gradient (Discrete Adjoint)")
    plot!(pl, a, grads_fin_dif, label="Gradient (Finite Diff)", linestyle=:dash)
    plot!(xlabel="α")
    plot!(title="Target using α=$p, T=$(tspan[end]), N=$(Ntimesteps)")

    return pl
end

pl = main()

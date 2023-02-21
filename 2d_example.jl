using LinearAlgebra
using Plots
using LaTeXStrings
using DifferentialEquations, Optimization, OptimizationPolyalgorithms, SciMLSensitivity
using ForwardDiff
using Zygote

E = 2 # Number of essential energy levels
S(t, a) = [0.0 0.0; 0.0 0.0]
dSda(t, a) = [0.0 0.0; 0.0 0.0]
K(t, a) = [0.0 a*cos(t); a*cos(t) 1.0]
dKda(t, a) = [0.0 cos(t); cos(t) 0.0]
M(t, a) = [S(t, a) -K(t,a)
           K(t, a) S(t,a)]
dMda(t, a) = [dSda(t, a) -dKda(t,a)
              dKda(t, a) dSda(t,a)]

function infidelity(Q, target_complex; obj_type=0)
    R = vcat(real(target_complex), imag(target_complex))
    T = vcat(imag(target_complex), -real(target_complex))
    if obj_type == 2
        return 0.5*norm(R-Q)^2
    elseif obj_type == 1
        return Q'*Q
    end
    return 1 - (1/E^2)*(tr(Q'*R)^2 + tr(Q'*T)^2)
end

"""
Overlap Function / Complex Inner Product
"""
function overlap(A, B)
    B_vu = vcat(B[1+end÷2:end,:], -B[1:end÷2,:])
    return tr(A'*B) + im*tr(A'*B_vu)
end

function complex_to_real(A)
    return vcat(real(A), imag(A))
end

function real_to_complex(A)
    return A[1:end÷2,:] + im*A[1+end÷2:end,:]
end

function eval_forward(a, Q0_complex; N=100, fT=1.0)
    dt = fT/N # Timestep size
    Q = vcat(real(Q0_complex), imag(Q0_complex)) 

    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt
        Q = (I - 0.5*dt*M(tnp1, a)) \ ((I + 0.5*dt*M(tn, a))*Q)
        #println("$tn, $tnp1")
    end

    return Q
end

"""
Objective Type:
- 0 for infidelity
- 1 for |u_T|^2
- 2 for "tracking"
"""
function disc_adj(a, Q0_complex, target_complex; N=100, fT=1.0, obj_type=0)
    dt = fT/N # Timestep size

    Qs = zeros(4,size(Q0_complex, 2),N+1)
    lambdas = zeros(4,size(Q0_complex, 2),N+1)

    # Forward eval saving all points
    Qs[:,:,1+0] = vcat(real(Q0_complex), imag(Q0_complex)) 
    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt
        Qs[:,:,1+n+1] = (I - 0.5*dt*M(tnp1, a)) \ ((I + 0.5*dt*M(tn, a))*Qs[:,:,1+n])
        #println("$tn, $tnp1")
    end

    # Get Λ_N using terminal condition
    R = vcat(real(target_complex), imag(target_complex))
    T = vcat(imag(target_complex), -real(target_complex))

    if obj_type == 2
        lambdas[:,:,1+N] = (I - 0.5*dt*M(fT, a)') \ (R-Qs[:,:,1+N])
    elseif obj_type == 1
        lambdas[:,:,1+N] = (I - 0.5*dt*M(fT, a)') \ (2*Qs[:,:,1+N])
    else
        lambdas[:,:,1+N] = (I - 0.5*dt*M(fT, a)') \ ((2/(E^2))*(tr(Qs[:,:,1+N]'*R)*R + tr(Qs[:,:,1+N]'*T)*T)) # guard-level term is 0 in 1D
    end

    # Backwards evolution of Λ
    for n in N-1:-1:1
        tn = n*dt
        lambdas[:,:,1+n] = (I - 0.5*dt*M(tn, a)') \ ((I + 0.5*dt*M(tn, a)')*lambdas[:,:,1+n+1]) # No guard-level forcing term because we are in 1D
    end

    # Compute gradient
    grad_disc_adj = 0.0
    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt
        grad_disc_adj += tr((dMda(tn, a)*Qs[:,:,1+n] + dMda(tnp1, a)*Qs[:,:,1+n+1])'*lambdas[:,:,1+n+1])
    end
    grad_disc_adj *= -0.5*dt

    #= Computing value of lagrangian
    lagrangian = 1 - 0.25*((tr(Qs[:,:,1+N]'*R))^2 + (tr(Qs[:,:,1+N]'*T))^2) # infidelity
    for n in 0:N-1 # lambda sum contributions (each one should be zero if ODE is satisfied)
        tn = n*dt
        tnp1 = (n+1)*dt
        Qn = Qs[:,:,1+n]
        Qnp1 = Qs[:,:,1+n+1]
        lambda_np1 = lambdas[:,:,1+n+1]
        lagrangian += tr(( (I-0.5*dt*M(tnp1, a))*Qnp1 - (I+0.5*dt*M(tn,a))*Qn)'*(lambda_np1))
    end
    =#

    return grad_disc_adj
end

function grad_derivative_method(a, Q0_complex; N=100, fT=1.0, obj_type=0)
    target_real = eval_forward(1.0, Q0_complex, 100)
    dt = fT/N # Timestep size

    # Forward eval saving all points
    Q_save = zeros(4,2,N+1)
    Q_save[:,:,1+0] = vcat(real(Q0_complex), imag(Q0_complex)) 
    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt
        Q_save[:,:,1+n+1] = (I - 0.5*dt*M(tnp1, a)) \ ((I + 0.5*dt*M(tn, a))*Q_save[:,:,1+n])
    end
    
    # Derivative of each Qn with respect to a
    dQda_save = zeros(4,2,N+1)
    dQda_save[:,:,1+0] = vcat(real(Q0_complex), imag(Q0_complex)) 
    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt
        dQda_save[:,:,1+n+1] = (I - 0.5*dt*M(tnp1, a)) \ (
            ((I + 0.5*dt*M(tn, a))*dQda_save[:,:,1+n])
            + 0.5*dt*(dMda(tn, a)*Q_save[:,:,1+n] + dMda(tnp1, a)*Q_save[:,:,1+n+1]) # Forcing terms
        )
    end

    S = overlap(target_real, Q_save[:,:,1+N])
    dSda = overlap(target_real, dQda_save[:,:,1+N])
    gradient = -2*real(S'*dSda)/E^2
    return gradient
    #return Qs, lambdas, grad_disc_adj
end


# Target gate




# Finite differnce gradient calculation
#=
da = 1e-10 # differential control parameter for finite difference gradient
Q_r = eval_forward(a+da, Q0_complex)
Q_l = eval_forward(a-da, Q0_complex)
infidelity_r = infidelity(Q_r, target_complex)
infidelity_l = infidelity(Q_l, target_complex)
fin_dif_grad = (infidelity_r - infidelity_l)/(2*da)

println("α = $a\nFinite Difference Gradient = $fin_dif_grad\nDiscrete Adjoint Gradient = $grad_disc_adj\nQs:")
display(Qs)
=#

function finite_diff_gradient(a, Q0_complex, target_complex; N=100, da=1e-5, obj_type=0, fT=1.0)
    Q_r = eval_forward(a+da, Q0_complex, N=N, fT=fT)
    Q_l = eval_forward(a-da, Q0_complex, N=N, fT=fT)
    infidelity_r = infidelity(Q_r, target_complex; obj_type=obj_type)
    infidelity_l = infidelity(Q_l, target_complex; obj_type=obj_type)
    fin_dif_grad = (infidelity_r - infidelity_l)/(2*da)
    return fin_dif_grad
end

"""
Plot Gradients using Finite Differences and Discrete Adjoint
"""
function graph1(;N=100, fT=1.0, return_data=false, obj_type=0)
    a = 1.0
    Q0_complex = [1.0+1.0im, 1.0+1.0im]
    Q0_complex = Q0_complex / norm(Q0_complex)
    #Q0_complex = [1.0 0.0; 0.0 1.0]
    target = eval_forward(a, Q0_complex, N=N)
    target_complex = target[1:end÷2,:] + (im .* target[1+end÷2:end,:])
    Nsamples = 101
    grads_fin_dif = zeros(Nsamples)
    grads_dis_adj = zeros(Nsamples)
    grads_gargamel = zeros(Nsamples)
    grads_fwd_dif = zeros(Nsamples)
    i = 1
    as = LinRange(-2,2,Nsamples)
    for i in 1:Nsamples
        #Q_r = eval_forward(as[i], Q0_complex, N)
        # grads_fin_dif[i] = infidelity(Q_r, target_complex; obj_type=obj_type)
        grads_fin_dif[i] = finite_diff_gradient(as[i], Q0_complex, target_complex, N=N, obj_type=obj_type)
        grads_dis_adj[i] = disc_adj(as[i], Q0_complex, target_complex, N=N, obj_type=obj_type)
        grads_gargamel[i] = grad_gargamel(as[i], Q0_complex)
        grads_fwd_dif[i] = grad_forward_diff(as[i], Q0_complex)
    end
    if return_data
        return as, grads_fin_dif, grads_dis_adj
    end
    pl = plot(as, grads_fin_dif, label="Fin Dif")
    plot!(pl, as, grads_dis_adj, label="Dis Adj")
    plot!(pl, as, grads_gargamel, label="Gargamel")
    plot!(pl, as, grads_fwd_dif, label="Fwd Dif")
    plot!(title="Gradients: Disc Adj vs Fin Diff", xlabel=L"\alpha",
          ylabel=L"\nabla(\alpha)")
    return pl

end

"""
Plot convergence of finite difference gradient to discrete adjoint gradient
"""
function graph2(;N=100, fT=1.0, a=1, return_data=false, obj_type=0)
    Q0_complex = [1.0 0.0; 0.0 1.0]
    Q0_complex = [1.0+1.0im, 1.0+1.0im]
    target = eval_forward(a, Q0_complex, 100)
    target_complex = target[1:end÷2,:] + (im .* target[1+end÷2:end,:])

    eps_vec = (0.5).^(-10:16);
    grads_fin_dif = zeros(length(eps_vec))
    for i = 1:length(eps_vec)
        grads_fin_dif[i] = finite_diff_gradient(a, Q0_complex, target_complex, N=N, da=eps_vec[i], obj_type=obj_type)
    end
    grad_dis_adj = disc_adj(a, Q0_complex, target_complex, N=N, obj_type=obj_type)
    rel_errors = abs.((grads_fin_dif .- grad_dis_adj) ./ grad_dis_adj)
    if return_data
        return eps_vec, grads_fin_dif, grad_dis_adj,rel_errors
    end
    return plot(eps_vec, rel_errors, xlabel=L"\epsilon",
                ylabel=L"\left|\frac{∇_{FD,\epsilon}(\alpha) - ∇_{DA}(\alpha)}{∇_{DA}(\alpha)}\right|",
                title="Fin Diff Convergence to Disc Adj, α=$a",
                label="", scale=:log10)
end

"""
Plot Infidelities
"""
function graph3(;N=100, fT=1.0, return_data=false, obj_type=0)
    Q0_complex = [1.0+1.0im, 1.0+1.0im]
    #Q0_complex = [1.0 0.0; 0.0 1.0]
    target = eval_forward(1.0, Q0_complex, 100)
    target_complex = target[1:end÷2,:] + (im .* target[1+end÷2:end,:])

    #target_complex = [0.6663705256153477+ 0.7456237308736332im,0.6663705256153477 + 0.7456237308736332im]
    #target_complex = target_complex ./ norm(target_complex) # Normalize

    Nsamples = 1001
    infidelities = zeros(Nsamples)
    i = 1
    as = LinRange(-2,2,Nsamples)
    for i in 1:Nsamples
        Q_N = eval_forward(as[i], Q0_complex, N=N)
        infidelities[i] = infidelity(Q_N, target_complex; obj_type=obj_type)
    end
    if return_data
        return as, infidelities
    end
    pl = plot(as, infidelities)
    plot!(title="Infidelity vs α", xlabel=L"\alpha", ylabel="Infidelity")
    return pl
end

"""
Under Construction
"""
function auto_diff(a_guess)
    Q0_complex = [1.0, 0.0]
    Q0_real = complex_to_real(Q0_complex)
    target_real = eval_forward(1.0, Q0_complex, 100)

    loss_func = function(a)
        sol = eval_forward(a, Q0_complex, 100)
        loss = (sol - target_real)'*(sol - target_real)
        return loss
    end

    callback = function(a, loss, sol)
        println("Loss: $loss")
        println("a: $a\n")
        return false
    end

    println("Zygote Gradient")
    display(loss_func'(1.0)) # Test Zygote

    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((u, p) -> loss_func(u), adtype)

    optprob = Optimization.OptimizationProblem(optf, a_guess)

    ## solve optimization problem (polyopt is the algorithm to use for the optimization)
    result_ode = Optimization.solve(optprob, PolyOpt(), callback = callback, maxiters=200)

    return result_ode
end

using LinearAlgebra

E = 2 # Number of essential energy levels
S(t, a) = [0.0 0.0; 0.0 0.0]
dSda(t, a) = [0.0 0.0; 0.0 0.0]
K(t, a) = a*cos(t).*[1.0 0.0; 0.0 1.0]
dKda(t, a) = cos(t).*[1.0 0.0; 0.0 1.0]
M(t, a) = [S(t, a) -K(t,a)
           K(t, a) S(t,a)]
dMda(t, a) = [dSda(t, a) -dKda(t,a)
              dKda(t, a) dSda(t,a)]
# Initial condition
Q0_complex = [1.0, 1.0] ./ sqrt(2) # Q0 = 1/sqrt(2)(1, 1) in C^2 complex values
# Target Gate (using the value obtained when evaluating [1.0,1.0]/sqrt(2) forward with a=1.0, N=10000)
target_complex = [0.4711924454937202 + 0.52723588667193im, 0.4711924454937202 + 0.52723588667193im]

function infidelity(Q, target_complex)
    R = vcat(real(target_complex), imag(target_complex))
    T = vcat(imag(target_complex), -real(target_complex))
    infidelity = 1 - (1/E^2)*(tr(Q'*R)^2 + tr(Q'*R)^2)
    return infidelity
end

function eval_forward(a, Q0_complex, N; fT=1.0)
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

function disc_adj(a, Q0_complex, target_complex, N; fT=1.0,)
    dt = fT/N # Timestep size

    # Forward eval saving all points
    Qs = zeros(4,N+1)
    Qs[:,1+0] = vcat(real(Q0_complex), imag(Q0_complex)) 
    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt
        Qs[:,1+n+1] = (I - 0.5*dt*M(tnp1, a)) \ ((I + 0.5*dt*M(tn, a))*Qs[:,1+n])
        #println("$tn, $tnp1")
    end
    # Get Λ_N using terminal condition
    lambdas = zeros(4,N+1)
    R = vcat(real(target_complex), imag(target_complex))
    T = vcat(imag(target_complex), -real(target_complex))
    lambdas[:,1+N] = (I - 0.5*dt*M(fT, a)') \ ((2/(E^2))*(tr(Qs[:,1+N]'*R)*R + tr(Qs[:,1+N]'*T)*T)) # guard-level term is 0 in 1D
    for n in N-1:-1:1
        tn = n*dt
        lambdas[:,1+n] = (I - 0.5*dt*M(tn, a)') \ ((I + 0.5*dt*M(tn, a)')*lambdas[:,1+n+1]) # No guard-level forcing term because we are in 1D
    end
    grad_disc_adj = 0.0
    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt
        grad_disc_adj += (dMda(tn, a)*Qs[:,1+n] + dMda(tnp1, a)*Qs[:,1+n+1])'*lambdas[:,1+n+1]
        display((dMda(tn, a)*Qs[:,1+n] + dMda(tnp1, a)*Qs[:,1+n+1])'*lambdas[:,1+n+1])
    end
    grad_disc_adj *= -0.5*dt
    return Qs, lambdas, grad_disc_adj
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

function finite_diff_gradient(a, Q0_complex, target_complex, N, da=1e-5)
    Q_r = eval_forward(a+da, Q0_complex, N)
    Q_l = eval_forward(a-da, Q0_complex, N)
    infidelity_r = infidelity(Q_r, target_complex)
    infidelity_l = infidelity(Q_l, target_complex)
    fin_dif_grad = (infidelity_r - infidelity_l)/(2*da)
    return fin_dif_grad
end

function graph(N, fT=1.0)
    Q0_complex = [1.0, 1.0]
    target_complex = [0.47119255134555293+0.5272358751693975im,0.47119255134555293+0.5272358751693975im]
    grads_fin_dif = zeros(1001)
    grads_dis_adj = zeros(1001)
    i = 1
    as = LinRange(0,2,1001)
    for i in 1:1001
        grads_fin_dif[i] = finite_diff_gradient(as[i], Q0_complex, target_complex, N)
        grads_dis_adj[i] = disc_adj(as[i], Q0_complex, target_complex, N)
    end
    return as, grads_fin_dif, grads_dis_adj
end

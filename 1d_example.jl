# Testing on 1D Problem
# dpsidt = i*cost*a*psi
using Plots
using LinearAlgebra
using LaTeXStrings

function compute_gradient(N, fT, psi0=1.0; use_IMR=false)
    S(t, a) = 0.0
    K(t, a) = cos(t)*a
    M(t, a) = [S(t, a) -K(t,a)
               K(t, a) S(t,a)]
    dMda(t) = [0.0 -cos(t)
               cos(t) 0.0]
    E = 1 # 1 essential energy level

    a = 1.0 # Current value of control parameter
    #Q0 = [1.0; 0.0] # psi_0= 1+0im, initial condition
    #psi0 = exp(1im*sin(0)*a)
    Q0 = [real(psi0), imag(psi0)]

    dt = fT/N # Timestep size
    Qs = zeros(2,N+1) # For storing state vector evolution (in larger problems wouldn't store them all)
    Qs[:,1] = Q0

    I = [1.0 0.0
         0.0 1.0]

    # Forward Evolve
    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt

        if use_IMR
            tn = 0.5*(tn+tnp1)
            tnp1 = tn
        end
        Qs[:,1+n+1] = (I - 0.5*dt*M(tnp1, a)) \ ((I + 0.5*dt*M(tn, a))*Qs[:,1+n])
        #println("$tn, $tnp1")
    end
    println("Qs")
    display(Qs)

    # Set target gate (to QN, in order to test the method; gradient should be zero)
    QN = Qs[:,1+N]
    R = QN[:] # copy
    T = [R[2], -R[1]]

    println("TRACE INFIDELITY : ", abs(1-(abs2(QN'*R)+abs2(QN'*T))/E^2))
    lambdas = zeros(2,N+1) # For storing adjoint state vector solutions

    # Adjust evaluation time depending on chosen scheme
    t_adj = fT
    if use_IMR
        t_adj = fT-0.5*dt
    end

    # Terminal condition
    lambdas[:,1+N] = (I - 0.5*dt*M(t_adj, a)') \ ((2/E^2)*((QN'*R)*R + (QN'*T)*T)) # guard-level term is 0 in 1D

    # Backward/Adjoint Evolve
    for n in N-1:-1:1
        tn = n*dt
        tnp1 = n*dt
        if use_IMR
            tnp1 =  n*dt + 0.5*dt
            tn = n*dt-0.5*dt
        end
        lambdas[:,1+n] = (I - 0.5*dt*M(tn, a)') \ (I + 0.5*dt*M(tnp1, a)')*lambdas[:,1+n+1] # No guard-level forcing term because we are in 1D
    end
    println("Λs:")
    display(lambdas)

    # Calculate gradient
    gradient = 0.0
    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt
        if use_IMR
            tn = 0.5*(tn+tnp1)
            tnp1 = tn
        end
        gradient += (dMda(tn)*Qs[:,1+n] + dMda(tnp1)*Qs[:,1+n+1])'*lambdas[:,1+n+1]
    end
    gradient *= -0.5*dt
    #println("Gradient: $gradient")
    return gradient, QN
end

function graph(Ns, fT, psi0=1.0; use_IMR=false)
    gradients = zeros(length(Ns))
    QNs = zeros(2,length(Ns))
    QN_lengths = zeros(length(Ns))
    for i in 1:length(Ns)
        N = Ns[i]
        gradients[i], QNs[:,i] = compute_gradient(N, fT, psi0, use_IMR=use_IMR)
        QN_lengths[i] = QNs[:,i]'*QNs[:,i]
    end

    psifT = exp(im*sin(fT))*psi0
    QN_true = [real(psifT), imag(psifT)]

    accuracies = [norm(QN - QN_true) for QN in eachcol(QNs)]*(1/norm(QN_true))
    pl_acc = plot(Ns, accuracies, xlabel="N", scale=:log10,
                  title=L"d\psi/dt = i \cos(t) \alpha \psi,\quad \psi_0=1, \alpha = 1", markershape=:+, label="Relative Error")
    plot!(pl_acc, Ns, (fT ./ Ns) .^ 2, linestyle=:dash, label="Δt ^ 2")
    #pl_grad = plot(Ns, abs.(gradients), xlabel="N", ylabel="|Gradient|", scale=:log10, marker=:+)
    pl_grad = plot(Ns, abs.(gradients), xlabel="N", ylabel="|Gradient|", markershape=:+, label="|Gradient|", scale=:log10)
    pl_length = plot(Ns, abs.(1 .- QN_lengths), markershape=:+, ylabel=L"|1-\psi_N_n^T\psi_N|", xlabel="N", label=L"|1-\psi_N^T\psi_N|")
    pl_combined_grad_lengths = plot(Ns, abs.(gradients), xlabel="N", markershape=:+, label="|Gradient|", scale=:log10)
    plot!(pl_combined_grad_lengths, Ns, abs.(1 .- QN_lengths), label=L"|1-\psi_N^T\psi_N|", markershape=:+)

    return [pl_acc, pl_grad, pl_length, pl_combined_grad_lengths], [accuracies, gradients, QN_lengths]
end

# An even simpler example
function compute_gradient_simpler(N, fT, psi0=1.0; use_IMR=false)
    S(t, a) = 0.0
    K(t, a) = a
    M(t, a) = [S(t, a) -K(t,a)
               K(t, a) S(t,a)]
    dMda(t) = [0.0 -1.0
               1.0 0.0]
    E = 1 # 1 essential energy level

    a = 0.0 # Current value of control parameter
    #Q0 = [1.0; 0.0] # psi_0= 1+0im, initial condition
    #psi0 = exp(1im*sin(0)*a)
    Q0 = [real(psi0), imag(psi0)]

    dt = fT/N # Timestep size
    Qs = zeros(2,N+1) # For storing state vector evolution (in larger problems wouldn't store them all)
    Qs[:,1] = Q0

    I = [1.0 0.0
         0.0 1.0]

    # Forward Evolve
    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt
        if use_IMR
            tn = 0.5*(tn+tnp1)
            tnp1 = tn
        end
        Qs[:,1+n+1] = (I - 0.5*dt*M(tnp1, a)) \ ((I + 0.5*dt*M(tn, a))*Qs[:,1+n])
        #println("$tn, $tnp1")
    end
    #println("Qs")
    #display(Qs)

    # Set target gate (to QN, in order to test the method; gradient should be zero)
    QN = Qs[:,1+N]
    R = QN[:] # copy
    T = [R[2], -R[1]]
    println("TRACE INFIDELITY : ", abs(1-(abs2(QN'*R)+abs2(QN'*T))/E^2))

    println("Qs:")
    display(Qs)
    println("Qₙ:")
    display(Qs[:,1+N])
    println("R:")
    display(R)
    println("T:")
    display(T)


    lambdas = zeros(2,N+1) # For storing adjoint state vector solutions

    t_adj = fT
    if use_IMR
        t_adj = fT-0.5*dt
    end

    # Terminal condition
    lambdas[:,1+N] = (I - 0.5*dt*M(t_adj, a)') \ ((2/E^2)*((QN'*R)*R + (QN'*T)*T)) # guard-level term is 0 in 1D
    println("Λₙ:")
    display(lambdas[:,1+N])

    # Backward/Adjoint Evolve
    for n in N-1:-1:1
        tn = n*dt
        tnp1 = tn
        if use_IMR
            tnp1 =  n*dt + 0.5*dt
            tn = (n)*dt-0.5*dt
        end
        lambdas[:,1+n] = (I - 0.5*dt*M(tn, a)') \ (I + 0.5*dt*M(tnp1, a)')*lambdas[:,1+n+1] # No guard-level forcing term because we are in 1D
    end
    #println("Λs:")
    #display(lambdas)

    # Calculate gradient
    gradient = 0.0
    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt
        if use_IMR
            tn = 0.5*(tn+tnp1)
            tnp1 = tn
        end
        gradient += (dMda(tn)*Qs[:,1+n] + dMda(tnp1)*Qs[:,1+n+1])'*lambdas[:,1+n+1]
        display(lambdas[:,1+n+1])
        println("gradient = $gradient")
    end
    gradient *= -0.5*dt
    #println("Gradient: $gradient")
    return gradient, QN
end

# Using quantities computed by hand
function analytic_test()
    cos0p5 = cos(1/2)
    cos1 = cos(1)

    Q1 = (1/(1+(1/16)*cos0p5^2))*[1-(1/16)*cos0p5, (1/4)*(1+cos0p5)]

    Q2 = begin
        Q2_1 = 1-(1/8)*cos0p5 -(1/16)*cos1 - (1/16)*cos0p5^2 - (1/8)*cos0p5*cos1 + (1/256)*cos0p5^2*cos1
        Q2_2 = (1/4)+(1/2)*cos0p5 + (1/4)*cos(1) - (1/64)cos0p5^2 - (1/32)*cos0p5*cos1 - (1/64)*cos0p5^2*cos1
        Q2_const_factor = (1/(1+(1/16)*cos1^2))*(1/(1+(1/16)*cos0p5^2))
        Q2_const_factor * [Q2_1, Q2_2]
    end
    println("Q1: $Q1")
    println("Q2: $Q2")

    R = [Q2[1], Q2[2]]
    T = [Q2[2], -Q2[1]]

    Λ2 = begin
        LHS_inv = (1/(1+(1/16)*cos1^2))*[1.0  (1/4)*cos1; -(1/4)*cos1 1.0]
        RHS = 2*((Q2'*R)*R + (Q2'*T)*T)
        LHS_inv * RHS
    end
    println("Λ2: $Λ2")

    Λ1 = (1/(1+(1/16)*cos0p5^2))*[1-(1/16)cos0p5^2  (1/2)*cos0p5; -(1/2)*cos0p5 1-(1/16)*cos0p5^2]*Λ2
    println("Λ1: $Λ1")
    # Everything up to this point agrees with computation, but the gradient does not (but neither is giving me zero)

    gradient = begin
        n1_term_1 = -cos0p5*Q1[2]*((1-(1/16)*cos0p5^2)*Λ2[1] + (1/2)*cos(1/2)*Λ2[2])
        n1_term_2 = (1+cos0p5*Q1[1])*(-(1/2)*cos0p5*Λ2[1] + (1 - (1/16)*cos0p5^2)*Λ2[2])
        n2_term_1 = -(cos0p5*Q1[2] + cos1*Q2[2])*Λ2[1]
        n2_term_2 = (cos0p5*Q1[1] + cos1*Q2[1])*Λ2[2]
        -(1/4) * (n1_term_1 + n1_term_2 + n2_term_1 + n2_term_2)
    end
    return gradient
end

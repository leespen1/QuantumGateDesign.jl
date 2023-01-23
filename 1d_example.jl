# Testing on 1D Problem
# dpsidt = i*cost*a*psi
using Plots
using LinearAlgebra
using LaTeXStrings

function compute_gradient(N, fT, psi0=1.0)
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
        Qs[:,1+n+1] = (I - 0.5*dt*M(tnp1, a)) \ ((I + 0.5*dt*M(tn, a))*Qs[:,1+n])
        #println("$tn, $tnp1")
    end
    println("Qs")
    display(Qs)

    # Set target gate (to QN, in order to test the method; gradient should be zero)
    QN = Qs[:,1+N]
    R = QN[:] # copy
    T = [R[2], -R[1]]


    lambdas = zeros(2,N+1) # For storing adjoint state vector solutions
    # Terminal condition
    lambdas[:,1+N] = (I - 0.5*dt*M(fT, a)') \ ((2/E^2)*((QN'*R)*R + (QN'*T)*T)) # guard-level term is 0 in 1D

    # Backward/Adjoint Evolve
    for n in N-1:-1:1
        tn = n*dt
        lambdas[:,1+n] = (I - 0.5*dt*M(tn, a)') \ (I + 0.5*dt*M(tn, a)')*lambdas[:,1+n+1] # No guard-level forcing term because we are in 1D
    end
    println("Λs:")
    display(lambdas)

    # Calculate gradient
    gradient = 0.0
    for n in 0:N-1
        tn = n*dt
        tnp1 = (n+1)*dt
        gradient += (dMda(tn)*Qs[:,1+n] + dMda(tnp1)*Qs[:,1+n+1])'*lambdas[:,1+n+1]
    end
    gradient *= -0.5*dt
    #println("Gradient: $gradient")
    return gradient, QN
end

function graph(Ns, fT, psi0=1.0)
    gradients = zeros(length(Ns))
    QNs = zeros(2,length(Ns))
    for i in 1:length(Ns)
        N = Ns[i]
        gradients[i], QNs[:,i] = compute_gradient(N, fT)
    end

    psifT = exp(im*sin(fT))*psi0
    QN_true = [real(psifT), imag(psifT)]

    accuracies = [norm(QN - QN_true) for QN in eachcol(QNs)]*(1/norm(QN_true))
    pl_acc = plot(Ns, accuracies, xlabel="N", ylabel="Relative Error", scale=:log10,
                  title=L"\dot{\psi} = i \cos(t) \alpha \psi(t),\quad \alpha = 1")
    pl_grad = plot(Ns, gradients, xlabel="N", ylabel=L"||\nabla_\alpha \mathcal{J}_h||")
    return [pl_acc, pl_grad]
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

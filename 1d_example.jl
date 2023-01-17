# Testing on 1D Problem
# dpsidt = i*sint*a*psi
using Plots
using LinearAlgebra
using LaTeXStrings

function compute_gradient(N, fT)
    S(t, a) = 0.0
    K(t, a) = sin(t)*a
    M(t, a) = [S(t, a) -K(t,a)
               K(t, a) S(t,a)]
    dMda(t) = [0.0 -sin(t)
               sin(t) 0.0]
    E = 1 # 1 essential energy level

    a = 1.0 # Current value of control parameter
    #Q0 = [1.0; 0.0] # psi_0= 1+0im, initial condition
    psi0 = exp(-1im*cos(0)*a)
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
    #println("Qs")
    #display(Qs)

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
    #println("Λs:")
    #display(lambdas)

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

function graph(Ns, fT)
    gradients = zeros(length(Ns))
    QNs = zeros(2,length(Ns))
    for i in 1:length(Ns)
        N = Ns[i]
        gradients[i], QNs[:,i] = compute_gradient(N, fT)
    end

    psifT = exp(-im*cos(fT))
    QN_true = [real(psifT), imag(psifT)]

    accuracies = [norm(QN - QN_true) for QN in eachcol(QNs)]*(1/norm(QN_true))
    pl_acc = plot(Ns, accuracies, xlabel="N", ylabel="Relative Error", scale=:log10,
                  title=L"\dot{\psi} = i \sin(t) \alpha \psi(t),\quad \alpha = 1")
    pl_grad = plot(Ns, gradients, xlabel="N", ylabel=L"||\nabla_\alpha \mathcal{J}_h||")
    return [pl_acc, pl_grad]
end

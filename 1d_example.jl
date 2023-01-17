# Testing on 1D Problem
# dpsidt = i*sint*a*psi

function compute_gradient(N)
    S(t, a) = 0.0
    K(t, a) = sin(t)*a
    M(t, a) = [S(t, a) -K(t,a)
               K(t, a) S(t,a)]
    dMda(t) = [0.0 -sin(t)
               sin(t) 0.0]
    E = 1 # 1 essential energy level

    a = 1.0 # Current value of control parameter
    Q0 = [1.0; 0.0] # psi_0= 1+0im, initial condition
    fT = 4 # Final time

    dt = fT/(N-1) # Timestep size
    Qs = zeros(2,N) # For storing state vector evolution (in larger problems wouldn't store them all)
    Qs[:,1] = Q0

    I = [1.0 0.0
         0.0 1.0]

    # Forward Evolve
    for n in 1:N-1
        tn = (n-1)*dt
        tnp1 = n*dt
        Qs[:,n+1] = (I - 0.5*dt*M(tnp1, a)) \ (I + 0.5*dt*M(tn, a))*Qs[:,n]
        #println("$tn, $tnp1")
    end
    #println("Qs: $Qs")

    # Set target gate (to QN, in order to test the method; gradient should be zero)
    QN = Qs[:,N]
    R = QN[:] # copy
    T = [R[2], -R[1]]


    lambdas = zeros(2,N) # For storing adjoint state vector solutions
    # Terminal condition
    lambdas[:,N] = (I - 0.5*dt*M(fT, a)') \ ((2/E^2)*((QN'*R)*R + (QN'*T)*T)) # guard-level term is 0 in 1D

    # Backward/Adjoint Evolve
    for n in N-1:-1:1
        tn = (n-1)*dt
        tnp1 = n*dt
        lambdas[:,n] = (I - 0.5*dt*M(tn, a)') \ (I + 0.5*dt*M(tn, a)')*lambdas[:,n+1] # No guard-level forcing term because we are in 1D
        #println("$tn, $tnp1")
    end
    #println("Λs: $lambdas")

    # Calculate gradient
    gradient = 0.0
    for n in 1:N-1
        tn = (n-1)*Δt
        tnp1 = n*Δt
        global gradient += (dMda(tn)*Qs[:,n] + dMda(tnp1)*Qs[:,n+1])'*lambdas[:,n+1]
    end
    gradient *= -0.5*Δt
    println("Gradient: $gradient")
    return gradient, QN
end

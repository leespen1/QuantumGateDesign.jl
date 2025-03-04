#==============================================================================
#
# This file contains utilities for setting up the CNOT3 problem as it is done
# in Juqbox, converting to QuantumGateDesign format, getting control functions,
# etc.
#
==============================================================================#
"""
Given JuqboxParams, construct BCarrier control functions with the same number of
control coefficients, but with the appropriate level of smoothness for the
method order.
"""
function get_controls(method_order::Integer, D1::Integer, Cfreq::AbstractMatrix{<: Real}, tf::Real)
    # A degree N BSpline has continuous derivatives up to order N-1 
    degree = method_order
    #base_control = FortranBSplineControl(degree, D1, tf)

    base_bspline = FortranBSpline(degree, D1)
    base_control = FortranBSplineControl2(base_bspline, tf)
    controls = [CarrierControl(base_control, freqs) for freqs in eachrow(Cfreq)]

    return controls
end

function get_D1(pcof_length::Integer, N_freq::Integer, N_controls::Integer)
    D1 = div(pcof_length, N_controls*N_freq)
    if pcof_length != D1*Nsig*Nfreq
        @warn "pcof_length (pcof_length) not divisible by N_freq*N_controls ($N_freq*$N_controls). D1 calculated ($D1) will give a different control vector length than the one originally given."
    end
    return D1
end

struct CNOT3Ret{T}
    juqbox_params::Juqbox.objparams
    juqbox_wa::Juqbox.Working_Arrays
    qgd_prob::T
    target::Matrix{ComplexF64}
    pcof0::Vector{Float64}
    D1::Integer
    amax::Float64
    tf::Float64
end
    

function setup_cnot3(; seed::Integer=0, atol=1e-10, rtol=1e-12, D1=15)
    #==============================================================================
    #
    # Juqbox Problem Setup
    #
    ==============================================================================#


    Ne1 = 2 # essential energy levels per oscillator # AP: want Ne1=Ne2=2, but Ne3 = 1
    Ne2 = 2
    Ne3 = 1

    Ng1 = 2 # Osc-1, number of guard states
    Ng2 = 2 # Osc-2, number of guard states
    Ng3 = 3 # 5 # Osc-3, number of guard states

    Ne = [Ne1, Ne2, Ne3]
    Ng = [Ng1, Ng2, Ng3]
    Nt = Ne + Ng

    N = Ne1*Ne2*Ne3; # Total number of nonpenalized energy levels
    Ntot = Nt[1]*Nt[2]*Nt[3]
    Nguard = Ntot - N # Total number of guard states

    Tmax = 550.0 # 700.0

    # frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
    fa = 4.10595
    fb = 4.81526  # official
    fs = 7.8447 # storage   # official
    rot_freq = [fa, fb, fs] # rotational frequencies
    xa = 2 * 0.1099
    xb = 2 * 0.1126 # official
    xs = 0.002494^2/xa # 2.8298e-5 # official
    xab = 1.0e-6 # 1e-6 official
    xas = sqrt(xa*xs) # 2.494e-3 # official
    xbs = sqrt(xb*xs) # 2.524e-3 # official

    # Note: The ket psi = kji> = e_k kron e_j kron e_i.
    # We order the elements in the vector psi such that i varies the fastest with i in [1,Nt1], j in [1,Nt2], , k in [1,Nt3]
    # The matrix amat = I kron I kron a1 acts on alpha in psi = gamma kron beta kron alpha
    # The matrix bmat = I kron a2 kron I acts on beta in psi = gamma kron beta kron alpha
    # The matrix cmat = a3 kron I2 kron I1 acts on gamma in psi = gamma kron beta kron alpha

    # construct the lowering and raising matricies: amat, bmat, cmat
    # and the system Hamiltonian: H0

    a1 = Array(Bidiagonal(zeros(Nt[1]),sqrt.(collect(1:Nt[1]-1)),:U))
    a2 = Array(Bidiagonal(zeros(Nt[2]),sqrt.(collect(1:Nt[2]-1)),:U))
    a3 = Array(Bidiagonal(zeros(Nt[3]),sqrt.(collect(1:Nt[3]-1)),:U))

    I1 = Array{Float64, 2}(I, Nt[1], Nt[1])
    I2 = Array{Float64, 2}(I, Nt[2], Nt[2])
    I3 = Array{Float64, 2}(I, Nt[3], Nt[3])

    # create the a, a^\dag, b and b^\dag vectors
    amat = kron(I3, kron(I2, a1))
    bmat = kron(I3, kron(a2, I1))
    cmat = kron(a3, kron(I2, I1))

    adag = Array(transpose(amat))
    bdag = Array(transpose(bmat))
    cdag = Array(transpose(cmat))

    # number ops
    num1 = Diagonal(collect(0:Nt[1]-1))
    num2 = Diagonal(collect(0:Nt[2]-1))
    num3 = Diagonal(collect(0:Nt[3]-1))

    # number operators
    Na = Diagonal(kron(I3, kron(I2, num1)) )
    Nb = Diagonal(kron(I3, kron(num2, I1)) )
    Nc = Diagonal(kron(num3, kron(I2, I1)) )

    H0 = -2*pi*(xa/2*(Na*Na-Na) + xb/2*(Nb*Nb-Nb) + xs/2*(Nc*Nc-Nc) + xab*(Na*Nb) + xas*(Na*Nc) + xbs*(Nb*Nc))

    # max coefficient amplitudes, rotating frame
    amax = 0.05
    bmax = 0.1
    cmax = 0.1
    maxpar = [amax, bmax, cmax] 

    # package the lowering and raising matrices together into an one-dimensional array of two-dimensional arrays
    # Here we choose dense or sparse representation
    use_sparse = true

    # dense matrices run faster, but take more memory
    Hsym_ops=[Array(amat+adag), Array(bmat+bdag), Array(cmat+cdag)]
    Hanti_ops=[Array(amat-adag), Array(bmat-bdag), Array(cmat - cdag)]
    H0 = Array(H0)

    # Estimate time step
    Pmin = 40 # should be 20 or higher
    nsteps = Juqbox.calculate_timestep(Tmax, H0, Hsym_ops, Hanti_ops, maxpar, Pmin)

    println("Number of time steps = ", nsteps)

    Nctrl = length(Hsym_ops)

    Nfreq = 3 

    om = zeros(Nctrl,Nfreq) # In the rotating frame all ctrl Hamiltonians have a zero resonace frequency

    # initialize the carrier frequencies
    @assert(Nfreq == 1 || Nfreq == 2 || Nfreq == 3)
    if Nfreq==2
        om[1,2] = -2.0*pi*xa # carrier freq for ctrl Hamiltonian 1
        om[2,2] = -2.0*pi*xb # carrier freq for ctrl Hamiltonian 2
        om[3,2] = -2.0*pi*sqrt(xas*xbs) # carrier freq for ctrl Hamiltonian #3
    elseif Nfreq==3
        # fundamental resonance frequencies for the transmons 
        om[1:2,2] .= -2.0*pi*xa # carrier freq's for ctrl Hamiltonian 1 & 2
        om[1:2,3] .= -2.0*pi*xb # carrier freq's for ctrl Hamiltonian 1 & 2
        om[3,2] = -2.0*pi*xas # carrier freq 2 for ctrl Hamiltonian #3
        om[3,3] = -2.0*pi*xbs # carrier freq 2 for ctrl Hamiltonian #3
    end

    println("Carrier frequencies 1st ctrl Hamiltonian [GHz]: ", om[1,:]./(2*pi))
    println("Carrier frequencies 2nd ctrl Hamiltonian [GHz]: ", om[2,:]./(2*pi))
    println("Carrier frequencies 3rd ctrl Hamiltonian [GHz]: ", om[3,:]./(2*pi))


    # target for CNOT gate between oscillators 1 and 2
    gate_cnot = zeros(ComplexF64, 4, 4)
    gate_cnot[1,1] = 1.0
    gate_cnot[2,2] = 1.0
    gate_cnot[3,4] = 1.0
    gate_cnot[4,3] = 1.0

    if Ne[3] == 1
        Utarg = gate_cnot
    else
        Ident3 = Array{Float64, 2}(I, Ne[3], Ne[3])
        Utarg = kron(Ident3, gate_cnot)
    end

    # Initial basis with guard levels
    U0 = Juqbox.initial_cond(Ne, Ng)
    # U0 has size Ntot x Ness. Each of the Ness columns has one non-zero element, which is 1.

    utarget = U0 * Utarg

    # rotation matrices
    omega1, omega2, omega3 = Juqbox.setup_rotmatrices(Ne, Ng, rot_freq)

    # Compute Ra*Rb*utarget
    rot1 = Diagonal(exp.(im*omega1*Tmax))
    rot2 = Diagonal(exp.(im*omega2*Tmax))
    rot3 = Diagonal(exp.(im*omega3*Tmax))

    # target in the rotating frame
    vtarget = rot1*rot2*rot3*utarget

    # NOTE: maxpar is now a vector with 3 elements: amax, bmax, cmax
    juqbox_params = Juqbox.objparams(Ne, Ng, Tmax, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=rot_freq,
                              Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, use_sparse=use_sparse)

    # setup the initial parameter vector, randomized
    nCoeff = 2*Nctrl*Nfreq*D1 # Total number of parameters.
    pcof0 = amax*0.01 * rand(MersenneTwister(seed), nCoeff)
    println("*** Starting from random pcof with amplitude ", amax*0.01)

    # min and max B-spline coefficient values
    minCoeff, maxCoeff = Juqbox.assign_thresholds(juqbox_params,D1,maxpar)

    # output run information
    println("*** Settings ***")
    println("Frequencies: Alice = ", fa, " Bob = ", fb, " Storage = ", fs)
    println("Anharmonic coefficients in the Hamiltonian: xa = ", xa, " xb = ", xb, " xs = ", xs)
    println("Coupling coefficients in the Hamiltonian: xab = ", xab, " xas = ", xas, " xbs = ", xbs)
    println("Essential states in osc = ", Ne, " Guard states in osc = ", Ng)
    println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
    println("Number of B-spline parameters per spline = ", D1, " Total number of parameters = ", nCoeff)
    println("Max parameter amplitudes: maxpar = ", maxpar)
    println("Tikhonov coefficients: tik0 (L2) = ", juqbox_params.tik0)
    println("Tolerance in Linear Solver = ", juqbox_params.linear_solver.tol)
    if use_sparse
        println("Using a sparse representation of the Hamiltonian matrices")
    else
        println("Using a dense representation of the Hamiltonian matrices")
    end

    juqbox_wa = Juqbox.Working_Arrays(juqbox_params,nCoeff)

    println("Initial coefficient vector stored in 'pcof0'")

    #==============================================================================
    # Convert Juqbox Problem to QGD problem, get target
    ==============================================================================#
    qgd_prob = convert_juqbox(
        juqbox_params,
        gmres_reltol=rtol,
        gmres_abstol=atol,
        preconditioner_type=QuantumGateDesign.DiagonalHamiltonianPreconditioner
    )

    target = juqbox_params.Utarget_r + im*juqbox_params.Utarget_i

    return CNOT3Ret(juqbox_params, juqbox_wa, qgd_prob, target, pcof0, D1, amax, Tmax)
end


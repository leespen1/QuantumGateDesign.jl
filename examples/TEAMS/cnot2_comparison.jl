#==========================================================
This routine initializes an optimization problem to recover 
a CNOT gate on a coupled 2-qubit system with 2 energy 
levels on each oscillator (with 1 guard state on one and 
2 guard states on the other). The drift Hamiltonian in 
the rotating frame is
    H_0 = - 0.5*ξ_a(a^†a^†aa) 
          - 0.5*ξ_b(b^†b^†bb) 
          - ξ_{ab}(a^†ab^†b),
where a,b are the annihilation operators for each qubit.
Here the control Hamiltonian in the rotating frame
includes the usual symmetric and anti-symmetric terms 
H_{sym,1} = p_1(t)(a + a^†),  H_{asym,1} = q_1(t)(a - a^†),
H_{sym,2} = p_2(t)(b + b^†),  H_{asym,2} = q_2(t)(b - b^†).
The problem parameters for this example are,
            ω_a    =  2π × 4.10595   Grad/s,
            ξ_a    =  2π × 2(0.1099) Grad/s,
            ω_b    =  2π × 4.81526   Grad/s,
            ξ_b    =  2π × 2(0.1126) Grad/s,
            ξ_{ab} =  2π × 0.1       Grad/s,
We use Bsplines with carrier waves with frequencies
0, ξ_a, 2ξ_a Grad/s for each oscillator.
==========================================================# 
using LinearAlgebra
using Ipopt
using Base.Threads
using Random
using DelimitedFiles
using Printf
using FFTW
using Plots
using SparseArrays

Base.show(io::IO, f::Float64) = @printf(io, "%20.13e", f)

using Juqbox # quantum control module

eval_lab = false # true
println("Setup for ", eval_lab ? "lab frame evaluation" : "rotating frame optimization")

Nctrl = 2 # Number of control Hamiltonians

Ne1 = 2 # essential energy levels per oscillator 
Ne2 = 2
Ng1 = 2 # 0 # Osc-1, number of guard states
Ng2 = 2 # 0 # Osc-2, number of guard states

Ne = [Ne1, Ne2]
Ng = [Ng1, Ng2]

N = Ne1*Ne2; # Total number of nonpenalized energy levels
Ntot = (Ne1+Ng1)*(Ne2+Ng2)
Nguard = Ntot - N # total number of guard states

Nt1 = Ne1 + Ng1
Nt2 = Ne2 + Ng2

Tmax = 50.0 # Duration of gate

# frequencies (in GHz, will be multiplied by 2*pi to get angular frequencies in the Hamiltonian matrix)
fa = 4.10595    # official
fb = 4.81526   # official
favg = 0.5*(fa+fb)
rot_freq = [fa, fb] # rotational frequencies
#rot_freq = [favg, favg] # rotational frequencies
x1 = 2* 0.1099  # official
x2 = 2* 0.1126   # official
x12 = 0.1 # Artificially large to allow fast coupling. Actual value: 1e-6 
  
# construct the lowering and raising matricies: amat, bmat

# Note: The ket psi = ji> = e_j kron e_i.
# We order the elements in the vector psi such that i varies the fastest with i in [1,Nt1] and j in [1,Nt2]
# The matrix amat = I kron a1 acts on alpha in psi = beta kron alpha
# The matrix bmat = a2 kron I acts on beta in psi = beta kron alpha
a1 = Array(Bidiagonal(zeros(Nt1),sqrt.(collect(1:Nt1-1)),:U))
a2 = Array(Bidiagonal(zeros(Nt2),sqrt.(collect(1:Nt2-1)),:U))

I1 = Array{Float64, 2}(I, Nt1, Nt1)
I2 = Array{Float64, 2}(I, Nt2, Nt2)

# create the a, a^\dag, b and b^\dag vectors
amat = kron(I2, a1)
bmat = kron(a2, I1)

adag = Array(transpose(amat))
bdag = Array(transpose(bmat))

# number ops
num1 = Diagonal(collect(0:Nt1-1))
num2 = Diagonal(collect(0:Nt2-1))

# number operators
N1 = Diagonal(kron(I2, num1) )
N2 = Diagonal(kron(num2, I1) )

# System Hamiltonian
if eval_lab
    H0 = 2*pi*( fa*N1 + fb*N2 -x1/2*(N1*N1-N1) - x2/2*(N2*N2-N2) - x12*(N1*N2) )
else
    H0 = 2*pi*( (fa-rot_freq[1])*N1 + (fb-rot_freq[2])*N2 -x1/2*(N1*N1-N1) - x2/2*(N2*N2-N2) - x12*(N1*N2) )
end
H0 = Array(H0)

if eval_lab
    Hunc_ops=[Array(amat+adag), Array(bmat+bdag)]
else
    Hsym_ops=[Array(amat+adag), Array(bmat+bdag)]
    Hanti_ops=[Array(amat-adag), Array(bmat-bdag)]
end

# max coefficients, rotating frame
amax = 0.040 # 0.014 # max amplitude ctrl func for Hamiltonian #1
bmax = 0.040 # 0.020 # max amplitude ctrl func for Hamiltonian #2
maxpar = [amax, bmax]

# Estimate time step
if eval_lab
    Pmin = 100 # should be 20 or higher
    nsteps = calculate_timestep(Tmax, H0, Hunc_ops, maxpar, Pmin)
else
    Pmin = 40 # should be 20 or higher
    nsteps = calculate_timestep(Tmax, H0, Hsym_ops, Hanti_ops, maxpar, Pmin)
end

println("Number of time steps = ", nsteps)

# package the lowering and raising matrices together into an one-dimensional array of two-dimensional arrays
# Here we choose dense or sparse representation
use_sparse = true
# use_sparse = false

Nfreq = 2 # number of carrier frequencies

om = zeros(Nctrl, Nfreq) # Allocate space for the carrier wave frequencies

@assert(Nfreq==1 || Nfreq==2 || Nfreq==3)
if Nfreq == 2
    # ctrl 1
    om[1,1] = 2*pi*(fa - rot_freq[1])
    om[1,2] = 2*pi*(fa - rot_freq[1] - x12) # coupling freq for both ctrl funcs (re/im)
    # ctrl 1
    om[2,1] = 2*pi*(fb - rot_freq[2])
    om[2,2] = 2*pi*(fb - rot_freq[2] - x12) # coupling freq for both ctrl funcs (re/im)
elseif Nfreq == 3
    om[1,2] = -2.0*pi*x1 # 1st ctrl, re
    om[2,2] = -2.0*pi*x2 # 2nd ctrl, re
    om[1:Nctrl,3] .= -2.0*pi*x12 # coupling freq for both ctrl funcs (re/im)
end
println("Carrier frequencies (lab frame) 1st ctrl Hamiltonian [GHz]: ", rot_freq[1] .+ om[1,:]./(2*pi))
println("Carrier frequencies (lab frame) 2nd ctrl Hamiltonian [GHz]: ", rot_freq[2] .+ om[2,:]./(2*pi))

# CNOT target for the essential levels
gate_cnot =  zeros(ComplexF64, N, N)
gate_cnot[1,1] = 1.0
gate_cnot[2,2] = 1.0
gate_cnot[3,4] = 1.0
gate_cnot[4,3] = 1.0

# Initial basis with guard levels
U0 = initial_cond(Ne, Ng)

utarget = U0 * gate_cnot

# rotation matrices
omega1, omega2 = Juqbox.setup_rotmatrices(Ne, Ng, rot_freq)

# Compute Ra*Rb*utarget
rot1 = Diagonal(exp.(im*omega1*Tmax))
rot2 = Diagonal(exp.(im*omega2*Tmax))

if eval_lab
    vtarget = utarget # target in the lab frame
else    
    vtarget = rot1*rot2*utarget # target in the rotating frame
end

# assemble problem description for the optimization
if eval_lab
    params = Juqbox.objparams(Ne, Ng, Tmax, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=rot_freq,
                              Hconst=H0, Hunc_ops=Hunc_ops, use_sparse=use_sparse)
else
    params = Juqbox.objparams(Ne, Ng, Tmax, nsteps, Uinit=U0, Utarget=vtarget, Cfreq=om, Rfreq=rot_freq,
                              Hconst=H0, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, use_sparse=use_sparse)
end

# initial parameter guess
if eval_lab
    startFromScratch = false
else
    startFromScratch = true
end
#startFile = "examples/drives/cnot2-pcof-opt-t100.jld2"
startFile = "../drives/cnot2-pcof-opt-t50-avg.jld2"

# dimensions for the parameter vector
D1 = 10 # number of B-spline coeff per oscillator, freq and sin/cos

nCoeff = 2*Nctrl*Nfreq*D1 # Total number of parameters.

Random.seed!(2456)
if startFromScratch
    pcof0 = amax*0.01 * rand(nCoeff)
    println("*** Starting from pcof with random amplitude ", amax*0.01)
else
    # the data on the startfile must be consistent with the setup!
    # use if you want to have initial coefficients read from file
    pcof0 = read_pcof(startFile)
    nCoeff = length(pcof0)
    D1 = div(nCoeff, 2*Nctrl*Nfreq) # factor '2' is for sin/cos
    nCoeff = 2*Nctrl*Nfreq*D1 # just to be safe if the file doesn't contain the right number of elements
    println("*** Starting from B-spline coefficients in file: ", startFile)
end

samplerate = 32 # for plotting
casename = "cnot2" # for constructing file names

# min and max B-spline coefficient values
minCoeff, maxCoeff = Juqbox.assign_thresholds(params,D1,maxpar)
println("Number of min coeff: ", length(minCoeff), "Max Coeff: ", length(maxCoeff))

maxIter = 50 # 0 #250 #50 # optional argument
lbfgsMax = 250 # optional argument

println("*** Settings ***")
# output run information
println("Frequencies: fa = ", fa, " fb = ", fb, " fa-favg = ", fa-favg, " fb-favg = ", fb-favg )
println("Coefficients in the Hamiltonian: x1 = ", x1, " x2 = ", x2, " x12 = ", x12)
println("Essential states in osc = ", [Ne1, Ne2], " Guard states in osc = ", [Ng1, Ng2])
println("Total number of states, Ntot = ", Ntot, " Total number of guard states, Nguard = ", Nguard)
println("Number of B-spline parameters per spline = ", D1, " Total number of parameters = ", nCoeff)
println("Max parameter amplitudes: maxpar = ", maxpar)
println("Tikhonov regularization tik0 (L2) = ", params.tik0)
if use_sparse
    println("Using a sparse representation of the Hamiltonian matrices")
else
    println("Using a dense representation of the Hamiltonian matrices")
end

new_tol = 1e-12
estimate_Neumann!(new_tol, params, maxpar);
println("Using tolerance", new_tol, " and ", params.linear_solver.max_iter, " terms in the Neumann iteration")

# Allocate all working arrays
wa = Juqbox.Working_Arrays(params, nCoeff)
juqbox_ipopt_prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter=maxIter, lbfgsMax=lbfgsMax, startFromScratch=startFromScratch)

#uncomment to run the gradient checker for the initial pcof
#=
if @isdefined addOption
    addOption( prob, "derivative_test", "first-order"); # for testing the gradient
else
    AddIpoptStrOption( prob, "derivative_test", "first-order")
end
=#

#uncomment to change print level
#=
if @isdefined addOption
    addOption(prob, "print_level", 0); # for testing the gradient
else
    AddIpoptIntOption(prob, "print_level", 0)
end
=#

println("Initial coefficient vector stored in 'pcof0'")

if eval_lab
    objf, uhist, trfid = traceobjgrad(pcof0, params, wa, true, false); # evaluate objective function, but not the gradient
    println("Trace fidelity: ", trfid);
end

prob = convert_juqbox(params)
controls = QuantumGateDesign.bspline_controls(prob.tf, D1, om)
controls_autodiff = QuantumGateDesign.bspline_controls_autodiff(prob.tf, D1, om)
target = vcat(params.Utarget_r, params.Utarget_i)

# Should this be for T=100 or T=50?
include("../drives/cnot2_t100.jl")

N_derivatives = 5
N_hermite_samples = 8
hermite_controls, hermite_pcof = QuantumGateDesign.sample_from_controls(
    controls_autodiff, pcof, N_hermite_samples, N_derivatives
)

hermite_pcof_L_mat = fill(-2e19, (1+N_derivatives, 
                                  N_hermite_samples*length(hermite_controls), 2))
hermite_pcof_U_mat = fill(2e19, (1+N_derivatives, 
                                  N_hermite_samples*length(hermite_controls), 2))

# For now, instead of doing a proper constraint just hack it by limiting
# vals at the sample points
hermite_pcof_L_mat[1,:,:] .= -amax
hermite_pcof_U_mat[1,:,:] .= amax
hermite_pcof_L = reshape(hermite_pcof_L_mat, :)
hermite_pcof_U = reshape(hermite_pcof_U_mat, :)


# Check infidelity after converting to hermite control
println(
    "Infidelity after hermite conversion is ",
    QuantumGateDesign.infidelity(prob, hermite_controls, hermite_pcof, target, order=8)
)

## For plotting controls
#i = 2
#pcof_old = QuantumGateDesign.get_control_vector_slice(pcof, controls, i)
#pcof_new = QuantumGateDesign.get_control_vector_slice(hermite_pcof, hermite_controls, i)
#plot(plot_control(controls[i], pcof_old), plot_control(hermite_controls[i], pcof_new))

run_convergence = true
if run_convergence
    params.nsteps = 5
    prob.nsteps = 5

    ret_qgd = get_history_convergence(prob, hermite_controls, hermite_pcof, 15)
    ret_juq = get_history_convergence(params, pcof, wa, 15)

    labels = ["Hermite-2" "Hermite-4" "Hermite-6" "Hermite-8"]
    pls = plot_history_convergence(ret_qgd..., include_orderlines=false, labels=labels)
    QuantumGateDesign.plot_history_convergence!(
        pls..., ret_juq..., labels="Juqbox", colors=:black, marker=:star
    )


    println(
        "Runtime Ratios: ",
        QuantumGateDesign.get_runtime_ratios(ret_qgd[2], ret_qgd[3], ret_juq[2], ret_juq[3])
    )
    #  3.7454890210116e+00
    #  7.7500209338029e-02
    #  1.4760277802926e-01
    #  1.7436255934397e-01
end

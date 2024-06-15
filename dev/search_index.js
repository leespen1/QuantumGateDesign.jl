var documenterSearchIndex = {"docs":
[{"location":"optimization/#Optimization","page":"Optimization","title":"Optimization","text":"","category":"section"},{"location":"optimization/","page":"Optimization","title":"Optimization","text":"To actually find an optimal control pulse, we interface with IPOPT using the optimize_gate function.","category":"page"},{"location":"optimization/","page":"Optimization","title":"Optimization","text":"optimize_gate","category":"page"},{"location":"optimization/#QuantumGateDesign.optimize_gate","page":"Optimization","title":"QuantumGateDesign.optimize_gate","text":"optimize_gate(schro_prob, controls, pcof_init, target, [order=4, pcof_L=missing, pcof_U=missing, maxIter=50, print_level=5, ridge_penalty_strength=1e-2, max_cpu_time = 300.0])\n\nPerform gradient-based search (L-BFGS) to find value of the control vector pcof which minimizes the objective function for the given problem and target. Returns a dictionary which contains the ipopt optimization problem object, as well as other information about the optimization.\n\nNOTE: Right now the GMRES tolerance is not implemented as a kwarg, so the default value of 1e-10 for abstol and reltol will be used. \n\nNOTE: to play around with IPOPT settings which are not accessible through this function call, could run the optimization with maxIter=1, then grab the IPOPT problem from the return dictionary, and change the IPOPT settings directly through the IPOPT API.\n\nArguments\n\nprob::SchrodingerProb: Object containing the Hamiltonians, number of timesteps, etc.\ncontrols: An AstractControl or vector of controls, where the i-th control corresponds to the i-th control Hamiltonian.\npcof::AbstractVector{<: Real}: The control vector.\ntarget::AbstractMatrix{Float64}: The target gate, in 'stacked' real-valued format.\norder::Int64=2: Which order of the timestepping method to use.\npcof_L=missing: Lower bounds of the control parameters. Can either be a single number, used for all parameters, or a vector the same length as pcof, which will set a lower limit on each parameter.\npcof_U=missing: Upper bounds of the control parameters.\nmaxIter=50: Maximum number of iterations to perform.\nprint_level=5: Print level of IPOPT.\nridge_penalty_strength: Strength of the ridge/Tikhonov regularization term in the objective function.\nmax_cpu_time: Maximum CPU time (in seconds) to spend on the optimization problem.\n\n\n\n\n\n","category":"function"},{"location":"forward_simulation/#Forward-Simulation","page":"Forward Simulation","title":"Forward Simulation","text":"","category":"section"},{"location":"forward_simulation/","page":"Forward Simulation","title":"Forward Simulation","text":"The function eval_forward can be used to simulate the evolution of a quantum state in time.","category":"page"},{"location":"forward_simulation/","page":"Forward Simulation","title":"Forward Simulation","text":"eval_forward(\n        prob::SchrodingerProb{M1, M2}, controls, pcof::AbstractVector{<: Real};\n        order::Int=2, saveEveryNsteps::Int=1,\n        forcing::Union{AbstractArray{Float64, 4}, Missing}=missing,\n        kwargs...\n    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}","category":"page"},{"location":"forward_simulation/#QuantumGateDesign.eval_forward-Union{Tuple{M2}, Tuple{M1}, Tuple{SchrodingerProb{M1, M2}, Any, AbstractVector{<:Real}}} where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}","page":"Forward Simulation","title":"QuantumGateDesign.eval_forward","text":"eval_forward(prob, controls, pcof; [order=2, saveEveryNsteps=1, forcing=missing, abstol=1e-10, reltol=1e-10])\n\nSimulate a SchrodingerProb forward in time. Return the history of the state vector for each initial condition as a 4D array.\n\nArguments\n\nprob::SchrodingerProb: Object containing the Hamiltonians, number of timesteps, etc.\ncontrols: An AstractControl or vector of controls, where the i-th control corresponds to the i-th control Hamiltonian.\npcof::AbstractVector{<: Real}: The control vector.\norder::Int64=2: Which order of the method to use.\nsaveEveryNsteps::Int64=1: Only store the state every saveEveryNsteps timesteps.\nforcing::Union{AbstractArray{Float64}, Missing}: Optional forcing array, ordered in same format as the returned history.\nabstol::Float64=1e-10: Absolute tolerance to use in GMRES.\nreltol::Float64=1e-10: Relative tolerance to use in GMRES.\n\n\n\n\n\n","category":"method"},{"location":"examples/#Rabi-Oscillator","page":"Examples","title":"Rabi Oscillator","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"In this example we set up a Rabi oscillator problem, and optimize for a Pauli-X gate.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"The Rabi oscillator consists of a single qubit in the rotating frame, with the rotation frequency chosen perfectly so that the state vector is constant-in-time unless a control pulse is applied.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"We have one control pulse available in the lab frame (and therefore two in the rotating frame), which will have a constant amplitude in the rotating frame.","category":"page"},{"location":"examples/#Setting-up-the-Problem","page":"Examples","title":"Setting up the Problem","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"First we construct the Hamiltonians and initial conditions, and put them together in a SchrodingerProb. For a Rabi oscillator the system/drift Hamiltonian is zero, and the control Hamiltonian has real part a+a^dagger and imaginary part a-a^dagger, where a is the lowering/annihilation operator","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"a = beginbmatrix 0  1  0  0 endbmatrix","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Consequently, the dynamics of the problem are","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"fracdUdt = left(p_0beginbmatrix 0  1  1  0 endbmatrix +iq_0\nbeginbmatrix 0  1  1  0 endbmatrixright)U","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"(NOTE: not sure if there should be a -i on the right-hand side when we work in the complex formulation).","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using QuantumGateDesign\n\nH0_sym = zeros(2,2)\nH0_asym = zeros(2,2)\na = [0.0 1;\n     0   0]\nHc_sym = a + a'\nHc_asym = a - a'\n\nsym_ops = [Hc_sym]\nasym_ops = [Hc_asym]\n\nU0_complex = [1.0 0;\n              0   1]\n\nu0 = real(U0_complex)\nv0 = imag(U0_complex)\n\nΩ = 0.5 + 0.0im\n\n# 5 Rabi Oscillations\ntf = 10pi / (2*abs(Ω))\nnsteps = 10\nN_ess_levels = 2\n\nprob = SchrodingerProb(\n    H0_sym, H0_asym, sym_ops, asym_ops, u0, v0, tf, nsteps, N_ess_levels\n)","category":"page"},{"location":"examples/#Setting-up-the-Control","page":"Examples","title":"Setting up the Control","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Next we establish the controls. We use a GRAPEControl, which implements a piecewise constant control (like what is used in the GRAPE method, with one control coefficient (per real/imaginary part) in order to achieve control pulses that are constant throughout the duration of the gate.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"N_control_coeff = 1\ncontrol = QuantumGateDesign.GRAPEControl(N_control_coeff, prob.tf)\n\npcof = [real(Ω), imag(Ω)]","category":"page"},{"location":"examples/#Performing-the-Optimization","page":"Examples","title":"Performing the Optimization","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Finally, we set up our target and interface to IPOPT (using optimize_gate)to perform the optimization.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"# Pauli-X gate, or 1-qubit SWAP\ntarget_complex = [0.0 1;\n                  1   0]\ntarget = vcat(real(target_complex), imag(target_complex))\n\nret_dict = optimize_gate(prob, control, pcof, target, order=4)","category":"page"},{"location":"control_functions/#Control-Functions","page":"Control Functions","title":"Control Functions","text":"","category":"section"},{"location":"control_functions/","page":"Control Functions","title":"Control Functions","text":"The only requirement we have of a control function is that ","category":"page"},{"location":"control_functions/","page":"Control Functions","title":"Control Functions","text":"Controls are implemented as subtypes of the AbstractControl abstract type.","category":"page"},{"location":"control_functions/","page":"Control Functions","title":"Control Functions","text":"Each control has an associated control vector length, which is included as a parameter of the control object. Some controls have half of their control parameters control the real part, while the reamaining half control the imaginary part. By convention, when this is the case we reserve the first half of the control vector for the real-associated parameters, and the second half for the imaginary-associated parameters (as opposed to alternating them).","category":"page"},{"location":"control_functions/","page":"Control Functions","title":"Control Functions","text":"When controls are gathered together in a vector, the collective control vector will just be the concatenation of all the individual control vectors.","category":"page"},{"location":"control_functions/","page":"Control Functions","title":"Control Functions","text":"AbstractControl\nBSplineControl\nHermiteControl\nHermiteCarrierControl","category":"page"},{"location":"control_functions/#QuantumGateDesign.AbstractControl","page":"Control Functions","title":"QuantumGateDesign.AbstractControl","text":"Abstract supertype for all controls.\n\nEvery concrete subtype must have the following methods defined:\n\nMethods\n\neval_p(control::AbstractControl, t::Real, pcof::AbstractVector{<: Real})\neval_q(control::AbstractControl, t::Real, pcof::AbstractVector{<: Real})\n\nEvery concrete subtype must have the following parameters:\n\nParameters\n\nN_coeff::Int\ntf::Float64\n\nThe following methods can also be handwritten for efficiency, but have defaults implemented using automatic differentiation (currently broken):\n\nOptional Methods\n\neval_p_derivative\neval_q_derivative\neval_grad_p_derivative\neval_grad_q_derivative\n\n\n\n\n\n","category":"type"},{"location":"control_functions/#QuantumGateDesign.BSplineControl","page":"Control Functions","title":"QuantumGateDesign.BSplineControl","text":"BsplineControl(tf, D1, omega)\n\nConstruct a control whose value is the sum of Bspline envelopes multiplied by carrier waves.\n\n\n\n\n\n","category":"type"},{"location":"control_functions/#QuantumGateDesign.HermiteControl","page":"Control Functions","title":"QuantumGateDesign.HermiteControl","text":"HermiteControl(N_points, tf, N_derivatives; [scaling_type=:Heuristic])\n\nConstruct a control that is a Hermite interpolating polynomial of the values and first N_derivatives derivatives at N_points evenly spaced points. The control vector gives the values and the derivatives (scaled depending on scaling_type).\n\nNotes\n\nWorking on making this non-allocating and non-repeating.\n\nAlso remember to eventually change when the 1/j! is applied, for better numerical stability\n\nNote: I can use a high-order hermite control for a low order method, and pcof still works the same way.\n\nAnd for pcof, it is convenient to just reshape a matrix whose columns are the derivatives\n\n\n\n\n\n","category":"type"},{"location":"control_functions/#QuantumGateDesign.HermiteCarrierControl","page":"Control Functions","title":"QuantumGateDesign.HermiteCarrierControl","text":"HermiteCarrierControl(N_points, tf, N_derivatives, carrier_wave_freqs; [scaling_type=:Heuristic])\n\nConstruct  a control that is the sum of Hermite interpolating polynomials of the values and first N_derivatives derivatives at N_points evenly spaced points, multiplied by carrier waves. The control vector gives the values and the derivatives (scaled depending on scaling_type) for each of the polynomials multiplied by carrier waves.\n\nNotes\n\nWorking on making this non-allocating and non-repeating.\n\nAlso remember to eventually change when the 1/j! is applied, for better numerical stability\n\nNote: I can use a high-order hermite control for a low order method, and pcof still works the same way.\n\nAnd for pcof, it is convenient to just reshape a matrix whose columns are the derivatives\n\nStatic arrays could be useful here. Wouldn't have such a big struct, could just construct them inline on the stack. Just need a N_derivatives struct parameter.\n\nIdea: Have a version for which we specify N derivatives in pcof, but we use N+M derivatives, which the remainder always being 0. That way we have highly smooth controls, but we don't have as many control parameters. (right now, 3 carrier waves, 2 points each, 3 derivatives each, uses 48 parameters.)\n\nAlso, as I have more derivatives actually controlled by pcof, the more parameters I have affecting each time interval. Not sure how much that matters, but with B-splines they seemed to think it was good that each point is affected by at most 3 parameters. In my case, that number is 2*(1+N_derivatives)\n\n\n\n\n\n","category":"type"},{"location":"gradient_evaluation/#Gradient-Evaluation","page":"Gradient Evaluation","title":"Gradient Evaluation","text":"","category":"section"},{"location":"gradient_evaluation/","page":"Gradient Evaluation","title":"Gradient Evaluation","text":"The gradient can be computed with a function call which is similar to that of  eval_forward, but with the additional target argument, which is the target gate we wish to implement.","category":"page"},{"location":"gradient_evaluation/","page":"Gradient Evaluation","title":"Gradient Evaluation","text":"The intended way of computing the gradient is the discrete adjoint method, which can be called using the discrete_adjoint function, but the functions eval_grad_forced and eval_grad_finite_difference can also be used, for example to check the correctness of the discrete adjoint method.","category":"page"},{"location":"gradient_evaluation/","page":"Gradient Evaluation","title":"Gradient Evaluation","text":"discrete_adjoint\neval_grad_finite_difference\neval_grad_forced","category":"page"},{"location":"gradient_evaluation/#QuantumGateDesign.discrete_adjoint","page":"Gradient Evaluation","title":"QuantumGateDesign.discrete_adjoint","text":"discrete_adjoint(prob, controls, pcof, target; [order=2, cost_type=:Infidelity, return_lambda_history=false, abstol=1e-10, reltol=1e-10])\n\nCompute the gradient using the discrete adjoint method. Return the gradient.\n\nArguments\n\nprob::SchrodingerProb: Object containing the Hamiltonians, number of timesteps, etc.\ncontrols: An AstractControl or vector of controls, where the i-th control corresponds to the i-th control Hamiltonian.\npcof::AbstractVector{<: Real}: The control vector.\ntarget::AbstractMatrix{Float64}: The target gate, in 'stacked' real-valued format.\norder::Int64=2: Which order of the method to use.\ncost_type=:Infidelity: The cost function to use (ONLY USE INFIDELITY, OTHERS HAVE NOT BEEN TESTED RECENTLY).\nreturn_lambda_history=false: Whether to return the history of the adjoint variable lambda.\nabstol::Float64=1e-10: Absolute tolerance to use in GMRES.\nreltol::Float64=1e-10: Relative tolerance to use in GMRES.\n\n\n\n\n\n","category":"function"},{"location":"gradient_evaluation/#QuantumGateDesign.eval_grad_finite_difference","page":"Gradient Evaluation","title":"QuantumGateDesign.eval_grad_finite_difference","text":"eval_grad_finite_difference(prob, controls, pcof, target; [dpcof=1e-5, order=2, cost_type=:Infidelity, abstol=1e-10, reltol=1e-10])\n\nCompute the gradient using centered difference for each control parameter. Return the gradient.\n\nArguments\n\nprob::SchrodingerProb: Object containing the Hamiltonians, number of timesteps, etc.\ncontrols: An AstractControl or vector of controls, where the i-th control corresponds to the i-th control Hamiltonian.\npcof::AbstractVector{<: Real}: The control vector.\ntarget::AbstractMatrix{Float64}: The target gate, in 'stacked' real-valued format.\ndpcof=1e-5: The spacing to be used in the centered difference method.\ncost_type=:Infidelity: The cost function to use (ONLY USE INFIDELITY, OTHERS HAVE NOT BEEN TESTED RECENTLY)\norder::Int64=2: Which order of the method to use.\nabstol::Float64=1e-10: Absolute tolerance to use in GMRES.\nreltol::Float64=1e-10: Relative tolerance to use in GMRES.\n\n\n\n\n\n","category":"function"},{"location":"gradient_evaluation/#QuantumGateDesign.eval_grad_forced","page":"Gradient Evaluation","title":"QuantumGateDesign.eval_grad_forced","text":"eval_grad_forced(prob, controls, pcof, target; [order=2, cost_type=:Infidelity, return_forcing=false, abstol=1e-10, reltol=1e-10])\n\nCompute the gradient by differentiating Schrodinger's equation w.r.t each control parameter (the GOAT method). Return the gradient.\n\nArguments\n\nprob::SchrodingerProb: Object containing the Hamiltonians, number of timesteps, etc.\ncontrols: An AstractControl or vector of controls, where the i-th control corresponds to the i-th control Hamiltonian.\npcof::AbstractVector{<: Real}: The control vector.\ntarget::AbstractMatrix{Float64}: The target gate, in 'stacked' real-valued format.\ncost_type=:Infidelity: The cost function to use (ONLY USE INFIDELITY, OTHERS HAVE NOT BEEN TESTED RECENTLY).\nreturn_forcing=false: Whether to return the forcing array computed for the last control parameter (for debugging).\norder::Int64=2: Which order of the method to use.\nabstol::Float64=1e-10: Absolute tolerance to use in GMRES.\nreltol::Float64=1e-10: Relative tolerance to use in GMRES.\n\n\n\n\n\n","category":"function"},{"location":"visualization/#Visualization","page":"Visualization","title":"Visualization","text":"","category":"section"},{"location":"visualization/","page":"Visualization","title":"Visualization","text":"Tools for visualizing the state populations and the control pulses are provided in the plot_populations and plot_control functions, respectively.","category":"page"},{"location":"visualization/","page":"Visualization","title":"Visualization","text":"plot_populations\nplot_control","category":"page"},{"location":"visualization/#QuantumGateDesign.plot_populations","page":"Visualization","title":"QuantumGateDesign.plot_populations","text":"plot_populations(history; [ts=missing, level_indices=missing, labels=missing])\n\nGiven state vector history, plot population evolution (assuming single qubit for labels).\n\n\n\n\n\n","category":"function"},{"location":"visualization/#QuantumGateDesign.plot_control","page":"Visualization","title":"QuantumGateDesign.plot_control","text":"plot_control(control, pcof; [npoints=1001, derivative_orders=0, convert_units=false, linewidth=2])\n\nPlot the pulse amplitudes over time for the given control and control vector. Return the plot.\n\nThe derivatives can also be plotted by supplying an integer or vector of integers as the arguemnt for derivative_orders.\n\n\n\n\n\n","category":"function"},{"location":"problem_setup/#Problem-Setup","page":"Problem Setup","title":"Problem Setup","text":"","category":"section"},{"location":"problem_setup/","page":"Problem Setup","title":"Problem Setup","text":"The parameters that determine the physics of the problem are held in a SchrodingerProb.","category":"page"},{"location":"problem_setup/","page":"Problem Setup","title":"Problem Setup","text":"For a state-transfer problem, please give the initial conditions as column matrices. There are currently bugs when giving them as vectors.","category":"page"},{"location":"problem_setup/","page":"Problem Setup","title":"Problem Setup","text":"SchrodingerProb","category":"page"},{"location":"problem_setup/#QuantumGateDesign.SchrodingerProb","page":"Problem Setup","title":"QuantumGateDesign.SchrodingerProb","text":"SchrodingerProb(system_sym, system_asym, sym_operators, asym_operators, u0, v0, tf, nsteps, N_ess_levels)\n\nSet up an object containing the data that defines the physics and numerics of the problem (the Hamiltonians, initial conditions, number of timesteps, etc).\n\nArguments\n\nsystem_sym::M: the symmetric/real part of the system Hamiltonian.\nsystem_asym::M: the antisymmetric/imaginary part of the system Hamiltonian.\nsym_operators::Vector{M}: a vector whose i-th entry is the symmetric part of the i-th control Hamiltonian.\nasym_operators::Vector{M}: a vector whose i-th entry is the antisymmetric part of the i-th control Hamiltonian.\nu0::M: the real part of the initial conditions. The i-th column corresponds to the i-th initial state in the gate basis.\nv0::M: the imaginary part of the initial conditions. The i-th column corresponds to the i-th initial state in the gate basis.\ntf::Real: duration of the gate.\nnsteps::Int64: number of timesteps to take.\nN_ess_levels::Int64: number of levels in the 'essential' subspace, i.e. the part of the subspace actually used for computation.\nguard_subspace_projector::Union{M, missing}=missing: matrix projecting a state vector in to the 'guard' subspace.\n\nwhere M <: AbstractMatrix{Float64}\n\nTODO: allow different types for each operator, to allow better specializations (e.g. for symmetric or tridiagonal matrices).\n\n\n\n\n\n","category":"type"},{"location":"#QuantumGateDesign.jl","page":"Home","title":"QuantumGateDesign.jl","text":"","category":"section"},{"location":"#Table-of-Contents","page":"Home","title":"Table of Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Problem-Description","page":"Home","title":"Problem Description","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"QuantumGateDesign.jl solves optimization problems of the form","category":"page"},{"location":"","page":"Home","title":"Home","text":"min_boldsymboltheta mathcalJ_1(U(t_f)) + int_0^t_f mathcalJ_2\nU(t) dt","category":"page"},{"location":"","page":"Home","title":"Home","text":"where the dynamics of U are governed by Schrodinger's equation:","category":"page"},{"location":"","page":"Home","title":"Home","text":"fracdUdt = -iH(tboldsymboltheta) U\nquad 0 leq t leq t_fquad U(0) = U_0 in mathbbC^N_s","category":"page"},{"location":"","page":"Home","title":"Home","text":"and the Hamiltonian can be decomposed into a constant 'system'/'drift' component, and several 'control' hamiltonians which are modulated by scalar functions:","category":"page"},{"location":"","page":"Home","title":"Home","text":"H(tboldsymboltheta) = H_d + sum_j=1^N_c c_j(t boldsymboltheta) cdot H_j","category":"page"},{"location":"#Workflow","page":"Home","title":"Workflow","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The basic workflow is:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Set up a SchrodingerProb.\nSet up the control functions.\nChoose an initial guess for the control vector.\nChoose a target gate.\nOptimize the control vector using ipopt.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The user can also directly simulate the time evolution of the state vector and compute gradients of the gate-infidelity, outside of the optimization loop.","category":"page"},{"location":"#Real-Valued-Representation","page":"Home","title":"Real-Valued Representation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"We perform all our computations using real-valued arithmetic. When the real and imaginary part of a state vector are not handled in separate data structures, we 'stack' them into one vector. For example, the complex-valued state boldsymbolpsi = 1+2i 3+4i^T becomes the real-valued state 1 2 3 4.","category":"page"},{"location":"","page":"Home","title":"Home","text":"In several parts of the source code, we use u to indicate the real part of a state, and v to indicate the imaginary part. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"For control pulses, the real part is indicated by p, and the imaginary part by q.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Eventually we would like to replace the u/v and p/q notation with something more descriptive, like real/imag.","category":"page"},{"location":"#State-Storage","page":"Home","title":"State Storage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Because we solve the initial value problem stated above using Hermite interpolation, after solving the problem we have computed the state vector and its derivatives at many points in time. For a problem with multiple initial conditions, the history is stored as a four-dimensional array. The four indices correspond respectively to","category":"page"},{"location":"","page":"Home","title":"Home","text":"the component of the state vector or derivative,\nthe derivative order (with a 1j factor),\nthe timestep index,\nand the initial condition.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Therefore, given a 4D array history, history[i,j,k,l] gives the  i-th component of the real part (for i leq N_s) of the j-th derivative (divided by j) of the state vector corresponding to the l-th initial condition after k-1 timesteps.","category":"page"},{"location":"function-index/","page":"Index","title":"Index","text":"Modules = [QuantumGateDesign]","category":"page"}]
}

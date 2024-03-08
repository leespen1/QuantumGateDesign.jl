#===============================================================================
Notes:

Use PyArray to wrap return numpy arrays

The fidelity error computed by qutip does not appear to be the regular fidelity error?
I am finding conflicting documentation on whether to square the dimension, and
whether 

# TIME DEPENDENT OPERATORS IN QUTIP

For time-dependent problems, H and c_ops can be a specified in a nested-list format where each element in the list is a list of length 2, containing an operator (qutip.qobj) at the first element and where the second element is either a string (list string format), a callback function (list callback format) that evaluates to the time-dependent coefficient for the corresponding operator, or a NumPy array (list array format) which specifies the value of the coefficient to the corresponding operator for each value of t in tlist.

Alternatively, H (but not c_ops) can be a callback function with the signature f(t, args) -> Qobj (callback format), which can return the Hamiltonian or Liouvillian superoperator at any point in time. If the equation cannot be put in standard Lindblad form, then this time-dependence format must be used.

# SOLVERS

sesolve solves schrodinger's equation
mesolve solves the Linblad master equation (or schrodinger's equation, if no collapse operators are given)

essolve exists (but will be deprecated in v5.0), and does the ODE via
EXPONENTIAL SERIES. Do they mean like a taylor expansion, similar to our method?

Looking at the source code for sesolve, looks like it uses SciPy to integrate,
and to that end uses zvode, which is written in fortran. So I think solving via
sesolve should be accurate (as long as there is no time-dependence, or if a
callback function is used) and fast. But presumably this method is *not* being
used for the GRAPE pulse optimization. But perhaps it is being used to calculate
the fidelity error? That could be the discrepancy.

Can do time-dependence using operators and callback functions. Will
PythonCall/JuliaCall allow me to use my julia controls?

Alternatively, could sample the julia controls and do the hamiltonian/amp list option.

Can also use Cython strings
===============================================================================#

using CondaPkg
using PythonCall
using QuantumGateDesign

CondaPkg.add("numpy")
CondaPkg.add("qutip")

np = pyimport("numpy")
qutip = pyimport("qutip")
pulseoptim = pyimport("qutip.control.pulseoptim")

convert_to_numpy(A::AbstractArray{<: Real}) = np.array(A, dtype=np.float64)
# Make sure complex matrices are stored in the correct format
# Note that numpy.complex128 stores the real and imaginary parts as 64-bit
# floating points, which is equivalent to Julia's ComplexF64.
convert_to_numpy(A::AbstractArray{<: Complex}) = np.array(A, dtype=np.complex128)

Qobj(A::AbstractArray) = qutip.Qobj(convert_to_numpy(A))

"""
Get the qutip.Qobj version of an array (which is used to store operators, kets,
etc)
"""
function Qobj(A_real::AbstractArray{<: Real}, A_imag::AbstractArray{<: Real})
    return Qobj(A_real .+ (im .* A_imag))
end

"""
Qobj instances contain data as a sparse numpy array. PythonCall can handle dense
numpy arrays, but not sparse ones, so we convert to dense.

Once we get the dense, numpy matrix, we call PyArray on it so we can deal with
it using Julia indices.
"""
function unpack_Qobj(qobj)
    return PyArray(qobj.data.todense())
end

"""
Time dynamics in qutip:
https://qutip.org/docs/latest/guide/dynamics/dynamics-time.html

Here, I will assume there is no time-dependence in the hamiltonian
"""
function simulate_prob_no_control(prob::SchrodingerProb; atol=1e-15, rtol=1e-15, kwargs ...)

    H = Qobj(prob.system_sym, prob.system_asym)


    #=
    # Get python list of q-objects for each control operator
    @assert length(prob.sym_operators) == length(prob.asym_operators) == prob.N_operators
    H_c = pylist([
        Qobj(prob.sym_operators[i], prob.sym_operators[i]) 
        for i in 1:prob.N_operators
    ])
    =#

    # Initial Condition (if it's a matrix, I think qutip needs it to be
    # unitary, so we may not be able to have guard levels)
    U_0 = Qobj(prob.u0, prob.v0)

    # Times
    tlist = np.linspace(0.0, prob.tf, 1+prob.nsteps)

    # Solve Schrodinger's Equation
    options = qutip.Options(atol=atol, rtol=rtol)
    solverResult = qutip.sesolve(H, U_0, tlist, options=options)
    # solverResult is a python list of Qobjs. We turn the Qobj's into column
    # vectors/matrices, then hstack them to get the 'history' array (complex).
    history_complex = reduce(hcat, unpack_Qobj.(solverResult.states))
    # Convert to 'tall' real-valued complex format
    return complex_to_real(history_complex)
    

    # Attempts at pulse optimization, not just simulation
    #=
    U_targ = Qobj(target)
    evo_time = prob.tf

    # Creates pulse optimizer. Optimization can be run with the `.run_optimizer()` method
    return pulseoptim.optimize_pulse(H_d, H_c, U_0, U_targ, num_tslots=num_tslots, evo_time=evo_time)
    #return pulseoptim.create_pulse_optimizer(H_d, H_c, U_0, U_targ, num_tslots=num_tslots, evo_time=evo_time)
    =#
end


# Example of how to pass on kwargs
function test(A; kwargs ...)
    return np.array(A; kwargs ...)
end

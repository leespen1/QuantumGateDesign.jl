module OrdinaryDiffEqInterface

println("Loading OrdinaryDiffEqInterface")

using QuantumGateDesign
using OrdinaryDiffEq

#===============================================================================

Extension for interfacing with DifferentialEquations package.

Given a `prob::SchrodingerProb`, `controls`, `pcof`, and initial condition `w0`, do:
```
p = ODEparams(prob, controls, pcof)
ode_prob = ODEProblem(ODE_f!, w0, (0, tf), p)
```

Or you can directly do:
```
ode_prob = construct_ODEProb(prob, controls, pcof)
```
But beware that in the second case, updating `tf` in `prob` will not update
`tspan` in `ode_prob`. But on the bright side, updating `pcof` in-place will
update it in `ode_prob`.

And then you can solve evolve the system using whatever methods
DifferentialEquations.jl allows! 

===============================================================================#

"""
Need to gather all the parameters I will need to pass to ODE solver in one struct.
"""
mutable struct ODEparams{Tschrodingerprob, Tcontrols}
    prob::Tschrodingerprob 
    controls::Tcontrols
    pcof::Vector{Float64}
end

"""
Compute derivative
"""
function ODE_f!(du::V, u::V, p, t) where {V <: AbstractVector}
    real_system_size = length(u)
    complex_system_size = div(real_system_size, 2)

    u_real = view(u, 1:complex_system_size)
    u_imag = view(u, complex_system_size+1:real_system_size)

    du_real = view(du, 1:complex_system_size)
    du_imag = view(du, complex_system_size+1:real_system_size)

    du .= 0
    apply_hamiltonian!(du_real, du_imag, u_real, u_imag, p.prob, p.controls, t, p.pcof)

    return nothing
end

"""
Matrix version
"""
function ODE_f!(du::M, u::M, p, t) where {M <: AbstractMatrix}
    for i in 1:size(u, 2)
        u_vec  = view(u,  :, i)
        du_vec = view(du, :, i)
        ODE_f!(du_vec, u_vec, p, t)
    end

    return nothing
end

function QuantumGateDesign.construct_ODEProb(prob::SchrodingerProb, controls, pcof::AbstractVector{<: Real})
    p = ODEparams(prob, controls, pcof)
    w0 = vcat(prob.u0, prob.v0)
    tf = prob.tf

    ode_prob = ODEProblem(ODE_f!, w0, (0, tf), p)
    
    return ode_prob
end


"""
For testing that ODE_f! computes the same first derivatives as we do.
"""
function test_agreement(p::ODEparams)
    history = eval_forward(p.prob, p.controls, p.pcof, order=2)

    du = zeros(p.prob.real_system_size)
    u  = zeros(p.prob.real_system_size)

    N_tests = size(history, 3)*size(history, 4)
    N_tests_passed = 0

    for initial_condition_index in 1:size(history, 4)
        for step in 1:size(history, 3)
            u .= history[:, 1, step, initial_condition_index]
            ODE_f!(du, u, p, t)
            if (du == history[:, 2, step, initial_condition_index])
                N_tests_passed += 1
            end
        end
    end

    println("Tests Passed: ", N_tests_passed, " / ", N_tests)
    return N_tests == N_tests_passed
end

end # module

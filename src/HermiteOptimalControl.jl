module HermiteOptimalControl

using LinearAlgebra, LinearMaps, IterativeSolvers, ForwardDiff, Plots

export SchrodingerProb, eval_forward, eval_forward_forced
export eval_grad_finite_difference, eval_grad_forced, discrete_adjoint

include("hermite.jl")

end # module HermiteOptimalControl

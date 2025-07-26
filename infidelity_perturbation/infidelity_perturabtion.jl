using LinearAlgebra, Statistics

"""
Given a state vector and a number epsilon, randomly generate another state
vector with a fidelity around epsilon between the two states.

The process may also be useful in characterizing the epsilon squared behavior
of the infidelity errors.

In fact, this analysis could be a useful part of the paper.
"""
function random_state_with_fidelity(x::Vector{<: Number}, ε::Real)
    @assert 0 ≤ ε ≤ 1 "Fidelity ε must be between 0 and 1"
    n = length(x)
    
    # Generate a random Haar-distributed state
    z = randn(ComplexF64, n)
    z /= norm(z)

    # Remove the component of z along x to ensure orthogonality
    z′ = z - (dot(x, z) * x)
    if norm(z′) > 1e-10  # Avoid division by zero
        z′ /= norm(z′)
    else
        # If z′ is nearly zero, pick another random vector
        return random_state_with_fidelity(x, ε)
    end

    # Construct y with the correct fidelity
    y = sqrt(ε) * x + sqrt(1 - ε) * z′

    return y
end



function randunit(n::Integer)
    # Using real instead of complex here made the 1/√N factor disappear. Interesting.
    vector = randn(ComplexF64, n) 
    vector ./= norm(vector)
    return vector
end

N = 1000
Nsamples = 1000
fidelity(x,y) = abs(x'*y)^2
infidelity(x,y) = 1 - fidelity(x,y)

#
# This experiment shows that in general, the error in the infidelity scales with
# the realative error in the numerical solution at the final time.
#
# A remaining question is: is this still the case for psif that is already
# pretty close to the target? That is a very small minority of the points
# tested in this example.
#

#=
Nsamples = 1_000
for eps in 10.0 .^ (-1:-1:-10)
    numerical_errors = Float64[]
    for i in 1:Nsamples
        target = randunit(N)
        psif = randunit(N)
        perturbartion = eps .* randunit(N)
        actual_fidelity = fidelity(target, psif)
        numerical_fidelity = fidelity(target, psif + perturbartion)
        numerical_error = abs(numerical_fidelity - actual_fidelity)
        push!(numerical_errors, numerical_error)
    end
    println("Epsilon = ", eps)
    println("\tMean Error = ", mean(numerical_errors))
    println("\tStddev Error = ", std(numerical_errors))
end

#
# Test for when the target and final state are close
#

for eps1 in 10.0 .^ (-1:-1:-5)
  println("\nEpsilon1 = ", eps1)
  for eps2 in 10.0 .^ (-1:-1:-5)
    numerical_errors = Float64[]
    for i in 1:Nsamples
        target = randunit(N)
        psif = target + eps1 * randunit(N)
        perturbartion = eps2 .* randunit(N)
        actual_fidelity = fidelity(target, psif)
        numerical_fidelity = fidelity(target, psif + perturbartion)
        numerical_error = abs(numerical_fidelity - actual_fidelity)
        push!(numerical_errors, numerical_error)
    end
    println("\tEpsilon2 = ", eps2)
    println("\t\tMean Error = ", mean(numerical_errors))
    println("\t\tStddev Error = ", std(numerical_errors))
  end
end
=#

#
# Final test should be when the fidelity specifically is close, not just the state
# (although the state should be more restrictive)
#

for target_infidelity in 10.0 .^ (-1:-2:-5)
  println("\nϵ = ", target_infidelity)
  for eps in 10.0 .^ (-1:-2:-5)
    numerical_errors = Float64[]
    for i in 1:Nsamples
        target = randunit(N)
        psif = random_state_with_fidelity(target, target_infidelity)
        perturbartion = eps .* randunit(N)
        actual_fidelity = fidelity(target, psif)
        numerical_fidelity = fidelity(target, psif + perturbartion)
        numerical_error = abs(numerical_fidelity - actual_fidelity)
        push!(numerical_errors, numerical_error)
    end
    println("\tδ = ", eps)
    println("\t\tMean Error = ", mean(numerical_errors))
    println("\t\t2δ√ϵ = ", 2*sqrt(target_infidelity)*eps)
    println("\t\t2δ√ϵ/√N = ", 2*sqrt(target_infidelity)*eps/sqrt(N))
    # This estimate seems to be pretty accurate 
    println("\t\tStddev = ", std(numerical_errors))
  end
end

nothing


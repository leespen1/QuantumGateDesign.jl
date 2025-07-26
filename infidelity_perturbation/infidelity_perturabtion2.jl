using LinearAlgebra, Statistics

N = 100
Nsamples = 1000
E = 100

"""
Given a state vector and a number epsilon, randomly generate another state
vector with a fidelity around epsilon between the two states.

"""
function random_state_with_fidelity(x::Vector{<: Number}, e::Real)
    @assert 0 ≤ e ≤ 1 "Fidelity ε must be between 0 and 1"
    n = length(x)
    @assert n > 1 "Dimension must be greater than 1 for orthogonality to make sense"
    
    # Generate a random Haar-distributed state
    z = randn(ComplexF64, n)
    z /= norm(z)

    # Remove the component of z along x to ensure orthogonality
    zc = z - (dot(x, z) * x)
    if norm(zc) > 1e-10  # Avoid division by zero
        zc /= norm(zc)
    else
        # If z′ is nearly zero, pick another random vector
        return random_state_with_fidelity(x, e)
    end

    # Construct y with the correct fidelity
    y = sqrt(e) * x + sqrt(1 - e) * zc

    return y
end

function random_state_with_fidelity(x::Matrix{<: Number}, ε::Real)
    ys = [random_state_with_fidelity(x[:,i], ε) for i in 1:size(x,2)]
    return reduce(hcat, ys)
end



function randunit(n::Integer, e::Integer)
    # Using real instead of complex here made the 1/√N factor disappear. Interesting.
    vectors = [randn(ComplexF64, n) for i in 1:e ]
    for i in 1:length(vectors)
        vectors[i] ./= norm(vectors[i])
    end
    return hcat(vectors...)
end

function fidelity(x, y)
    return abs(dot(x, y))^2 / size(x,2)^2
end

function infidelity(x,y)
    return 1 - fidelity(x,y)
end



for target_fidelity in 10.0 .^ (-1:-2:-5)
  println("\nϵ = ", target_fidelity)
  for eps in 10.0 .^ (-1:-2:-5)
    numerical_errors = Float64[]
    for i in 1:Nsamples
        target = randunit(N, E)
        psif = random_state_with_fidelity(target, target_fidelity)
        perturbartion = eps .* randunit(N, E)
        actual_fidelity = fidelity(target, psif)
        numerical_fidelity = fidelity(target, psif + perturbartion)
        numerical_error = abs(numerical_fidelity - actual_fidelity)
        push!(numerical_errors, numerical_error)
    end
    println("\tδ = ", eps)
    println("\t\tMean Error =\t", mean(numerical_errors))
    println("\t\t2δ√ϵ/ √N = \t", 2*sqrt(target_fidelity)*eps/sqrt(N))
    println("\t\t*2δ√ϵ/ √N*√E = \t", 2*sqrt(target_fidelity)*eps/ (sqrt(N)*sqrt(E)))
    println("\t\t2δ√ϵ/ √N*E = \t", 2*sqrt(target_fidelity)*eps / (sqrt(N)*E))
    println("\t\t2δ√ϵ/ √N*E*√E = \t", 2*sqrt(target_fidelity)*eps/ (sqrt(N)*E^1.5))
    println("\t\t2δ√ϵ/ √N*E^2 = \t", 2*sqrt(target_fidelity)*eps/ (sqrt(N)*E^2))
    # This estimate seems to be pretty accurate 
    println("\t\tStddev= \t", std(numerical_errors))
  end
end

nothing


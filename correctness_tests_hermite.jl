function rabi_osc(Ω::ComplexF64=1.0+0.0im, tf::Float64=NaN; nsteps::Int64=100)
    #Ks::Matrix{Float64} = [0 0; 0 1]
    Ks::Matrix{Float64} = [0 0; 0 0] # Rotating frame
    Ss::Matrix{Float64} = [0 0; 0 0]
    a_plus_adag::Matrix{Float64} = [0 1; 1 0]
    a_minus_adag::Matrix{Float64} = [0 1; -1 0]
    p(t,α) = real(Ω)
    q(t,α) = imag(Ω)
    u0::Vector{Float64} = [1,0]
    v0::Vector{Float64} = [0,0]
    # Default time to pi/2 pulse
    if isnan(tf)
        tf = pi/(2*abs(Ω))
    end
    return SchrodingerProb(Ks,Ss,a_plus_adag,a_minus_adag,p,q,u0,v0,tf,nsteps)
end


function convergence_test()
    α = 1.0 # Doesn't matter, control doesn't depend on α

    # Use 10000 timesteps for "true solution"
    prob10000 = rabi_osc(nsteps=10000)
    history10000 = eval_forward(prob10000, α)
    prob200 = rabi_osc(nsteps=200)
    history200 = eval_forward(prob200, α)
    prob100 = rabi_osc(nsteps=100)
    history100 = eval_forward(prob100, α)

    # Change stride so all histories have same shape
    history10000 = history10000[:,1:100:end]
    history200 = history200[:,1:2:end]

    error100 = abs.(history10000 - history100)
    error200 = abs.(history10000 - history200)

    log_ratio = log2.(error100 ./ error200)
    println("Log₂ of error ratio between 100 step and 200 step methods (10000 steps used for true value)")
    display(log_ratio)
    return log_ratio
end

# Use analytic solution
function analytic_rabi(r,θ)
end


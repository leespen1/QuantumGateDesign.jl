"""
Construct a 'SchrodingerProb' corresponding to a Rabi Oscillator, with default
time corresponding to a pi/2 pulse.
"""
function rabi_osc(Ω::ComplexF64=1.0+0.0im, tf::Float64=NaN; nsteps::Int64=100)
    #Ks::Matrix{Float64} = [0 0; 0 1]
    Ks::Matrix{Float64} = [0 0; 0 0] # Rotating frame
    Ss::Matrix{Float64} = [0 0; 0 0]
    p(t,α) = α*real(Ω)
    q(t,α) = α*imag(Ω)
    dpdt(t,α) = 0.0
    dqdt(t,α) = 0.0
    dpda(t,α) = real(Ω)
    dqda(t,α) = imag(Ω)
    d2p_dta(t,α) = 0.0
    d2q_dta(t,α) = 0.0
    u0::Vector{Float64} = [1,0]
    v0::Vector{Float64} = [0,0]
    # Default time to pi/2 pulse
    if isnan(tf)
        tf = pi/(2*abs(Ω))
    end
    return SchrodingerProb(Ks,Ss,
                           p,q,dpdt,dqdt,dpda,dqda,d2p_dta,d2q_dta,
                           u0,v0,tf,nsteps)
end



"""
Evolve intitial condition according to rabi oscillation H_c = Ωa + Ω̄a† (in rotating frame).
Input initial condition is *real-valued*, with the first half of the vector
being the real part and the second half being the negative imaginary part.
Output is *real-valued*, in the same representation. 

Assumed to be 2-level system.
"""
function analytic_rabi(Ω::ComplexF64,t::Float64,ψ0::Vector{Float64})
    @assert length(ψ0) == 4
    ψ0_complex = ψ0[1+0:1+1] - im*ψ0[1+2:1+3]
    r = abs(Ω)
    θ = angle(Ω)
    ψt_0 = cos(r*t)ψ0[1+0] + (sin(θ) - im*cos(θ)*sin(r*t))ψ0[1+1]
    ψt_1 = -(sin(θ) + im*cos(θ)*sin(r*t))ψ0[1+0] + cos(r*t)ψ0[1+1] 

    ψt_real = [real(ψt_0), real(ψt_1), -imag(ψt_0), -imag(ψt_1)]
    return ψt_real
end


function rabi_convergence_test(Ω::ComplexF64=1.0+0.0im; α=1.0, analytic=true, order=2)

    prob200 = rabi_osc(Ω, nsteps=200)
    history200 = eval_forward(prob200, α, order=order)
    # Change stride to match 100 timestep result
    history200 = history200[:,1:2:end]
    prob100 = rabi_osc(Ω, nsteps=100)
    history100 = eval_forward(prob100, α, order=order)

    if analytic
        # Use analytic true solution
        ψ0 = vcat(prob100.u0, prob100.v0)
        tf = prob100.tf
        dt = tf/100
        t = 0
        history_true = zeros(4,101)
        history_true[:,1] = ψ0
        t += dt
        for i in 1:100
            history_true[:,1+i] = analytic_rabi(Ω,t,ψ0)
            t += dt
        end
    else
        # Use 10000 timesteps for "true solution"
        prob10000 = rabi_osc(nsteps=10000, order=order)
        history_true = eval_forward(prob10000, α)
        history_true = history_true[:,1:100:end]
    end

    error100 = abs.(history_true - history100)
    error200 = abs.(history_true - history200)

    log_ratio = log2.(error100 ./ error200)
    println("Log₂ of error ratio between 100 step and 200 step methods")
    if analytic
        println("(Analytic solution used for 'true' value)")
    else
        println("(10000 steps used for true value)")
    end


    display(log_ratio)
    return log_ratio, error100, error200
end

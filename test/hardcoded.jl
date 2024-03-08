# dw/dt = θcos(t)[0 1;-1 0]w
# 0 ≤ t ≤ pi/2 (end time chosen so that cos(tf) = 0)
# Take one timestep, Δt = tf = pi/2

using LinearAlgebra: dot

Hc = [0 1;-1 0]

function final_state(u0, v0, θ)
    w0 = vcat(u0, v0)
    wf = w0 + (θ^2 * pi/4)*Hc*w0
    return wf
end

function final_state_partial(u0, v0, θ)
    w0 = vcat(u0, v0)
    dwf_dθ = (2*θ*pi/4)*Hc*w0
    return dwf_dθ
end

function guard_penalty_partial(u0, v0, θ)
    wf = final_state(u0, v0, θ)
    dwf_dθ = final_state_partial(u0, v0, θ)

    return dot(wf, dwf_dθ)
end



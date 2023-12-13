struct SinCosControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    frequency::Float64
    function SinCosControl(tf::Float64, frequency::Float64=1.0)
        N_coeff = 0
        new(N_coeff, tf, frequency)
    end
end

function eval_p(control::SinCosControl, t::Real, pcof::AbstractVector{Float64})
    return sin(t*control.frequency)
end


function eval_q(control::SinCosControl, t::Real, pcof::AbstractVector{Float64})
    return cos(t*control.frequency)
end

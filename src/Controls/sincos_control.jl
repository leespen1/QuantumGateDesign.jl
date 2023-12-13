struct SinCosControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    frequency::Float64
    function SinCosControl(tf::Real, frequency::Real=1.0)
        tf_f64 = convert(Float64, tf)
        frequency_f64 = convert(Float64, frequency)
        N_coeff = 0
        new(N_coeff, tf_f64, frequency_f64)
    end
end

function eval_p(control::SinCosControl, t::Real, pcof::AbstractVector{<: Real})
    return sin(t*control.frequency)
end


function eval_q(control::SinCosControl, t::Real, pcof::AbstractVector{<: Real})
    return cos(t*control.frequency)
end

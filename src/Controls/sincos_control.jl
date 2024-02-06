"""
Control with 2 coefficients. One to control amplitude of sin, one to control
amplitude of cos.
"""
struct SinCosControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    frequency::Float64
    function SinCosControl(tf::Real; frequency::Real=1.0)
        tf_f64 = convert(Float64, tf)
        frequency_f64 = convert(Float64, frequency)
        N_coeff = 2
        new(N_coeff, tf_f64, frequency_f64)
    end
end

function eval_p(control::SinCosControl, t::Real, pcof::AbstractVector{<: Real})
    return sin(t*control.frequency)*pcof[1]
end


function eval_q(control::SinCosControl, t::Real, pcof::AbstractVector{<: Real})
    return cos(t*control.frequency)*pcof[2]
end

"""
Control with 2 coefficients. One to control amplitude of sin, one to control
amplitude of each sin wave.

Really, this is better than SinCos, because the controls should all start at
zero. (they should really end at zero too, but whatever)
"""
struct SinControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    frequency::Float64
    function SinControl(tf::Real; frequency::Real=1.0)
        tf_f64 = convert(Float64, tf)
        frequency_f64 = convert(Float64, frequency)
        N_coeff = 2
        new(N_coeff, tf_f64, frequency_f64)
    end
end

function eval_p(control::SinControl, t::Real, pcof::AbstractVector{<: Real})
    return sin(t*control.frequency)*pcof[1]
end


function eval_q(control::SinControl, t::Real, pcof::AbstractVector{<: Real})
    return sin(t*control.frequency)*pcof[2]
end

struct CosControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    frequency::Float64
    function CosControl(tf::Real; frequency::Real=1.0)
        tf_f64 = convert(Float64, tf)
        frequency_f64 = convert(Float64, frequency)
        N_coeff = 2
        new(N_coeff, tf_f64, frequency_f64)
    end
end

function eval_p(control::CosControl, t::Real, pcof::AbstractVector{<: Real})
    return cos(t*control.frequency)*pcof[1]
end


function eval_q(control::CosControl, t::Real, pcof::AbstractVector{<: Real})
    return cos(t*control.frequency)*pcof[2]
end

struct SquaredAmpCosControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    frequency::Float64
    function SquaredAmpCosControl(tf::Real; frequency::Real=1.0)
        tf_f64 = convert(Float64, tf)
        frequency_f64 = convert(Float64, frequency)
        N_coeff = 2
        new(N_coeff, tf_f64, frequency_f64)
    end
end

function eval_p(control::SquaredAmpCosControl, t::Real, pcof::AbstractVector{<: Real})
    return cos(t*control.frequency)*(pcof[1])^2
end


function eval_q(control::SquaredAmpCosControl, t::Real, pcof::AbstractVector{<: Real})
    return cos(t*control.frequency)*(pcof[2])^2
end

struct SingleSymCosControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    frequency::Float64
    function SingleSymCosControl(tf::Real; frequency::Real=1.0)
        tf_f64 = convert(Float64, tf)
        frequency_f64 = convert(Float64, frequency)
        N_coeff = 1
        new(N_coeff, tf_f64, frequency_f64)
    end
end

function eval_p(control::SingleSymCosControl, t::Real, pcof::AbstractVector{<: Real})
    return cos(t*control.frequency)*pcof[1]
end


function eval_q(control::SingleSymCosControl, t::Real, pcof::AbstractVector{<: Real})
    return 0.0
end

struct ZeroControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
end

function eval_p(control::ZeroControl, t::Real, pcof::AbstractVector{<: Real})
    return 0
end

function eval_q(control::ZeroControl, t::Real, pcof::AbstractVector{<: Real})
    return 0
end

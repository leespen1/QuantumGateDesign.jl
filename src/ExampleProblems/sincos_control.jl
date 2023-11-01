#=
#
# A pulse using sines for p and cosines for q
# May or may not be useful (check out that control CRAB used, I should do that)
# Main purpose is to have infinitely smooth controls, for testing numerical
# convergence.
#
=#
function sincos_control(N_coeff_per_control)

    function sin_pulse(t, pcof)
        result = 0.0
        for k in 1:N_coeff_per_control
            # Use k-th principal frequency, pcof adjusts strength
            result += pcof[k]*sin(k*t)
        end
        return result
    end

    function sin_pulse_derivative(t, pcof)
        result = 0.0
        for k in 1:N_coeff_per_control
            # Use k-th principal frequency, pcof adjusts strength
            result += k*pcof[k]*cos(k*t)
        end
        return result
    end

    function cos_pulse(t, pcof)
        result = 0.0
        for k in 1:N_coeff_per_control
            result += pcof[N_coeff_per_control+k]*cos(k*t)
        end
        return result
    end

    function cos_pulse_derivative(t, pcof)
        result = 0.0
        for k in 1:N_coeff_per_control
            result += -k*pcof[N_coeff_per_control+k]*sin(k*t)
        end
        return result
    end

    N_derivatives = 4

    return Control([sin_pulse, sin_pulse_derivative], [cos_pulse, cos_pulse_derivative], 2*N_coeff_per_control)
end

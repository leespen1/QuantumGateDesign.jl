struct RichardsonExtrapolation{T}
    order::Int64
    abs_err_L1::Float64
    abs_err_L2::Float64
    rel_err_L1::Float64
    rel_err_L2::Float64
    abs_err_Linf::Float64
    sol::T
    err::T
    """
    Given numerical approximations of `A` using a method of order `order` and
    stepsizes `h` and `2h`, use Richardson extrapolation to obtain a higher-order
    approximation of `A`, and a higher-order estimate of the error in `Aₕ`. 
    """
    function RichardsonExtrapolation(Aₕ, A₂ₕ, order::Integer)
        sol = ((2^order)*Aₕ - A₂ₕ)/(2^order-1)
        err = sol - Aₕ
        abs_err_L1 = norm(err, 1)
        abs_err_L2 = norm(err, 2)
        rel_err_L1 = abs_err_L1 / norm(sol, 1)
        rel_err_L2 = abs_err_L2 / norm(sol, 2)
        abs_err_Linf = norm(err, Inf)
        new{typeof(sol)}(order, abs_err_L1, abs_err_L2, rel_err_L1, rel_err_L2, 
                         abs_err_Linf, sol, err)
    end
end

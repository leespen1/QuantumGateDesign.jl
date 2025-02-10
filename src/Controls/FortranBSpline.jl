const fortrain_lib_str = Base.Filesystem.dirname(pathof(@__MODULE__)) * "/Controls/bspline_lib.so"

"""
Considering adding a "current_t", which would check if the t is the current t before
evaluating. Because the fortran subroutines involve only the basis functions, the

Also, because Fortran is also column major, I think I can det away with
providing a large output array, which will only be filled partially if I
evaluate fewer derivatives than necessary.

The evaluations will be done for the "clamped" bspline. The end knots are
considered repeated by fortran, we don't have to repeat them ourselves.

A clamped bspline with order k and N knots should have N-k basis functions
"""
struct FortranBSplineControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    N_basis_functions::Int64
    N_knots::Int64
    N_distinct_knots::Int64
    degree::Int64
    bspline_order::Int64
    knot_vector::Vector{Float64}
    work_array::Matrix{Float64}
    output_array::Matrix{Float64}
    jbsplvd::Base.RefValue{Int64}
    deltal::Vector{Float64}
    deltar::Vector{Float64}
    #function FortranBSplineControl(degree::Integer, N_distinct_knots::Integer, tf::Real)
    function FortranBSplineControl(degree::Integer, N_basis_functions::Integer, tf::Real)
        degree = convert(Int64, degree)
        tf = convert(Float64, tf)
        N_basis_functions = convert(Int64, N_basis_functions)

        N_coeff = 2*N_basis_functions

        order = degree+1
        N_knots = N_basis_functions + order
        N_distinct_knots = N_knots - 2*(order-1) 
        #=
        order = degree+1
        N_knots = N_distinct_knots + 2*(order-1)
        N_basis_functions = N_knots - order
        N_coeff = 2*N_basis_functions
        =#

        N_basis_functions = N_knots - order

        work_array = zeros(order, order)
        output_array = zeros(order, 20) # just make it large enough, see if that works

        order = degree+1
        knot_vector = Vector(LinRange(0, 1, N_distinct_knots))
        # First and last knots should be repeated 'order' times
        knot_vector = vcat(
            repeat([knot_vector[1]], order-1),
            knot_vector,
            repeat([knot_vector[end]], order-1)
        )

        jbsplvd = Ref(1)
        deltal = fill(NaN, 20)
        deltar = fill(NaN, 20)

        new(N_coeff, tf, N_basis_functions, N_knots, N_distinct_knots,
            degree, order, knot_vector, work_array, output_array, jbsplvd, deltal, deltar)
    end
end


"""
Could do a check for t_scaled = t_current, but first let's see if this is fast or not 

For now, let's just check that results agree with BasicBSplineControl

In general, shouldn't call this this way, since it repeats lower order derivative
computations. Better to use fill_p_vec!
"""
function eval_p_derivative(control::FortranBSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, order+1)
    
    pcof_offset = floor(Int64, t_scaled*(control.N_distinct_knots-1) + 1)
    pcof_offset = min(pcof_offset, control.N_distinct_knots-1)

    val = 0.0
    for i in 0:control.bspline_order-1
        val += pcof[pcof_offset+i] * control.output_array[1+i,1+order]
    end
    val /= control.tf ^ order
    return val
end

function fill_p_vec!(
        vals_vec::AbstractVector{<: Real}, control::FortranBSplineControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    # calculate derivatives up to (but not including) nderiv
    nderiv = length(vals_vec)  
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, nderiv)

    pcof_offset = floor(Int64, t_scaled*(control.N_distinct_knots-1) + 1)
    pcof_offset = min(pcof_offset, control.N_distinct_knots-1)

    for derivative_order in 0:nderiv-1
        val = 0.0
        for i in 0:control.bspline_order-1
            val += pcof[pcof_offset+i] * control.output_array[1+i,1+derivative_order]
        end
        # Chain rule and getting 1/j! factor in p⁽ʲ⁾(t)/j!
        vals_vec[1+derivative_order] = val / ((control.tf ^ derivative_order)*factorial(derivative_order))
    end
    return vals_vec
end
    
function eval_q_derivative(control::FortranBSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, order+1)
    
    pcof_offset = floor(Int64, t_scaled*(control.N_distinct_knots-1) + 1)
    pcof_offset = min(pcof_offset, control.N_distinct_knots-1)
    pcof_offset += div(control.N_coeff, 2)

    val = 0.0
    for i in 0:control.bspline_order-1
        val += pcof[pcof_offset+i] * control.output_array[1+i,1+order]
    end
    val /= control.tf ^ order
    if isnan(val) #REMOVETHIS
        println("order=",order, ",\tt=",t, "\tt_scaled=", t_scaled)
    end
    return val
end

function fill_q_vec!(
        vals_vec::AbstractVector{<: Real}, control::FortranBSplineControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    # calculate derivatives up to (but not including) nderiv
    nderiv = length(vals_vec)  
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, nderiv)

    pcof_offset = floor(Int64, t_scaled*(control.N_distinct_knots-1) + 1)
    pcof_offset = min(pcof_offset, control.N_distinct_knots-1)
    pcof_offset += div(control.N_coeff, 2)

    for derivative_order in 0:nderiv-1
        val = 0.0
        for i in 0:control.bspline_order-1
            val += pcof[pcof_offset+i] * control.output_array[1+i,1+derivative_order]
        end
        # Chain rule and getting 1/j! factor in q⁽ʲ⁾(t)/j!
        vals_vec[1+derivative_order] = val / ((control.tf ^ derivative_order)*factorial(derivative_order))
    end
    return vals_vec
end

function eval_grad_p_derivative!(
        grad::AbstractVector{Float64}, control::FortranBSplineControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Integer
    )
    # calculate derivatives up to (but not including) nderiv
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, order+1)

    pcof_offset = floor(Int64, t_scaled*(control.N_distinct_knots-1) + 1)
    pcof_offset = min(pcof_offset, control.N_distinct_knots-1)

    # Control is linear in the pcof coefficients
    grad .= 0
    for i in 0:control.bspline_order-1
        # Chain rule (no 1/j! factor needed)
        grad[pcof_offset+i] = control.output_array[1+i,1+order] / (control.tf ^ order)
    end

    return grad
end

function eval_grad_q_derivative!(
        grad::AbstractVector{Float64}, control::FortranBSplineControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Integer
    )
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, order+1)

    pcof_offset = floor(Int64, t_scaled*(control.N_distinct_knots-1) + 1)
    pcof_offset = min(pcof_offset, control.N_distinct_knots-1)
    pcof_offset += div(control.N_coeff, 2)

    # Control is linear in the pcof coefficients
    grad .= 0
    for i in 0:control.bspline_order-1
        # Chain rule (no 1/j! factor needed)
        grad[pcof_offset+i] = control.output_array[1+i,1+order] / (control.tf ^ order)
    end

    return grad
end

function fill_grad_p_mat!(
        grad_mat::AbstractMatrix{Float64}, control::FortranBSplineControl, t::Real,
        pcof::AbstractVector{<: Real}
    )
    # calculate derivatives up to (but not including) nderiv
    nderiv = size(grad_mat, 2)  
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, nderiv)

    pcof_offset = floor(Int64, t_scaled*(control.N_distinct_knots-1) + 1)
    pcof_offset = min(pcof_offset, control.N_distinct_knots-1)

    # Control is linear in the pcof coefficients
    grad_mat .= 0
    pow_tf = 1.0
    for k in 0:nderiv-1
        for i in 0:control.bspline_order-1
            # Chain rule (no 1/j! factor needed)
            grad_mat[pcof_offset+i,1+k] = control.output_array[1+i,1+k] / pow_tf
        end
        pow_tf *= control.tf
    end

    return grad_mat
end

function fill_grad_q_mat!(
        grad_mat::AbstractMatrix{Float64}, control::FortranBSplineControl, t::Real,
        pcof::AbstractVector{<: Real}
    )
    # calculate derivatives up to (but not including) nderiv
    nderiv = size(grad_mat, 2)  
    # calculate derivatives up to (but not including) nderiv
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, nderiv)

    pcof_offset = floor(Int64, t_scaled*(control.N_distinct_knots-1) + 1)
    pcof_offset = min(pcof_offset, control.N_distinct_knots-1)
    pcof_offset += div(control.N_coeff, 2)

    # Control is linear in the pcof coefficients
    grad_mat .= 0
    pow_tf = 1.0
    for k in 0:nderiv-1
        for i in 0:control.bspline_order-1
            # Chain rule (no 1/j! factor needed)
            grad_mat[pcof_offset+i,1+k] = control.output_array[1+i,1+k] / pow_tf
        end
        pow_tf *= control.tf
    end

    return grad_mat
end


"""
Bezier degree elevations: 
    https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node13.html
B-spline degree elevation description: 
    https://pages.mtu.edu/~shene/COURSES/cs3621/LAB/curve/elevation.html
Knot Insertion and Removal for BSplines:
    https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node18.html

Degree elevation of a BSpline can be done by knot insertion:
1. Insert knots at internal knots until each segment of the B-spline is a bezier curve.
2. Perform degree elevation of the bezier curves (which also results in new
   control points, so pcof is updated)
3. "combining them together back to a single B-spline". Does combining them
    together mean knot removal? Because I want smoothness. So I can't have
    internal knots with multiplicity > 1. But I think the knot removal may be exact.
"""
function elevate_degree(control::FortranBSplineControl, pcof::AbstractVector{<: Real})
    
end



"""
From Netlib pppack

From  * a practical guide to splines *  by c. de Boor (7 may 92)    
calls bsplvb
calculates value and deriv.s of all b-splines which do not vanish at x

******  i n p u t  ******
  t     the knot array, of length left+k (at least)
  k     the order of the b-splines to be evaluated
  x     the point at which these values are sought
  left  an integer indicating the left endpoint of the interval of
        interest. the  k  b-splines whose support contains the interval
               (t(left), t(left+1))
        are to be considered.
  a s s u m p t i o n  - - -  it is assumed that
               t(left) .lt. t(left+1)
        division by zero will result otherwise (in  b s p l v b ).
        also, the output is as advertised only if
               t(left) .le. x .le. t(left+1) .
  nderiv   an integer indicating that values of b-splines and their
        derivatives up to but not including the  nderiv-th  are asked
        for. ( nderiv  is replaced internally by the integer  m h i g h
        in  (1,k)  closest to it.)

******  w o r k   a r e a  ******
  a     an array of order (k,k), to contain b-coeff.s of the derivat-
        ives of a certain order of the  k  b-splines of interest.

******  o u t p u t  ******
  dbiatx   an array of order (k,nderiv). its entry  (i,m)  contains
        value of  (m-1)st  derivative of  (left-k+i)-th  b-spline of
        order  k  for knot sequence  t , i=1,...,k, m=1,...,nderiv.

******  m e t h o d  ******
  values at  x  of all the relevant b-splines of order k,k-1,...,
  k+1-nderiv  are generated via  bsplvb  and stored temporarily in
  dbiatx .  then, the b-coeffs of the required derivatives of the b-
  splines of interest are generated by differencing, each from the pre-
  ceding one of lower order, and combined with the values of b-splines
  of corresponding order in  dbiatx  to produce the desired values .

"""
@inline function bsplvd!(t::Vector{Float64}, k::Int64, x::Float64, left::Int64,
        a::Matrix{Float64}, dbiatx::Matrix{Float64}, nderiv::Int64,
        jbsplvd::Ref{Int64}, deltal::Vector{Float64}, deltar::Vector{Float64})
    # jsplvd should be passed in as a reference, since I want it to be changed
    # by the program ()
    ccall(
        (:bsplvd_, fortrain_lib_str),
        Cvoid, # Return

        (Ref{Float64}, Ref{Int64}, Ref{Float64}, Ref{Int64},
         Ref{Float64}, Ref{Float64}, Ref{Int64}, 
         Ref{Int64}, Ref{Float64}, Ref{Float64}), # Argument Types

        t, Ref(k), Ref(x), Ref(left),
        a, dbiatx, Ref(nderiv),
        jbsplvd, deltal, deltar  # Arguments
    )
end


@inline function bsplvd!(control::FortranBSplineControl, x::Float64, nderiv::Int64)
    # It is assumed that x ∈ [0,1]
    #left::Int64 = floor(Int64, x*(control.N_distinct_knots-1) + 1)
    #left = min(left, control.N_distinct_knots-1)
    left = floor(Int64, x*(control.N_distinct_knots-1) + control.bspline_order)
    left = min(left, control.N_knots-control.bspline_order)
    #@assert control.knot_vector[left] < control.knot_vector[left+1]

    bsplvd!(control.knot_vector, control.bspline_order, x, left,
            control.work_array, control.output_array, nderiv, control.jbsplvd, control.deltal, control.deltar)
end




################################################################################

struct FortranBSpline
    N_basis_functions::Int64
    N_knots::Int64
    N_distinct_knots::Int64
    degree::Int64
    bspline_order::Int64
    current_t::Base.RefValue{Float64} # Should be between 0 and 1
    current_derivative_order::Base.RefValue{Int64}
    jbsplvd::Base.RefValue{Int64}
    knot_vector::Vector{Float64}
    work_array::Matrix{Float64}
    output_array::Matrix{Float64} # For storing the output of bsplvd
    deltal::Vector{Float64}
    deltar::Vector{Float64}
    function FortranBSpline(degree::Integer, N_basis_functions::Integer)
        degree = convert(Int64, degree)
        N_basis_functions = convert(Int64, N_basis_functions)

        order = degree+1
        N_knots = N_basis_functions + order
        N_distinct_knots = N_knots - 2*(order-1) 
        #=
        order = degree+1
        N_knots = N_distinct_knots + 2*(order-1)
        N_basis_functions = N_knots - order
        N_coeff = 2*N_basis_functions
        =#

        N_basis_functions = N_knots - order

        work_array = zeros(order, order)
        output_array = zeros(order, 20) # just make it large enough, see if that works

        order = degree+1
        knot_vector = Vector(LinRange(0, 1, N_distinct_knots))
        # First and last knots should be repeated 'order' times
        knot_vector = vcat(
            repeat([knot_vector[1]], order-1),
            knot_vector,
            repeat([knot_vector[end]], order-1)
        )

        jbsplvd = Ref(1)
        deltal = fill(NaN, 20)
        deltar = fill(NaN, 20)

        current_t = Ref(NaN)
        current_derivative_order = Ref(-1)

        new(N_basis_functions, N_knots, N_distinct_knots, degree, order,
            current_t, current_derivative_order, jbsplvd, knot_vector,
            work_array, output_array, deltal, deltar)
    end
end

@inline function bsplvd!(bspline::FortranBSpline, x::Float64, nderiv::Int64)
    # It is assumed that x ∈ [0,1]
    #left::Int64 = floor(Int64, x*(bspline.N_distinct_knots-1) + 1)
    #left = min(left, bspline.N_distinct_knots-1)
    left = floor(Int64, x*(bspline.N_distinct_knots-1) + bspline.bspline_order)
    left = min(left, bspline.N_knots-bspline.bspline_order)
    #@assert bspline.knot_vector[left] < bspline.knot_vector[left+1]

    bsplvd!(bspline.knot_vector, bspline.bspline_order, x, left,
            bspline.work_array, bspline.output_array, nderiv, bspline.jbsplvd, bspline.deltal, bspline.deltar)
end



@inline function update!(bspline::FortranBSpline, t::Real, nderiv::Integer)
    updated::Bool = false
    if (t != bspline.current_t[]) || (nderiv > bspline.current_derivative_order[])
        if !(0.0 <= t <= 1.0)
            throw(DomainError(t, "Attempted to update FortranBSpline with t outside range [0,1]."))
        end
        bspline.current_t[] = t
        bspline.current_derivative_order[] = nderiv
        # bsplvd!(a,b,N) evaluates derivatives upto, but not including order N
        bsplvd!(bspline, Float64(t), Int64(nderiv+1))
        updated = true
    end
    return updated
end



struct FortranBSplineControl2 <: AbstractControl
    N_coeff::Int64
    tf::Float64
    bspline::FortranBSpline
    function FortranBSplineControl2(bspline::FortranBSpline, tf::Real)
        N_coeff::Int64 = 2*bspline.N_basis_functions
        tf = convert(Float64, tf)
        new(N_coeff, tf, bspline)
    end
end


"""
Could do a check for t_scaled = t_current, but first let's see if this is fast or not 

For now, let's just check that results agree with BasicBSplineControl

In general, shouldn't call this this way, since it repeats lower order derivative
computations. Better to use fill_p_vec!
"""
function eval_derivative(control::FortranBSplineControl2, t::Real, pcof::AbstractVector{<: Real}, order::Int64, pcof_extra_offset::Integer)
    check_pcof_length(control, pcof)
    t_scaled::Float64 = t / control.tf
    update!(control.bspline, t_scaled, order)
    
    pcof_offset = floor(Int64, t_scaled*(control.bspline.N_distinct_knots-1) + 1)
    pcof_offset = min(pcof_offset, control.bspline.N_distinct_knots-1)
    pcof_offset += pcof_extra_offset

    val = 0.0
    #for i in 0:control.bspline.bspline_order-1
    # TODO Make sure this is safe!
    @turbo for i in 0:control.bspline.bspline_order-1
        val += pcof[pcof_offset+i] * control.bspline.output_array[1+i,1+order]
    end
    val /= control.tf ^ order # Chain Rule

    return val
end


@inline function eval_p_derivative(control::FortranBSplineControl2, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    offset = 0
    return eval_derivative(control, t, pcof, order, offset)
end


@inline function eval_q_derivative(control::FortranBSplineControl2, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    offset = div(control.N_coeff, 2)
    return eval_derivative(control, t, pcof, order, offset)
end


function fill_derivative_vec!(
        vals_vec::AbstractVector{<: Real}, control::FortranBSplineControl2,
        t::Real, pcof::AbstractVector{<: Real}, extra_offset::Integer
    )
    check_pcof_length(control, pcof)
    # calculate derivatives up to and including nderiv
    nderiv = length(vals_vec) - 1
    t_scaled::Float64 = t / control.tf
    update!(control.bspline, t_scaled, nderiv)

    pcof_offset = floor(Int64, t_scaled*(control.bspline.N_distinct_knots-1) + 1)
    pcof_offset = min(pcof_offset, control.bspline.N_distinct_knots-1)
    pcof_offset += extra_offset
    #println("pcof_offset=", pcof_offset)

    pow_tf_derivative_order = 1.0
    for derivative_order in 0:nderiv
        val = 0.0
        #TODO MAKE SURE THE BELOW IS SAFE!!!!
        @turbo for i in 0:control.bspline.bspline_order-1
            #val += pcof[pcof_offset+i] * control.bspline.output_array[1+i,1+derivative_order]
            val += pcof[pcof_offset+i] * control.bspline.output_array[1+i,1+derivative_order]
        end
        # Chain rule and getting 1/j! factor in p⁽ʲ⁾(t)/j!
        val /= pow_tf_derivative_order*factorial(derivative_order)
        vals_vec[1+derivative_order] = val
        pow_tf_derivative_order *= control.tf
    end
    return vals_vec
end


@inline function fill_p_vec!(
        vals_vec::AbstractVector{<: Real}, control::FortranBSplineControl2,
        t::Real, pcof::AbstractVector{<: Real}
    )
    offset = 0
    return fill_derivative_vec!(vals_vec, control, t, pcof, offset)
end


@inline function fill_q_vec!(
        vals_vec::AbstractVector{<: Real}, control::FortranBSplineControl2,
        t::Real, pcof::AbstractVector{<: Real}
    )
    offset = div(control.N_coeff, 2)
    return fill_derivative_vec!(vals_vec, control, t, pcof, offset)
end

    
function eval_grad_derivative!(
        grad::AbstractVector{Float64}, control::FortranBSplineControl2, t::Real,
        pcof::AbstractVector{<: Real}, order::Integer, extra_offset::Integer
    )
    grad .= 0
    check_indices_are_1_to_size(grad)

    t_scaled::Float64 = t / control.tf
    update!(control.bspline, t_scaled, order)

    pcof_offset = floor(Int64, t_scaled*(control.bspline.N_distinct_knots-1) + 1)
    pcof_offset = min(pcof_offset, control.bspline.N_distinct_knots-1)
    pcof_offset += extra_offset
    tf_pow_order = control.tf ^ order
    @turbo for i in 0:control.bspline.bspline_order-1 
        # Control is linear in the pcof coefficients
        # Chain rule (no 1/j! factor needed)
        grad[pcof_offset+i] = control.bspline.output_array[1+i,1+order] / tf_pow_order
    end

    return grad
end


@inline function eval_grad_p_derivative!(
        grad::AbstractVector{Float64}, control::FortranBSplineControl2, t::Real,
        pcof::AbstractVector{<: Real}, order::Integer
    )
    offset = 0
    return eval_grad_derivative!(grad, control, t, pcof, order, offset)
end


@inline function eval_grad_q_derivative!(
        grad::AbstractVector{Float64}, control::FortranBSplineControl2, t::Real,
        pcof::AbstractVector{<: Real}, order::Integer
    )
    offset = div(control.N_coeff, 2)
    return eval_grad_derivative!(grad, control, t, pcof, order, offset)
end

function fill_grad_mat!(
        grad_mat::AbstractMatrix{Float64}, control::FortranBSplineControl2, t::Real,
        pcof::AbstractVector{<: Real}, extra_offset::Integer
    )
    check_pcof_length(control, pcof)
    check_indices_are_1_to_size(grad_mat)
    # calculate derivatives up to (but not including nderiv)
    nderiv = size(grad_mat, 2)
    t_scaled::Float64 = t / control.tf
    update!(control.bspline, t_scaled, nderiv)

    pcof_offset = floor(Int64, t_scaled*(control.bspline.N_distinct_knots-1) + 1)
    pcof_offset = min(pcof_offset, control.bspline.N_distinct_knots-1)
    pcof_offset += extra_offset

    # Control is linear in the pcof coefficients
    grad_mat .= 0
    pow_tf = 1.0
    for k in 0:nderiv-1
        @turbo for i in 0:control.bspline.bspline_order-1
            # Chain rule (no 1/j! factor needed)
            grad_mat[pcof_offset+i,1+k] = control.bspline.output_array[1+i,1+k] / pow_tf
        end
        pow_tf *= control.tf
    end

    return grad_mat
end

function fill_grad_p_mat!(
        grad_mat::AbstractMatrix{Float64}, control::FortranBSplineControl2, t::Real,
        pcof::AbstractVector{<: Real}
    )
    offset = 0 
    fill_grad_mat!(grad_mat, control, t, pcof, offset)
end

function fill_grad_q_mat!(
        grad_mat::AbstractMatrix{Float64}, control::FortranBSplineControl2, t::Real,
        pcof::AbstractVector{<: Real}
    )
    offset = div(control.N_coeff, 2)
    fill_grad_mat!(grad_mat, control, t, pcof, offset)
end

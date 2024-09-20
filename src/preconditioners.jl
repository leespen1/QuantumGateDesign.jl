struct DiagonalHamiltonianPreconditioner
    LHS_diagonal::Vector{Float64} 
    LHS_upper_diagonal::Vector{Float64}
    LHS_lower_diagonal::Vector{Float64}
    real_system_size::Int64
    complex_system_size::Int64
end



"""
Construct preconditioner based on schrodinger prob.
"""
function DiagonalHamiltonianPreconditioner(prob::SchrodingerProb, order::Int; adjoint=false)
    return DiagonalHamiltonianPreconditioner(form_LHS_no_control(prob, order, adjoint=adjoint))
end



"""
Construct preconditioner based on arbitrary LHS matrix.
E.g. preconditioner for LHS*x = b
"""
function DiagonalHamiltonianPreconditioner(LHS::AbstractMatrix)
    @assert size(LHS, 1) == size(LHS, 2)
    @assert iseven(size(LHS, 1))
    real_system_size = size(LHS, 1)
    complex_system_size = div(real_system_size, 2)

    LHS_diagonal = LinearAlgebra.diag(LHS)
    @assert all(LHS_diagonal .!= 0) # Make sure no diagonal entries are zero

    LHS_upper_diagonal = LinearAlgebra.diag(LHS, complex_system_size)
    LHS_lower_diagonal = LinearAlgebra.diag(LHS, -complex_system_size)

    ## Technically we won't get an error if this is not the case, but still good to check
    #@assert LHS_upper_diagonal == -LHS_lower_diagonal
        
    return DiagonalHamiltonianPreconditioner(LHS_diagonal, LHS_upper_diagonal, LHS_lower_diagonal,
                            real_system_size, complex_system_size)
end



function LinearAlgebra.ldiv!(y::AbstractVector, P::DiagonalHamiltonianPreconditioner, x::AbstractVector)
    y .= x
    ldiv!(P, y)
end



function LinearAlgebra.ldiv!(P::DiagonalHamiltonianPreconditioner, x::AbstractVector)
    N = P.complex_system_size
    # Should be able to do this very broadcasted
    for i in eachindex(P.LHS_lower_diagonal)
        # Clear lower-left block diagonal
        ratio = P.LHS_lower_diagonal[i] / P.LHS_diagonal[i]
        x[N+i] -= x[i]*ratio
        # Get bottom-right block diagonal to ones
        x[N+i] /= (P.LHS_diagonal[N+i] - P.LHS_upper_diagonal[i]*ratio)
    end
    for i in eachindex(P.LHS_lower_diagonal)
        # Clear diagonal of of upper-right block
        x[i] -= P.LHS_upper_diagonal[i] * x[N+i]
        # Get upper-left block diagonal to ones
        x[i] /= P.LHS_diagonal[i]
    end
      
    return x
end



function Base.:\(P::DiagonalHamiltonianPreconditioner, b::AbstractVector)
    x = similar(b)
    ldiv!(x, P, b)
end

function lu_preconditioner(prob::SchrodingerProb, order; adjoint=false)
    LHS = form_LHS_no_control(prob, order, adjoint=adjoint)
    preconditioner = lu(LHS)
    return preconditioner
end

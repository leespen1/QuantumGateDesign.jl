# Forward Simulation
The function [`eval_forward`](@ref) can be used to simulate the evolution of a
quantum state in time.
```@docs
eval_forward(
        prob::SchrodingerProb{M1, M2}, controls, pcof::AbstractVector{<: Real};
        order::Int=2, saveEveryNsteps::Int=1,
        forcing::Union{AbstractArray{Float64, 4}, Missing}=missing,
        kwargs...
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}
```

using Documenter, QuantumGateDesign

makedocs(sitename="My Documentation")
push!(LOAD_PATH,"../src/")
makedocs(
    modules = [QuantumGateDesign],
    sitename = "QuantumGateDesign.jl",
    doctest = true
)
deploydocs(
    repo = "github.com/leespen1/QuantumGateDesign.jl.git",
)

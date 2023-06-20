using Documenter, HermiteOptimalControl

makedocs(sitename="My Documentation")
push!(LOAD_PATH,"../src/")
makedocs(
    modules = [HermiteOptimalControl],
    sitename = "HermiteOptimalControl.jl",
    doctest = true
)
deploydocs(
    repo = "github.com/leespen1/HermiteOptimalControl.jl.git",
)

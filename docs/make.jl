using Documenter, QuantumGateDesign

makedocs(sitename="My Documentation")
push!(LOAD_PATH,"../src/")
makedocs(
    modules = [QuantumGateDesign],
    sitename = "QuantumGateDesign.jl",
    doctest = true,
    pages = [
        "Home" => "index.md",
        "Problem Setup" => "problem_setup.md",
        "Forward Simulation" => "forward_simulation.md",
        "Gradient Evaluation" => "gradient_evaluation.md",
        "Optimization" => "optimization.md",
        "Control Functions" => "control_functions.md",
        "Examples" => "examples.md",
        "Visualization" => "visualization.md",
        "Index" => "function-index.md",
    ],
)
deploydocs(
    repo = "github.com/leespen1/QuantumGateDesign.jl.git",
)

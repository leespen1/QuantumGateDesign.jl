var documenterSearchIndex = {"docs":
[{"location":"#QuantumGateDesign.jl","page":"Home","title":"QuantumGateDesign.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Workflow","page":"Home","title":"Workflow","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Right now the user can set up Schrodinger problems, and compute the gradients of them with the control vector and target gate of their choosing. In a complete package, there would also be an optimization procedure which uses this gradient calculation.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The basic workflow is:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Set up a SchrodingerProblem, either yourself using the constructor SchrodingerProb, or by using one of the example problems provided. Schrodinger Problem Examples\nChoose a control vector and target.\nCompute a gradient using one of the methods provided. Gradient Evaluation","category":"page"},{"location":"#Functions","page":"Home","title":"Functions","text":"","category":"section"},{"location":"#Schrodinger-Problem-Definition","page":"Home","title":"Schrodinger Problem Definition","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SchrodingerProb","category":"page"},{"location":"#Schrodinger-Problem-Examples","page":"Home","title":"Schrodinger Problem Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"rabi_osc\ngargamel_prob\nbspline_prob","category":"page"},{"location":"#Forward-Evolution","page":"Home","title":"Forward Evolution","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Functions for evolving the state vector in a problem forward in time according to Schrodinger's equation, with or without forcing.","category":"page"},{"location":"","page":"Home","title":"Home","text":"eval_forward\neval_forward_forced","category":"page"},{"location":"#QuantumGateDesign.eval_forward","page":"Home","title":"QuantumGateDesign.eval_forward","text":"kwargs should be propogated all the way down to the vector version of eval_forward.\n\n\n\n\n\nEvolve a vector SchrodingerProblem forward in time. Return the history of the state vector (u/v) in a 3-index array, where the first index of corresponds to the vector component, the second index corresponds to the derivative to be taken, and the third index corresponds to the timestep number.\n\n\n\n\n\n","category":"function"},{"location":"#Gradient-Evaluation","page":"Home","title":"Gradient Evaluation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"discrete_adjoint\neval_grad_forced\neval_grad_finite_difference\ninfidelity","category":"page"},{"location":"#QuantumGateDesign.discrete_adjoint","page":"Home","title":"QuantumGateDesign.discrete_adjoint","text":"Arbitrary order version, should make one with target being abstract vector as well, so I can do state transfer problems.\n\n\n\n\n\n","category":"function"},{"location":"#QuantumGateDesign.infidelity","page":"Home","title":"QuantumGateDesign.infidelity","text":"Calculates the infidelity for the given state vector 'ψ' and target state 'target.'\n\nReturns: Infidelity\n\n\n\n\n\nCalculates the infidelity for the given matrix of state vectors 'Q' and matrix of target states 'target.'\n\nReturns: Infidelity\n\n\n\n\n\n","category":"function"},{"location":"#Bsplines","page":"Home","title":"Bsplines","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"bcparams\nbcarrier2\nbcarrier2_dt\ngradbcarrier2!\ngradbcarrier2_dt!","category":"page"},{"location":"function-index/","page":"Index","title":"Index","text":"Modules = [QuantumGateDesign]","category":"page"}]
}
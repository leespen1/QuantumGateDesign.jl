# Hermite Optimal Control
Julia package for quantum optimal control using Hermite methods.

Documentation located at: https://leespen1.github.io/HermiteOptimalControl.jl/dev/

## To Do
- [ ] Implement larger systems:
    - [ ] 3-level qubit
    - [ ] Qubit and cavity
    - [ ] 2 qubits with cross talk
- [ ] Incorporate guard level population penalty
- [ ] Evolve entire basis, not just a single vector
- [ ] Add check on the size of control vector when doing forward eval (make sure
      it matches the problem)
- [ ] Implement arbitrarily high order evolution and gradient calculation using
      Hermite
- [X] Add (automatic) test comparing discrete adjoint and forward
      differentiation gradients to nearly machine precision
- [ ] Break code into smaller functions where appropriate (discrete adjoint is
      quite long)
- [ ] Improve documentation, have documentation pages track project releases

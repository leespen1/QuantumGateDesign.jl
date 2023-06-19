# Hermite Optimal Control
Julia package for quantum optimal control using Hermite methods.

## To Do
- [ ] Fix 4th order discrete adjoint
- [ ] Implement arbitrarily high order evolution and gradient calculation using
      Hermite
- [ ] Implement larger systems:
    - [ ] 3-level qubit
    - [ ] Qubit and cavity
    - [ ] 2 qubits with cross talk
- [ ] Incorporate guard level population penalty
- [ ] Add check on the size of control vector when doing forward eval (make sure
      it matches the problem)
- [ ] Add (automatic) test comparing discrete adjoint and forward
      differentiation gradients to nearly machine precision
- [ ] Add higher order derivates for bsplines

# Hermite Optimal Control
Julia package for quantum optimal control using Hermite methods.

Documentation located at: https://leespen1.github.io/HermiteOptimalControl.jl/dev/

## To Do
- [ ] Make helper functions:
    - [ ] Create real valued target from complex target
    - [ ] Transform things from lab frame to rotating frame
    - [ ] Extract populations
    - [ ] Plotting helpers
- [ ] Implement larger systems:
    - [ ] 3-level qubit
    - [ ] Qubit and cavity
    - [ ] 2 qubits with cross talk
- [ ] Incorporate guard level population penalty
- [ ] Add check on the size of control vector when doing forward eval (make sure
      it matches the problem)
- [ ] Implement arbitrarily high order evolution and gradient calculation using
      Hermite
- [ ] Break code into smaller functions where appropriate (discrete adjoint is
      quite long)
- [ ] Improve documentation, have documentation pages track project releases
- [ ] The RWA and lab frames are kind of different. In the lab frame there is
      just one function, not p and q. Should there be different control objects
      for lab frame (would be fairly easy to dispatch across two control types)?
- [ ] Implement carrier frequencies
- [ ] Right now the entire gradient of p and q is computed whenever we really
      only need one partial derivative. Should improve this if it's at all
      significant to performance (I guess it depends on the complexity of the
      control function).
- [X] Evolve entire basis, not just a single vector
- [X] Integrate with an optimization package
- [X] Add (automatic) test comparing discrete adjoint and forward
      differentiation gradients to nearly machine precision

# Hermite Optimal Control
Julia package for quantum optimal control using Hermite methods.

Documentation located at: https://leespen1.github.io/HermiteOptimalControl.jl/dev/

## To Do
- [ ] Update forced gradient method to work with new control scheme.
- [ ] Update tests
- [ ] Incorporate guard level population penalty
- [ ] Compare timing with QuTip
- [o] Break code into smaller functions where appropriate (discrete adjoint is
      quite long)
- [o] Make helper functions:
    - [X] Create real valued target from complex target
    - [ ] Transform things from lab frame to rotating frame
    - [X] Extract populations
    - [ ] Plotting helpers
- [ ] Implement larger systems:
    - [ ] 3-level qubit
    - [ ] Qubit and cavity
    - [ ] 2 qubits with cross talk
- [ ] Add check on the size of control vector when doing forward eval (make sure
      it matches the problem)
- [ ] Implement arbitrarily high order evolution and gradient calculation using
      Hermite
- [.] Improve documentation, have documentation pages track project releases
- [ ] The RWA and lab frames are kind of different. In the lab frame there is
      just one function, not p and q. Should there be different control objects
      for lab frame (would be fairly easy to dispatch across two control types)?

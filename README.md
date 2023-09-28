# Hermite Optimal Control
Julia package for quantum optimal control using Hermite methods.

Documentation located at: https://leespen1.github.io/HermiteOptimalControl.jl/dev/

## To Do
- [ ] Implement larger systems:
    - [ ] 3-level qubit
    - [ ] Qubit and cavity
    - [ ] 2 qubits with cross talk
- [ ] Incorporate guard level population penalty
- [X] Evolve entire basis, not just a single vector
- [ ] Add check on the size of control vector when doing forward eval (make sure
      it matches the problem)
- [ ] Implement arbitrarily high order evolution and gradient calculation using
      Hermite
- [X] Add (automatic) test comparing discrete adjoint and forward
      differentiation gradients to nearly machine precision
- [ ] Break code into smaller functions where appropriate (discrete adjoint is
      quite long)
- [ ] Improve documentation, have documentation pages track project releases
- [X] Integrate with an optimization package
- [ ] Reorganize coed structre. There should be three objects:
    1. System, with system hamiltonian, N levels.
    2. Control, with the control functions, number of coefficients.
    3. Optimization, with initial and target states.
- [ ] The RWA and lab frames are kind of different. In the lab frame there is
      just one function, not p and q. Should there be different control objects
      for lab frame (would be fairly easy to dispatch across two control types).
- [ ] Implement carrier frequencies
- [ ] Make helper functions:
    - [ ] Create real valued target from complex target
    - [ ] Transform things from lab frame to rotating frame
- [ ] Right now the entire gradient of p and q is computed whenever we really
      only need one partial derivative. Should improve this if it's at all
      significant to performance (I guess it depends on the complexity of the
      control function).

The three object structure seems like a bit much, but I think it makes sense for
the following reasons:
1. The same control may be applied to different systems, and different controls
   may be applied to the same system. (maybe the raising and lowering operators
   should be in the system? I'm not sure. I think they should. "B-spline with
   carrier waves" has nothing to do with the matrices involved. On the other
   hand, what if the number of control functions isn't 2? Need to match number
   of control functions with number of associated control matrices.)
2. I can do a forward evolution without knowing anything about the optimization
3. It could be easy to dispatch over a parameter N for the number of derivatives
   to take
4. The more I separate things, the less mutable I need to make the objects
5. I can have some granularity in what I provide to functions but not need giant
   lists of arguments.

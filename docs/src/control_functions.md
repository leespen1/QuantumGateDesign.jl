# Control Functions
The only requirement we have of a control function is that 

Controls are implemented as subtypes of the [`AbstractControl`](@ref) abstract type.

Each control has an associated control vector length, which is included as a
parameter of the control object. Some controls have half of their control
parameters control the real part, while the reamaining half control the imaginary part.
By convention, when this is the case we reserve the first half of the control
vector for the real-associated parameters, and the second half for the
imaginary-associated parameters (as opposed to alternating them).

When controls are gathered together in a vector, the collective control vector
will just be the concatenation of all the individual control vectors.

```@docs
AbstractControl
BSplineControl
HermiteControl
HermiteCarrierControl
```

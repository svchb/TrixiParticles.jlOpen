# [Rigid Bodies](@id rigid_body)

Rigid bodies in TrixiParticles.jl are represented by particles whose motion is evolved
with rigid-body translation and rotation. This allows fluid-structure interaction while
keeping the structure kinematics rigid.

## API

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "structure", "rigid_body", "system.jl")]
```

### Contact Models

Rigid contact is configured through the contact model. This is separate from the
boundary model used for fluid-structure interaction; see
[Boundary Models](@ref boundary_models) for that part of the rigid-body setup.

`RigidContactModel` is the shared runtime model for both rigid-wall and rigid-rigid
contact. The shared normal spring-dashpot force and the contact diagnostics are active
for both paths.

Tangential history is updated through [`UpdateCallback`](@ref). This callback is required
when a rigid contact model uses history-dependent tangential terms, that is, whenever
`tangential_stiffness > 0` or `tangential_damping > 0`. The history keys are shared
between rigid-wall and rigid-rigid contact, but the history-dependent force evaluation is
currently enabled only for the rigid-wall path in this porting step.

The runtime model also stores later wall-specific controls such as
`resting_contact_projection` and `normalize_force_by_contact_patch`. These options remain
part of the shared runtime representation, but their wall-specific behavior is added in
later steps of the port.

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "structure", "rigid_body", "contact_models.jl")]
```

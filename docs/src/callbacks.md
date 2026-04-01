# Callbacks

```@autodocs
Modules = [TrixiParticles]
Pages = map(file -> joinpath("callbacks", file), readdir(joinpath("..", "src", "callbacks")))
```

`UpdateCallback` is also responsible for callback-driven rigid-contact state updates.
When a `RigidBodySystem` uses tangential history-dependent rigid contact terms such as
`tangential_stiffness` or `tangential_damping`, include `UpdateCallback()` in the solve
call so the tangential displacement history is updated and inactive contact pairs are
removed between time steps.

# [Custom Quantities](@id custom_quantities)

The following pre-defined custom quantities can be used with the
[`SolutionSavingCallback`](@ref) and [`PostprocessCallback`](@ref).

```@autodocs
Modules = [TrixiParticles]
Pages = ["general/custom_quantities.jl"]
```

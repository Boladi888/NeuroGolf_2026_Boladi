# DONE: task185 baked coord scale - scalar intermediate elimination (2026-06-12)

**Lesson**: Pre-baking a shared scalar multiplier into an initializer eliminates the scalar intermediate that would otherwise be produced by a chain of scalar Mul ops.

**Context**: In the task185 tree6 grid construction:
```
step_coord_h = Mul(step_h, coord_scale_h)  # 2B F16 [] intermediate
grid_offsets_h = Mul(grid_pos_h, step_coord_h)  # [1,6,1,2] F16
```

`coord_scale_h` is a constant (2/29). By pre-multiplying it into `grid_pos_h` at build time:
```
grid_pos_h_scaled = grid_pos_h * (2/29)  # baked into init, zero cost
grid_offsets_h = Mul(grid_pos_h_scaled, step_h)  # direct, no step_coord_h
```

The `step_coord_h` intermediate (2B F16 []) is eliminated. Net: -2B memory.

**Applicability**: Any computation `A = Mul(dynamic_scalar, const_scalar)` followed by `B = Mul(param_tensor, A)` can be rewritten as `B = Mul(param_tensor_scaled, dynamic_scalar)` where `param_tensor_scaled = param_tensor * const_scalar` is baked into the init. The cost savings is the intermediate `A` itself.

**Constraint**: `coord_scale_h` must still be kept if it is used in other dynamic computations (here, for `grid_start_scaled_h`). Only the Mul connecting it to the static `grid_pos_h` is folded.

**Verified**: task185 cost 1100 → 1098 (saves 2B F16 scalar intermediate). All gates pass.

**Family**: scalar-constant-fold / init-baking. Related to: cost_floor_reference.md §Constants and initializers.

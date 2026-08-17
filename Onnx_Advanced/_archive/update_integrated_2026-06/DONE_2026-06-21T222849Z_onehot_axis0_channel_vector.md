## OneHot(axis=0) channel-vector shrink

Task064 v12 used `OneHot` with `axis=0` on a scalar-ish color index shaped `[1,1]` to emit an F32 channel vector shaped `[C,1,1]`. In NCHW consumers this broadcasts as `[1,C,1,1]`, so it can feed both `Where(mask, color_vec, input)` and `Einsum(input, color_vec, ...)` without a Reshape/Unsqueeze.

This replaced the common scalar color path:

`Cast(index -> F32) -> Equal(index, color_ramp[1,C,1,1]) -> Cast(bool -> F32)`

with one `OneHot(index, depth, values, axis=0)`. On task064 it removed two scalar casts, two 10-cell bool Equals, the 10-param color ramp, and several duplicate scalar inits while preserving byte-equality. Use when the desired tensor is a broadcast channel selector, not a full batched one-hot plane.

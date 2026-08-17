# task200 - INT8/QLinear tail is dead

- Live bar: manifest `ng_current_bar.py --task 200` on 2026-07-14 reports cost **346**, sha256 `4daf5030fb8e55425bf26ccdc7e8a6af2b4aa414449eddb15c25593990c1a672`, owner Boladi.
- Attempted route: keep the current F32 scalar-moment marker decode, but narrow the post-`OneHot` tail (`seed_position`, `states`, `colors`, `rows`) to INT8/UINT8 using QLinearConv and an integer terminal writer.
- Exact failed gate: pre-build ORT support/price gate. `ort_reality.md` records `OneHot` cannot produce BOOL/UINT8 and `Einsum(INT8)` has no ORT kernel; task025's don't-do log gives the same `NOT_IMPLEMENTED : Einsum(12)` failure for UINT8 `Einsum`.
- Why dead: task200's current graph relies on a single terminal `Einsum` to assemble the full `[1,10,30,30]` output without charging a full canvas. Replacing only the Conv state carrier with QLinearConv still needs that terminal `Einsum`; INT8/UINT8 cannot run there, and casting back to F16 gives back the saved bytes while adding QLinear scalars/casts.
- One remaining hypothesis: a completely different terminal writer, such as a direct QLinearConv/ConvTranspose stamp with dynamic color routing, would have to avoid the terminal `Einsum`; naive dense routing is far above 346 params, so only revisit if a concrete low-rank dynamic-color writer is found first.

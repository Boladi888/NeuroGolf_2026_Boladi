# task279 ringtemplateqlinear hard record

Live bar: manifest/workbook Vamshi team-best cost 5,057, sha256
`4f9a0651cbb5bf227feb6d794dab4e84df585bd83cdfb7fa1a7e2928c70574f9`.

Attempted route: one 5x5 `QLinearConv` exact closed-rectangle seed bank
(`ringtemplate_probe.onnx`) replacing Vamshi's `core_0..core_3` and
`jump_0..jump_1`, followed by two gated regrowth rounds and the existing
`Where(color8, input)` output.

Cost result: static `_score_v2` cost 8,289 (`memory=7,812`, `params=477`),
which is +3,232 above the authoritative live bar. The named regression is the
materialized `ring_bank` tensor `[1,16,16,16]` UINT8 = 4,096 bytes; it deletes
six 256-byte U8 erosion/jump tensors but pays back more than it removes.

Failed gate: `python tools/ng_gate.py visible --task 279 --cand
The_400_Tasks/Task_by_Task/task279/prototypes/ringtemplate_probe.onnx`
returned `[visible] 264/266 (99.25%)  FAIL`.

Why dead: exact 3x3..5x5 hole-centered templates miss two official visible
pairs with larger/irregular closed components. A bounded no-false-positive
template search needed roughly ten or more kernels to cover the official
quirks, which already exceeds the 5,057 bar through the `N*256` seed-bank
plane. This kills the multi-channel exact-template representation and the
single-channel exact-count floor claim.

One remaining hypothesis: a genuinely single-channel signed 5x5 separator
that is not an exact template bank. Patch-conflict checks against Vamshi's
`jump_1` seed showed identical 5x5 blue patches can require different labels,
so this would need a different seed/regrowth target rather than a compression
of the live seed.

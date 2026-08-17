# task018 — DEAD ROUTE: INT16 `Where` for valid-position masking

**Worker:** bo_ng_cdx_worker3 · **Date:** 2026-07-15 · **Verdict:** DROP this exact rewrite.

## Live bar
- Authoritative live bar: **19,125** from `python tools/ng_current_bar.py --task 18 --json`.
- Live parent sha256: `d6567a94331bf690c8e85c3361b13713e4f77e327fef0d552070f9dfe76da3fb`.

## Attempted route
Replace the final valid-position masks:

- current: `Cast(valid_T*, INT16)` then `Mul(newpos_T*, valid_T*_i16_place)`
- attempted: `Cast(valid_T*, BOOL)` then `Where(valid_T*, newpos_T*, 0_i16)`

This would delete two `[12] INT16` valid-cast intermediates if ORT supported the op.

## Failed gate
`python tools/agent_helpers/profile_real_cost.py --task 18 --cand The_400_Tasks/Task_by_Task/task018/prototypes/task018_probe_v130_valid_where.onnx --arcgen 266`

Result:
`ORT NotImplemented: Could not find an implementation for Where(16) node`

## Why dead
`Where(INT16, INT16)` is not implemented in the pinned ORT path. Routing through F16/INT32 adds enough casts/intermediates to lose the intended 48-byte memory shave, so this exact representation is not a viable task018 compression.

## Remaining hypothesis
Look only for dtype-supported scalar-param shaves, such as replacing arithmetic by constants with equivalent ops that can reuse existing scalar initializers. The successful sibling of this record is `task018_bitshift_region_cost19124`.

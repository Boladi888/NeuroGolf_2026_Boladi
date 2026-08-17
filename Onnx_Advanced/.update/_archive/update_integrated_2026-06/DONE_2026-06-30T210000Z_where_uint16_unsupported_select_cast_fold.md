# ORT has no `Where` kernel for UINT16 — blocks the obvious "select-cast → Where" fold

## What I discovered (transferable ONNX/ORT quirk)

A very common micro-shave is to fold `Mul(mask, Cast(bool_selector -> int{0,1}))` into
`Where(bool_selector, mask, 0)`, deleting the per-use `Cast` node. When `mask` is **UINT16**,
this fold is **blocked**: the pinned ORT build raises

```
NOT_IMPLEMENTED : Could not find an implementation for Where(16) node ...
```

i.e. there is **no `Where` kernel for UINT16 operands** (opset 16/18). The deliberate
workaround is exactly what these graphs already do: `selected_u16 = Cast(bool->uint16)`
then `Mul(uint16_mask, selected_u16)` (Mul *does* have a UINT16 kernel). So the cast is
**not removable** when the gated value is a uint16 bit-mask.

Knock-on: you also cannot gate a uint16 mask via `BitwiseAnd(mask, sel)` because `sel` from
`Cast(bool->uint16)` is `{0,1}`, not `{0x0000,0xFFFF}`; producing the full-bits selector
(`Sub(0, Cast)` underflow trick) costs *more* nodes than the Cast+Mul it replaces.

This generalizes the partial, task-local note already in `task219.md`'s don't-do log
("Where(BOOL, UINT16, UINT16) is rejected by ORT") into an ORT-op-support fact.

## Where I encountered it

- task219 (HugeGraphRuleRewrite), worker `bo_ng_cld_worker3`, 2026-06-30.
- Builder: `The_400_Tasks/Task_by_Task/task219/prototypes/build_task219_structural_v5.py`
  (the per-(band,height,offset) `selected_u16` casts at the two pattern-emit sites).
- The fold would have deleted 66 uint16 select-casts = **−132 B memory** (7094 → ~6962),
  which would have been a clean strictly-below-bar win on a byte-equal-to-LIVE graph.
  Build rejected by `score_network_v2` (ORT session init NOT_IMPLEMENTED) — proven dead end.

## Why it matters

- Saves the next worker from re-attempting this exact fold on any uint16-bitmask graph and
  from burning a build+score cycle to discover the ORT gap.
- ORT op-support: `Where` supports BOOL/INT32/INT64/FLOAT/… but **NOT UINT16** (and, per the
  existing library note, INT16/UINT16 have no kernels broadly). Gate uint16 masks with
  `Cast(bool->uint16)+Mul`, not `Where`.

## What's missing from the library

- `lib_query.py op Where BOOL` returns "no ORT row for op Where" — there is **no ORT
  op-support row for `Where` at all**. Suggest adding a `Where` dtype-support row to the ORT
  op DB / `ort_reality` (supported: BOOL/INT8/INT32/INT64/FLOAT/FLOAT16/DOUBLE; **unsupported:
  UINT16, INT16**). The task219 don't-do log has the symptom but it is not generalized.

# task299 — `concat_bool_inputs` (bool-Concat keeping on a near-floor task)

| Field | Value |
|---|---|
| Task | task299 |
| Family | Pattern_Recognition / Filling Regions |
| Cost before | 2,678 |
| Cost after | 1,058 |
| Bytes saved | -1,620 (60% reduction) |
| Estimated LB | ~+0.92 |
| Pattern name | `concat_bool_inputs` (techniques.md §1 rule 7) |
| Staged | `.hermes/board/staged_submits/task299_boolconcat_t_fa59524e_on_5835_30/` |

## Why this exemplar matters

**Workers commonly dismiss "near-floor" tasks (cost < 5,000) as not worth optimization effort.** task299 is the canonical counter-example: a 2,678-cost task dropped 60% with a focused bool-Concat rewrite. Per-task LB gain ~+0.92 is comparable to many HUGE tasks.

Lesson: ALWAYS check for bool-eligible Concat/Pad chains before declaring a small task done.

## Before

```
... bool ops → Cast(bool_a, F32) → cast_a [F32, 360 bytes]
... bool ops → Cast(bool_b, F32) → cast_b [F32, 360 bytes]
... bool ops → Cast(bool_c, F32) → cast_c [F32, 360 bytes]
zero_map_init [F32, 360 bytes]   # 0/1-valued F32 padding init

Concat([cast_a, cast_b, cast_c, zero_map_init], axis=...) → concat_out [F32, [1, 10, 6, 6] = 1,440 bytes]
```

Concat output: `[1, 10, 6, 6]` F32 = 1,440 bytes intermediate.

## After

```
zero_map_bool_init [BOOL, 360 / 4 = 90 bytes]   # convert 0/1 F32 init → BOOL init

Concat([bool_a, bool_b, bool_c, zero_map_bool_init], axis=...) → concat_out [BOOL, [1, 10, 6, 6] = 360 bytes]
```

- Concat now operates on BOOL. Output dtype: BOOL.
- The 4 input F32 Casts are dropped (rewired to the underlying bool source).
- The `zero_map_init` is converted to BOOL.
- Concat output intermediate: 360 bytes (was 1,440). 4× reduction on this single intermediate.

Cumulative across the chain: cost 2,678 → 1,058.

## Why it works

When ALL Concat inputs are bool-eligible (each is either a bool source directly OR a 0/1-valued F32 initializer that can be converted to BOOL losslessly), the Concat output stays BOOL. Concat is dtype-preserving when all inputs share dtype.

The trick is converting the F32 0/1 initializer to a BOOL initializer with the same values. `(arr > 0.5).astype(np.bool_)` does this losslessly.

## Downstream requirements

For the rewrite to be safe, the Concat's downstream consumers must be bool-tolerant:
- **Where (CONDITION input only)**. `Where(BOOL, BOOL, BOOL)` is rejected by ORT. If both data branches are BOOL, rewrite as `Or(And(c, a), And(Not(c), b))` BEFORE the propagated BOOL hits the Where.
- **Equal(BOOL, BOOL)** works at all opsets ≥ 1. **`Greater(BOOL)` and `Less(BOOL)` are ALWAYS REJECTED by ORT** (BOOL is not order-comparable). Cast to UINT8/INT32 before any Greater/Less.
- **Pad (opset 13+)** accepts BOOL per the ONNX schema; opset 11 Pad rejects BOOL.
- **Reduce ops (Max/Min/Sum/Mean) all REJECT BOOL** — must `Cast(BOOL, UINT8)` before ReduceMax/ReduceMin (1 byte at opset 13+) or `Cast(BOOL, F16/INT32)` before ReduceSum/ReduceMean.
- Cast (explicit conversion downstream is fine — we just moved one Cast).
- Graph output (bool output works under `(out > 0.0)` decode).

ORT's `Where(BOOL, BOOL, BOOL)` rejection (see `ort_reality.md §2`) is the main risk — if a downstream Where has bool data inputs, the rewrite fails ORT load. Rewrite the Where as `Or(And(c, a), And(Not(c), b))` first.

## Generalizes to

`techniques.md` §1 rule 7 (`concat_bool_inputs`). Detector: `pass_concat_bool_inputs.py`.

**Especially valuable for small tasks** (cost < 5,000). The proportional savings are huge even when absolute bytes are small.

## Reproducibility

Detector runs in `tools/spread_scan.py`. Manual check: find Concat nodes, verify EVERY input is either:
- Output of a Cast(bool_op, F32), OR
- A 0/1-valued F32 initializer.

If ALL inputs check, the rewrite emits.

For any small task in your portfolio — especially Pattern_Recognition + Filling Regions — open the graph and look for Concat chains. The bool-Concat pattern is more common than expected.

## Artifacts

- Build prototype: `The_400_Tasks/Task_by_Task/task299/prototypes/task299_bool_concat_FA59524E.onnx` — the actual rewritten ONNX from the worker. Study its construction.
- Staged manifest: `.hermes/board/staged_submits/task299_boolconcat_t_fa59524e_on_5835_30/manifest.json`.

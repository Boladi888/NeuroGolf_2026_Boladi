# task273 / task342 / task250 — `drop_cast_before_pad_bool` (ORT Pad-13 accepts BOOL)

| Field | Value |
|---|---|
| Tasks | task250, task273, task342 |
| Pattern name | `drop_cast_before_pad_bool` (techniques.md §1 rule 5) |
| Cost reduction | -4,000 each on task250 and task342; smaller on task273 |
| Detector | `tools/spread_patterns/pass_drop_cast_before_pad_bool.py` |

## Before

```
Cast(bool_op_output, F32) → Pad → [graph output OR Greater(0)]
```

- Cast produces F32 [..., H+pad, W+pad] = 4 bytes/elem.
- Pad output goes to either the graph output (declared F32) OR a downstream Greater(0) threshold.

## After

```
Pad(bool_op_output) → [graph output (now BOOL) OR drop the Greater entirely]
```

- Cast is dropped.
- Pad operates directly on BOOL (1 byte/elem).
- Graph output dtype changed to BOOL.
- If a `Greater(_, 0)` consumer existed downstream, it becomes identity on BOOL (True > 0 = True, False > 0 = False) and gets dropped.

## Why it works

**Pad-13 accepts BOOL per the ONNX schema.** `tensor(bool)` is in the opset-13 Pad T type constraint (verified via `onnx.defs.get_all_schemas_with_history()`). The earlier claim "ORT-specific behavior despite spec" was wrong — this rewrite is standard at opset 13+ across ORT and other compliant runtimes. **Opset 10/11 Pad rejects BOOL** — if your source graph is at opset 10/11, you must either bump to opset 13 or use UINT8 (`Pad(UINT8)` works at opset 11+).

`run_network` does `(out > 0.0).astype(float)`. On BOOL {True, False} this produces {1.0, 0.0} — same as the original F32 {1.0, 0.0} would. Output byte-exact against the original under official decode.

## Safety conditions

- Pad's `constant_value` attribute (opset 10) or input[2] (opset 11+) must be 0 (or absent — default 0). A non-zero constant would need a typed equivalent.
- Pad's `mode` must be "constant" — "reflect" / "edge" with bool input are not tested.
- The Cast output must be consumed only by Pad (else orphaning).
- If the Pad output flows to anything OTHER than graph output or `Greater(0)`, the rewrite is skipped (potential dtype mismatch downstream).

## Cross-ref

See `ort_reality.md §1` for the full discussion of Pad-13 BOOL acceptance. This rewrite is portable across opset-13-compliant runtimes (verified in current ORT 1.24.4).

For models stuck at opset 10/11 (UINT8 instead of BOOL), use the opset-bump variant — see `task319_opset_bump_pad_uint8.md`.

## Generalizes to

`techniques.md` §1 rule 5. Mechanically detected. Common across PR_HUGE and any task with `Cast(bool, F32) → Pad → output` chain.

## Reproducibility

Run `tools/spread_scan.py --tasks taskNNN`. The detector requires:
- A `Cast` node with `to=FLOAT`, input from bool-producing op.
- That Cast's output consumed only by a `Pad` node.
- The Pad's `constant_value` is 0 or absent.
- Pad's output is either graph output OR a single `Greater(_, 0)` consumer.

When the rewrite applies, update `graph.output[0].type.tensor_type.elem_type = TensorProto.BOOL` to match the new Pad output dtype.

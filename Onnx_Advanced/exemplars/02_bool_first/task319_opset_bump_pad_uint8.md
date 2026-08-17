# task319 — `opset_bump_pad_uint8` (opset 10 → 11 + UINT8 instead of BOOL)

| Field | Value |
|---|---|
| Task | task319 |
| Pattern name | `opset_bump_pad_uint8` (techniques.md §1 rule 6) |
| Cost reduction | -742 |
| Detector | `tools/spread_patterns/pass_opset_bump_pad_uint8.py` |

## When to use this vs `drop_cast_before_pad_bool`

Both rewrites tackle the same `Cast(bool, F32) → Pad` pattern. Use:

- **`drop_cast_before_pad_bool`** when downstream is bool-tolerant (graph output or `Greater(0)`). 1 byte/elem.
- **`opset_bump_pad_uint8`** when downstream needs a numeric dtype and every immediate consumer accepts UINT8, or when you insert an explicit Cast before the first non-UINT8 op. UINT8 is 1 byte/elem (same as BOOL) but works through more ops than BOOL. Requires opset bump.

## Before (opset 10)

```
model.opset_import = [{"": 10}]
Cast(bool_op, F32) → Pad(attr pads=[...], mode="constant", value=0)
  → downstream numeric ops
```

- Pad in opset 10 attribute form: `pads` is an INTS attribute, length `2 × rank`.
- F32 intermediate between Cast and Pad: 4 bytes/elem.

## After (opset 11)

```
model.opset_import = [{"": 11}]
pads_init = numpy_helper.from_array(np.array([...], dtype=np.int64), name="pads_init")
Cast(bool_op, UINT8) → Pad(inputs=[Cast_out, pads_init], mode="constant")
  → downstream UINT8-compatible ops OR explicit Cast before unsupported ops
```

- Pad in opset 11+ input form: pads is input[1], an INT64 init.
- UINT8 intermediate between Cast and Pad: 1 byte/elem (4× smaller).

## Why it works

ORT supports both Pad-10 (attribute form) and Pad-11+ (input form). The opset bump is purely a re-encoding — same scoring, same semantics.

Pad-11+ supports many input dtypes including UINT8. Casting bool to UINT8 (instead of F32) preserves {0, 1} values exactly: bool True → UINT8 1, bool False → UINT8 0.

ORT does **not** implicitly promote unsupported UINT8 inputs to F32. This rewrite is safe only if the downstream consumers accept UINT8 (`Pad`, `Gather`, `Concat`, `Where(BOOL, UINT8, UINT8)`, `Equal`, `Greater`/`Less`, `MaxPool`, `ReduceMax`/`ReduceMin`, shape ops) or if you explicitly insert `Cast(UINT8 → F16/F32/INT32)` before the first unsupported op. `Add`/`Mul`/`Sub` on UINT8 require opset 14+, and `ReduceSum(UINT8)` is rejected.

## Critical: update graph output dtype

If the rewritten Pad's output is the graph output, the graph output dtype declaration MUST be updated to UINT8:

```python
graph.output[0].type.tensor_type.elem_type = TensorProto.UINT8
```

ORT enforces dtype consistency at load time. Without this, the load fails with:

```
Type Error: Type (tensor(float)) of output arg (output) of node (pad_input_form_obu_3) does not match expected type (tensor(uint8)).
```

(This is exactly what the detector hit during development; fixed by updating the graph output dtype after the rewrite.)

## Savings

- Cast output drops from F32 (4 bytes/elem) to UINT8 (1 byte/elem). For a `[1, 1, 30, 30]` intermediate that's 3,600 → 900 bytes.
- One pads_init INT64 added at ~16 bytes (4 INT64 values for a 2D Pad). The init's 4 elements count as 4 params.
- Net: -742 bytes on task319 after accounting for new init and output dtype changes.

## Generalizes to

`techniques.md` §1 rule 6. Apply when:
- Model is opset 10.
- Has `Cast(bool, F32) → Pad(attr form, mode=constant, value=0 or absent)`.
- Downstream needs a numeric dtype and is UINT8-compatible, or the first unsupported consumer is explicitly cast back to a supported dtype (otherwise prefer `drop_cast_before_pad_bool` for full BOOL output).

## Reproducibility

Detector is fully automated. The rewrite needs to:
1. Bump `opset_import` for the default domain (`""`) from 10 to 11.
2. Convert the Pad node from attribute form (pads as attr) to input form (pads as input).
3. Change the Cast target dtype from FLOAT to UINT8.
4. Update graph output dtype if the Pad output is the graph output.

All four are mechanical. The detector at `tools/spread_patterns/pass_opset_bump_pad_uint8.py` is conservative — only fires when all four can be applied safely.

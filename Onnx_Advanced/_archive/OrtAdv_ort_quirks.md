# ort_quirks.md — ORT/schema behaviors that matter in NeuroGolf

These are runtime and schema/opset facts verified during NeuroGolf work. Some are ORT-specific kernel behavior; others are ordinary ONNX schema boundaries that were previously misremembered as ORT quirks. Use them for Kaggle/ORT, and re-verify before assuming another runtime behaves the same way.

---

## 1. Pad-13 accepts BOOL input (in spec; opset-bump required from 10/11)

**The ONNX schema for opset 13 Pad INCLUDES `tensor(bool)` in the T constraint** (verified via `onnx.defs.get_all_schemas_with_history()`). The previous claim that "ORT accepts BOOL despite the spec" was incorrect — it's standard at opset 13+.

What is correct:
- **Pad(BOOL) requires opset 13+**. Local ORT (1.24.4 under `ORT_DISABLE_ALL`) rejects `Pad(BOOL)` at opsets 10 and 11.
- Opset 13+ accepts `Pad(BOOL)` and produces a BOOL output.

Implication: a `Cast(bool, F32) → Pad → output` chain at opset 10/11 must either (a) bump opset_import to 13+ and use `Pad(bool_op) → output`, OR (b) replace the F32 cast with UINT8 (`Pad(UINT8)` works at opset 11+).

- Saves 4 bytes/elem on the post-Pad tensor when going F32 → BOOL.
- Safe under `(out > 0.0)` thresholding in `run_network` — BOOL {True, False} produces the same {0, 1} mask after thresholding as F32 {0.0, 1.0}.
- **Portable across runtimes that implement the opset-13 schema** (not just ORT) — earlier "ORT-specific" warning is removed.

Cross-ref: `techniques.md` rule 5 (`drop_cast_before_pad_bool`).
Empirical: task250, task342 each saved -4,000 bytes via this.

---

## 2. ORT does NOT support `Where(bool, bool, bool)`

The ONNX spec for `Where` says the condition input is BOOL and the data inputs (x, y) must be `tensor(numeric)` OR `tensor(bool)`. ORT enforces this more strictly than the spec implies: **if both data inputs are BOOL, ORT rejects the model at session creation time**.

Workaround — rewrite with logical ops:

```
# Equivalent to Where(c, a, b) when all three are BOOL:
Or(And(c, a), And(Not(c), b))
```

If the structure is one-hot (often the case in NeuroGolf), exploit it to skip the Not:

```
And(one_hot_bool, pick_row_b) + Or(row_mask, col_mask)
```

Cross-ref: task110 worker hit this mid-rebuild and rewrote with And/Or/Not.
Empirical: task359 measurement — 9,000 bytes BOOL chain vs 36,000 bytes F32 Where equivalent (4× ratio holds).

**Where also rejects some narrow integer data dtypes.** Re-tested 2026-05-23 under ORT 1.24.4:
- `Where(BOOL, INT8, INT8)` → "Could not find an implementation for Where(9)"
- `Where(BOOL, INT16, INT16)` → same error
- `Where(BOOL, BOOL, BOOL)` → rejected
- Works: `Where(BOOL, UINT8, UINT8)`, `Where(BOOL, F16, F16)`, `Where(BOOL, F32, F32)`, `Where(BOOL, INT32, INT32)`, `Where(BOOL, INT64, INT64)`.

Practical implication: when you want to do a Where on integer color indices (range 0-9), `UINT8` is the cheapest valid data dtype if the surrounding chain is also UINT8-compatible. Use F16 only when later arithmetic/reduction requires it.

---

## 3. OneHot index/depth dtypes must match; INT32 works if depth is INT32

The ONNX spec lists INT32 and INT64 as valid index dtypes. ORT's kernel accepts both under
`ORT_DISABLE_ALL`, but the index dtype must match the scalar `depth` input dtype. The old
"INT64-only" rule was a misdiagnosis of an `INT32 indices + INT64 depth` mismatch.

Implication: an INT32 index map can save 4 bytes/element versus INT64, but make `depth` INT32 too.
If either side is mismatched, ORT can fail with a generic `NOT_IMPLEMENTED` load error.

Empirical update: task383 v1 re-probed this under the scorer path; INT32 indices + INT32 depth load
and run. task124 v3's older INT32 attempt failed because the depth/input dtype pairing was wrong.

**Sentinel behavior**: out-of-range index values produce all-zero channels at that position. Use
`index == depth` as a padding/no-color sentinel. Do not rely on negative indices unless you have
explicitly probed the exact opset/runtime pair; a non-negative `depth` sentinel is clearer and portable
within ORT's observed behavior.

**Placement trap (task075 2026-05-23)**: when you use OneHot to expand color indices to multi-channel output, the OneHot intermediate is a full `[1, N_classes, H, W]` FLOAT tensor that gets counted toward memory. Common cost: 10-channel × 30×30 = 36,000 bytes per OneHot intermediate. **Only place OneHot as the FINAL graph output node** (where the named-output rule skips counting it), or replace OneHot with `Equal(label, [0..N-1]) → BOOL` for direct multi-channel mask emission. task075's `convcode_index` ship explicitly avoided OneHot in favor of UINT8 color indices + final `Equal(label, color_constants)` BOOL output to save ~36,000 B vs a OneHot intermediate.

---

## 4. Opset-10 Equal accepts BOOL/INT32/INT64 ONLY (not float)

The ONNX `Equal` op had this history (verified via `onnx.defs`):
- **Opset 1/7/10 T constraint**: `BOOL, INT32, INT64`. (Note: BOOL was ALWAYS in Equal — the older "opset 11 added BOOL" claim was incorrect.)
- **Opset 11 T constraint**: adds the full numeric set — `UINT8/UINT16/UINT32/UINT64, INT8/INT16, FLOAT16/FLOAT/DOUBLE`. (Floats arrive here, NOT at opset 13.)
- **Opset 13 T constraint**: adds `BFLOAT16` only.
- **Opset 19+ T constraint**: adds `STRING`.

**ORT enforces this strictly for the declared opset version.** A model with `opset_import = [{"": 10}]` and `Equal(F32, F32)` will fail at load. Equal(BOOL, BOOL) works at opset 10+.

Workaround:
- Cast inputs to INT32 first.
- When the same value is compared against N constants, **share a single Cast** rather than per-comparison Cast — otherwise the N extra Cast nodes outweigh the savings from dropping Sub/Abs/Less chains.

```
# Bad: N Casts, one per Equal — net bytes neutral or negative
Equal(Cast(value, INT32), Cast(c1, INT32))
Equal(Cast(value, INT32), Cast(c2, INT32))
...

# Good: 1 shared Cast
v_int = Cast(value, INT32)
Equal(v_int, c1_int)   # c1_int is an INT32 initializer
Equal(v_int, c2_int)
...
```

Empirical: task368 netted -7,200 bytes via the shared-Cast variant on a 10-channel color compare.
Cross-ref: `techniques.md` §5 (Equal-fold).

---

## 5. Pad opset-10 attribute form vs opset-11+ input form

These are different ONNX representations of the same op. ORT supports both.

**Opset 10 (attribute form)**:
```python
helper.make_node("Pad", inputs=[data_name], outputs=[out_name],
                 pads=[0, 0, 0, 0, 0, 0, 0, 0],      # INTS attribute, length 2*rank
                 mode="constant",                      # STRING attribute
                 value=0.0)                            # FLOAT attribute (optional)
```

**Opset 11+ (input form)**:
```python
helper.make_node("Pad", inputs=[data_name, pads_init, constant_value_init], outputs=[out_name],
                 mode="constant")
# pads_init: INT64 tensor of length 2*rank
# constant_value_init: optional, scalar matching data dtype
```

Opset 11+ Pad also supports more input dtypes including UINT8. Bumping opset from 10 → 11 + converting Cast(bool, F32) to Cast(bool, UINT8) can net positive even though the structure looks the same.

Critical when bumping: update `graph.output[0].type.tensor_type.elem_type` if the graph output dtype changes (UINT8 instead of F32) — ORT enforces dtype consistency at load time and rejects mismatches.

Cross-ref: `techniques.md` rule 6 (`opset_bump_pad_uint8`).
Empirical: task319 saved -742 bytes via this opset bump.

---

## 6. `pads` attribute layout is grouped, not interleaved

For both attribute-form and input-form Pad, the `pads` array of length `2 × rank` is ordered:

```
[x1_begin, x2_begin, ..., xN_begin, x1_end, x2_end, ..., xN_end]
```

NOT `[x1_begin, x1_end, x2_begin, x2_end, ...]`. Many people get this wrong intuitively. ORT will produce shape-inference errors with otherwise-correct surgery if the order is mixed up.

---

## 7. Reduce ops have per-reducer dtype rules (split by reducer)

Re-verified 2026-05-23 (ORT 1.24.4 under `ORT_DISABLE_ALL`, opset 13). The earlier blanket "Reduce requires ≥16-bit numeric" rule was an over-correction. Actual behavior:

| Reducer | BOOL | INT8 | UINT8 | INT16 | INT32 | INT64 | F16 | F32 |
|---|---|---|---|---|---|---|---|---|
| `ReduceMax` | ✗ through opset 17; **✓ at opset 20** | **✓** | **✓** | (untested) | ✓ | ✓ | ✓ | ✓ |
| `ReduceMin` | ✗ | **✓** | **✓** | (untested) | ✓ | ✓ | ✓ | ✓ |
| `ReduceSum` | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| `ReduceMean` | ✗ | ✗ | ✗ | ✗ | **✓** | **✓** | ✓ | ✓ |

(Schemas claim `UINT32/UINT64` for all Reduce ops, but the CPU kernel is not implemented — load fails.)

**Practical implications**:
- For `BOOL → ReduceMax`: at opset <=17, `Cast(BOOL, UINT8)` is the cheapest 1-byte path. At
  **opset 20**, direct `ReduceMax(BOOL)` works and can remove the cast entirely. This is
  Kaggle-confirmed by task191 v6 (2026-06-02, public +0.29); it requires the opset-18+ axes-as-input
  schema form. `ReduceMin(BOOL)` at opset 20 is not yet confirmed.
- For `BOOL → ReduceSum/ReduceMean`: must go through `Cast(BOOL, F16/INT32/INT64)`. F16 is the cheapest 2-byte path; INT32 if the sum can overflow F16 (>2048 in unsafe regions).
- For `UINT8 → ReduceMax/ReduceMin`: stay UINT8 (1 byte) at opset 13+.
- For `UINT8 → ReduceSum/ReduceMean`: must `Cast(UINT8, INT32)` (4 bytes) before reduction.

The earlier "ReduceSum(BOOL) → INT" claim was wrong (rejected at load); the over-correction "all Reduce needs ≥16-bit" was also wrong (Max/Min accept INT8/UINT8).

Cross-ref: `ort_op_dtype_matrix.md` for the full per-op dtype-support table.

---

## 11. F16 integer-exactness threshold (precision danger zone)

F16 represents integers exactly only up to a threshold:

- 0 to 2048: **all integers exact**.
- 2049 to 4096: only **even** integers exact (rounds to nearest even).
- 4097 to 8192: only **multiples of 4** exact.
- Doubles each octave; max F16 value is ±65504.

**Implications**:
- Hash keys, position indices, sums-over-large-regions in F16 lose discrimination. NEVER F16 a hash-decode pipeline's MatMul weights (cross-ref `techniques.md §4 Rule C`).
- For NeuroGolf [1,10,30,30], a `ReduceSum(input, axes=[1,2,3])` can reach 900 — F16 stores 900 exactly. Safe. ReduceSum over a `[1, 100, 100, 100]` tensor would NOT be safe in F16.
- F16 `Mul(big, big)` can saturate to inf. Check max magnitude before casting.
- Color indices 0–9 in NeuroGolf: F16 is fully safe.

Empirical: task366 v10 uses F16 throughout the spatial pipeline because all values are colors 0–10. See `exemplars/01_whole_graph_rewrites/task366_template_vote_to_sparse_stamps.md` sub-technique 3.

---

## 12. ORT does NOT constant-fold under `ORT_DISABLE_ALL`

The execution mode Kaggle uses (`ORT_DISABLE_ALL`) skips ALL graph optimizations including constant folding. Implications:

- `Add(x, 0)`, `Mul(x, 1)`, `Sub(x, 0)` — ORT keeps the op AND allocates the intermediate. Remove these by hand in your build script.
- `Cast(x, F32) → Cast(F32, F32)` — both Casts execute. Same dtype = wasted op.
- `Concat(x)` (single input) — ORT keeps the Concat op, allocates an intermediate copy. Use Identity wire-through or just rename.
- `Slice` with starts=full / ends=full / step=1 = identity. Drop.
- Two identical Slices on the same input with same params = two intermediates. **CSE manually** (`techniques.md §13`).
- `Cast(x, F16) → Cast(F16, F32)` chain (or F32→F16→F32): both Casts execute, both intermediates materialize. Cast once at the boundary where the dtype change actually matters; don't bounce.

`onnxsim` catches some of these BEFORE ORT runs, but not all. Run `onnxsim` + manual pass on the residue.

---

## 13. Axes-attr → axes-input transitions across opsets (NOT uniform across Reduce ops)

Subtle compatibility trap when bumping opsets. The transition is **per-op**, not uniform:

| Op | axes is attribute through | axes is input from |
|---|---|---|
| `Squeeze` | opset 12 | opset 13 |
| `Unsqueeze` | opset 12 | opset 13 |
| `ReduceSum` | opset 12 | **opset 13** |
| `ReduceMean` | opset 12 | (opset 13 still uses attr; spec change for ReduceMean came at opset 18+) |
| `ReduceMax` | opset 17 | **opset 18** |
| `ReduceMin` | opset 17 | **opset 18** |
| `ReduceL1`/`ReduceL2`/`ReduceLogSum`/`ReduceLogSumExp`/`ReduceProd`/`ReduceSumSquare` | opset 12 | (per-op; check schema) |

**Trap**: applying the "opset 13 schema break" advice uniformly to all Reduce ops is wrong. ReduceMax/ReduceMin keep `axes` as an attribute until opset 18. Building a `ReduceMax` at opset 13 with `axes` as an input tensor will fail.

```python
# opset 11–17 for ReduceMax/ReduceMin: axes is an ATTRIBUTE
helper.make_node("ReduceMax", ["x"], ["y"], axes=[2,3], keepdims=1)

# opset 13 for ReduceSum/Squeeze/Unsqueeze: axes is an INPUT
helper.make_node("ReduceSum", ["x", "axes_init"], ["y"], keepdims=1)
```

**Symptom**: ORT load error like `"axes input is missing"` or `"unexpected attribute axes"` after an opset bump. Check the specific op's schema-since-version before bumping.

Also at opset 11: `Equal` accepts FLOAT (was INT/BOOL only). See quirk #4 — floats arrive at opset 11, not opset 13.

---

## 8. `Where` operand broadcasting

ORT enforces strict broadcasting semantics on Where's three inputs (condition, x, y). All three must be broadcast-compatible. Bool condition broadcasts the same way as numeric tensors (numpy rules). If your condition is `[1, 1, 30, 30]` and x is `[1, 10, 30, 30]`, that's fine — condition broadcasts to `[1, 10, 30, 30]` and selects per-channel.

**Critical limitation (ORT 1.24.4, Where version 9): `Where([] BOOL, BOOL_tensor, BOOL_tensor)` — scalar BOOL condition with BOOL data tensors — raises "Could not find an implementation for Where(9) node" at session init.** Same error for `Where([] BOOL, [1,1,12,1] BOOL, [1,1,12,1] BOOL)`. Workaround: replace with `Or(And(cond, x), And(Not(cond), y))` for BOOL data. F16 data with scalar BOOL condition works fine (`Where([] BOOL, [] F16, [] F16)` verified OK). Verified 2026-06-04 during task119 v1 build.

---

## 9. Where, OneHot are 0 MACs in onnx-tool 1.0.1

`WhereNode` and `OneHotNode` in `onnx_tool/node.py` subclass `Node` directly with no `profile()` override. They inherit `Node.profile() = [0, 0]`. Some research notes bill these at `output_elems` based on "treat mov as 1 MAC" — reading the source contradicts the doc. **For NeuroGolf scoring, treat Where and OneHot as cost-free in MACs.**

Combined: `OneHot(int_indices, depth=10, values=[off, on])` followed by `Where(bool_mask, onehot_result, zero)` is a cost-efficient way to expand an integer color index to a 10-channel one-hot output. The OneHot output tensor IS counted in memory (it's the largest intermediate), but the op itself adds no MACs.

Keep this separate from the "shape-only cost floor": Where and OneHot are 0-MAC, but their outputs still count toward `memory` unless the output tensor is named exactly `output`. Do not treat them as memory-free shape-only ops.

---

## 10. Shape-inference cache: `value_info` must be cleared after surgery

ORT loads a model and uses `graph.value_info` (the pre-saved shape inference) to validate operator shape compatibility. **If you modify the graph (insert/remove nodes, change dtypes) without updating `value_info`, ORT loads the stale cached shapes and rejects the model with `[ShapeInferenceError] Incompatible dimensions`** — even though the modified graph would shape-infer correctly from scratch.

```python
# After any non-trivial graph mutation:
graph.ClearField("value_info")
# Then save. Or, if you want fresh inferred shapes:
model = onnx.shape_inference.infer_shapes(model)
```

This single line is the difference between an ORT load failure and a working model after a crop-to-active-grid or dtype-change rewrite. Cross-ref: `techniques.md` §2 (crop) and §15 (separable_mask_collapse).

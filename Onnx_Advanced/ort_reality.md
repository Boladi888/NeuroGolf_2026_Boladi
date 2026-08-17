# ort_reality.md — the ORT-empirical layer (Phase-2 merge of Ort_Advanced/, 2026-06-05)

> ~90% A-grade. Every dtype/opset/error fact here is PROBED on our build or grader-confirmed.



# ===== ort_reality.md =====

# ort_reality.md — per-op dtype support under ORT

What ORT actually accepts at runtime per op. **Re-probed 2026-05-23** under `onnx 1.21.0`, `onnxruntime 1.24.4`, `ORT_DISABLE_ALL`, CPUExecutionProvider. Each row notes the OPSET at which the listed support holds; failures are split between schema rejection (load fails in `check_model`) and kernel rejection (load fails in `InferenceSession` with "NOT_IMPLEMENTED").

Use this table before designing a chain in a narrow dtype. Several "obvious" dtype reductions don't compile.

---

## Table (2026-05-23 re-probe; 2026-06-02 targeted additions)

| Op | Accepted input dtypes | Rejected (notable) | Notes |
|---|---|---|---|
| `ReduceSum` | F16, F32, F64, INT32, INT64 (opset 13+) | **BOOL, INT8, UINT8, INT16, UINT32, UINT64** (UINT32/UINT64 in schema but kernel missing) | Cast BOOL→F16 (2 B) before reduction. axes is an INPUT from opset 13. |
| `ReduceMax` | F16, F32, F64, INT32, INT64, **UINT8 (opset 12+)**, **INT8 (opset 12+)**, **BOOL (opset 20)** | BOOL at opset <=17, INT16, UINT32, UINT64 (UINT32/UINT64 in schema; kernel missing) | UINT8 and INT8 BOTH work at opset 12/13. Earlier "INT8 needs opset 16+" was wrong. axes is an ATTRIBUTE through opset 17; INPUT from opset 18. **BOOL ReduceMax at opset 20 is Kaggle-confirmed** (task191 v6, 2026-06-02). If a BOOL->U8 Cast only feeds ReduceMax, reduce BOOL then Cast the smaller result for ArgMax; opset >=18 reducers need axes inputs, not attrs. |
| `ReduceMin` | F16, F32, F64, INT32, INT64, **UINT8 (opset 12+)**, **INT8 (opset 12+)** | **BOOL, INT16, UINT32, UINT64** | Same as ReduceMax. axes-attr → input at opset 18. |
| `ReduceMean` | F16, F32, F64, **INT32 (opset 13+)**, **INT64 (opset 13+)** | BOOL, UINT*, INT8, INT16 | Earlier "float-only" claim was wrong; INT32/INT64 work at opset 13. |
| `CumSum` (opset 11/14) | F32, F64, INT32, INT64, UINT32, UINT64; **F16 at opset 14+ (CPU kernel EXISTS)** | UINT8, INT8, BOOL (kernel/schema missing). **F16 at opset ≤13: the SCHEMA itself rejects it** (F16 added to the type constraint at 14) | **Correction 2026-06-06 (task055)**: the old "NO F16 kernel" row was an opset artifact — at opset ≤13 `infer_shapes(strict)` rejects F16 before any kernel lookup (likely what the 2026-05-31 task070 probe hit). At opset 14+ F16 CumSum loads AND runs: LIVE task055 (Kaggle-scored) + staircase v1–v7 execute `CumSum(F16 [1,1,30,1], axis)` across 200k+ corpus pairs. 1-D prefix chains CAN be 2 B/elem — build at opset ≥14. Has a native `reverse=1` attr (opset 11+) → backward fill in one op, no reverse-Slice. `axis` is an INPUT. |
| `Where(c, x, y)` | c: BOOL; x,y matched: F16, F32, F64, INT32, INT64, **UINT8 (opset 13+)** | **x,y as BOOL, INT8, INT16, UINT16** | Earlier "UINT8 at opset 14+" was wrong; opset 13 works. BOOL-BOOL chain workaround: `Or(And(c, a), And(Not(c), b))`. See `ort_reality.md §2`. |
| `OneHot` | indices: **INT32 or INT64 (must match `depth` dtype)**; values: **F16, F32, INT64 only** | values: **BOOL, UINT8 (kernel missing)**; indices int8/int16/uint8 (`NOT_IMPLEMENTED`) | OneHot CANNOT produce BOOL/UINT8 output; F16 is the memory floor for one-hot marker tails (task391). **INT32 indices DO work** (re-verified 2026-06-01 task383) provided `depth` is also INT32 — a dtype mismatch (INT32 indices + INT64 depth) → `NOT_IMPLEMENTED`. **Out-of-range index (==depth or <0) → all-zero channels at that position**: free "no-color/padding" sentinel — pad the index map with `depth` and those cells decode to all-zero, no extra mask op. Use `Equal(label, channel_constants) → BOOL` if you need a BOOL multi-channel mask. See `ort_reality.md §3` for placement trap. |
| `Gather` | data: any numeric, BOOL; indices: INT32 or INT64 | — | Broadly supported, including BOOL data (task079). **Fixed Gather is NOT cost 0** — the index initializer counts in `params`. If a later `ArgMax`/`Reduce` needs numeric data, gather the BOOL plane first and cast only the small gathered result. **Dtype inheritance**: `Gather` output dtype == `data` dtype (task383, 2026-06-22). `Gather(F32_input, idx, axis=1)` → F32 output — NOT a free U8 mask; a channel-slice of the F32 one-hot is still F32. The dilated Conv (#98) that never materializes the F32 plane is cheaper than any Gather-of-F32 + Cast chain. |
| `Pad` (opset 13+) | data: F16, F32, F64, INT8, INT32, INT64, UINT8, **BOOL** | INT16, UINT16 (schema allows some int widths but ORT kernel missing) | BOOL is IN the opset-13 ONNX schema (no longer ORT-specific). |
| `Pad` (opset 11) | data: F16, F32, F64, INT8, INT32, INT64, UINT8 | **BOOL, INT16, UINT16** | Bump to opset 13 for BOOL Pad. |
| `Pad` (opset 10) | F16, F32, F64 (attr-form only) | BOOL, UINT8, INT32, INT64 in local ORT | opset-10 attr-form rejects most non-float per kernel. |
| `Equal` (opset 11+) | F16, F32, F64, INT*, UINT*, BOOL (full numeric+bool) | — | Floats arrive at opset 11, NOT opset 13. |
| `Equal` (opset 1/7/10) | **BOOL, INT32, INT64 only** | F32, F16, UINT8, INT8 | BOOL has ALWAYS been in Equal since opset 1. Cast non-int inputs to INT32 at opset 10. |
| `Greater` / `Less` (opset 10+) | F16, F32, F64, INT32, INT64, **UINT8, INT8, INT16 (opset 10+)** | **BOOL (all opsets)** | Earlier "UINT8 needs opset 14+" was wrong; opset 10 works. Order-comparisons on BOOL are always rejected. |
| `GreaterOrEqual` / `LessOrEqual` (opset 12+) | **UINT8 at opset 13 verified** (plus the Greater/Less numeric set) | BOOL | task199 v3 2026-06-06: `LessOrEqual(u8,u8)` ran the full scoring path + 200k corpus + 266 officials at opset 13. Same kernel family as Greater/Less. |
| `Mod` | INT32, INT64, **UINT8 (opset 13 probed)** | mixed input dtypes | Both inputs must have the same dtype: `Mod(U8,U8)` works, `Mod(U8,INT32)` rejects. U8 Mod is useful for small periodic index chains, but `Gather` still requires an INT32/INT64 index cast before use (task110). |
| `Max(UINT8, UINT8) → UINT8` | opset 12+ | opset ≤ 11 | task054 verified. |
| `Min(UINT8, UINT8) → UINT8` | opset 17 (likely 12+, same kernel family as Max) | — | task055 staircase v7 verified 2026-06-06 (200k corpus executions): `Min(u8,u8)` = AND on 0/1 masks in ONE op, replaces `And(bool)`+`Cast(U8)`. |
| `Sub(UINT8, UINT8)` / `Add(UINT8, UINT8)` / `Mul(UINT8, UINT8)` | **opset 14+** | opset ≤ 13 | Same for INT8. Schema permits earlier per ONNX, but ORT kernel requires 14. |
| `And` / `Or` / `Not` / `Xor` | BOOL only | — | Cheapest mask combiners. |
| `Mul` / `Add` / `Sub` / `Div` (float/int32/int64) | F16, F32, F64, INT32, INT64 (all opsets); **UINT8/INT8 at opset 14+** | BOOL, UINT8/INT8 **below** opset 14, **INT16/UINT16 (no kernel, any opset)** | BOOL chain: convert to `And`/`Or`. **Correction 2026-06-21 (task209 v14, Kaggle byte-equal 200266 + profile==static):** INT32 is the floor **only below opset 14**. At opset 14+ `Mul`/`Add`/`Sub` run on **INT8/UINT8** (consistent with the dedicated U8/I8 row above), so small-int index/constraint arithmetic whose intermediates all fit `[-128,127]` (U8: `[0,255]`) CAN use a 1 B dtype — incl. full **I8 constraint chains** (`Sub(anc, Mul(iota, mag)) → Add(…, extent)` + `Greater`/`LessOrEqual`, comparisons already I8 at opset 10+). task209: 78 B I32→I8 saving, net −73 score. **Still true:** **INT16/UINT16 has NO Mul/Add kernel at ANY opset** (task221 probe 2026-06-07) — INT8 is the sub-INT32 floor, not INT16. NB `Less`/`Greater` accept int8/int16, so below opset 14 the rejection is the arithmetic, not the comparison. Value-range guard: verify every intermediate fits the dtype range before converting (`max_mag·max_iota`, `anchor−mag·steps`, bound±1 ≤ 127). |
| `Conv` | F16, F32, F64 | INT*, BOOL, **mixed-dtype X/W** | `X`, `W`, `Y` share the same type parameter — `Conv(F32 input, F16 weights)` FAILS load. Cast input first or keep weights at input dtype. **W input can be runtime tensor** under `ORT_DISABLE_ALL`. Negative pads are accepted in ORT; a 1x1 Conv can channel-collapse and crop in one node (task079). **F16 output on an F32 input is structurally impossible**: Conv inherits input dtype, so a smaller-dtype output needs a smaller-dtype input — but `Cast(input→F16)` materializes `[1,10,30,30] F16 = 18,000 B` (measured +16,236 vs baseline 9,892 on task017, 2026-06-24). For a one-op full-plane F32 read, the F32 Conv output (4 B/elem over the active grid) is the floor; only a different op class or a formulation that never reads the full F32 plane can beat it. |
| `ConvInteger` | INT8/UINT8 input; INT32 output | F16/F32 | Use for U8 stamps/detection (task075 `convcode_index` pattern). Pinned ORT accepts scalar/size-1 `x_zero_point` only; a real per-input-channel 1-D `x_zero_point` can pass checker but fail at runtime, while per-output-channel `w_zero_point` is OK (task219). |
| `QLinearConv` | INT8/UINT8 input + scale/zp pairs | F32 input rejected; sparse weights rejected by `shape_inference(strict_mode=True)` | Empirical task136/042/task321 — sparse weights pass ORT but fail official scorer; direct F32 graph input must be narrowed first, and a full input Cast usually costs more than it saves. |
| `Gemm` | F16, F32, F64 | **INT32 rejected by current ORT kernel even with C supplied**, BOOL, INT8 | transA/transB attributes work. C input required at opset ≤ 10, optional at opset 11+. |
| `MatMul` | F16, F32, F64, INT32, INT64 | BOOL, INT8, UINT8 | F16 MatMul works for hash-decode pipelines; INT32/INT64 MatMul also load/run in local ORT. |
| `MatMulInteger` | INT8/UINT8 input; INT32 output | F16/F32 | Quantized matmul for U8 chains. |
| `Cast` | broad → broad | a few obscure pairs | The escape hatch for dtype mismatches. |
| `Concat` | any matching dtype across inputs | mixed dtype inputs | All inputs must share dtype. |
| `Slice` | any | — | Shape-only, zero MACs. Runtime `starts`/`ends`/`axes`/`steps` may be INT32 or INT64, but all four index inputs must use the same Tind. |
| `Reshape` / `Squeeze` / `Unsqueeze` / `Transpose` | any | — | Shape-only. **`Squeeze`/`Unsqueeze` axes-attr → input at opset 13**. |
| `Identity` | any | — | Same. Cost 0 only if it's also the graph output (no `value_info` intermediate). |
| `Tile` | any numeric, BOOL | — | Broadly supported, but the tiled output is a charged intermediate unless it is the graph output or fully absorbed into a free-output op. |
| `Range` (opset 11+) | F32, F64, INT16, INT32, INT64 | BOOL, UINT8 | Generates index ranges. |
| `Einsum` | F16, F32, INT32 | UINT8, INT8, **INT16** (kernel missing); **mixed operand dtypes** | Useful for projection compression — see `by_symptom.md`. **INT16 also has NO kernel** (task281 2026-06-09, checker-valid but ORT `NOT_IMPLEMENTED`) — narrow-dtype Einsum bottoms out at INT32; F16 is the 2 B floor for the moment/separable families (#93/#119). **All operands must share one dtype** (task351 2026-06-20, onnx 1.21/ort 1.24): a contraction mixing the **F32 graph input with F16 selector operands** fails shape-inference (`ShapeInferenceError: Inputs has inconsistent type tensor(float16)`). So a free-output Einsum reading the F32 one-hot input is **F32-locked**; if the final Einsum never takes the F32 input, init/Cast-derived small operands may be F16 and decode under `>0` (task195). O8 source-read noted `OperatorKernels.md` omits F16, but local 2026-07-03 mini-probe `ab,bc->ac` loaded/ran F16 on onnx 1.21.0 / ORT 1.24.4 CPU. See techniques.md #109. |
| `BitShift` | UINT8, UINT32, UINT64 | UINT16, signed ints | For packed-bit unpacks. |
| `DynamicQuantizeLinear` | F32 input | F16 (kernel missing) | Use for runtime quantization. |
| `GatherElements` / `ScatterElements` | BOOL, UINT8, F16, INT32 (and broader; not exhaustively tested) | — | Probed working at opset 13. |
| `ScatterND` | BOOL, UINT8, F16, F32, INT32, INT64 | duplicate-target reductions | UINT8/BOOL work at opset 13 — F16 is no longer the smallest path. Use for unique sparse writes; O2 marks duplicate-target reductions unsafe on ORT CPU 1.24.4. |
| `MaxPool` | F16, F32, F64, INT8, UINT8 | BOOL | Use UINT8 (1 B) for bool-grid pooling at opset 12+. Negative END-pads are accepted and can crop stride-pooled outputs in-node (task213). INT8 pads use INT8_MIN; if upstream values can be negative, test `Greater(pool,0)`, not `Cast(pool,BOOL)` (task162). |
| `AveragePool` | F16, F32 | **UINT8 (rejected — "Type Error: Type tensor(uint8) of input parameter")** | Float-only at all opsets. Use `MaxPool(U8)` for solidity/BFS; `AveragePool` is unavailable for integer grids (task158, 2026-06-22). |
| `TopK` (opset 10+) | F16, F32, F64, INT32, INT64 | **BOOL, UINT8, INT8 (no kernel)** | K is INPUT from opset 10 (earlier "opset 11" claim was wrong). **U8/I8 are REJECTED — TopK forces F16/F32 values; TopK indices are INT64 by spec, so cast them to INT32 immediately if a downstream OneHot/Gather accepts it** (task391). |
| `ArgMax` / `ArgMin` | F16, F32, F64, INT*, UINT8 | **BOOL (invalid)** | task281 2026-06-09: `ArgMax(BOOL)` rejected — Cast a bool mask to U8/F16 before arg-selection. |

---

## How to use this when designing a chain

1. **Start in the narrowest dtype your chain's first op accepts.** For masks/predicates, that's BOOL (1 byte/elem). For 0-255 integer counts, UINT8.
2. **Walk the chain forward; promote at the first op that rejects your current dtype.** From BOOL the cheapest promotion is to UINT8 (1 B) if downstream is `Where`/`Gather`/`Pad`/`Concat`/`Equal`/`MaxPool`/`ReduceMax`/`ReduceMin`. Promote to F16 (2 B) only if you hit `ReduceSum`/`ReduceMean` or arithmetic.
3. **For Reduce ops on BOOL masks (Max/Min)**: `Cast(BOOL, UINT8)` is the cheapest 1-byte path at opset 13+. F16 is wasteful here.
4. **For Reduce ops on BOOL masks (Sum/Mean)**: `Cast(BOOL, F16/INT32)` first. F16 cheapest if sum stays < 2048; otherwise INT32.
5. **For `Where` on color indices (0-9)**: UINT8 (1 B) is preferred at opset 13+. F16 is the fallback only if you need Add/Sub/Mul arithmetic at opset ≤ 13.

## How this table was built

Each row was empirically tested by attempting to build + load + run a one-op model with the relevant input dtype combination. Distinction made between **schema rejection** (caught in `onnx.checker.check_model`) and **kernel rejection** (caught in `ort.InferenceSession` with "NOT_IMPLEMENTED"). Versions: `onnx==1.21.0`, `onnxruntime==1.24.4`, `ORT_DISABLE_ALL`, `CPUExecutionProvider`. Re-probed 2026-05-23 after audit.

When you discover a dtype rejection not in this table, drop a DONE file with the empirical error message, opset, ONNX/ORT versions, and the op signature you tested.

## Related references

- `ort_reality.md` — for behavioral divergences from the spec (Pad-BOOL, OneHot dtype matching, Equal opset-10).
- `ort_reality.md` — for error message → root cause lookup.
- `ort_reality.md` — for "which opset for which feature".
- `techniques.md §1` (BOOL-first family calibration) — quantifies when BOOL chains beat F16 even with more nodes.

- **TopK: UINT8 input UNSUPPORTED on ORT CPU** (measured task280, 2026-06-04 — a uint8 TopK
  shortcut had to be abandoned; use F16/F32/INT32 inputs for TopK).

- **GatherND: CPU indices are INT64-ONLY** (Gather/GatherElements accept INT32) — a GatherND
  index plane costs 8 B/elem vs 4; prefer Gather/GatherElements when addressing allows
  (`_archive/Research_nary_fanin_catalog.md`, 2026-06-04; extraction: techniques #40).
- **Mean: float-only on CPU**; **Sum is not a bool reducer** (cast masks numeric upstream);
  Conv/GlobalAveragePool/GlobalMaxPool/GlobalLpPool = float-only (same source).

| `BitwiseAnd`/`BitwiseOr`/`BitwiseXor` (opset 18) | UINT8, INT32, UINT32 all OK (probed 2026-06-04, task338 route-2 audit) | — | Kernels exist on CPU EP; passed checker full_check + execution. Wrap semantics on u8 Xor confirmed (255^2=253). |
| `BitShift` (opset 11+, probed at 18) | UINT8, UINT32 OK (LEFT probed; u8 wraps 255<<1=254) | signed INT8/INT32 (schema: unsigned only) | Usable for packed-row tricks, BUT: exact 25-bit packing has no ≤625B entry path and per-cell unpack broadcast is ≥2,500B INT32 — packed XOR-prefix priced DEAD for [25,25] per-cell rules (task338.md route-2 verdict). |
| `MatMul` (opset 13) | INT32/UINT32 + INT64/UINT64 registered on CPU (`BuildKernelDefConstraints<int32_t,uint32_t>()` — J1 research, source-verified) | — | Enables exact integer packs: u32 row dot-products vs F32's 2^24 ceiling. |
| `Einsum` | float, double, INT32, INT64 in the ORT 1.24.4 CPU kernel table; local probe notes above and a 2026-07-03 mini-probe observed F16 load/run on target builds | U8, unsigned, INT16 | Repeated-subscript diagonal READS implemented (J2); output-repeated labels rejected. Treat F16 `Einsum` as runtime-probed, not source-table guaranteed: O8 source-read found v1.24.4 `OperatorKernels.md` omits F16 despite local F16 execution. |


# ===== ort_reality.md =====

# ort_reality.md — ORT/schema behaviors that matter in NeuroGolf

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
- `Where(BOOL, BOOL, BOOL)` → rejected (also at opset 16: `Could not find an implementation for Where(16)`, task177 2026-06-21 — a BOOL-branch "Where-as-output one-hot" fails to LOAD; route through a U8 `Where` -> `Equal(channels)` tail instead, see techniques.md #67)
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
| `ReduceProd` | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | **✓** | ✓ |

(`ReduceProd` row fully probed 2026-06-06, opset 12/13: identical profile to `ReduceSum` — BOOL/INT8/UINT8/INT16 rejected at load (INVALID_GRAPH), INT32/INT64/F16/F32 run. F16 used in task078 v1 channel-pair sign product.)

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

Cross-ref: `ort_reality.md` for the full per-op dtype-support table.

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
- O8 narrowing gate (2026-07-03): moving `Cast` before `Gather`/`Slice` narrows
  more values than the selected result, so selected-scalar equality is
  insufficient. Require `absmax <= 65504` for arithmetic consumers (unless a
  documented `+inf` carrier goes only to final `>0` decode), tensor dumps with
  no new `nan` or undocumented infinities, and no dataflow path to invalid
  patterns such as `0*inf` or `inf+(-inf)`. In positive decode, `+inf > 0`
  behaves true while `nan > 0` behaves false.

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


# ===== ort_reality.md =====

# ort_reality.md — error message → meaning → fix

When ORT or the validator rejects a model, the error message is often cryptic. This table is the lookup. Add to it when you debug a new one (drop a `DONE_*.md` in `.update/`).

## How to triage an unfamiliar error

1. **Run `onnx.checker.check_model(model, full_check=True)`** first. Many errors surface here before ORT, with clearer messages.
2. **Run `onnx.shape_inference.infer_shapes(model)`** next. If this raises, the issue is a shape/dtype mismatch.
3. **Try `onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])`** as a final load-time check.
4. If steps 1-3 all pass and the model still fails at score time, the issue is at execution (usually an op-specific dtype mismatch or unsupported op).

## Error table

| Error fragment | Likely cause | Fix |
|---|---|---|
| `"two nodes with same node name"` | Multiple emitted nodes share `name` field. Common when batch-generating with `helper.make_node("Foo", name=f"{x.name or 'fallback'}")` and multiple x's have empty names. | Use a counter: `def _uniq(p): counter[0] += 1; return f"{p}_{counter[0]}"`. Every node gets a unique name. |
| `"Type Error: Type X of output arg (name) does not match expected type Y"` | The graph output's declared dtype doesn't match the dtype produced by the node that emits it. Common after dtype-changing surgery (BOOL/UINT8 Pad rewrites). | Update `graph.output[0].type.tensor_type.elem_type` to match the producing node's output dtype. |
| `"[ShapeInferenceError] Incompatible dimensions"` | Stale `value_info` cached shapes from before graph surgery. ORT loads the cached shapes and they conflict with the new graph structure. | `graph.ClearField("value_info")` before save. (Or `model = onnx.shape_inference.infer_shapes(model)` if you want fresh inferred shapes.) |
| `"[ShapeInferenceError] (op: Op_Type)"` with no other detail | The named op's spec violation — usually wrong input count, wrong dtype, or wrong attribute. | Look up the op in `onnx.defs.get_schema("Op_Type")`. Check input/output counts and dtype constraints. |
| `"Op (Pad) [ShapeInferenceError]"` opset 10 | `pads` attribute length is wrong, or wrong layout. | `pads` is `INTS` of length `2 × rank`. Layout: `[x1_begin, x2_begin, ..., xN_begin, x1_end, x2_end, ..., xN_end]`. NOT interleaved. |
| `"Op (Equal) [ShapeInferenceError]"` opset 10 | Equal received float inputs. Opset 10 Equal accepts BOOL/INT32/INT64 only. | Cast inputs to INT32 first or bump to opset 11+ for float/numeric Equal. See `ort_reality.md §4` for the shared-Cast trick to net positive bytes. |
| `"Invalid model. opset_import for domain '' is null"` | The model has no `opset_import` for the default domain. | Always include `opset_imports=[helper.make_opsetid("", N)]` in `helper.make_model`. |
| `"Op (X) is not a registered function/op"` | Op name typo, or op exists in newer opset than declared. | Verify op name spelling. If the op exists in a newer opset, bump opset_import. If banned (LOOP/SCAN/NONZERO/UNIQUE/SCRIPT/FUNCTION/COMPRESS) — the validator rejects regardless of opset. |
| `"Constant node ... has no value attribute"` | A Constant node missing its `value`/`value_floats`/`value_ints`/`value_strings`/`sparse_value` attribute. | Provide one. Or, preferred: move the value into `graph.initializer` (no node-overhead). See `techniques.md` §18 (const_to_init). |
| `"Cannot find node input 'foo'"` | A node references a tensor name that isn't an input, initializer, or earlier node's output. | Reorder nodes so producers come before consumers. Or fix the name — common typo source. |
| `"Cannot find node input 'foo'"` after deletion | You deleted a producer but downstream consumers still reference its output. | Rewire consumers' input names before/during the delete. Build a producer→consumer map first; mutate atomically. |
| `"Initializer name conflicts with input name"` | An initializer shares a name with the model input or output. NeuroGolf validator's `calculate_memory` returns None (model rejected). | Rename the initializer. |
| ORT_DISABLE_ALL OneHot error | OneHot index/depth dtype mismatch, most often INT32 indices with INT64 depth. | Make `indices` and scalar `depth` the same dtype (INT32+INT32 or INT64+INT64). See `ort_reality.md §3`. |
| `"InferenceSession failed: [ONNXRuntimeError] FAIL : Node (X) Op (Where) [TypeInferenceError]"` | `Where(bool, bool, bool)` — ORT does not support BOOL data inputs. | Rewrite as `Or(And(c, a), And(Not(c), b))`. See `ort_reality.md §2`. |
| `"InferenceSession failed: [ONNXRuntimeError] FAIL"` (generic, no detail) | Many possible causes; ORT didn't surface specifics. | (1) Run `onnx.checker.check_model(model, full_check=True)`. (2) Run `onnx.shape_inference.infer_shapes(model)`. (3) Check every initializer is referenced. (4) Check no banned ops. (5) Check opset_import is set. |
| `"check_model: ValidationError"` (with detail) | Schema violation in your construction. | Read the detail. Almost always tells you the exact node and constraint. |
| `"Sparse tensor with no values"` | Sparse initializer's values are empty. | Provide values in `init.sparse_tensor.values` and indices in `init.sparse_tensor.indices`. |
| `"dimensions: size <= 0"` | A `dim_value <= 0` in some tensor's shape. Validator rejects. | Fix the shape. Or, if symbolic, ensure all dims are concrete `dim_value > 0` (no `dim_param`). |
| Score returns `None` after model loads | One of the validator's rejection paths triggered AFTER load: e.g., more than 1 input, name collision i/o ↔ init, subgraphs, functions, non-default opset domain, scalar tensor type issues. | See `core.md` "Validator rejection criteria" — work through the list. |

## When to suspect a Kaggle-specific issue (not ORT)

If the candidate scores fine locally (`score_network_v2` returns a number) but Kaggle returns ERROR for that task:

- **Cross-task memory cascade**: see `core.md` "Cross-task memory cascade bug". A high-memory ONNX in slot N can ERROR slot N+1. Local validators cannot detect this.
- **Kaggle scoring formula drift**: rare but happens (rewrite on 2026-05-04). Check forum thread before debugging further.

## Diagnostic snippets (copy-paste)

```python
# Full pre-flight check
import onnx, onnxruntime as ort
model = onnx.load("candidate.onnx")
onnx.checker.check_model(model, full_check=True)
model = onnx.shape_inference.infer_shapes(model)
sess = ort.InferenceSession("candidate.onnx", providers=["CPUExecutionProvider"])
print(f"Loaded OK. Inputs: {[i.name for i in sess.get_inputs()]}")
```

```python
# List all initializer names + shapes + dtypes
for init in onnx.load("x.onnx").graph.initializer:
    print(f"{init.name:<40} shape={list(init.dims)} dtype={init.data_type}")
```

```python
# List all node op_types in order
for n in onnx.load("x.onnx").graph.node:
    print(f"{n.op_type:<15} in={list(n.input)} out={list(n.output)}")
```

```python
# Show where shape inference fails
try:
    onnx.shape_inference.infer_shapes(onnx.load("x.onnx"), strict_mode=True)
except onnx.shape_inference.InferenceError as e:
    print(f"Shape inference broken: {e}")
```


# ===== ort_reality.md =====

# ort_reality.md — opset matrix for NeuroGolf

The project's `neurogolf_utils.py` declares `_IR_VERSION=10, _OPSET_IMPORTS=10` as defaults for `build_template_network`. **The validator does NOT enforce opset 10** — confirmed empirically by building opset-11 models that score correctly.

The actual constraint at scoring time is the banned-op list, regardless of opset version.

## Banned ops (validator rejects regardless of opset)

```
LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS
```

Source: `_EXCLUDED_OP_TYPES` in `neurogolf_utils.py`. These cannot appear in any submitted graph. CC analysis via Unique-based label compaction is permanently blocked.

**Plus a separate validator rule (core.md): any op type whose name contains the substring `"Sequence"` is rejected** — `SequenceAt`/`SequenceConstruct`/`SplitToSequence`/`ConcatFromSequence` AND `ReverseSequence` (the substring catches it). R6 (2026-06-19) scorer-printed `Op type SplitToSequence is not permitted` and `Op type ReverseSequence is not permitted`, so the entire sequence/optional/string/sparse container family (N2) is dead with **no exceptions** — every "delay materialization in a sequence" idea rematerializes the same bytes at the tensor bridge anyway. `ReverseSequence` additionally can't even load for NCHW H/W reversal (ORT rejects `time_axis>=2`); use `Slice` negstep (#44) for flips.

## Opset 11+ ops verified working under the scorer

Confirmed via opset-bump tests (2026-05-12). All of these score correctly:

- **CumSum** — opset 11 (F32, F64, INT32, INT64). **F16 CumSum requires opset 14** — bumping from opset 11→14 drops the `Cast(F16→F32) → CumSum(F32) → Cast(F32→F16)` roundtrip, saving ~3,200B per CumSum block on NeuroGolf shapes. **Common falsified claim**: "F16 CumSum is unsafe if output feeds Gather as indices." Verify the Gather connection FIRST — `Gather(data, indices)` only cares about precision on the `indices` input (position 1), not `data` (position 0). F16 data + INT64 indices is exact. Empirical: task280 v1 saved -25,600B via opset 14 bump after debunking the wrong claim.
- **ScatterND** — opset 11. Scatter values into a tensor at index locations. task118 uses bool ScatterND (-38,341 bytes).
- **Round** — opset 11. Round half-to-even.
- **TopK with runtime K** — available from **opset 10** (not 11). `TopK-10` already makes K a required input; opset 11 only adds `largest`/`sorted` attributes. Verified at opset 10 under `ORT_DISABLE_ALL` 2026-05-23.
- **Equal expanded** — opset 11 added the full numeric set (UINT*/INT*/FLOAT16/FLOAT/DOUBLE). **BOOL has been in Equal since opset 1** — NOT added at opset 11. Opset 13 added BFLOAT16. Under opset 11+, `Equal(F32, F32)` works; at opset 10 floats are rejected (use INT32 cast).
- **GatherElements / ScatterElements** — opset 11.
- **BitShift** — opset 11.
- **DynamicQuantizeLinear** — opset 11.
- **Clip expanded** — opset 11 made min/max inputs (previously attributes).
- **Pad with input pads** — opset 11. Switches from attribute form to input form. Also accepts UINT8 input dtype.
- **OneHot** — opset 9 originally, multiple subsequent updates. Works fine; INT32 indices are allowed
  when the `depth` scalar is also INT32, and INT64 indices need INT64 depth (see `ort_reality.md §3`).
- **ReduceMax(BOOL)** — **opset 20**. This is the first Kaggle-confirmed use of a direct BOOL reducer
  in this family: task191 v6 replaced 8 `Equal` + 6 `Or` membership plumbing with two broadcast
  `Equal` tensors and two `ReduceMax(BOOL)` nodes, scoring 6455.23 on 2026-06-02 (+0.29 over the
  current base). Because ReduceMax axes are an input from opset 18 onward, use the axes-input schema.
  At opset <=17, keep the old `Cast(BOOL, UINT8) -> ReduceMax(UINT8)` path.

## When to bump opset

Bumping `opset_import` from 10 → 11 (or higher) is occasionally a net win for cost. Examples:

- **opset 10 Pad-attr + Cast(bool→F32) → opset 11 Pad(input) + Cast(bool→UINT8)** — UINT8 is 1 byte/elem (vs F32 4 bytes), Pad-11+ supports it. Saved -742 bytes on task319. Cross-ref: `techniques.md` rule 6.
- **opset 10 float Equal → opset 11 numeric Equal**: opset 10 `Equal` accepts BOOL/INT32/INT64 only under ORT. BOOL Equal already returns BOOL at opset 10; bump to opset 11+ only when the inputs are F16/F32/F64 or other numeric dtypes rejected by opset 10.
- **CumSum-based algorithms**: replace a chain of cumulative-add ops with one CumSum. task382 used CumSum+Gather for geometric encoding (-73,443 bytes from-scratch rebuild).
- **BOOL membership reductions**: if the graph is already opset-bumped and the membership plane is
  naturally BOOL, opset 20 can remove the `Cast(BOOL, UINT8/F16)` before `ReduceMax`. This is not just
  locally valid: the Kaggle grader accepted task191 v6.

When you bump opset:
- Update `opset_import` for the default domain (`""` or `"ai.onnx"`).
- The IR version (`model.ir_version`) typically stays the same — IR is a separate concept from opset.
- Re-verify: some ops have semantics that changed between opsets (e.g., Pad's `pads` moving from attr to input, ReduceMax's `axes` becoming optional in opset 18+).

**Ceiling: ORT 1.24.4 loads ai.onnx through opset 25 only.** opset 26 (`BitCast`, `CumProd`) passes the ONNX checker but is rejected at model load BEFORE kernel lookup (R6/N1 2026-06-19) — never set `opset_import` > 25. The opset 23-26 surface is otherwise barren for our metric (techniques #94); `Attention-23/24` is smoke-probed viable but parked because O3 found zero current Softmax/attention chains.

## ORT-CPU op × narrow-dtype matrix (2026-06-01, task286 v1 — empirically probed)

For shrinking a one-hot-input graph (the common NeuroGolf shape: `input` F32 [1,10,30,30]),
you want intermediates in UINT8/INT8 (1 B) not F16 (2 B) / F32 (4 B). What ORT CPU actually
accepts at `ORT_DISABLE_ALL` (the strict scoring mode):

| Op | UINT8 / INT8 | Notes |
|---|---|---|
| `Cast` | ✓ | F32↔UINT8↔INT8↔F16↔BOOL↔INT32 all OK |
| `Gather` | ✓ | data any dtype; **indices must be INT32/INT64** (traps.md §8c) |
| `ReduceMax` | ✓ (UINT8/INT8) | one-hot: `ReduceMax(x,axis=ch)` == "active" mask == `ReduceSum`, but stays 1 B |
| `Max` / `Min` | ✓ | UINT8 and INT8 both load |
| `Greater` / `Less` / `Equal` | ✓ → BOOL | the **AND-NOT trick**: `valid AND NOT wall` (0/1 masks) = `Greater(valid_u8, wall_u8)` in ONE op — replaces a 7-op BOOL `Cast/Not/And` chain |
| `MaxPool` | ✓ **opset ≥ 12** | rejected at opset 10/11; **BOOL always rejected** — 1-byte floor for a flood/BFS is INT8/UINT8 |
| `ConvInteger` | ✓ (UINT8 in → INT32 out) | weighted channel-sum (e.g. color-index `[0..9]`) without an F16 input cast; output is INT32 (4 B), narrow it immediately |
| `Mul` / `Add` | ✗ at opset ≤ 13, ✓ **at opset ≥ 14** | the lever for INT8 arithmetic chains; colors/small-ints stay exact |
| `Sub` | ✗ UINT8 **and** INT8 (all opsets) | use `Greater`/`Max` reformulation instead (see AND-NOT trick) |
| `Where` | ✗ INT8 data in ORT CPU | condition BOOL is fine; data branches may be **UINT8**, F16/F32, INT32, or INT64. INT8/INT16 and BOOL data branches reject. Prefer UINT8 for color-index select tails at opset 13+. |

### Reusable pattern: "UINT8-cast Phase A" for one-hot inputs
`Cast(input, F16)` [10-ch = 18,000 B on 30×30] is usually the single biggest non-algorithmic
tensor. Halve it: `Cast(input, UINT8)` [9,000] and route consumers through the int-friendly
ops above —
- per-cell **color index** → `ConvInteger(xu8, [0..9]_u8)` → INT32 → narrow,
- **active** mask → `ReduceMax(xu8, axis=ch)` (UINT8),
- **channel** picks (bg/wall) → `Gather(xu8, [k])` (UINT8),
- **mask algebra** (`a AND NOT b`, `a OR b`) → `Greater`/`Max` (UINT8→BOOL), never `Sub`/`Mul`.
Caveat: **initializer params are counted by element count, not dtype** (`calculate_params`
sums `prod(dims)`), so DON'T keep both an F16 and an INT8 copy of a mask param — pick one
dtype and derive a BOOL view with `Cast`. task286 v1: 119,906 → 107,150 byte-equal (+0.1125 LB).

## What stays out of scope

Opset 11+ enables some new ops, but the banned list still applies. So:

- **Loops / conditional branching**: blocked. LOOP, SCAN, COMPRESS all banned. No control flow.
- **Functions / custom subgraphs**: blocked. `model.functions` must be empty; nodes cannot have `AttributeProto.GRAPH` or `GRAPHS`.
- **NonZero**: blocked. Cannot extract non-zero indices. Use alternative approaches (ReduceSum + Greater, etc.).
- **Unique**: blocked. Cannot deduplicate values. Label compaction in CC analysis requires alternative.

## Practical guidance

Most NeuroGolf models historically built against opset 10. The validator accepts opset 11+ silently.
As of 2026-06-02, the highest publicly confirmed Kaggle-scored opset is **20** (task191 v6,
`ReduceMax(BOOL)`). Bumping is safe as long as:

1. You don't introduce a banned op.
2. You update any ops whose representation changed (e.g., Pad attr→input form).
3. You run the full verify gate after the bump.

`tools/spread_patterns/pass_opset_bump_pad_uint8.py` is the only auto-detector that bumps opset. Other patterns leave opset_import alone; if you write a new pattern that needs opset 11+ ops, document the opset bump in its docstring and add a guard so it only fires on already-opset-11+ models OR explicitly bumps and re-validates.


# ===== K2 source-read compendium (2026-06-06) =====

# ORT 1.24.4 CPU exact semantics <!-- wave-add below --> — source-cited (from `_archive/Research_k2_ort_exact_semantics.md`)

**Opset 19-22 audit (M1, 2026-06-08, `_archive/Research_m1_opset19_22_op_audit.md`)**: ORT 1.24.4 CPU has NO kernel for ImageDecoder / DeformConv / Bernoulli / main-domain GroupNormalization (contrib `GroupNorm` only). Opset-22 bf16 schema widening is NOT honored on CPU for Conv/ConvTranspose/AveragePool/GlobalAveragePool/GlobalMaxPool/GridSample/Dropout/Elu/Det (all stay float/double-only). Usable post-18: bool ReduceMax/ReduceMin @ opset 20 (CPU bool-reduce); Pad(mode="wrap") @ opset 19 (circular pad, but PadCPU is dtype-narrower than the ONNX schema). **`GridSample` is GRADER-VALIDATED (Boladi task351, 2026-06-10): the Kaggle grader CREDITS it (own-stack delta-validation +3.04, 31,271→1,498) — it is no longer an op to avoid; float-only on CPU, cast input+grid to float. See techniques.md #121.**

**Generator + spatial op CPU dtypes (N3/N4 source-read of ORT v1.24.4 + local probes, 2026-06-19)** — ONNX schema is wider than the ORT CPU kernels; build to the ORT cell, not the spec. `EyeLike` = f32/f64/i32/i64/u64 (**NO bool/u8** — the common trap; the old zero_param_generators `EyeLike(BOOL)` examples were wrong); `Trilu` = bool/f32/f64/i64 (**NO u8**); `OneHot` output values = f32/i64 (**NO bool/u8** — use `Equal(label, ch_ids)` for a bool channel mask); `Range` = f32/f64/i16/i32/i64 (**NO u8/f16**); `CumSum` = f32/f64/i32/i64 (no bool/u8; F16 needs opset 14); `Resize` = f32/i32/i8/u8 (no bool/f16); `DepthToSpace` = f64/f32/**u8** but `SpaceToDepth` = f64/f32 **only (no u8)** — the pair is NOT symmetric; `MaxPool` = f32/f64/i8/u8 at opset≥12. Rule of thumb: for a **bool/u8 diagonal or triangle**, use `Trilu(bool_ones)` or `Equal(iota_row, iota_col)`, never `EyeLike(bool)`.

**Wave additions (task209 probes, 2026-06-07)**: `Where(bool,i8,i8)` NOT_IMPLEMENTED on CPU (route selects through u8 — wrapped negatives only touch unselected lanes); `Min(u8)` kernel exists at opset 13; i8 arith (Sub/Div/Less/GreaterOrEqual/Cast) all kernel-backed at opset 14, integer Div(i8) truncates toward zero like i32.

Research card K2 read the **v1.24.4 GitHub tag** kernel sources. Distilled rules (full tables,
citations, and counterexamples in the deliverable):

## Act-now rules (the two decisive ones)
- **`TopK(sorted=0)` ORDER IS UNSTABLE — never rely on it.** ORT uses `std::nth_element` and
  SKIPS the final sort: the selected SET honors the lower-index-tie rule, but the order inside
  it is an STL implementation detail. `sorted=1` is safe (ties = lower original index first,
  source: `top_k.cc::GreaterValueCmp` + `std::sort`).
- **`Where(cond, X, Y)` has NO bool kernel** (registrations: u8/i32/i64/float/double/string at
  9–15; u8/i32/i64/float at 16; F16/BF16 commented out). Confirms trap §2 at source level.

## Integer arithmetic (native C++ semantics, not "model-portable")
- **`Mod(fmod=0)`: result sign = DIVISOR sign** (`%` + corrective add): `Mod(-3,2)=1`.
  **`Mod(fmod=1)` = `std::fmod`: sign = DIVIDEND**: `Mod(-3.0,2.0,fmod=1)=-1.0`.
  Float inputs REQUIRE fmod=1.
- **Integer `Div` truncates TOWARD ZERO** (native `/`): `Div(-3,2) = -1`, NOT floor's -2.
- **`Pow` = `std::pow` + `static_cast` back**: `Pow(int 2, int -1) = int(0.5) = 0`. Exponents
  2/3 special-cased to `x*x`/`x*x*x`. Treat integer Pow edges as hostile.
- **Signed overflow = C++ UB; unsigned wraps mod 2^N.** Never let int paths overflow.

## Selection / rounding / NaN
- **ArgMax/ArgMin ties: FIRST index by default; `select_last_index=1` flips** — source-confirmed
  (`reduction_ops.cc`); matches our #62b usage.
- **`Resize`: PIN EVERY ATTRIBUTE explicitly.** Defaults: `coordinate_transformation_mode=
  half_pixel` (`x_orig=(x+0.5)/scale-0.5`), `nearest_mode=round_prefer_floor` (half-ties go DOWN).
  K2 ranks Resize-mode mismatch the #1 silent-divergence risk — a one-index sample difference
  is one wrong boolean cell after `>0`.
- **`Min`/`Max` elementwise PROPAGATE NaN** (`Eigen::PropagateNaN`, F32 and F16 paths).
  Integer-valued graphs are immune; never feed a possible-NaN into min/max chains.
- **Negative pads (crop) confirmed in the common Pad path** (`pads`+`slices` both plumbed into
  `PadImpl` regardless of mode) — source backing for §41 negpad.
- Comparisons with NaN are false; `Equal(+0.0,-0.0)` is true (standard ordered semantics;
  not spec-prose-confirmed, flagged [DERIVED] in K2).

## Determinism
- Single-thread CPU + ORT_DISABLE_ALL: **no intentional nondeterminism found** in the covered
  kernels. The hazards are under-specification (TopK sorted=0), not randomness. Multi-threaded
  reductions can reorder float accumulation — our intra_op=1 config avoids the class.

## Still open ([SPECULATIVE] in K2 — don't build on these without a local probe)
Hardmax tie line · ReduceMax/Min NaN aggregator · exact f32→f16 / float→int Cast helper lines ·
QuantizeLinear clamp/round line · legacy opset-10 Pad negative-pad path.

## Wave additions (2026-06-07)
- **`Mod(F32, fmod=1)` with a CONSTANT divisor is exact on integer-valued floats** (probed
  opset 13, values ≤1010; task078 shipped on it). The Div/Mod-by-INPUT→SIGFPE trap is about
  runtime-zero divisors; const divisors are safe.
- Grader-validated this wave: **BitwiseAnd (205 v31)**, **QLinear saturating casts @ opset 19
  (256 v12)**, **C13 pad-fill @ opset 12 (256 v11)** — all exact-priced.

## Wave-2 additions (2026-06-07 morning)
- **ConvInteger GRADER-VALIDATED** (task357 v8 @ 375, exact) — the integer-counting family
  above QLinearConv's u8-255 bound is open (K5's recommendation realized).
- **QLinearConv-as-output**: y_zero_point MUST be u8 (INT8-y NOT_IMPLEMENTED, probed);
  all 8 inputs mandatory (no omission); u8 output decodes via >0 (205/125 precedent).
- **opset-14 u8/i8 probe extensions** (task125): Mul(U8), MaxPool(U8/I8), Pad(U8 value-init),
  Cast(F32→U8/I8) all run. Extends #74's lane list.
- **Negative Gather indices legal** (wrap semantics) — probed via zeros-profile.
- **i32 graph output decodes** through the official `>0` pipeline (task345 887 exact).
- **Einsum(INT8): NO kernel** (task354 v9 probe, score None) — einsum operands stay F32/F16-castable class.

## Wave-3 additions (2026-06-12…19 lesson/DONE sweep)
- **`Cast(F32→UINT8)` is MODULAR, not saturating** (task281): `1027→3`, `256→0` (mod 256), NOT clamped to 255. Breaks any "large-F32 → U8 → ArgMax" foreground detection that assumes saturation — keep the comparison in F32, or clamp explicitly before the cast.
- **`ScatterND` `updates` dtype is HARD-bound to the data tensor** (task154): F16 `updates` into an F32 data tensor is REJECTED (no implicit cast); narrowing updates to save bytes means narrowing the whole scatter. A rank-4 scalar index also cannot drop its batch coordinate without breaking the free-`input` passthrough.
- **O2 scatter-reduction split (2026-06-19):** `ScatterElements` reductions are the safe duplicate-target shelf on CPU; ORT walks a deterministic flattened fold, and spec-clean `min`/`max` require opset 18. Duplicate-target `ScatterND` reductions (`add`/`mul`/`min`/`max`) are unsafe for exact histograms/overlap/priority routing on ORT CPU 1.24.4 because the visible implementation parallelizes shared-output writes without conflict resolution. Unique-target `ScatterND` sparse patches remain valid.
- **opset-18 `Pad` `axes` input MUST be INT64** (task048): INT32 axes → `INVALID_GRAPH` at load. The axes form still pays off (carry only the padded axes — 4 elems vs the full 8-element `pads`, saving `2·rank − len(axes)` params; task183), but pin INT64.
- **Skip `Cast(INT64→U8)` after ArgMax when the sole consumer is `Equal` vs an index vector** (task192): `Equal(INT64, INT64)` is valid at opset 13+, so the cast is dead weight.
- **`BitwiseAnd/Or/Xor` are opset-18 ONLY — re-confirmed** (probe 2026-06-19): a lesson claimed opset-14 u8 Bitwise; FALSE — checker AND ORT reject `BitwiseAnd` at opsets 13/14/17 ("No Op registered … domain_version of 14"); only opset 18 loads/runs. The library's opset-18 statements (#112b, op_cost_patterns) stand.
- **Cheap "pick-one-of-two" recolor/select expressions that DON'T score** (task040 floor probes, all score None / ORT-reject — use a U8 mask + arithmetic select instead): **rank-0 (scalar-index) `GatherND`**, **data-dependent dynamic `Reshape`** (shape from a runtime tensor), and **`If`-branch** (also confirmed scorer-None on task390 #93 boundary). When a half-axis/conditional recolor tempts you toward one of these, it won't credit — express the branch as `mask*a + (1-mask)*b` or `Where(maskbool, a, b)` on U8 planes.
- **Slice/Gather/Conv worker probes (task079):** `Slice` index tensors can be all-INT32 if Tind matches across starts/ends/axes/steps; `Gather` accepts BOOL data, so gather a small BOOL slice before casting for `ArgMax`/reductions; `Conv` accepts negative pads, allowing 1x1 channel-collapse plus crop in one node.
- **U8 reductions at low opset (task324):** `ReduceMax(U8)` and `Max(U8)` work at opset 12/13, while `ReduceSum(U8)` is still rejected. Bump to opset 12+ when a 1-byte OR/max carrier replaces an F16 plane; do not expect U8 sums.
- **U8 Mod+Pad index chains (task110):** `Mod(U8,U8)`, `Pad(U8, UINT8_const)`, `Cast(INT32->U8)`, and `Cast(U8->INT32)` all run at opset 13. `Mod(U8,INT32)` rejects, and `Gather` indices must be INT32/INT64, so the U8 chain only pays when it replaces more than one pre-Gather INT32 intermediate.

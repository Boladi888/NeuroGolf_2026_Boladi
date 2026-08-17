# I8 constraint arithmetic chains — practical proof and library correction

## What I discovered

At opset 14+, `Mul`/`Add`/`Sub` on INT8 ARE kernel-backed in ORT (same row 41 note in ort_reality.md
correctly says "below opset 14" = rejected). At opset 18, full arithmetic works. This enables
CONSTRAINT CHAINS (not just render/indexing chains) to run in I8.

Task209 v14 proves it on a real constraint chain:
- Row chain: `Sub(minr_i8, Mul(iota3c_i8[3,1], mag_i8)) → orir_i8[3,1]` then `Add(orir_i8, maggh_i8) → endr_i8[3,1]`
- Col chain: `Sub(anc_i8, Mul(iota5r_i8[1,5], mag_i8)) → oric_i8[1,5]` then `Add(oric_i8, maggw_i8) → endc_i8[1,5]`
- Comparison: `Greater/LessOrEqual(I8[3,1], I8[])` — confirmed kernel-backed at opset 14+.

Savings: 40B I32 → 10B I8 (row) + 64B I32 → 16B I8 (col) = 78B saved at the cost of ~10B new intermediates (I8 bounds casts). Net: -73B memory = -73 score on task209 (4995→4922).

## Library correction needed

`ort_reality.md` line 41 says: "so small-int index arithmetic (e.g. `row*z+col`, values <128)
can't use a 1–2 B dtype; INT32 (4 B) is the floor." This statement is WRONG for opset 14+. The
correct statement is: INT32 is the floor **below opset 14**. At opset 14+, INT8 arithmetic
(Mul/Add/Sub) works, and values that fit [-128,127] can use I8.

The clarification: "INT16/UINT16 has no kernel ANY opset" still holds. Only INT8 works at opset 14+.

## Value-range checklist for I8 constraint chains

When converting I32 constraint chains to I8, verify:
1. All intermediate values fit [-128, 127].
2. `Mul(iota_i8[K], mag_i8)`: max = max_mag * max_iota_val. For task209: 4*4=16 ✓
3. `Sub(anchor_i8, iota_mul_i8)`: min = 0 - max_mag*max_K_steps. For task209: 0-16=-16 ✓
4. `Add(sub_result_i8, extent_i8)`: max = max_anchor + max_mag*max_extent. For task209: 17+20=37 ✓
5. Bounds for comparisons: bbot+1, bright+1 — check ≤127.

## Params saving: iota init replacement

If an existing I8 `[N]` init can be `Reshape`-d to `[1,N]` for broadcasting, the original
`[1,N]` I32 init can be removed (-N params). The Reshape may be folded by onnxsim into a
const I8[1,N] init (same params as the folded constant, net 0 params). But the I32 init IS
removed (-N params). Combined with removing I32 shape init (e.g. `sh15_rc` if onnxsim folds
it), net may be 0 params. Task209: removing `iota5r I32[1,5]` (-5 params) + onnxsim folds
Reshape to const I8[1,5] (new init, +5 params) = 0 net params from that specific pattern.

## Where encountered

- Task 209, `The_400_Tasks/Task_by_Task/task209/prototypes/build_task209_compress_v14.py`
- Staged candidate: `staged_submits/task209_i8constraints_cost4922/`
- All 5 gates passed (byte-equal vs submission.zip, 99.21% truth, profile=static=4922).

## What's missing from the library

`ort_reality.md` §41 needs the clarification that INT32 is NOT the floor at opset 14+; INT8
arithmetic works, just not INT16. Also worth noting: `Greater`/`Less`/`LessOrEqual`/`GreaterOrEqual`
work on INT8 at opset 10+ (already documented), confirming full constraint-chain I8 chains at opset 14+.

## Sorter note: Onnx_Advanced/ort_reality.md — extend existing row (Mul/Add/Sub) to clarify INT8 works at opset 14+ and "INT32 is the floor" only applies below opset 14. Cross-ref task209 v14 as proof exemplar.

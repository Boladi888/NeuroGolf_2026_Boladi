# task363 stretch (1600–1700) — WALL: memory floor is the two forced F32 channel slices

- **Live bar at record time:** cost **1882** (OnnxlyFans `manifest.csv` + workbook, owner Aditya, sha `6af5050c28ba7fc7261133c64a03f2e54f28a77610b78871e99aa9b7fa637855`, updated 2026-06-15T19:08:38Z).
- **Current staged win:** `staged_candidates/task363_regionfillrewrite_safe_cost1881/task363.onnx`, sha `67444d6d0d64e06c15a11b262a8fc882855f98f8d32e92a144560eb1bdb4508f`, **cost 1881** (memory 1790 + params 91), profile=1881. **−1 vs bar.** Byte-equal to legal full-bias reference (1888). This record is about the *stretch to 1600–1700*, which is a wall.

## Exact charge accounting (from `neurogolf_utils.calculate_memory` / `calculate_params`, read 2026-07-03)
`calculate_memory` = Σ over **every** intermediate tensor of `max(static_bytes, observed_bytes)`, where the **`input` and `output` tensors are explicitly excluded** (`if tensor_name in ['input','output']: continue`). `calculate_params` = Σ initializer **element counts** (dtype-free). Verified: summing all value_info bytes minus the 9000 B `output` = **1790**, matching the measured memory exactly.

Memory breakdown of the staged 1881 graph (bytes):

| tensor | dtype | shape | bytes |
|---|---|---|---:|
| `m5_f32` | F32 | [1,1,10,10] | **400** |
| `m2_f32` | F32 | [1,1,10,10] | **400** |
| `render_x_u8` | U8 | [1,2,10,10] | 200 |
| `valid_u8` / `valid_masked_u8` / `stamp_special_u8` / `m5_u8` / `m2_u8` | U8 | [1,1,10,10] | 100 ×5 = 500 |
| `patch_rows_u8` | U8 | [1,1,4,10] | 40 |
| ~9 small [1,1,4,4] / [4] tensors | U8/I32/BOOL | — | ~144 |
| remaining scalars/vectors | — | — | ~106 |
| **output** `[1,10,30,30]` U8 | — | — | **FREE (excluded)** |
| **total charged memory** | | | **1790** |

## Why 1600–1700 is a wall in this representation family
Reaching 1600–1700 needs ≥180 B off memory. The two candidate levers both fail:

1. **The 800 B of F32 channel slices are structurally forced.** The grader input is F32 one-hot `[1,10,30,30]` (`encode()` writes float32). Extracting the color-2 and color-5 10×10 sub-planes as U8 requires a **Slice on the F32 input → [1,1,10,10] F32 = 400 B**, then Cast → U8. Slice preserves input dtype, so the 400 B F32 intermediate is unavoidable per channel. Alternatives are all worse:
   - Cast whole input F32→U8 first: `[1,10,30,30]` U8 = **9000 B** intermediate.
   - Collapse channels (MatMul/Gather to a label grid): `[1,30,30]` F32 = 3600 B, or slice channels 2:6 = `[1,4,10,10]` F32 = 1600 B.
   - Cast to F16/BOOL: the F32 slice still materializes first; BOOL is still 1 B like U8 with no upstream saving.
   So `m2_f32`+`m5_f32` = 800 B is a hard floor. (Confirmed: computing `n_seed` from `m2_u8` instead of `m2_f32` does not remove `m2_f32`, because the Cast→`m2_u8` still consumes the F32 slice.)

2. **The ~500 B of U8 planes are correctness-required.** `valid_u8`→`valid_masked_u8` (ScatterND) → `stamp_special_u8` carry the line tie-break and diamond center-on-5 / row-major sparse-block suppression. The don't-do log forbids omitting either special branch, and the no-original-guard probes (cost 1865/1864) that shave this machinery **diverge 139/200265 vs the legal reference** and drop synth to 99.32% — below-safe. `render_x_u8` (200 B, 2-channel Concat) feeds the final 1×1 render conv that maps {stamp, blocker}→10-way one-hot; the two masks are not separable into 1 channel without adding a Gather/expand that costs more than the 100 B saved.

## Verdict
- **Primary lever (1882→1881 exact recapture): DONE and re-validated** — visible 265/265 (2 stable runs), profile 1881, capped ab vs legal 1888 reference byte-equal 2265/2265, capped truth 4994/5000 (99.88%).
- **Stretch (1600–1700): WALL.** No cheaper legal graph exists in the slice→cast→conv render family; the memory floor is pinned by the two forced 400 B F32 channel slices plus the correctness-required special-case planes. A sub-1700 rung would need a fundamentally different input decode that reads the F32 one-hot without a per-channel F32 slice — none found is cheaper than the two 400 B slices.
- **One remaining hypothesis (unproven, for next batch):** if a single op could emit a U8 10×10 mask directly from the F32 one-hot channel without a 400 B F32 intermediate (e.g. a quantized channel-gather kernel available in this ORT), the 800 B floor could drop toward 400 B. No such op path was found available under the local ORT.

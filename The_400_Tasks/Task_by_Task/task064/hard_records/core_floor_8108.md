# DEAD ROUTE — the task064 renderer CORE is forced at ~6852; 8108 is the floor

**Recorded:** 2026-07-12 by `bo_ng_cld_worker1` (Claude lane), forced 2nd pass.
**Verdict:** The staged **8108** (`task064_robust2cand_scalarisrect_cost8108`, byte-equal to LIVE 8695, −587) is the floor. Do NOT re-explore the core-attack routes below — each is priced/library/history-confirmed WORSE.

## Live bar
- live_cost **8695** (`Team_Best_Onnx/task064.onnx` sha `7562db13`, manifest-authoritative). My staged win **8108** (mem 7994 + params 114).

## Forced CORE = 6852 (information-theoretic; identical in Sho 8695 and my 8108)
The rule paints dot-color rays from scattered dots to a box wall. The near end of each ray is the **per-row / per-col dot extent** — data-dependent and multi-rank (task064.md Don't-do 2026-06-19, 60k-corpus). Rendering it forces:
1. **Presence plane `SD` F32 [1,1,24,24] = 2304.** A per-cell plane is required for the per-row dot extent; it is a channel-reduction of the F32 one-hot input, and every channel-reduction op (Conv/Einsum/ReduceSum/MatMul) outputs **F32** (traps 8f: F16 is the reduce floor and only *leaf* ops narrow; Conv F16-out on F32-in impossible). Integer alternatives are worse: `ArgMax(input,axis=1)` index plane = INT64 [30,30] = 7200 (worker3 fast-probe); `QLinearConv` needs a U8 input frame = 9000 (traps 8g / INPUT-DTYPE-LOCK). Casting SD→U8 keeps the 2304 F32 first, then adds 576. **2304 forced.**
2. **4× ArgMax walls I64 [1,1,24,1] = 768.** ArgMax finds the wall position with NO product plane; the ReduceMax-of-(colidx·presence) alternative needs an extra [24,24] plane (≥576) per direction. **768 forced.**
3. **Ray fill 5×[24,24] = 2880** (hoff/fullH/voff/fullV/B24). Two-sided range test = 1 Sub + 1 LessOrEqual (U8-underflow) per orientation + OR = minimal.
4. **Output mask `B30` BOOL [1,1,30,30] = 900.** The `Where(B30, doh, input)` condition must broadcast over the 30×30 padded grid to preserve the pad ring; a 24×24 mask can't broadcast to 30×30. **900 forced.**

Detection+geometry (~1256) is already minimized to scalar in the 8108 win (see manifest). Core 6852 + detection ~1256 = **8108**.

## Ruled-out CORE formulations (priced/confirmed WORSE than 8108)
| route | cost | evidence |
|---|---|---|
| **Packed ray-fill (#110), H-band hybrid** | **9937** (mem 9655) — BUILT this pass | `prototypes/w1_2026-07-12/build_task064_packedH_fill.py`. Output Where needs a full [30,30] mask, so each band must UNPACK to [24,24] before the OR; unpack(1152) ≥ direct-compare(1152) and packing ADDS a pack round-trip + INT64 index casts. Strictly loses. |
| Packed dual-pack (v6..v21) | 9743 | task064.md history; dominated by Sho's moment+Conv renderer. |
| CumSum "fill-between-extremes" | ~4× fill | prefix+suffix CumSum are F32 [24,24]=2304 each (traps: CumSum is f32/f64/i32/i64 only). |
| U8/BOOL presence plane | impossible | channel-reduction outputs F32; casting keeps the 2304 F32 first (traps 8f/8g). |
| Index-plane collapse (ArgMax axis=1) | 7200+ | INT64 [30,30] (worker3 `color_index_collapse.md`). |
| Box-scalar-wall + dot-only presence | = same +Min/Max | dot-only presence is still F32 [24,24]=2304; combined ArgMax (Sho) is strictly cheaper. |
| Directional MaxPool / spatial scan | 19889 | W9 scanB. |
| Morphology box/dot discrimination | ~36 KB plane | a depthwise 3×3 neighbor Conv is [1,10,30,30] F32. |

## Theorem (why lower is unreachable by this route)
Any correct renderer must (a) materialize a per-cell F32 [24,24] presence plane (2304, forced by the multi-rank per-row dot extent + F32 channel reduction), (b) produce a full [30,30] output mask for the pad-preserving Where (900), and (c) fill two ray bands as full [24,24] masks (packing them costs a strictly-worse round-trip). So **core ≥ 6852**, and robust 3-color detection adds ≥ ~1130. **8108 is the floor; the prompt's ~1500 floor_est is a FALSE FLOOR (the presence plane alone is 2304).**

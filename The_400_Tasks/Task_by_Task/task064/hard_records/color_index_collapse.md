# DEAD ROUTE — color-index-collapse (Direction / Fast-probe) is a FALSE FLOOR

**Recorded:** 2026-07-10 by `bo_ng_cld_worker3` (Claude lane).
**Verdict:** DROP. Do not re-explore the "collapse to one ArgMax index plane, spatial logic on it, Gather/Equal at the boundary, floor ~900" route.

## Live bar (refreshed 2026-07-10)
- **live_cost = 8695** (manifest, owner **Sho**, updated 2026-07-09T01:09Z; authoritative per `ng_current_bar.py`).
- live model `Team_Best_Onnx/task064.onnx` **sha256 `7562db13c022f4c62c1db4f2a7603a121142480de2e5d01a5050f994f1f6a14e`**, graph name `t064v3`, opset 17.
- Re-scored locally: `score_network_v2 = (macs 0, memory 8581, params 114) => cost 8695`. ✓ matches manifest.
- Prompt/context_pack STATUS (14706 / 9819) and the local "byte-equal floor 9743" (v20) are STALE — Sho's `t064v3` (8695) supersedes and DEFEATS the old v1..v21 packed-Where floor proof.

## The Direction's charged component is ALREADY GONE in the live graph
Direction (prompt): "perform spatial logic on ONE index channel; Gather/Equal only at the boundary; delete the charged carrier/canvas/template." Fast probe: "replace the per-color path with an ArgMax/index-plane prototype."

`t064v3` has **already collapsed the per-color path**:
- box/dot color are SCALAR indices `b64`/`d64` (ArgMax over the *color* axis of [1,10] moment vectors — NOT a spatial index plane).
- rectangle detection is done via per-color MOMENTS (Einsum `cnt/Sr/Sr2/Sc/Sc2`, ~43× [1,10] F32 tensors @40 B ≈ 2 KB) → `isrect` (count==h·w and integer h,w).
- there is **no per-color spatial manipulation** and no 10-channel carrier/canvas/template left to delete.

The remaining charge is NOT a per-color path — it is the irreducible single-channel renderer (see forced floor below).

## Fast-probe result — PRICED WALL (kill rule fired)
Built `prototypes/build_task064_fastprobe_indexplane{,_v2}.py`:
- `idx = ArgMax(input, axis=1, keepdims=1)` → **INT64 [1,1,30,30] = 7,200 B**. ONNX ArgMax output is ALWAYS INT64 (no dtype attr); grid is the padded 30×30.
- v2 (idx as a real INTERMEDIATE consumed by a boundary `Equal`): `score_network_v2 = (0, 8100, 1)` → **cost 8101** = 7,200 (idx) + 900 (boundary BOOL) + 1. `largest_intermediate`: idx = 7,200 (100%).
- You cannot cheaply get a 24×24 index plane: ArgMax@30×30 (7,200) then Slice→24×24 (4,608) charges BOTH; Slice the F32 input first = F32 [1,10,24,24] = 23,040. Route floor via ArgMax ≥ 7,200.

**So the literal index-plane route prices ≥ 7,200 B before ANY rectangle detection, wall-finding, or ray fill — already ~8× the ~900 floor_est and at/above live 8695.** Kill rule ("prices above 900 → stop, do not grind") fires.

## Why ~900 is a FALSE FLOOR (refreshed forced-component floor)
Memory model (`neurogolf_utils.calculate_memory`): only tensors named exactly `input`/`output` are free; every other intermediate is charged at `max(static, profiled-runtime)`. Forced, all charged intermediates:
1. **Per-cell single-channel projection ≈ 2,304 B** (F32 [1,1,24,24]). The rule needs, per in-span row, the leftmost/rightmost box-or-dot column (the ray extent), which is DATA-DEPENDENT per row (dots are scattered) — a per-row reduction REQUIRES a per-cell plane. Cheapest from the F32 one-hot input is an F32 [24,24] projection (live `SD`, Conv). The INT64 ArgMax alternative is worse (7,200). Moments give only the box (global scalars), never per-row dot extents.
2. **Multi-rank ray canvas ≥ 576×2 B** (BOOL [24,24] `fullH`/`fullV`, +`B24` OR). Established non-separable: the painted region's near end is the per-row min/max dot col (per-row-varying) → NOT a low-rank outer product, so no free-output row⊗col Einsum (task064.md Don't-do 2026-06-19 / W9 2026-06-22, 60k-corpus verified).
3. **Output-extent mask `B30` BOOL [1,1,30,30] = 900 B** — forced: the final `Where(B30, doh, input)` must copy the 10-ch F32 canvas on the padded border, so the mask can't live at 24×24.

Floor ≈ 2,304 + ~1,152 + 900 + walls/detection (~2–3 KB) ≈ **live 8695**. The ~900 floor_est counted only the output tail and ignored (1)+(2). The forced per-cell projection + multi-rank ray canvas are the false-floor components.

## Corroborating dead routes (don't re-run)
- **Packed bit-plane ray-fill** (v6..v21, my lane's prior floor): byte-equal floor **9743 > 8695** — the packed route is DOMINATED by Sho's moment+projection route, and still needs a per-cell plane before packing. (task064.md COMPRESSED HISTORY / floor proof.)
- **Directional MaxPool/spatial-scan** (W9 scanB): 19,889. **Count-derived in-box span** (v14/v19/v21): DIRTY vs LIVE 200k (padding leak), Path-B RISK only.

## One remaining hypothesis (only if the scorer/route genuinely changes)
Shrink the ~2 KB moment-based rectangle detector in `t064v3` (e.g., a cheaper "box = largest non-bg contiguous color" classifier) — but that is a **rung-epsilon shave of Sho's route, NOT a false-floor crack**, explicitly out of scope for this SWEEP prompt, and risks the LIVE adversarial oracle (legacy padding can set ch0 → multi-channel marker, decoded `11`; task064.md Don't-do 2026-05-22). Do not pursue without a Kaggle-confirmed reason.

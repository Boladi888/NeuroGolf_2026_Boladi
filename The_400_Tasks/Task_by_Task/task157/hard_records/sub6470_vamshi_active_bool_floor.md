# Hard record — sub-6470 wall on the Vamshi active-BOOL tail (batch 42, bo_ng_cld_worker1, 2026-07-03)

## Live bar / state at time of record
- Team-best (manifest.csv + xlsx): **Vamshi cost 6925**, sha `17e73ca4e6d94039e27a8da4f0da8dcffadae4fc32a863c000d160b43b0596c6`, updated 2026-07-02T02:35:15Z.
- **Standing win already staged** (by `bo_ng_cdx2_worker3`, 2026-07-02): `staged_candidates/task157_vamshi_bool3_pad_cost6470/`, cost **6470**, profile 6470, sha `165ddccd6190ed6ca963043c0a21beffa103fde2122321be3e24c57170966d4b`. Δ vs bar = **−455**.
- Re-verified this pass: visible 265/265 PASS; profile static=6470 profile=6470; `ab --ref Team_Best_Onnx/task157.onnx --n 8000` = 8265/8265 divergences=0 **byte_equal=True** (matches manifest full 107007/107007).

## Attempted route
Push below 6470 while staying **byte-equal to Vamshi** (mandatory: task157 is ambiguous-class, LIVE truth 98.83%, so byte-equal-vs-team-best is the gold gate — a non-byte-equal "improvement" is a learning artifact).

## Why it walls — forced-component floor (per-tensor profile of the 6470 graph)
Total intermediate memory 6053 + params 417 = 6470. Top charged tensors, each ORT-forced:
- **450B `onehot_3x10x15_bool`** — the output payload itself: 3 colors × 10 × 15 × 1B. BOOL is already the 1-byte minimum; the Codex worker already replaced the 900B UINT8 30×30 canvas with this active 3-channel plane (the −455 that produced 6470). Irreducible.
- **256B + 256B `place_indices_3d` / `place_indices`** (int32) — `place_indices_3d = Add(final_base_411[4,1,1], place_offsets[1,4,4]) → [4,4,4]` reshaped to `place_indices[64]`, consumed by `ScatterElements` into `base_state[150]` (rank-1). ScatterElements requires indices rank == data rank (1), so the flat `[64]` reshape is mandatory; the Add inherently produces 64 int32 (4 creatures × 16 cells) in **any** shape = 256B. Fusing the reshape into the Add inputs cannot drop below 64 int32 — no net save. Both materialize = 512B forced.
- **240B `bottom_f`** (float32, `[1,1,4,15]`) — `Slice(input,...) → Cast(uint8)`. Input is F32; an F32-input Slice stays F32 (ORT dtype fact: cast-full-input loses), so the 240B F32 slice is forced before the downstream uint8 cast.
- Remainder is a long flat tail of 60–150B forced scalars/vectors (color maps, shift/pow tables, TopK/ArgMax int32/int64 index outputs). No dominant tensor.

Every large tensor is pinned by an ORT op-spec/dtype constraint. The one clean lever (canvas→active-BOOL) is already taken.

## Independent from-scratch formulations (all worse or truth-unsafe)
1. **Enumeration / exact-cover structural** (`prototypes/task157_v1.onnx`, `task157_match_v1.onnx`): both profile **37,310** (~5.8× the bar). Truth-safe but far worse. Dead as a cost route.
2. **Lookup-table uint64/uint16 tail** (docs' scout 4808 / stretch 3581): these are the OLD template-lookup family. Confirmed **learning artifacts** — `task157_key_holes_tail_bool3_u64_port` (cost 3581): visible 265/265 PASS but **truth 261/20000 = 1.305%** and ab-vs-Vamshi byte_equal=False (526/5265). Memorizes the 265 official templates; collapses on the 200k ARC-GEN corpus. The 4808 "BOOL 3-channel active-output" figure is the same lookup family (not the Vamshi-structural active-BOOL) and inherits the same truth failure. **The scout/stretch targets are unachievable truth-safe.**
3. **Vamshi-structural + active-BOOL tail** = 6470. The only truth-safe sweet spot; already staged.

## Verdict: WALL at 6470 (forced-component floor for the truth-safe family)
No byte-safe candidate strictly below 6470 exists on the Vamshi-structural representation; all top tensors are ORT-forced. The only sub-6470 numbers in the docs (4808/3581) belong to the truth-unsafe lookup family. **Nothing new staged** (6470 win stands, preserved).

## One remaining hypothesis (low confidence)
A ground-up truth-safe structural graph that fuses the final render into the scatter (eliminating the separate `onehot` materialization) *might* shave the 450B, but it would rewrite Vamshi's placement/scatter core and near-certainly break byte-equality on the ambiguous cases — high risk, and the fused render still has to emit ≥3×10×15 bool of output somewhere. Not worth a siege per the prompt ("not a floor-proving task; stop when the named lever is exhausted").

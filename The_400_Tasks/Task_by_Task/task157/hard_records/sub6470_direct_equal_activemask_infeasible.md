# Hard record — sub-6470 wall on the "direct Equal/Pad/Gather active-mask" formulation (batch 46, bo_ng_cld_worker3, 2026-07-03)

## Live bar / state at time of record
- Team-best (manifest.csv + xlsx): **Vamshi cost 6925**, sha `17e73ca4e6d94039e27a8da4f0da8dcffadae4fc32a863c000d160b43b0596c6`, updated 2026-07-02T02:35:15Z.
- **Standing win already staged**: `staged_candidates/task157_vamshi_bool3_pad_cost6470/`, cost **6470**, profile 6470, sha `165ddccd6190ed6ca963043c0a21beffa103fde2122321be3e24c57170966d4b`. Δ vs bar = **−455**. Re-verified this pass: SHA matches; `profile_real_cost --arcgen 266` = `static=6470 profile=6470 (memory 6053 + params 417; 265 real inputs)`.
- Effective bar to beat = **strict-below 6470**.

## Route attempted (this batch's assigned NEW formulation)
Prompt batch 46 directed a from-scratch alternative to polishing the table tail: **a direct 3-channel Equal/BOOL Pad or Gather-style active mask** that removes the two int32 Scatter index tensors *and* the `bottom_f` F32 carrier, rather than reusing the Vamshi scatter core.

## Per-tensor profile of the 6470 graph (this batch, `prototypes/per_tensor_cost.py`)
Total intermediate memory 6053 + params 417 = 6470. The tensors this formulation targets:
- **450B bool** (`safe_name_306` = `onehot_3x10x15_bool`) — the output payload, 3 colors × 10 × 15 × 1B. BOOL is the 1-byte minimum. Irreducible.
- **256B + 256B int32** (`safe_name_299`/`300` = `place_indices_3d` / `place_indices`) — the Scatter index pair.
- **240B f32** (`safe_name_95` = `bottom_f`) — the input staging-region slice.
- Remainder is a flat tail of forced 32–192B scalars/vectors (color maps, shift/pow tables, TopK/ArgMax int32/int64 index outputs, base_state/state_flat 150B each). No dominant tensor.

## Why the new formulation is infeasible or strictly worse

**1. ch1 (placed-creature channel) cannot be a free-output / separable Equal.**
The creature top-anchor positions are **runtime-decoded from the input** — `final_base` = `final_row15 + final_col`, built from `Gather(top_start_idx, final_slot)` where the slot/row/col come from `TopK`/`ArgMax` over the bottom staging region. Library verdict #10 (one-hot-dissolve rank gate + runtime-selector double-kill): *"Runtime-decoded selectors — if K/R/C are computed from live input, free-output fails regardless of rank."* So there is **no static Equal/Pad formula** for the placement plane; you must scatter/gather at computed indices.

**2. The two 256B int32 index tensors are a rank-forced floor — cannot collapse to one.**
`ScatterElements` requires `indices.rank == data.rank`. `base_state` is rank-1 `[150]`, so indices must be rank-1 `[64]`. The placement index is `Add(final_base_411[4,1,1], place_offsets[1,4,4]) → [4,4,4]` (256B) then `Reshape → [64]` (256B) — both materialize. You cannot feed the `[4,4,4]` directly to Scatter (rank-3 vs rank-1 = checker/ORT reject). Pushing the 4×16 expansion upstream (make `final_slot/row/col/base` all `[64]` instead of `[4]`) inflates ~5 per-creature tensors from 16B→256B each = **+~1200B**, strictly worse. int32 is the index-dtype floor (no int8/int16 index kernel — floor-proof + library #5). **512B forced.**

**3. Shift-and-OR Pad/Gather is byte-heavier than the scatter it replaces.**
The scatter-free alternative — take each creature's `[4,4]` BOOL footprint, dynamic-`Pad`/`Gather`-roll it to `[10,15]` at the computed anchor, then `Or` the 4 planes — materializes 4 × `[10,15]` bool planes = **600B**, worse than the 512B index pair + 150B state it removes, and still needs runtime pad/gather index tensors (same runtime-selector cost). No win.

**4. `bottom_f` (240B f32) is the sole forced input read of creature shapes.**
`bottom_f = Slice(input, rows=4, cols=15) → Cast(uint8)`. Input is F32; an F32-input Slice stays F32 (ORT dtype fact — cast-full-input loses), so `[1,1,4,15]`×4B = 240B is forced *before* the uint8 cast. Removing it = not reading the creatures from the input. The 4-row slice is already near-minimal; slicing fewer rows is data-dependent (can't statically narrow). Forced.

## Independent from-scratch formulations (all three ruled out)
1. **Vamshi-structural + active-BOOL tail** (scatter core) = **6470** — the truth-safe sweet spot, already staged. Every top tensor ORT-forced (this record + batch-42 `sub6470_vamshi_active_bool_floor.md`).
2. **Enumeration / exact-cover structural** (`prototypes/task157_v1.onnx`, `task157_match_v1.onnx`) = **37,310** (~5.8×). Truth-safe but far worse. Dead as a cost route.
3. **Lookup-table uint64/uint16 tail** (docs' 4808/3581) = **truth-unsafe learning artifact** (`task157_key_holes_tail_bool3_u64_port` truth 261/20000 = 1.305%, ab byte_equal=False). Memorizes the 265 official templates; collapses on the 200k ARC-GEN corpus.
4. **Direct Equal/Pad/Gather active mask** (this batch's assigned new formulation) = **infeasible** (ch1 placement is runtime-decoded → no free-output Equal) or **byte-heavier** (shift-and-OR planes 600B > 512B scatter indices). bottom_f is the sole forced input read.

## Verdict: WALL at 6470 (truth-safe forced-component floor)
No byte-safe candidate strictly below 6470 exists on any of the three independent representations. The scatter indices (512B), output payload (450B bool), and bottom_f (240B f32) are each pinned by an ORT op-spec/dtype constraint; the placement plane is provably not free-output (runtime-selector kill). **Nothing new staged** (6470 win stands, preserved).

## One remaining hypothesis (low confidence, not pursued)
A ground-up graph that fuses the final render into the scatter (eliminating the separate 450B `onehot` materialization) *might* shave the payload, but the fused render still must emit ≥3×10×15 bool somewhere, and it rewrites the placement/scatter core → near-certain byte-equality break on the ambiguous cases (task157 is ambiguous-class, LIVE truth 98.83%, so byte-equal-vs-team-best is the gold gate). Not worth a siege per the prompt.

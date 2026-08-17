# Transferable insight — column-position→color remap: single-Einsum rank-2 is the floor; slice-to-shrink-params loses to the charged intermediate

**Source**: task262 batch 53 (`bo_ng_cld_worker2`, 2026-07-07). LIVE 80 (Vamshi, sha `560f5b67…4095`) confirmed at forced floor; no improvement. Perfect-25 rumor disproven.

**Task**: each 3×3 input row has one `5`; its **column** selects a row-fill color (col0→2, col1→4, col2→3, non-monotonic). One-hot `[1,10,30,30]` canvas.

**Pattern (the live cost-80 architecture, at floor)**: a **single** `Einsum bkrj,bkrh,blrc,jp,po->borc` with `Q[30,2]`=60 + `V[2,10]`=20.
- `input`×`input` (`bkrj,bkrh`) computes per-row match-counts → **avoids a channel-5 selector** (folding channel-select into count arithmetic; an explicit `sel[10]` selector Einsum costs 90).
- `Q[30,2]` reads absolute column position (rank-2); `V[2,10]` maps rank-2 → 10 channels.
- The third `input` (`blrc`, Σ over channels = 1) **reuses input as the free full-canvas broadcast stencil** — same trick as task129: output is column-constant, no stored spatial mask.

**Why 80 is the floor (forced-component argument)**:
- 3 column-classes → 3 distinct output channels needs **rank ≥ 2**: rank-1 gives `output = scalar·V`, whose `argmax_o` depends only on `sign(scalar)` → ≤ 2 classes. Rank-2 over the 30-position column axis forces `Q[30,2]`=60; rank-2 → 10 channels forces `V[2,10]`=20 ⇒ 80.
- **Perfect-25 cost≤1 impossible**: output alphabet {2,3,4} is disjoint from input {0,5}; no value-preserving op (Identity/Transpose/Flip/Rotate) can synthesize new colors → a payload initializer is forced. Don't grind the zero-param sweep.

**The lever / anti-pattern (measured)** — the reusable lesson:
- **Slice-to-shrink-params is a NET LOSS when the output is full-canvas.** Slicing the 30-col axis to 3 shrinks `Q[30,2]→[3,2]` (or a direct `M[3,10]`=30) but the Slice **output is a charged intermediate**: `[1,1,30,3]` F32 = **360 B** ≫ the params saved. Measured `Slice(ch5,cols0:3)+Einsum M[3,10]`: visible 11/11 PASS but **profile 396** (36 params + 360 mem). LOSS.
- **`input` used as broadcast source is a graph INPUT → not charged; a node's output between nodes IS charged.** So a zero-intermediate **single node** beats any 2-node slice here. Charge asymmetry to remember: **params = element count (dtype-free); memory = bytes (dtype matters)** — a 90-element F32 intermediate (360 B) outweighs 90 params.
- **Compact `[1,10,3,3]` output ⇒ `visible 0/11`** (grader compares the full `[1,10,30,30]` encoded tensor). Same guard as task129/DONE_task129. Always render full-canvas; never shrink the output to shrink the intermediate.

**Applies to**: positional-selector → color-fill tasks (Pattern_Recognition / Filling Regions; 143 similar to task262). Catalog #119 free-output Einsum micro-floors + #132 (materialized-vector caveat). Rule of thumb: if the output is full-canvas, prefer a **single zero-intermediate Einsum** at the rank floor over any Slice/Gather route that trades params for a charged intermediate.

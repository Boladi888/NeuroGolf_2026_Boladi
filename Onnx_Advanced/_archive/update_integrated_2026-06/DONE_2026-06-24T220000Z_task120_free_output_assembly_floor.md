# task120 free-output assembly floor — single-Conv beats erosion+assemble (measured)

## What I discovered (dead-end + a measured floor law)

For a per-cell recolor whose output must be the full `[1,10,30,30]` one-hot tensor
(byte-equal decode), a **single free-output Conv that writes directly to the graph
`"output"` tensor is the unique zero-memory structure.** Decomposing the rule into
"cheap morphology + assemble the output" is a trap: every assembly op materializes a
charged intermediate, and the sum dwarfs the dense Conv's params.

task120 rule: rectangle interiors → cyan(8), borders keep color. Interior = plus-kernel
erosion of the colored mask (all 4 orth neighbours occupied). The prompt insisted the
910 dense `[10,10,3,3]` Conv was a "formulation floor" and that a tiny plus-kernel
erosion + free-output recolor would reach <500 / <200.

I **built and measured** the prompt's two headline angles:

- **Angle A** (occ = Greater(ReduceSum(in ch1..9)) → Cast → plus-Conv → Greater →
  Slice/Mul/Concat assemble `[1,10,30,30]`): **score_v2 = 114,325** (114,300 memory +
  25 params). The Slice/Mul/Concat assembly charges multiple `[1,10,30,30]` and
  `[1,1,30,30]` F32 intermediates.
- **Leanest Where route** (fold occupancy into the conv: one `[1,10,3,3]` plus-conv
  reading in{1,2,3} directly → Greater → `Where(mask, eight_onehot, input)`):
  **score_v2 = 4602** (4500 memory + 102 params). Two unavoidable charged
  `[1,1,30,30]`: conv raw F32 = 3600 + BOOL mask = 900.

Both >> the 910 dense single-Conv.

## Why the single-Conv wins (the law)

`calculate_memory` SKIPS the tensors named exactly `"input"` and `"output"`. A single
op whose output IS the graph output pays **0 memory** — only its initializers count as
params. The dense `Conv[10,10,3,3]+B[10]` produces the entire `[1,10,30,30]` in one op
→ params 910, memory 0.

Any decomposition needs ≥1 intermediate of shape `[1,1,30,30]` (the per-cell interior
decision over the full grid) PLUS the conv-raw before its threshold — interior detection
is inherently *spatial-reduction-then-threshold* = 2 ops = 2 charged tensors. Minimum
4500 memory. You cannot make the interior mask the SOLE charged tensor, and you cannot
reach `[1,10,30,30]` from a `[1,1,30,30]` mask without an assembly op that charges more.

Weight can't shrink either: group=1 forces `[10,10,3,3]` (output 10ch, input 10ch);
group>1 traps out-ch8 in a group reading only the always-zero in{5..9}; a sub-3×3 kernel
can't hold the plus stencil; slicing input to fewer channels charges a `[1,k,30,30]`.
The `[10]` bias is load-bearing (only Conv mechanism emitting a constant at all-zero
padding cells, where ch8's −4 threshold must NOT fire). ⇒ exactly 910, == team-best.

## Where / why it matters

task120 (`Task_by_Task/task120.md`, prototypes `task120_refit_biasconv.py` /
`task120_wherePB_MEASURED_LOSS.py`). This is the THIRD worker session re-tasked on this
already-proven floor (2026-06-10 Worker-9, 2026-06-20 Worker-2, 2026-06-24 Worker-9).
The refit anchor re-gated this session: visible 266/266, AB byte-equal vs team-best LIVE
5266/5266 (0 div). LIVE is already correct AND at the floor; staging 910 adds 0.000
(== team-best, HOME-COST TRAP).

## What may be missing from the library

A crisp **"free-output single-op vs decomposed-assembly" floor law** in
`cost_floor_reference.md` / `scoring_internals.md`: when the output must be the full
`[1,C,H,W]` one-hot, a single free-output op (Conv/Einsum writing `"output"`) pays only
params; ANY multi-op decomposition pays ≥1×`[1,1,H,W]` per intermediate, and a
reduce-then-threshold predicate forces ≥2. `traps.md §FREE-OUTPUT` covers the curve/region
distinction but not this "don't decompose a free-output Conv into morphology+assemble"
measured trap with the 910 vs 4602 vs 114,325 ladder. Worth a one-line entry so the next
prompt-writer stops re-tasking task120.

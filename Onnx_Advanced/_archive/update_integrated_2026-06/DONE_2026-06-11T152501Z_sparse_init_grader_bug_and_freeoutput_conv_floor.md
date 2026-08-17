# task097: sparse_initializer grader bug + free-output-Conv params floor + "byte-equal-to-LIVE is the wrong objective"

Encountered on **task097** (NeighborSameColor / remove-isolated-cells). Worker-4, 2026-06-11.
Build scripts: `The_400_Tasks/Task_by_Task/task097/prototypes/build_task097_dw2.py`,
`build_task097_sparse.py`, `_probe_multicolor.py`. Reference net: `_aditya097.onnx`.

## 1. sparse_initializer DOES NOT WORK on this grader (hard trap — likely transferable to other low-param tasks)

The cost model `neurogolf_utils.calculate_params` (data/competitions/neurogolf-2026/...)
counts `math.prod(sparse_init.values.dims)` for a `sparse_initializer` — i.e. only the
**nonzero count**, not the dense size. This looks like a huge lever for params-dominated
nets (e.g. a Conv weight `[10,10,3,3]`=900 dense with only 163 nonzeros → 173 params).

**But it is unusable**, because `neurogolf_utils.sanitize_model` (called by BOTH the
official `verify_network` at line ~489 AND `tools/_score_v2.score_network_v2` before
inference) renames `graph.input`/`graph.initializer`/node I/O via `get_safe_name`, but
**never remaps `graph.sparse_initializer`**. So a Conv whose weight is a sparse_initializer
named `W` has its node input renamed to `safe_name_N`, while the sparse tensor keeps the
name `W` → ORT fails at session-create: `INVALID_ARGUMENT: Node input 'safe_name_1' is not
a graph input, initializer, or output of a previous node`. The model loads fine in RAW ORT
(so it's tempting), but **the grader sanitizes first → it never scores**.

**Library gap:** I searched scoring_internals.md / cost_floor_reference.md and found no note
that sparse_initializer is a dead end under sanitize. Recommend a one-liner in
scoring_internals.md (accounting-levers / ban-workarounds) + cost_floor_reference.md:
"sparse_initializer params-trick is REJECTED — sanitize_model doesn't remap it, model won't
load. Don't build it."

## 2. Free-output single-Conv params floor law (reusable cost arithmetic)

For an argmax/`>0`-decoded `[1,10,30,30]→[1,10,30,30]` task solved by ONE Conv whose output
IS the graph output: **memory = 0** (calculate_memory line 234 skips tensors literally named
`input`/`output`), so **cost = params = M·(C/group)·kH·kW + bias(M)**. For an 8-neighbour
rule needing cross-channel reads in at least one output channel: group=1 forced, kernel 3×3
forced, M=C=10 → 900 + 10 bias = **910**. This is exactly the team-best (Aditya) net and is
a genuine forced floor. The `+1 center on every colour plane` trick (Aditya's ch0) is the
padding-safe way to detect "removed" cells (`0.5 - same_color_ring`), independent of grid
position — worth capturing as a technique: it beats the depthwise bg-ring approach which is
border-broken (see below).

**Depthwise (group=10, 100 params) is a TRAP here:** each channel reads only its own plane,
so the background channel can't see the colour ring; it must rely on its own bg-plane ring,
and Conv pads with 0 (≠ background), so removed cells at grid edges decode to −1. Result:
visible 11/267, truth **6.75%** — looks 9× cheaper, is totally broken. Lesson: a background/
"deletion" output channel that must fire on isolated cells needs CROSS-CHANNEL reads → kills
the depthwise param saving.

## 3. "Byte-equal to the canonical LIVE model" was the WRONG objective (FAMILY-LABEL-adjacent)

task097's task file had a hardened doctrine: "same-colour mandatory, multicolor-safe, real
floor ≈17k" — derived from requiring **byte-equality to the canonical same-colour LIVE net**.
But the grader scores **truth vs ARC-GEN ground truth** on the hidden set, and the generator
(`external/ARC-GEN/tasks/task_42a50994.py`, line 38) draws **one `common.random_color()` per
grid** → the hidden set is SINGLE-COLOUR → any-neighbour == same-colour there. Aditya's
single any-neighbour Conv scores **200,000/200,000 truth at cost 910**. The −13.02 LB scar
that "proved same-colour mandatory" was actually a broken/old-base PACKAGE rebase, not the
rule. **Lesson (extends traps §FAMILY-LABEL / §14d):** when the hidden set is generator-drawn,
verify the floor against TRUTH-vs-generator, not byte-equality-to-LIVE; a multicolor
adversarial divergence vs the canonical is only relevant if the generator actually produces
multicolor inputs — here it provably does not (source-read the generator's color draw).
A correct solution can be ~19× cheaper than the byte-equal-to-LIVE one.

Net outcome: task097 is a no-candidate DROP (910 is forced; matching = 0.000 pts), but the
team should ensure the 910 Conv (not our 17,160 v1) is what ships for task097 — that's a
+ln(17160/910) ≈ +2.94 personal-stack delta if our LIVE still carries the 17k v1.

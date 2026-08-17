# task018 — sparse-CC flood is a hard floor when colour-identical components are near

**Context:** task018 (sprite-clone rebuild, Object_Based family). Team-best = `compress_v123`,
score `(0, 18536, 639) = 19,175`, SHA `a9f3074e8401780f1e66e38d6ef9af92c635794cea2b50975e69350737f952c6`.
123+ documented compression versions; I rebuilt v123 byte-identically and re-gated the cheaper probes.

## Transferable trap (candidate for traps.md / cellular_automata.md)

**A multi-round clamped-MaxPool "flood" that separates two connected components is irreducible to fewer/larger
rounds when the two components can be COLOUR-IDENTICAL and spatially NEAR.** Reducing rounds OR widening the
kernel to reach the same radius in fewer steps makes it strictly WORSE, not cheaper:

- A wider kernel (or single big round) reaches across the inter-component gap and **MERGES** the two
  components into one. Downstream sparse placement then computes positions for a merged blob and the
  flat-index `ScatterElements` **CRASHES in ORT** (`indices element out of data bounds, idx=601 must be
  within [-576,575]`) — visible/score gate ERRORs, not just wrong output.
- A narrower/shorter flood under-separates and **fails visible** (98.1–98.9%, never 100%) → certain hidden 0
  by the VISIBLE-FAIL axiom.

**Why no escape:** ONNX has no connected-components op. NonZero/Unique are banned. So CC separation must be an
iterative dilate-and-clamp (`MaxPool(k) → Min(region)`) seeded from one cell, and the number of rounds is
pinned by `ceil(max_component_radius / per_round_radius)` with a per-round kernel small enough NOT to bridge
the minimum inter-component gap. For task018: 3 rounds of clamped MaxPool (radius ≤2/round) ≈ 3,456 B is the
floor; 2 rounds or any kernel ≥7×7 single-round bridges. This is a general law for any rule that must split
same-colour blobs (gravity/translation/clone families).

## Concrete dead-ends re-gated 2026-06-24 (don't re-walk)
34 prototypes scoring 15,135–18,218 (all below the 19,175 team-best): `*_v116_nocomponent_flood*`,
`*_v116_bboxsplit*`, `*_v117_nodil_region*`, `*_v123_oneround_k*` (k7..k21), `*_v114_flood1round_k*`.
Every one CRASHES or fails visible. Logged in `The_400_Tasks/Task_by_Task/task018.md` don't-do log.

## Floor note
The prompt's `<1,000` target is disproven structurally: a 30×30 label-canvas renderer lower-bounds at
~1,030 B (900 U8 padded label plane + 10 one-hot target labels + 24×4 INT32 scatter index + 24 colours)
before any detection/flood/rotation work. Already in task018.md v117-refresh; restating for the library.

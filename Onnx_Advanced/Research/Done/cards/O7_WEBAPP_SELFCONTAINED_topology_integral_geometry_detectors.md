# Research Card O7 — EXACT topology / integral-geometry detectors as free-input LOCAL-pattern reductions (the algebraic alternative to connected components)

**Deliverable**: one copyable Markdown research note (no file writes, no code execution). Save under `Research/Done` with basename `o7_topology_integral_geometry_detectors`. **Priority: HIGH** — this targets features we currently can only get via banned/expensive connected-component machinery.

## Researcher contract

Internet access, NO repo access. ONE self-contained Markdown note; no follow-up questions. Human-readable citations (author-year-title). Tag every nontrivial claim `[PROMPT]`/`[PRIMARY]`/`[SECONDARY]`/`[DERIVED]`/`[SPECULATIVE]`.

## Project facts (all `[PROMPT]`)

- **Task**: minimal static ONNX graph that EXACTLY computes a known discrete grid predicate/feature. Input `[1,10,30,30]` F32 one-hot (10 colors, grid ≤30×30, padding cells all-zero on every channel); output decoded by `> 0.0`. Exactness mandatory on the full generated distribution.
- **Cost** (minimize): `cost = memory + params`. `params` = Σ ELEMENT COUNTS of initializer + Constant tensors (unique counted once, dtype-blind). `memory` = Σ bytes of intermediate `value_info` tensors (`max(static, runtime)`); the single `input` and single `output` are NOT charged.
- **THE lever this card rides (our biggest win — do NOT re-derive, exploit it):** an op that READS the free graph `input` and writes toward the free `output` can compute a fixed LINEAR functional of local patterns at ~0 charged memory. Concretely: a small fixed `Conv` (kernel = a local-pattern weight stencil) or a multi-operand `Einsum` over the input one-hot produces a per-window quantity whose global `ReduceSum` is a single scalar — memory ≈ 0, params = the tiny stencil. We already use this for COUNT/centroid/variance/area moments. **This card is about TOPOLOGICAL / INTEGRAL-GEOMETRY features that are ALSO exact local sums but which we have NOT encoded** — and which otherwise look like they need connected-component labeling.
- **Why this is distinct from the rest of the board**: card O1 (general algorithm mining) covers **summed-area / integral-image moment features** (rectangle sums, variance, moments) — DO NOT re-cover those here. O1 and the global policy DECLINE "connected-component labeling and pathfinding as generic topics." This card's whole value is the **algebraic shortcut that AVOIDS connected components**: many topological invariants are LOCAL sums (no labeling, no flood) and so collapse to one cheap free-input reduction.
- **Environment**: ORT 1.24.4 CPU, `ORT_DISABLE_ALL`. opset bumpable (ORT loads ≤ opset 25). Banned: `Loop/Scan/NonZero/Unique/Compress`, any `"Sequence"`-substring op, Functions, `If`/subgraphs. Static shapes. F16 exactly represents integers up to 2048; a sum of local-pattern indicators over ≤900 cells can exceed that — flag where INT32/F32 accumulation is forced.

## The question this card answers

For binary masks / color-masks on a ≤30×30 grid, give the COMPLETE set of TOPOLOGICAL and INTEGRAL-GEOMETRY features that are EXACTLY computable as a single (or tiny fixed number of) free-input local-pattern reductions — with the exact local weight table, the ONNX lowering (`Conv` stencil + `ReduceSum`, or Einsum), the param/memory cost, the dtype-exactness verdict, and a proof-or-counterexample — PLUS the provable boundary of features that are NOT local and genuinely need iteration/labeling.

## Required answer order

1. **Euler characteristic χ via local patterns.** State the exact digital-topology results: χ of a binary image equals a weighted sum of local 2×2 (or vertex/edge/face) configuration counts (Gray 1971 "bit-quad" weights; Pratt; the cubical-complex V−E+F formula). (a) Give the exact local weight table for 4- and 8-connectivity (they differ — state both and the connectivity caveat). (b) Show the ONNX lowering: a fixed `Conv` (2×2 stencil family) over the mask → per-window indicator → `ReduceSum` → χ, priced (stencil params, ~0 memory). (c) State F16/INT32 exactness for the accumulation magnitude on 30×30.
2. **Hole count (genus) exactly, without labeling.** For a binary mask, `holes = components − χ` (β1 = β0 − χ in 2D). (a) If the shape is known/assumed single-component (common in ARC: one object), `holes = 1 − χ` — a single free-input reduction. State this and its exactness. (b) When multiple components exist, is component COUNT (β0) itself a local-sum-expressible quantity for any useful restricted class (e.g. convex blobs, axis-aligned rectangles), or does it strictly require labeling? Give the boundary.
3. **Other Minkowski functionals / intrinsic volumes as local sums.** In 2D the Minkowski functionals are {area, perimeter, Euler characteristic}. (a) Area = count (known). (b) PERIMETER / boundary length: exact local-edge-count formulas (4- vs 8-neighbor boundary, the standard crack-following vs local-pattern weights) and their `Conv`-stencil lowering + cost. (c) Any other intrinsic-volume / integral-geometry identity (e.g. local convexity indicators, corner counts, connectivity number / crossing number used in thinning) that yields an exact cheap detector. Note which detect ARC-relevant predicates (is-hollow / has-a-hole, is-convex, boundary-length thresholds, number-of-corners).
4. **The provable boundary.** Characterize predicates that are NOT finite local sums and genuinely need iteration/labeling/flood: connectedness, "are cells A and B in the same component", reachability/shortest-path, per-component property aggregation. Give the locality/lower-bound reasoning (why χ is local but connectivity is not). For each, name our existing non-algebraic route (iterated MaxPool flood, etc.) so builders stop hunting for a local trick.
5. **Overall verdict** — one blunt paragraph: the 3–5 highest-value exact local-reduction detectors to add to our toolkit (each with its weight table, ONNX lowering, param cost, dtype verdict, and the ARC family it serves), prioritizing hole/Euler detection if it is confirmed exact-and-cheap.

## Required table schema (sections 1–3)

| feature | exact local formula (2×2 / vertex-edge-face weights) | connectivity (4/8) caveat | ONNX lowering (Conv stencil + reduce / einsum) | params | charged memory | F16-exact or needs INT32/F32 | ARC predicate served | proof-or-counterexample given? | source tags |

## Demands

- The marquee deliverable is Q1+Q2: give the EXACT Euler-characteristic local weight table and an unambiguous yes/no on whether hole count is a single free-input reduction for the single-component case. If yes, give the full `Conv`-stencil + `ReduceSum` graph sketch and its cost.
- Every "exact" claim needs a one-line proof sketch OR a counterexample marking the boundary. Approximate features are excluded.
- State the 4-connectivity vs 8-connectivity weight difference explicitly (it changes χ and is a classic silent bug).
- Price each as element-count params + (stated) memory; apply the 2048 F16 bound to the actual accumulation magnitude on 30×30.
- Do NOT re-cover moments / summed-area / rectangle-sum features — those are card O1's scope. Stay on topology + integral geometry (Euler, holes, perimeter, intrinsic volumes, corner/crossing numbers).
- No code beyond ≤15-line stencil/einsum sketches.

## Source policy

Primary: digital-topology / integral-geometry references (Gray 1971 bit-quads; Serra, *Image Analysis and Mathematical Morphology*; Schneider–Weil, *Stochastic and Integral Geometry*; standard results on Euler-characteristic locality and Minkowski functionals). ONNX spec for any op cited. Mark cost-model deductions `[DERIVED]`.

## Final checklist

- [ ] Q1 gives the EXACT Euler-characteristic local weight table for BOTH 4- and 8-connectivity, with the Conv-stencil lowering and cost.
- [ ] Q2 answers unambiguously whether hole count is a single free-input reduction (single-component case), with exactness proof.
- [ ] Q3 covers perimeter and at least one further intrinsic-volume/corner detector with exact local formulas.
- [ ] Q4 names the non-local predicates (connectivity/flood/path) with locality reasoning and our existing route.
- [ ] No overlap with O1's moment/summed-area scope; verdict lists 3–5 NEW topology/integral-geometry detectors with cost + ARC family.

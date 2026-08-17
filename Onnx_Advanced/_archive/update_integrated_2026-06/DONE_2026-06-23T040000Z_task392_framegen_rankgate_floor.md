# task392 ConcentricFrameGenerator — closed-form / free-output dissolve is DEAD by the rank gate (3rd-angle floor confirm)

## What I discovered
The mission framing for task392 ("dissolve the 50-row hash LUT with a 4-scalar analytic
generator, target <500") is **structurally impossible**, and I confirmed it from a third
independent angle that the prior two floor notes had not actually run: the **#109/#183 RANK
GATE on the per-channel output mask**.

- Color-channel output mask rank over the 266-pair corpus:
  `{4:14, 5:49, 6:102, 7:25, 8:22, 9:34, 10:20}` — commonly rank 6–10, **full-rank-10 in
  20/266 (7.5%)**. This is exactly the task183 kill condition (rank reaches grid dimension on
  a meaningful corpus fraction).
- Distinct binary output SHAPES in the corpus = **38** ( = offset 0..9 × edge{top,left} ×
  thick{1,2} ≈ 40 minus impossibles). The output is a near-**bijective LUT over ~40 full 10×10
  patterns**, NOT a low-rank/separable function of position.

## Why every "cheaper" route fails
1. **Per-cell closed-form render (#108/#130):** materializes dozens of charged [1,1,10,10]
   F32 planes (`calculate_memory` sums EVERY named node output; input/output excluded only).
   Prior worker's build measured mem 93763 / params 378 = 94141. Irreducible below ~2400 even
   maximally fused.
2. **Free-output multi-term Einsum (#109/#119/#132):** needs T = rank terms. Here rank is up
   to 10, and the selectors `R,C[T,30]` are (a) **F32-locked** (#109 carrier tax) and (b)
   **data-dependent / runtime** — they're functions of the JOINT decoded (center, thick, edge)
   scalars, not fixed initializers (§225 role-binding wall, §132 "placement count must be
   structural, not data-dependent"). So they'd be computed ON-GRAPH as planes, reintroducing the
   render memory. Even ignoring that, one render's selectors = `2·T·30·4` up to 2400 B > 1009.
3. **Shape-table LUT (the LIVE 1009):** params 235 = `gather_idx[1,22,3]=66` (hash addr) +
   `hash_w[22]=22` + `row_table64a/b[50]+[50]=100` (packed-U64 shapes, 2×U64 per 100-bit shape)
   + 47 small consts. memory 774 (a [1,1,10,10] U16=200 + F16 sign10=200 + BOOL=100 + GatherND
   [1,22] F32=88 + scalars). The ~76–80 packed shape params ARE the information floor because the
   output is a 40-way lookup, not a formula.

## Why it matters (trap avoided)
This is the canonical case where "the rule HAS a clean closed form" (it does — fully decoded,
verified 200000/200000) but the **cost model makes the closed form far more expensive than the
LUT it would replace**. The generator math is a red herring for cost-golf. The deciding test is
the rank gate on the OUTPUT mask, run in ~30 s — NOT building the closed form first (the prior
worker burned a full build to reach 94141 before concluding the same thing). 

**General lesson for any "concentric frame / nested ring / generative-pattern" task:** before
attempting an analytic or free-output dissolve of a shape LUT, (1) count distinct output shapes
and (2) measure output-mask rank over the corpus. If shapes ≈ the full parameter-product count
AND rank is commonly near-full, the packed shape LUT is the floor — stop. The selectors for a
data-dependent (runtime-decoded) generator can never be fixed initializers, so #109/#119/#132
free-output is off the table regardless of rank.

## Where
- LIVE team-best: `../K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task392.onnx`
  (1009 = mem 774 + params 235; truth 200000/200000 re-verified this session).
- Prior CF build: `The_400_Tasks/Task_by_Task/task392/prototypes/build_closedform/` (94141).
- Generator: `external/ARC-GEN/tasks/task_f8c80d96.py`.
- task392.md Attempts table updated with the rank distribution + byte-law rows.

## What's missing from the library
#109/#183 has the rank-gate doctrine for SCATTER masks, but no entry pins the case of a
**generative pattern whose selectors are runtime-decoded scalars** (vs fixed-init separable
stripes). That double-kill (rank gate AND runtime-selector non-init-ability) is the task392
signature and worth a one-line cross-reference between #109's role-binding note and #132's
"placement count must be structural" note.

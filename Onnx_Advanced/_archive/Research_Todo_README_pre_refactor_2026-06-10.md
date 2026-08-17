# Research Todo — two card types

1. **Internet-only cards** (M-series, prefix `M`): designed for plain ChatGPT / a deep-research model with web access but NO repo access. Paste the card body; it returns a self-contained Markdown note you drop in `../Done/`. Strict contract below.
2. **REPO-ACCESS cards** (R-series, prefix `R_REPOACCESS_`): assigned to a worker WITH full repo + tool access. They read our generators/onnx/corpus, run the gates, build tooling, and write findings to `Logs/research/` (+ stage gated candidates for the captain). These act on our real data — higher leverage. They follow `WORKER_PROTOCOL.md`, never submit.

## How a card works

1. Paste the card body into a ChatGPT conversation.
2. ChatGPT produces a **copyable Markdown research note** (no file writes).
3. You copy the note back into this repo at the suggested filename under `Onnx_Advanced/Research/Done/`.

Each card follows a strict contract so ChatGPT doesn't drift into broad answers:

- Deliverable type stated upfront (research note, not implementation)
- Environment & constraints fixed (opset, runtime, banned ops, dtype rules)
- Cost model defined with formulas, not just intuitions
- Required answer order
- Required table schemas
- Mandatory dtype/opset caveats per op
- Source policy (ONNX spec vs ORT docs)
- Code snippets limited to recommended patterns only
- Final verification checklist


## Card-authoring guidance (updated 2026-06-04 after the G1/G2 deliverables came back decision-grade)

Two cards returned exactly what we needed; keep doing what the contract enforced, plus three upgrades observed from the deliverables themselves:

1. **Source-tag standard (adopt G2's 5-class set)**: require tags `[PROMPT]` (fact given in the card), `[PRIMARY]` (peer-reviewed/primary source), `[SECONDARY]` (docs/expository), `[DERIVED]` (deduction from cited sources + our stated cost model), `[SPECULATIVE]` (unproven construction). This superset replaces ad-hoc per-card tag sets.
2. **Citations must be human-readable** (author-year-title or repo file/function path), not just link tokens — G1 cited `sequential_executor.cc::CalculateTotalOutputSizes`; G2 cited Vincent 1993 / Robinson / Liu-Tarjan-Zhong by name. Both survive the paste back into the repo; bare link artifacts don't.
3. **Demand a blunt bottom line**: each card's checklist should require a one-paragraph "overall verdict" the captain can act on without reading the tables (G2's "there is no generally exact O(log K) shortcut" settled three open prompt hedges in one sentence).
4. **Standing card requirements that proved their worth** (keep): researcher-contract preamble (internet-only, no file access, single Markdown note); all project facts inline incl. the cost-model formulas; required answer ORDER; required TABLE SCHEMAS; counterexample-or-proof demands; verification checklist.

**Filing protocol (captain, when a deliverable lands as `deep-research-report.md`)**:
rename to the card's suggested filename under `Onnx_Advanced/Research/Done/` → extract actionable findings into `techniques.md`/`traps.md`/`cost_floor_reference.md` (dated, with a pointer back to the note) → update any OUTSTANDING worker prompts the findings obsolete → mark the card Done in the ledger below → retire the card file to `../Done/` → CHANGELOG entry.

## Active cards (L-series, authored 2026-06-07 — born from the einsum-final breakthrough)

| Card | Topic | Why it's load-bearing now | Priority |
|---|---|---|---|
| ~~L1~~ | Exact CP/rank decomposition — **Done 2026-06-07**: `../Done/l1_exact_cp_decomposition.md` (verdict: unrestricted-real rank is ∃R-hard, skip; the ONE regime = bounded-dyadic MIQCP + dictionary + symmetry breaks; flattening ranks = instant lower-bound kill test; permutation banks CANNOT OR-compress) → techniques #87 | — |
| ~~L2~~ | ORT Einsum internals — **Done 2026-06-07**: `../Done/l2_ort_einsum_internals.md` (in-kernel temporaries = temp-space, SCORER-INVISIBLE; operand order IS the left-deep path — kill big non-output labels early; no operand cap; split only for ~10x; diagonal/permute slow paths; #18654 edge) → techniques #89 | — |

## Active cards (R-series, REPO-ACCESS, authored 2026-06-08 — highest leverage, act on real data)

| Card | Topic | Deliverable | Priority |
|---|---|---|---|
| ~~R1~~ | Big-fish formulation-recon — **DONE 2026-06-08**: `Logs/research/R1_bigfish_*` (66 rows, 40 decoded + 26 compaction-skip; top-EV 342/371/362/183/019/289/046/343/029; 11 label-misnomers flagged). Card retired to `../Done/`. | — |
| ~~R2~~ | Selector-bank rank reduction — **DONE 2026-06-08**: `Logs/research/R2_selector_bank_rank.md`. task312 residual REAL — exact rank-18 {0,+1,-1} CP shipped 600->570; ranks 10-17 OPEN (needs M2 solver stack). task304 DEAD (no cheaper factorization). task357 N/A (packed-lookup already cheaper than any bank). Card -> `../Done/`. | — |
| ~~R3~~ | Peephole rewriter — **DONE 2026-06-08 (v1)**: `tools/peephole_rewrite.py` built (--task/--sweep/--check); swept 219 workable, found+staged 5 add_chain_to_sum cuts (185/190/205/208/308, all byte-equal + profile-gap-0). v1 ships ONE safe rewrite; future v2 adds free-output/dtype/negpad/groups rules -> re-run as the library grows. `Logs/research/R3_peephole_sweep.md`. Card -> `../Done/`. | — |
| ~~R4~~ | task312 deep rank harvest — **DONE 2026-06-08**: `Logs/research/R4_task312_deep_rank.md` + runner `tools/scripts/task312_rank_cpsat.py`. CP-SAT proved rank<=17 infeasible over {0,1}/{0,1,2}; signed 17->13 OPEN-timeout, no witness -> rank-18 cost 570 stays best, NO gain. The +0.46 was theoretical; a signed factorization was not found in budget. Card -> `../Done/`. **Board now empty (all K/L/M/R cards filed).** | — |

Card bodies: `R1_REPOACCESS_*.md`, `R2_REPOACCESS_*.md`, `R3_REPOACCESS_*.md`, `R4_REPOACCESS_*.md`. They pair with the M-cards: M2 researched the solver recipe -> **R4 runs it** on the real 312 bank; M3 derives the linear-map theory -> R1 applies it to the real big-fish.

## Active cards (M-series, authored 2026-06-08 — gap-targeted post-batch-16)

| Card | Topic | Why it is a real gap | Priority |
|---|---|---|---|
| ~~M1~~ | Opset 19-22 audit — **DONE 2026-06-08**: `../Done/m1_opset19_22_op_audit.md` (verdict: mostly barren; 3 levers = ConstantOfShape param-shrink / bool-Reduce@20 / Pad-wrap@19; traps = bf16-not-honored-on-CPU, no-kernel DeformConv/Bernoulli/GroupNorm, AffineGrid loses) -> techniques #94 + ort_reality. | — |
| ~~M2~~ | Exact-rank tooling recipe — **DONE 2026-06-08**: `../Done/m2_exact_rank_tooling_recipe.md` (verdict: OR-Tools CP-SAT, runnable script included; S={0,±1}+table-products+symmetry-breaks+2^24 guards) -> techniques #87. UNBLOCKS the 312 ranks-16-13 harvest. | — |
| ~~M3~~ | Min-cost integer linear map — **DONE 2026-06-08**: `../Done/m3_min_cost_integer_linear_map.md` (7-step decision procedure: boundary -> zero-param -> selector -> conv-sharing -> block -> low-rank -> sparse -> dense; structure->op table + kill-numbers) -> techniques #104. Pairs with R1 (applies it to the real big-fish). | — |

Card bodies: `M1_*.md`, `M2_*.md`, `M3_*.md` in this folder. File deliverables to `../Done/` and extract to techniques/traps on return (filing protocol above).

## Completed ledger (deliverables + retired cards live in `../Done/`; don't re-research)

**K-series (2026-06-06/07, five for five decision-grade)**:

| Card | Deliverable (in `../Done/`) | Verdict + extraction |
|---|---|---|
| ~~K1~~ | `k1_superoptimization_survey.md` | SKIP all superoptimizer ports (XL build, float-aggressive rules); sanctioned fallback = tiny ONNX-native peephole over the exact-rule shortlist → techniques #66 |
| ~~K2~~ | `k2_ort_exact_semantics.md` | v1.24.4 source-read; TopK sorted=1 mandatory, Where-bool no kernel, integer Div/Mod/Pow native-C++ semantics, pin Resize attrs → ort_reality.md K2 section + traps §K2 ACT-NOW PAIR |
| ~~K3~~ | `k3_unexploited_op_audit.md` | 72 residual ops; finds: Sum/RoiAlign/LpPool-p1/Hardmax; CenterCropPad CONFIRMED kernel-less; NMS validator-fatal → techniques #75 |
| ~~K4~~ | `k4_compression_native_minimal_nets.md` | Nothing beats hand construction; ONE transfer = offline ESPRESSO/ABC logic-synthesis pass → 1×1-Conv compile → techniques #76 |
| ~~K5~~ | `k5_accumulation_exactness.md` | Load-bearing F16-Conv claim REFUTED at source (NO native F16 Conv kernel; cast-insertion→F32); REAL hazard = native F16 Gemm stepwise rounding; QLinearConv ≤255 confirmed → techniques #77 |

**Earlier series (all delivered + extracted; bodies in `Onnx_Advanced/_archive/`, prefixed `Research_*` / `TodoDone_*`)**:

- **A1/A2/A3, C1, E1, F1, F2** — deliverables promoted to `Onnx_Advanced/` root docs: `symmetry_and_affine.md`, `arc_rule_taxonomy.md`, `synth_oracle_strategies.md`, `cellular_automata.md`, `mlp_rule_extraction.md`, `radius_encoded_weighted_cover.md`, `row_signature_rank_reduction.md`.
- **F3/F4** — `multistage_key_first_compositional.md`, `direct_detection_multi_qlinearconv.md` (root; extractions in techniques). Caveat both: worked examples were researcher inferences from public ARC hints, not project-verified specs.
- **G1** (ORT profiler vs dynamic shapes → techniques §39-UPGRADE), **G2** (log-round reconstruction: doubling provably NON-exact → techniques §37-ADDENDUM), **G3** (n-ary fan-in catalog → techniques §40), **H1** (spatial data-movement → #44), **H2** (bit-packing law → #45), **J1** (row-packed u32 morphology → #47), **J2** (Einsum equation catalog → #46) — deliverable bodies at `_archive/Research_<name>.md`, card bodies at `_archive/TodoDone_Card_*.md`.

**Declined topics (don't re-propose)**: ONNX quantized ops, bit-packing, FunctionProto/If subgraphs (validator-banned), connected-component labeling, pathfinding/wavefront, ARC-AGI 2024 winners, ARC Prize 2025+ research, discriminating-input synthesis, ARC-AGI-2 rule families. Reasons: standard knowledge in `techniques.md`, validator-banned per `core.md`, or covered by sibling project K-Arc2.

**Cross-project reference**: `../Done/k_arc2_cross_reference_2026-05-26.md` documents the sibling project `C:\path\to\K-Arc2` (ARC Prize 2026 / ARC-AGI-2) which has substantial ARC research applicable to NeuroGolf rule-decoding work. Read this BEFORE proposing new ARC-related research cards.

## Common environment block (referenced by every card)

```
Environment:
- ONNX opset: ≤ 18
- Runtime target: ONNX Runtime 1.24
- Banned ops: Loop, Scan, NonZero, Unique, Compress
- Layout: NCHW
- Typical tensor: [1, C, H, W] with C ≤ 10, H,W ≤ 30
```

## Common cost model block (referenced by every card)

```
Cost model:
- Score = intermediate_memory_bytes + param_elements
- param_elements = sum of element counts across all initializer and Constant-node tensors
  (unique tensors only; an initializer referenced N times counts once)
- intermediate_memory_bytes = sum of byte sizes of value_info tensors materialized between
  input and final output. Excludes the input tensor; excludes the final output tensor.
- Express memory symbolically using C, H, W, bytes_per_element where applicable.
- Separate static-shape and dynamic-shape cases when they change the pattern or cost.
- Ignore ONNX Runtime kernel fusion unless explicitly documented.
```

## Common boolean-op caveats block

```
Boolean-op caveats:
- And, Or, Not, Xor operands must be bool dtype (no implicit cast).
- ReduceMax/ReduceMin/ReduceSum on bool: requires Cast to UINT8/INT first.
- Where(bool, bool, bool) is rejected by ORT — rewrite as Or(And(c, a), And(Not(c), b)).
- Bitwise ops (BitwiseAnd/Or/Xor/Not, BitShift) require integer dtypes, not bool.
- Bool Pad requires opset 13+.
```

## Source policy (every card uses)

```
Sources:
- Use official ONNX operator spec pages (https://onnx.ai/onnx/operators/) as primary sources.
- Cite the ONNX spec for every operator used.
- Cite ONNX Runtime documentation only for runtime-specific behavior.
- For algorithmic claims, cite primary academic papers or canonical repositories.
- Mark any inference, estimate, or extrapolation that is not directly specified by the docs.
```

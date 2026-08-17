# Research Todo — the card board (two card types)

1. **Internet-only cards** (historically M-series; current plain O-series cards use this contract): designed for plain ChatGPT / a deep-research model with web access but NO repo access. Paste the card body; it returns a self-contained Markdown note you drop in `../Done/`. Strict contract below.
2. **REPO-ACCESS cards** (historically R-series; current `O*_REPOACCESS_` cards use this contract): assigned to a worker WITH full repo + tool access. They read our generators/onnx/corpus, run the gates, build tooling, and write findings to `Logs/research/` (+ stage gated candidates for the captain). These act on our real data — higher leverage. They follow `WORKER_PROTOCOL.md`, never submit.

## Filename convention

- `*_WEBAPP_SELFCONTAINED_*.md` — paste into a web app / deep-research chat with internet access and no access to this repo or machine.
- `*_REPOACCESS_*.md` — assign only to a worker with this repo, tools, and local artifacts.

## Board status

**REOPENED 2026-07-10 for two repo-access librarian cards.** O7 and O8 are delivered, extracted, archived, and retired. O1-O8 are delivered/extracted. Completed-card ledger (what was researched, verdicts, where each body lives — **don't re-propose these**): `../Done/COMPLETED_CARDS_LEDGER.md`. Completed audit/refactor deliverables now live in `../Done/audits/`.

Open cards:

- `SHO_REPOACCESS_CURRENT_TEAM_BEST_IDIOM_HARVEST.md` — repo-access harvest of the 2026-07-10 Sho Team_Best promotion plus low-cost obscure-op examples; expected output is `Logs/research/SHO_CURRENT_TEAM_BEST_IDIOM_HARVEST_20260710.md` with reviewed fact candidates and task-index corrections.
- `ROOT_UPDATE_REPOACCESS_BACKLOG_DRAIN_20260710.md` — repo-access drain of the 22 root `.update/DONE_*.md` drops that official librarian digest does not see; expected output is `Logs/research/ROOT_UPDATE_BACKLOG_DRAIN_20260710.md` with reviewed fact candidates, doc patch plan, and archive plan.

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

1. **Source-tag standard (G2's 5-class set)**: require tags `[PROMPT]` (fact given in the card), `[PRIMARY]` (peer-reviewed/primary source), `[SECONDARY]` (docs/expository), `[DERIVED]` (deduction from cited sources + our stated cost model), `[SPECULATIVE]` (unproven construction).
2. **Citations must be human-readable** (author-year-title or repo file/function path), not just link tokens.
3. **Demand a blunt bottom line**: each card's checklist requires a one-paragraph "overall verdict" the captain can act on without reading the tables.
4. **Standing card requirements**: researcher-contract preamble (internet-only, no file access, single Markdown note); all project facts inline incl. the cost-model formulas; required answer ORDER; required TABLE SCHEMAS; counterexample-or-proof demands; verification checklist.
5. **Before proposing an ARC-related card**: read `../Done/k_arc2_cross_reference_2026-05-26.md` (the sibling K-Arc2 project covers substantial ARC research).

**Filing protocol (captain, when a deliverable lands)**:
rename to the card's suggested filename under `Onnx_Advanced/Research/Done/` → extract actionable findings into `techniques.md`/`traps.md`/`ort_reality.md`/`cost_floor_reference.md` (dated, with a pointer back) → **regenerate the catalog** (`python Onnx_Advanced/.update/tools/build_techniques_catalog.py`) + add the by_symptom route → update any OUTSTANDING worker prompts the findings obsolete → add a row to `../Done/COMPLETED_CARDS_LEDGER.md` → **`git mv` the full deliverable body to `_archive/Research_<name>.md` in the same pass** (rule 7; the extraction + ledger row are what stay live) → retire the card file to `../Done/cards/` → CHANGELOG entry → run `lib_lint.py`.

**Declined topics (don't re-propose)**: ONNX quantized ops, bit-packing, FunctionProto/If subgraphs (validator-banned), connected-component labeling, pathfinding/wavefront, ARC-AGI 2024 winners, ARC Prize 2025+ research, discriminating-input synthesis, ARC-AGI-2 rule families. Reasons: standard knowledge in `techniques.md`, validator-banned per `core.md`, or covered by sibling project K-Arc2.

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

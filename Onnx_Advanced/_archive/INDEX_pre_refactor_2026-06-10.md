# INDEX.md — the Onnx_Advanced router (READ THIS FIRST)

This is the map of the whole knowledge base. **You do not need to browse folders** — find your situation below and open the 2–4 files it names. Every durable doc lives at the top level of `Onnx_Advanced/`; the only subfolders are `Ort_Advanced/`, `exemplars/`, `Algorithm_Library/` (all routed below) and `Research/` (research-card pipeline: `Todo/` open cards + `Done/` delivered, not required reading).

> Convention: a bare filename like `techniques.md` or `arc_rule_taxonomy.md` always means the file at the root of `Onnx_Advanced/`.

---

## Start here (every task, ~15 min)

| File | Read it for |
|---|---|
| `INDEX.md` (QUICKSTART merged here Phase-2) | the 5-minute orientation + the standard workflow |
| `core.md` | the scoring formula, the input/output encoder, the 1-ch0 trap, the validator's banned-op list, cross-task memory cascade |
| `00_COMPETITION_SPEC.md` | the competition rules + what `submission.zip` must contain |
| `cost_floor_reference.md` | per-tensor byte tables + the full-canvas-plane floor taxonomy — **price your op chain before building** |

## The two routers (use whichever entry point you have)

| File | Read it for |
|---|---|
| `by_symptom.md` | **"I see pattern X in my graph"** → jumps you to the exact exemplar/technique (graph-shape → fix) |
| `task_techniques_index.md` | **"what was done to task N (or a similar task)"** → task ID → technique applied |

## The toolkit (the workhorses)

| File | Read it for |
|---|---|
| `techniques.md` | the full rewrite toolkit: bool-first family, F16/U8/BOOL placement, single-channel canvas, dilated crop-collapse (§8b), Einsum/ScatterElements, priority-scatter, and the §-numbered patterns by_symptom points into |
| `traps.md` | what breaks your model: the SHIP DECISION REFERENCE (incl. the Path-B/op-swap discriminator + §CROP-CONTRACT), build-time traps (`Where(bool,bool,bool)`, sub-INT32 gather idx, dynamic-shape Slice→None, Div/Mod-by-input→SIGFPE), runtime traps |
| `scoring_internals.md` | **how the scorer actually works** (onnx-tool profiler internals, cost-metric exploits, ban workarounds, grader replication). Cost-golf ground truth |
| `patterns.md` | recurring solver shapes |
| `cascade_strategy.md` | only if cost > 150k (cascade-unlock + single-task-wildcard submit rules) |
| `diagnostics_recipes.md` | the `tools/agent_helpers/` catalog (extract_task, inspect_graph, largest_intermediate, find_op, …) |
| `scoring_internals.md` (merged jewel) | refresher on ONNX fundamentals if you're new |
| `cost_floor_reference.md` (salvage section) | raw cost measurements (reference) |

## Decode the rule (Phase 1 — "I don't know what this task does")

| File | Read it for |
|---|---|
| `arc_rule_taxonomy.md` | the 18-family (ConceptARC names) taxonomy + decision tree — **short-circuits rule-decoding to <15 min** |
| `arc_rule_taxonomy.md` (alt archived — won 3/23; its rows noted in front-matter) | alternative 15-family (output-shape grouping) taxonomy if the primary doesn't fit |
| `arc_rule_taxonomy.md` (merged as front-matter) | which families the shipped tasks actually map to |

## Build patterns (after the rule is decoded)

| File | Read it for |
|---|---|
| `tiny_architectures.md` | hand-crafting sub-1k graphs: hand-set weights, LUT-as-network, weight sharing, nonlinearities, static-shape tricks, iteration without `Loop` |
| `mlp_rule_extraction.md` | LIVE is a small trained MLP/lookup → extract the symbolic rule, rewrite without trained weights |
| `direct_detection_multi_qlinearconv.md` | UINT8 QLinearConv detector banks |
| `radius_encoded_weighted_cover.md` | weighted-cover / vote-bank detectors |
| `row_signature_rank_reduction.md` | collapse per-row/signature MatMul churn (rank-4↔rank-2) |
| `multistage_key_first_compositional.md` | multi-step / key-first compositional rules |
| `zero_param_generators.md` | generate masks/indices at runtime with no param cost |
| `function_proto_subgraphs.md` | FunctionProto / `If` subgraph structuring (advanced) |
| `cellular_automata.md` | CA-style iterative-fill rules |

## Op references (reach for a specific op)

| File | Read it for |
|---|---|
| `op_cost_patterns.md` (spec reprints archived) | upstream ONNX 1.21 op schemas (BitShift, BitwiseAnd/Or/Xor, Gather, Mod) + opset coverage |
| `op_cost_patterns.md` (salvage; archived) | quick-ref for ops not in your usual vocabulary |
| `einsum.md` | Einsum as a free (MACs-uncounted) reduction/projection |
| `op_cost_patterns.md` + techniques #45 (salvage; archived) | modulo + bitwise op tricks (bitpacking) |
| `op_cost_patterns.md` (salvage; archived) | GatherND / multi-axis Slice / scatter indexing |
| techniques **#44** (canonical catalog; original archived) | crop / pad / tile / transpose geometry |
| `symmetry_and_affine.md` | reflections, rotations, D4, affine transforms |
| `op_cost_patterns.md` (API tour archived) | graph-surgery utilities |
| `op_cost_patterns.md` (salvage; archived) | misc ONNX build helpers |

## Validation

| File | Read it for |
|---|---|
| `synth_oracle_strategies.md` | building an INDEPENDENT synth oracle (avoid the self-consistency trap) |

## Subfolders (still routed — open the file the table names)

| Folder | Read it for |
|---|---|
| `Ort_Advanced/` | ORT-specific behavior: `ort_reality.md` (deviations), `ort_reality.md` (error→fix), `ort_reality.md` (first section) / `ort_reality.md` (what dtype/opset each op accepts), `scoring_internals.md` (profiling section merged Phase-2) |
| `exemplars/` | worked case studies by category (`01_whole_graph_rewrites/`, `02_bool_first/`, `03_crop_and_f16/`, `04_algebraic_identities/`, `05_static_to_dynamic/`, `06_cse_dead_op/`, `07_conv_reformulation/`, `08_gather_lookup/`, `09_periodicity_detection/`) — by_symptom links to the exact one |
| `Algorithm_Library/` | `algorithm_patterns.md` (bbox, mode-color, permutation, hash, CC) + `rule_discovery_methodology.md` |
| `Research/` | **NOT required reading.** Research-card pipeline: `Todo/` = open card prompts for outside researchers; `Done/` = returned deliverables + retired cards + process notes. Durable findings get *promoted* up to the root docs above (+ added to this INDEX). Browse only when hunting for something brand-new. |
| `.update/` | pending library updates dropped at session end (promoted into the toolkit docs over time) |

---

*Maintenance: when you add or promote a durable doc, add a one-line row here. Use `tools/refactor_paths.py` for any move/rename so references follow. This INDEX is the contract — if a file isn't listed here, agents won't find it.*


## EVERY-TASK SET (Phase-1, 2026-06-05 — size-budgeted; this is the WHOLE required read)

| file | budget | role |
|---|---|---|
| `INDEX.md` | 7 KB | this router |
| `core.md` | 25 KB | scoring formula / encoder / validator / hard rules — THE canonical home |
| `cost_floor_reference.md` | 40 KB | floors + per-op costs (ship-ledger evicted to `_archive/`) |
| `traps.md` | 82 KB | numbered traps + doctrine blocks (dated diary → `traps_appendix.md`, grep-only) |
| your `taskNNN.md` | — | task history |

Everything else is grepped per-section on demand (techniques.md by §number/title, by_symptom.md routing,
ort dtype matrix). Ship-decision strategy lives in `The_400_Tasks/current_strategy.md` §ship-decision.
Adding a file to this set requires evicting equivalent KB (charter §8b rule 9).


## 10-minute path (merged from QUICKSTART, Phase-2)


If you have time to read everything, follow `README.md`. If you don't, follow this.

## The minimum path before touching ONNX (10 minutes total)

1. **Read your task's graph diagnostic (2 min)**
   Open `The_400_Tasks/Task_by_Task/taskNNN.md`. Scroll to the `## Graph Diagnostic` section at the bottom. Note:
   - Cost / memory / params
   - Max active grid (from the JSON)
   - Quick-look flags (has_Conv? first_node_is_Slice? ends_with_Cast_to_F32?)
   - Suggested techniques

2. **Extract the canonical ONNX (10 sec)**
   ```
   .venv/Scripts/python tools/agent_helpers/extract_task.py NNN
   ```
   Drops `taskNNN.onnx` into your CWD.

3. **Read your task's entry in the router (2 min)**
   Open `task_techniques_index.md`. Find your task's family section. Find your task ID. Read the linked exemplar file in full.

4. **Read the trap catalog (3 min)**
   `traps.md`. Five minutes of trap-scan prevents hours of debugging. Especially:
   - ORT does NOT support `Where(bool, bool, bool)`.
   - OneHot: INT32 and INT64 indices BOTH work — but `depth` dtype must MATCH indices (the old "INT64 only" rule was a task124 misdiagnosis; traps §2).
   - `g.ClearField("value_info")` after any shape-changing surgery.
   - 1-ch0 trap for foreground detection.

5. **Read `core.md` (3 min)**
   NeuroGolf-specific facts. Scoring formula, encoder details, validator rejection criteria.


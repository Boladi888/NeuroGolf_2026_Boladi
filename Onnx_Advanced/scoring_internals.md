> **CORRECTION BANNER (2026-06-05, body moved to `_archive/` 2026-06-10)**: MACs were REMOVED from scoring on
> 2026-05-04 (see `core.md`). **cost = memory + params** — nothing else. The `_score_v2`
> tuple still RETURNS `(macs, memory, params)` for API compatibility; position 0 is ignored
> by every gate and by the grader. MAC-era body text, the stale dtype-packing sections, the
> competitor-exploit long-forms, and the hand-replication setup were moved to `_archive/` 2026-06-10
> (refactor pass 4) — full originals: `_archive/scoring_internals_pre_refactor_2026-06-10.md`.

# scoring_internals.md — how the NeuroGolf scorer actually works

> Profiler internals, scorer accounting laws, ban workarounds, grader-replication anchors, and the scoring-discrepancy history. Cost-golf ground truth. Single home (consolidated 2026-06-01).

## Contents
- [A.1 — profiler internals](#a1_onnx_tool_profiler_internals)
- [A.2 — accounting levers + decoded exploits (ethical-play: documented, not used)](#a2_cost_metric_exploits)
- [A.3 — ban workarounds](#a3_ban_workarounds)
- [A.4 — grader replication anchors](#a4_grader_replication)
- [How `memory` is actually measured (the §39-mechanism home)](#ort_profiling)
- [Scoring mechanics quick-list (the 9-line jewel)](#jewel)

---
<a id="a1_onnx_tool_profiler_internals"></a>

# A.1 — profiler internals

**Pinned version:** `onnx-tool==1.0.1` (validator dep), effectively frozen upstream. But note: the LIVE scorer for params/memory is `neurogolf_utils.calculate_params/calculate_memory`, NOT raw onnx-tool — every accounting question below is answered against `neurogolf_utils`.

`neurogolf_utils.calculate_*` calls `onnx_tool.Model(path).graph.shape_infer({"input": ...}); graph.profile()` after `loadmodel(..., {'constant_folding': True})`:

- `params` — sum of *unique* initializer tensor sizes — **ELEMENT COUNT, dtype-blind** (an initializer referenced twice counts once; a sparse `Constant.sparse_value` counts stored values).
- `memory` (custom in validator) — sum of `value_info` intermediate-activation byte sizes plus declared IO bytes, **excluding** the `"input"`/`"output"` tensors themselves. NOT onnx-tool's native memory field. Initializers do **not** count toward `memory`.

Cost fed to the score formula = `memory + params`. Score `= max(1.0, 25.0 - log(max(1.0, cost)))`. Natural log.

## Quirks that bite

- **No fusion assumed.** Conv→BN→ReLU bills as three ops. Pre-fuse BN into Conv weights (`W' = W·γ/√(var+ε)`, `b' = (b−mean)·γ/√(var+ε) + β`); never ship a separate BN.
- **Constant folding is enabled, but not a license to ignore generated tensors** (host-confirmed
  2026-04-28; `neurogolf_utils.py:209` loads with `constant_folding: True`). Some all-initializer
  subgraphs fold away; others, such as all-init `Tile`/`Range`/`Gather`/`ConstantOfShape` results
  that survive as graph values, are still charged by `calculate_memory`. Build inputs as initializers
  where it removes runtime tensors, but profile every generator rewrite.
- **Shared initializers count once.** Two Convs referencing the same weight name = one params charge. The tied-weights / cyclic-shift-kernel lever.
- **Forum thread `/discussion/692827` ("scoring discrepancies")** remains unread (anti-bot wall) — see A.4 for the known anomaly evidence.

---
<a id="a2_cost_metric_exploits"></a>

# A.2 — accounting levers (corrected 2026-06-10) + decoded exploits

Validator gates: opset `""`/`"ai.onnx"` only, static shapes, no banned ops, exclusive `>` 1,509,949-byte file cap.

## Levers that actually work (post-MAC, element-count reality)

| Lever | Reduces | Notes |
|---|---|---|
| Shape compression of initializers (separable masks, scalar broadcast, computed-on-the-fly) | params | THE params lever — params = element count, so only SHAPE helps (see cost_floor_reference.md) |
| Dtype narrowing of *intermediates* (F32→F16→U8/BOOL where kernels allow) | memory | dtype helps memory ONLY; per-op dtype floors in traps §8f + the ort_reality matrix |
| Output/input exemption (free-output families) | memory | `"input"`/`"output"` tensors are skipped — techniques #67/#78 catalog |
| Pre-fuse Conv+BN; drop Identity/Dropout; prune dead initializers | memory+params | pure local rewrites, free |
| Share one initializer across N nodes (alias by name) | params | up to N× on the shared tensor |
| `ConstantOfShape(shape_init, value=v)` instead of a dense constant | params + file bytes | output memory is charged if the generated tensor survives; use for param/file shrink, graph output, or inside a free-output contraction; re-measure post-fold |
| Strip `doc_string`/`metadata_props`/`training_info`/unused initializers; 1-char names | file bytes only | matters when racing the 1.5 MB cap |

**Measured (R6 + O4, 2026-06-19):** final-output `ConstantOfShape` locally charges the shape initializer only; the scalar `value` attribute is uncharged. R6's `[1,1,4,4]` u8 probe scored `memory=0 params=4` vs dense `Constant` params=16. O4 repeated the A/B at full NeuroGolf output shape `[1,10,30,30]`: dense final `Constant` scored 9000 params, while default/explicit F32 zero and explicit U8-seven `ConstantOfShape` scored 4 params and 0 memory. **Caveat:** this is still local scorer behavior, not Kaggle-confirmed doctrine. O4 found no public task whose correct final output is a scalar-filled tensor, so do not submit the probe models; exploit this only when a real task-valid graph preserves the one-hot output and passes the gates.

**Estimation discipline (task152, 2026-06-13):** an intermediate's bytes = product of ALL dims × itemsize — the channel dim is NEVER free in an intermediate (only the graph `input`/`output` are exempt). A BOOL `Equal([1,10,6,6])` is **360 B, not 60 B**; the frequent reviewer under-count is forgetting the ×10 channel factor.

**No-liveness plane trap (task079, 2026-06-19):** per-color 2-D window planes and
re-expanded color axes are still fully charged even when a later op reads only a
few winners. Collapse spatial axes to 1-D projections first when possible, while
keeping colors separated only as long as they are needed. Do not assume a dead-looking
color plane is free unless it is inside a free-output contraction or removed from
`value_info`.

**Dead levers (proven, do not spend time):** dtype-packing initializers (uint8/uint4/f16) does NOT cut params — `calculate_params` counts ELEMENTS, dtype-blind (task055 Range-vs-init probe; task180 burned 30 min re-disproving; the stale "4–8× params" table is archived). Sub-byte packing buys file bytes only. `sparse_initializer` as an op input → scorer `None` (3 independent gates, traps §8e); sparse NNZ-counting is unreachable for any real consumer. Packed-bool bit-fields: see the #45 packing law (only ≥4 packed-executable consumer sites escape).

**Quantized op variants** (`QLinearConv`, `MatMulInteger`, `ConvInteger`): useful as TYPE-NARROWING for memory (U8 lanes) and as free compute gadgets — techniques #41/#72/#85/#120/#137. Params unchanged (elements). Symmetric scale=1/zp=0 folds the q-params to nothing.

**`external_data`: do not use.** Single-file submission; the cap is per file; no upside, DQ risk.

## File-byte slimming (the cap, separate from cost)

The 1,509,949-byte cap is on the serialized file (exclusive `>`). Strip: `doc_string` everywhere, `metadata_props`, `training_info`, unused initializers, dead `value_info`, long node/tensor names (`node.name = ""` is legal). Protobuf overhead per tensor is small but real; typical models are 500–5,000 B so this matters only near the cap.

## What the validator does NOT count

- **Initializer bytes → not `memory`** (params only).
- **Actually folded subgraphs** → vanish except surviving initializers. All-init generator outputs that remain
  in `value_info` still count as memory.
- **`If`/`Loop` bodies** → banned outright, not a counting question.

## Decoded competitor exploits — documented so we can read competitor cost-tables; OWNER STANCE: ethical play, never implement

- **kq5y `shape_infer(None)` gap (LIVE, by-design upstream):** `ConstantOfShape` whose shape derives from input data infers `[…,0,0]` under the scorer's zero-dummy → downstream ops bill ~0. Survives the 2026-04-28 update. This is the mechanism behind "positive-cost wrappers": `Shape(internal)→ConstantOfShape(0)→Add(internal, cos)` — 3 ops that *increase* runtime cost yet *decrease* profiled cost (decoded from a public "ONNX hacker" script; two sibling hacks: static→symbolic dims on output/value_info, initializer→Constant conversion — the latter neutralized by the Apr-28 folding fix).
- **Compress-with-fake-zero-batch:** mechanism confirmed locally (all-False condition collapses downstream shapes to 0) but **unreachable today** — `Compress` is banned and the ban check walks ALL nodes (3 bypass routes tested and rejected; unknown op_types escape the name-match but die in shape_infer). Likely a historical exploit against a pre-Apr-21 scorer.
- **History:** evaluator fixed + leaderboard recalibrated **2026-04-29**; treat any cost numbers from before then as suspect. codex#19594 flagged a 5148→7600.42 jump citing `ConstantOfShape`/dynamic/dtype accounting on tasks 157/177/002/337/276/309. Yiheng Wang (3rd) explicitly pinned 1.0.1 *to avoid* exploiting bugs.
- **Detection duty:** we don't use Constant nodes, `dim_param`, or data-derived ConstantOfShape — clean by construction; keep it that way so our own ships never accidentally trigger re-pricing.
- onnx-tool <1.0.1 bug table (Expand/negative-Slice/Less-inference/Constant-memory): all fixed or moot on 1.0.1 + folding. `EinSum` without ellipsis does not broadcast (spec-correct, don't rely on it); `Pow` mismatched-broadcast inference unverified.

## Cost ladder cheat sheet (architecture → cost → points; single home, SPEC/core point here)

| Architecture | Cost | Score |
|---|---|---|
| Static Identity (cost=0) | 0 | **25.0** |
| Channel Gather | ~10–54 | **21.0–22.7** |
| Slice/Pad chains | ~200–500 | **19–21** |
| Spatial Gather | ~4,563 | **16.6** |
| 1×1 Conv | ~90,500 | **13.6** |
| 3×3 CNN | ~814,500 | **11.4** |

"Lower local cost ≠ better LB score. Always bisect on LB before accepting a swap." (silent-fail lesson; gates exist now — WORKER_PROTOCOL.)

---
<a id="a3_ban_workarounds"></a>

# A.3 — Ban workarounds

The validator rejects: banned ops (`Loop, Scan, NonZero, Unique, Script, Function, Compress`), subgraph-bearing nodes (`If`, anything with `GRAPH`/`GRAPHS` attr), non-stdlib domains, and files over the cap. The common thread: **dynamic shapes and control flow are blocked; static shape-determined dataflow is fine.**

| Ban | Why banned | What we use instead |
|---|---|---|
| `Loop` / `Scan` | subgraph attrs → `calculate_memory` = None | **unrolled stacked ops** (bounded: flood ≤60 dilations; power-iteration doubling reaches diameter-30 in ~6 steps — techniques §37 has the corpus-measured floors; tiny_architectures B.6 the catalog) |
| `If` | body subgraph | `Where(mask, a, b)` (one node, one output intermediate — always prefer over mask-multiply, which pays 3-4 intermediates); `Min`/`Max` for compare-select. Both branches are always computed — keep them cheap/shared |
| `Function` | local custom ops | inline each use; share initializers by name (count once) |
| `NonZero` | dynamic output shape | `TopK` (static `[k]`) or `ArgMax` + threshold |
| `Compress` | dynamic output shape | `Where(mask, value, sentinel)` — keep full grid, mark instead of filter |
| `Unique` | dynamic output shape | static cardinality (≤12 colors): `Equal`-bank / OneHot + ReduceMax presence vector |
| `Script` / custom domain | embedded code / non-stdlib | rewrite in stdlib; `domain ∈ {"", "ai.onnx"}` only — watch for `com.microsoft.*` contrib ops |

**Unroll cost (post-MAC):** params are shared-kernel-free (name dedup); the bill is MEMORY — every unrolled step's intermediate counts (`[1,10,30,30]` F32 = 36,000 B/step, U8 mask = 900 B/step). This is why §37 iteration floors and the log2 power-iteration schedule matter: 6 doubling steps ≈ full-grid reach vs 30–60 linear steps.

**Every "find-where" becomes "mark-where"** — we always carry the full grid, so position-extraction ops are never needed.

## Banned-op emission footguns (when not hand-building)

`tensor[mask]` → NonZero+Gather · `torch.unique` → Unique · `torch.nonzero` → NonZero · `torch.cond`/scripted control flow → If/Loop (**never `torch.jit.script`**; tracing-based export unrolls loops, which is what we want). The project default is hand-construction via `onnx.helper` (zero exporter risk) — capture trained weights only, build the graph manually.

**Post-export surgery checklist:** (1) `onnx.checker.check_model`; (2) assert no banned `op_type`; (3) assert `domain ∈ {"", "ai.onnx"}`; (4) assert no `GRAPH`/`GRAPHS` attr; (5) assert every dim has `dim_value` (no `dim_param`); (6) `score_network_v2` returns non-None.

---
<a id="a4_grader_replication"></a>

# A.4 — Grader replication anchors

**Never hand-replicate — the tooling exists**: `tools/_score_v2.score_network_v2` (exact local cost; 99%+ within 1 byte of Kaggle) and `tools/ng_gate.py` (visible/truth/ab/battery). This section keeps the FACTS the tools embody. (The original kaggle-pull + headless-stub + monkey-patch setup recipe: archived snapshot.)

- Validator = `neurogolf_utils.py`, pinned deps `numpy==2.4.4 onnx==1.21.0 onnxruntime==1.24.4 onnx-tool==1.0.1`.
- `verify_network` runs ORT over train+test+arc-gen pairs, thresholds `(result > 0.0).astype(float)`, `np.array_equal` against expected.
- Points: `max(1.0, 25.0 − ln(max(1.0, memory+params)))`. Correct floors at 1.0; **incorrect = literal 0** (not 0.001) — a 95%-right heuristic is WORSE than a correct expensive model on that task.
- Sanity anchors: Identity scores 25.000 on tasks 179/241; 1×1 identity Conv = 90,100 → 13.591 pts; color-perm Gather ≈ 22.6974 pts/task.

## ARC-GEN + the hidden distribution

Each `taskNNN.json` bundles `train` + `test` + `arc-gen` lists; the grader evaluates ALL THREE for public score; private adds a held-out set — **most likely a fresh sample from the same mimetic generators** (Moffitt authors both), possibly edge-weighted. Generators vary grid sizes within the task's natural envelope — any pattern that only works on 30×30 fails. Design from the *transformation*, not the visible examples' properties. The deterministic 200k corpus cache + `ng_gate truth` is the local proxy (WORKER_PROTOCOL gates).

**Official-`None` oversize examples are skipped, not unprofiled inputs (`profile_real_cost --arcgen`, task366/task184/task381, 2026-06-22).** The official `convert_to_numpy`/encoder treats any bundled example whose max dimension exceeds 30 as **`None`** (it cannot encode it into the `[1,10,30,30]` frame), so the grader never scores it. A full-arc-gen profiler that tries to encode those rows CRASHES; the honest "full arc-gen profile" is the count of **scorer-encodable** examples with the official-`None` (>30) rows SKIPPED — e.g. task366 profiled 255 valid and skipped 11 official-`None`; task184 used 169 encodable rows and skipped 97 `>30x30` rows; reviewers accept the filtered set as the honest cost basis. The skipped rows are official `None`, NOT valid hidden-size inputs you failed to profile.

## Submission format (hard facts)

One `taskNNN.onnx` per task · I/O tensor names exactly `"input"`/`"output"` · input `[1,10,30,30]` F32 (output one-hot compared after `>0.0` threshold; channel 10 = padding sentinel, 11 = too-many-colors) · all dims `dim_value` · per-file ≤ 1,509,949 B · zip of 400 files, 1:1 (no ensembling) · 5 submits/day, Kaggle keeps highest.

---
<a id="ort_profiling"></a>

# How `memory` is actually measured (the §39-mechanism home)

The scorer's `calculate_memory(model, trace_path)` combines static shape inference with the ORT runtime profiler trace:

1. **Validation**: rejects >1 input/output, name collisions, subgraphs, sequence types, non-default domains, functions.
2. **Static pass**: walks `graph.input + value_info + output`; per tensor `num_elements × dtype.itemsize` — tensors named literally `'input'`/`'output'` are **SKIPPED**; dynamic dims → model rejected (`None`).
3. **Profiler pass**: for each trace `Node` event's `output_type_shape`, computes bytes and takes **`max(static, profiler-observed)`** per tensor.
4. Returns the sum.

## Key implications (each one is a shipped technique family)

- **Initializers don't count toward memory** (params only, element count) — shape-compress params, dtype-narrow memory.
- **`'input'`/`'output'` skipped** → the F16/BOOL-output trick (techniques #3) and the whole free-output catalog (#67/#78): make the expensive tensor BE the output. For expansion/paint tails, land the final broadcast/scatter on graph `output`; task286's large UINT32 one-hot output is free while predecessors would be charged.
- **Growth-op predecessors are not skipped** (task057/task120): `Pad`/`Concat`/`Tile`/`Expand` inputs are ordinary charged value_info tensors. Build-smaller-then-grow wins only when `param_save > predecessor_bytes`; a single free-output Conv/Einsum often beats decomposed assembly.
- **3+ operand non-output Einsums can carry a Kaggle markup** (task246): if a static graph scores higher on Kaggle than local while final-output Einsums do not, try 2-op all-channel moment rewrites; profile every candidate on real/arcgen inputs.
- **Scalars cost `dtype.itemsize`** (BOOL []=1 B … INT64 []=8 B) — small-dtype your sentinels.
- **`max(static, runtime)` is THE grader-gap mechanism** (traps §GRADER-GAP, techniques §39 MECHANISM): static graphs price EXACTLY; dynamic/broadcast-inflated shapes get charged at the runtime max profiled on REAL inputs. Stale value_info that understates a broadcast result gets out-voted by the profiler — `ClearField("value_info")` after surgery.
- **`calculate_memory` RE-INFERS value_info** (`infer_shapes(strict_mode=True)`) — real intermediates count even if you save with empty value_info; clearing is for load-time conflicts, NOT for hiding memory. Declaring EXTRA dead value_info entries is wasted bytes.
- **Single-op `input → Op → output` graphs have memory = 0** — zero-cost class (Identity/Transpose ships on 179/241).

## Local-vs-Kaggle calibration

Per-task cost: exact (99%+ within 1 byte) for static graphs. Aggregate LB deltas follow the regime table in `cost_floor_reference.md` (single-task 1.0×, megastack ~0.48×, cascade = rare step-function). The cross-task memory cascade only manifests in Kaggle's pipeline (local runs are isolated) — trigger set in traps §20.

---
<a id="jewel"></a>

# Scoring mechanics quick-list (the 9-line jewel)

- `Constant` node `value` attr → params via `attr.t.dims` element count. Sparse `sparse_value` counts **stored values**, not logical dense shape.
- `graph.initializer` → params via `init.dims`. `graph.sparse_initializer` counts `values.dims`, but is not submission-usable: strict shape-infer treats sparse-fed op inputs as rank-0/unknown, value_info fixes re-bill dense memory, and sanitizer misses sparse names (traps §8e).
- Both count toward the file cap (1,509,949 B).
- Prefer `initializer` over `Constant` nodes — same params, no node overhead.
- `calculate_memory` re-infers value_info (strict) before counting — real intermediates always count.
- Each value_info entry bills `num_elements × dtype.itemsize`; scalars bill itemsize (1/2/4/8 B by dtype).
- Extra/dead value_info entries are pure waste; real ones are counted regardless.

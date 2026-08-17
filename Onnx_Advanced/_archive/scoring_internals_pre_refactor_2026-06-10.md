> **CORRECTION BANNER (2026-06-05 Phase-0 hotfix)**: MACs were REMOVED from scoring on
> 2026-05-04 (see `core.md`). **cost = memory + params** — nothing else. The `_score_v2`
> tuple still RETURNS `(macs, memory, params)` for API compatibility; position 0 is ignored
> by every gate and by the grader. Formula sites below were corrected in place; any stray
> "macs" in prose describes the dead pre-05-04 metric.

# scoring_internals.md — how the NeuroGolf scorer actually works

> onnx-tool profiler internals, cost-metric exploits, ban workarounds, grader replication, and the leaderboard-scoring discrepancy. Cost-golf ground truth.
> Consolidated from `Onnx_Advanced/scoring_research/` on 2026-06-01 for discoverability (see INDEX.md). Originals removed; this is the single home.

## Contents
- [A1_onnx_tool_profiler_internals.md](#a1_onnx_tool_profiler_internals)
- [A2_cost_metric_exploits.md](#a2_cost_metric_exploits)
- [A3_ban_workarounds.md](#a3_ban_workarounds)
- [A4_grader_replication.md](#a4_grader_replication)
- [leaderboard-scoring-discrepancy.md](#leaderboard-scoring-discrepancy)


---
<a id="a1_onnx_tool_profiler_internals"></a>

# A.1 — onnx-tool profiler internals

**Pinned version:** `onnx-tool==1.0.1` (validator dep). Source: `ThanatosShinji/onnx-tool`. Canonical reference: `data/Profile.md` in the repo. MIT-licensed. Last upstream release ~Feb 2025; project is "Inactive" per Snyk → 1.0.1 behavior is effectively frozen for our purposes.

## What the scorer actually invokes

`neurogolf_utils.calculate_*` calls `onnx_tool.Model(path).graph.shape_infer({"input": ...}); graph.profile()` after `loadmodel(..., {'constant_folding': True})`. Three numbers come back:

- `macs` — sum over nodes of each node's profiler `macs` return.
- `params` — sum of *unique* initializer tensor sizes (the same initializer referenced twice counts once).
- `memory` (custom in validator) — sum of `value_info` intermediate-activation byte sizes plus declared IO bytes, **excluding** the `"input"`/`"output"` tensors themselves. This is **not** onnx-tool's native `memory` field (which is weights+output per layer); validator overrides it. Initializers do **not** count toward `memory` — they are `params` only.

Cost fed to the score formula = `memory + params` (MACs removed 2026-05-04). Score `= max(1.0, 25.0 - log(max(1.0, cost)))`. Natural log. So `cost = 1` → 25.0 (only achievable by the floor); `cost = e^24 ≈ 2.6e10` → 1.0.

*(MAC definition + Conv formula + op-by-op MAC table — MAC-era content archived to `_archive/scoring_internals_pre_phase3.md`; MACs left scoring 2026-05-04.)*

## Quirks / exploits / things that bite

- **No fusion assumed.** Conv→BN→ReLU is billed as three separate ops. Pre-fuse manually: fold BN into Conv weights (`W' = W * gamma/sqrt(var+eps)`, `b' = (b-mean)*gamma/sqrt(var+eps) + beta`), and ReLU is unavoidable except via Greater+Where (same cost). Don't ship a separate BN.
- **Constant folding runs first.** Anything whose inputs are all initializers gets pre-evaluated and replaced with a Constant. So `Reshape` with a constant shape input, `Slice` with constant indices, `Transpose` of a constant tensor, etc. — they all disappear. Build inputs as initializers wherever possible.
- **Shared initializers count once.** If two Conv nodes reference the same weight tensor name, the param count adds once. Useful for tied weights / cyclic-shift-tied kernels (see B.3).
- **`raw_data` vs typed fields:** weights stored as `raw_data` (bytes) are byte-counted by `params` based on declared dtype size × element count. `int4`/`uint4` packed two-per-byte initializers are halved (verify in A.2). Sub-byte storage is the lever for params.
- **MACs are dtype-blind.** A Conv with int8 weights costs the same MACs as a Conv with float32 weights (per the "1 MAC = float(a)*float(b)+float(c)" definition — onnx-tool doesn't downgrade for narrower dtypes). So **quantization helps `params` and `memory`, not `macs`**. To cut `macs`, only operator/topology changes work.
- **Special-function MACs are inaccurate.** `Exp`, `Log`, `Sigmoid`, `Tanh`, `Softmax` are billed at 1 MAC/element by onnx-tool but real cost is much higher on hardware. Doesn't matter for us — only the billed number matters. So Exp is no costlier than Add per onnx-tool. But Softmax is still ~3× because it composes max+exp+sum+div.
- **Custom ops fall back to 0 MACs** if no profiler is registered — but we can't ship custom ops anyway (`Function` is banned).
- **Sparse tensors:** `sparse_initializer` is supported by the format. Whether onnx-tool's `params` counter reduces for sparse storage is a A.2 follow-up — current assumption: it counts the dense element count, not the bytes.
- **Forum thread `/discussion/692827`** flags "scoring discrepancies" between onnx-tool and ground truth. Still unread — possible exploit surface. Worth pulling next.

## Tooling / replication recipe

```python
import onnx_tool
m = onnx_tool.Model("model.onnx", {"constant_folding": True})
m.graph.shape_infer({"input": np.zeros((1,10,30,30), dtype=np.float32)})
m.graph.profile()
m.graph.print_node_map()  # per-node MACs/params/memory table
# Validator-style sums:
total_macs   = sum(n.macs   for n in m.graph.nodemap.values())
total_params = sum(t.numpy.nbytes for t in m.graph.tensormap.values() if t.is_initializer)  # rough
```

For the validator's *custom* memory metric, walk `value_info` byte sizes excluding `"input"`/`"output"` — see `neurogolf_utils.calculate_memory()` directly.

## Action items downstream

- **A.2** confirms cost-metric exploits (sub-byte packing, sparse, external_data, ConstantOfShape free zeros).
- **A.4** confirms ARC-GEN sampling and the `/discussion/692827` loophole.
- **C.1** confirms which onnx-optimizer passes the scorer effectively pre-applies (constant folding) vs which we must apply ourselves (Conv+BN fusion, Identity removal, dead initializer pruning).

*(MAC-era line removed — MACs left scoring 2026-05-04.)*

---
<a id="a2_cost_metric_exploits"></a>

# A.2 — Cost-metric exploits

The score is `25 - log(memory + params)` (MACs removed 2026-05-04). Each term has different exploits. Knowing what onnx-tool actually counts (A.1) is the ground truth; what follows is everything we can stack on top of it within the validator's gates: opset `""`/`"ai.onnx"` only, static shapes, no banned ops, exclusive `>` 1,509,949 byte cap.

## Levers, ranked by expected leverage

| Lever | Reduces | Risk | Expected leverage |
|---|---|---|---|
| Stay zero-MAC (Identity / Transpose / Slice / Gather / Reshape / Cast / Resize-nearest / Concat / Split / Pad / Tile / ConstantOfShape) | macs | none — these are the safe vocabulary | **infinite** (cuts macs from 9k+ to 0) |
| Constant-fold weight-only structure (validator does this for us) | memory+params | none | already applied, build for it |
| Pre-fuse Conv+BN, drop inference-time Dropout, eliminate no-op Identity/Reshape | memory+params | none — pure local rewrites | meaningful for any non-trivial graph |
| Pack initializers as `uint4` instead of `float32` | params (8×) | opset≥21 required — **risk: validator may reject** | ~80% of params budget |
| Pack initializers as `uint8` instead of `float32` | params (4×) | opset 13+, almost certainly safe | ~75% of params budget |
| Pack initializers as `float16`/`bfloat16` instead of `float32` | params (2×) | opset 11+ for fp16, fp16 well-supported | ~50% of params budget |
| `ConstantOfShape` with value=0/1 instead of literal Constant | params, file bytes | opset 9+, zero risk | tiny — saves a kernel-sized initializer per replaced Constant |
| Share one initializer across N nodes (alias by name) | params | none | up to N× on params for the shared tensor |
| Strip `doc_string`, `metadata_props`, training fields, unused initializers | file bytes | none | small, but the cap is binary so every byte matters |
| Sparse_initializer (COO) | params (only at sparsity > 67%) | overhead from int64 indices | only worth it for very sparse / small NNZ |
| `external_data` files | file bytes — but probably not score | **almost certainly DQ** — see below | do not use |

*(zero-MAC vocabulary exploit decode — MAC-era content archived to `_archive/scoring_internals_pre_phase3.md`; MACs left scoring 2026-05-04.)*

## Sub-byte initializer storage (params reduction)

> **⚠️ STALE — section describes the RETIRED onnx-tool counter (annotated 2026-06-07, task180 probe).** The canonical scorer (`neurogolf_utils.calculate_params`, core.md) counts **ELEMENTS, dtype-blind** — confirmed by the task055 Range-vs-init probe (`zero_param_generators.md` correction). Sub-byte/dtype weight packing buys file bytes only, NEVER params. Do not spend build time here (task180's sub-210 hunt burned 30 min re-disproving it). Original text kept below for the file-bytes question only.

The `params` term is computed by onnx-tool as the sum of *unique* initializer tensor sizes, in **bytes of the declared dtype** times element count. This is dtype-aware. Therefore:

- `float32` → 4 bytes/elem
- `float16`, `bfloat16` → 2 bytes/elem (½× params)
- `int8`, `uint8`, `bool` → 1 byte/elem (¼× params)
- `int4`, `uint4` → ½ byte/elem (⅛× params), packed 2 per byte in `raw_data`
- `int2`, `uint2` (opset 24) → ¼ byte/elem — extreme but newest; risk

The full bytes of a tensor in raw_data form: `ceil(numel × bits / 8)`. So a 30×30 float32 weight = 3600 bytes; same shape uint4 = 450 bytes — same `params` cost, same MAC cost (Conv MACs are dtype-blind), but lower file-byte footprint and lower memory if the activation also goes through Cast.

**Procedure:** keep the *activation* path in float32 (so Conv/Add still work), but store *weights* as the smallest dtype that holds the values you need, then `Cast` to float32 right before use. The `Cast` node has zero MACs and zero params. Activations through it stay float32.

**Caveats:**

1. **Opset gating.** INT4/UINT4 require opset 21. Bfloat16 requires opset 13+. Float16 has been supported since early opsets. The starter pins opset 10. We need to test whether the validator accepts opset 21 — A.4 follow-up. If not, uint8 is the floor we can rely on (4× params reduction, 100% safe).
2. **Negative INT4 protobuf bug** ([onnx/onnx#6160](https://github.com/onnx/onnx/issues/6160)): negative INT4 values stored via `int32_data` rather than `raw_data` blow up byte size 5×. Always use UINT4 with offset, or write INT4 to `raw_data` directly via `make_tensor` with the bytes pre-packed.
3. **Cast cost.** Cast itself is zero MACs but the resulting *float32* activation tensor is what onnx-tool's normal `memory` tracks. Validator's custom memory metric uses `value_info` byte sizes (which reflect declared dtype) — so a uint4 tensor that is then cast to float32 produces a float32 intermediate. The full chain still pays float32 memory for the activation. The win is purely on `params`.

## Identity-aliasing dedup / shared-name initializers

ONNX initializers are referenced by name. If two Conv nodes both reference an initializer named `"W_shared"`, the bytes count once in `params`. This works for:

- **Tied weights** across layers — repeat the same kernel in stacked Convs (B.3).
- **Color permutation reused** across channels — one 10-entry table, multiple Gather ops.
- **Diagonal kernels** — one 1×1 identity initializer reused as residual injection.

Within one node, repeated weight references are rarer but the validator's loader is just a protobuf parser; nothing prevents the same tensor name from appearing in `inputs` multiple times.

## ConstantOfShape as free zeros/ones

`ConstantOfShape(shape, value=v)` allocates a tensor of `shape` filled with `v`. The `shape` input is typically an initializer (so it folds), and the `value` attribute is one scalar. Compared to:

- `Constant(value=zeros((1,10,30,30)))` → an initializer of 9000 floats (36 KB params).
- `ConstantOfShape(shape=[1,10,30,30], value=0.0)` → an initializer of 4 ints (32 bytes) for the shape, plus a scalar attribute. **Constant-folding usually runs anyway and replaces this with a literal Constant**, so the win can evaporate. Exception: if onnx-tool's folding skips ConstantOfShape (some versions did), the savings stand.

**Use case:** generating a 30×30 mask of 1s as an attention/select template, without paying for a dense initializer. Worth measuring locally with onnx-tool 1.0.1.

## Sparse initializers (`sparse_initializer`)

ONNX format: COO with values tensor + 2D indices tensor (int64). Per nonzero element, cost ≈ `(rank × 8) + value_bytes`. For our 30×30 grids (rank 4), each NNZ = 32 + 4 = 36 bytes (vs 4 for dense fp32). Crossover: dense `n` floats = `4n` bytes; sparse with NNZ = k = `36k` bytes. Worth it only when `k < n/9`, i.e. **>89% sparsity**.

For a 1×1 identity Conv (10 input channels × 10 output channels = 100 weights, identity diagonal has 10 nonzeros = 90% sparse): dense = 400B, sparse = 360B. Marginal. For a 3×3 channel-mixing identity-projector (900 weights, diagonal-only = 30 nonzeros = 96.7% sparse): dense = 3600B, sparse = 1080B. Worth it.

**Caveat:** whether onnx-tool's `params` counter for sparse initializers reports the dense element count or the storage byte count is **unverified**. If it reports dense element count, sparse buys file bytes only, not params score. **Test this locally before relying on it.**

## `external_data` — almost certainly disqualifying

External data stores tensor bytes in *separate files* referenced by relative path. Submission format is *one ONNX file per task*. The competition's file-byte cap of 1,509,949 is an exclusive `>` check on the submitted file. If validator accepts a model whose `.onnx` file is small but whose external `.data` files contain MB of weights:

- The submission upload is single-file; external data files cannot be uploaded.
- Even if multi-file uploads worked, the validator's profile/score path (`onnx_tool.Model(path)`) calls `onnx.load(path)` which auto-loads external data from sibling paths — but the cap is per file.

**Do not use external_data.** Confirmed risk; no upside.

## Packed-bool / bit-fields

ONNX has `bool` (1 byte/elem in raw_data — *not* 1 bit). A packed boolean mask via `uint8` with 8 bits/byte requires manual unpack via Cast+BitwiseAnd+Greater chains, which themselves cost MACs. Net loss for masks under ~1000 elements. Skip.

## Quantized op variants

`QLinearConv`, `QLinearMatMul`, `MatMulInteger`, `ConvInteger` use int8 inputs. Per onnx-tool's "1 MAC = float(a)*float(b)+float(c)" definition, they bill the **same MAC count as their float counterparts**. So:

- QLinearConv on int8 weights: same MACs as fp32 Conv, **half the params** (int8 weight bytes), but adds extra inputs (scale, zero_point) which are tiny but nonzero.
- MatMulInteger: needs int8 inputs, so we'd Cast in/out — adds activation memory pressure.

Only worthwhile if we can quantize statically (zero_point=0, fixed scale=1 → bias and scale tensors can be folded out). Symmetric quantization with scale=1 zero_point=0 is essentially free type-narrowing. Saves params, costs nothing in MACs.

## File-byte slimming (separate from params/memory)

The 1,509,949 byte cap is on the serialized file. `params` byte cost ≠ file byte cost; protobuf has overhead. Things to strip from the proto before submission:

- `doc_string` on every node, tensor, and the model
- `metadata_props`
- `training_info` (we never train inside)
- Unused initializers (anything not consumed by a node)
- `value_info` for tensors that aren't used (shape-inference rebuilds these)
- Long node `name`s — single-char names cost less. Set `node.name = ""` or 1 char.
- Long tensor names referenced only internally

Single-byte node names plus `doc_string` stripping can shave a few hundred bytes on small models. With typical models being ~500–5000 bytes, this matters when racing toward the cap from the upper edge.

## What the validator does NOT count

- **Initializer bytes do not count toward `memory`.** Only activations + IO. So big weight tensors don't push the memory term — only MACs and params do.
- **Folded subgraphs.** Anything reduced to an initializer by `constant_folding` doesn't show up as MACs/memory at all; only as `params` (if it remains as an initializer in the folded graph).
- **`If` and `Loop` subgraph bodies.** Banned outright — not a counting question.

## Exploit checklist (apply in this order)

1. Shape-only solution if any pattern allows it. (~25 pts.)
2. Apply Conv+BN fusion, Identity/Dropout removal, dead-tensor pruning before submission. (Free; cuts memory+params on any non-trivial graph.)
3. Quantize weights to lowest opset-compatible dtype (uint8 if opset 10–20, uint4 if opset≥21 is accepted). Keep activations float32. (Cuts params 4–8×.)
4. Reuse one shared initializer across multiple nodes wherever symmetry allows. (Cuts params multiplicatively.)
5. Strip proto metadata, shorten names. (Saves file bytes only.)
6. Avoid: `external_data`, `Function`, `If`, `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Compress`, custom domains.

## Discussion-thread bugs and 2026-04-28 host metric update

Sourced from the Kaggle discussion thread "Issues in onnx-tool" by `shinh` (posted 14 days before 2026-05-01) plus its branches. Different thread from the `/discussion/692827` "scoring discrepancies" thread that's still open in our follow-ups below.

### 2026-04-28 host metric update — scoring now does constant folding

Michael D. Moffitt (Competition Host, "3 days ago" relative to a comment dated ~2026-04-28): *"This issue should now be fixed (see also: our metric update from today). In particular, our scoring engine will perform constant folding on all networks before scoring."*

Confirmed locally: `data/competitions/neurogolf-2026/neurogolf_utils/neurogolf_utils.py:209` already loads via `onnx_tool.loadmodel(m, {'verbose': False, 'constant_folding': True})`. So our 0.01-pt local-vs-Kaggle tracking across banana → fig was *not* coincidence — we've been folding the way the server now does. Submissions affected by the change were ones that relied on **un-folded** Constant/ConstantOfShape nodes whose accounting differed pre-folding vs post-folding. Wild evidence: a publicly-shared notebook is named `neurogolf-4808-21-post-apr-28-update.ipynb`, vs an earlier notebook `neurogolf-2026-4-27-lb-5126.ipynb` — bracketing roughly a 6% downward shift on at least one solution.

**Implications for us:** none today (already aligned). Future patterns that use `Constant` literals or `ConstantOfShape` with data-dependent shapes must still be cost-checked under `constant_folding=True` (which we do by default), and any "ConstantOfShape is cheaper than Constant" trick from the section above should be re-measured post-folding before relying on it.

### Bugs in onnx-tool < 1.0.1 — mostly moot for us

| Bug | Effect | Status |
|---|---|---|
| `Expand` mis-sized post-op profiling | `Sqrt(Expand(input, [30]))` scored `(720, 248, 1)` vs equivalent `Sqrt(input)` `(216000, 36000, 0)` | Fixed in newer onnx-tool by author Yu Luo — verify whether the fix is in our pinned 1.0.1 if we ever use Expand |
| Negative-step `Slice` off-by-one | `Slice(starts=29, ends=-100, steps=-1)` on length-30 axis inferred shape 29 instead of 30; downstream Add reported 8,700 MACs vs correct 9,000 | Fixed shortly after post; 1.0.1 fix unverified |
| `Less()` shape inference wrong | `[1,10,30,30]` runtime → `[1,1,30,30]` profiled in older versions | Confirmed fixed in 1.0.1 by `robga` (11th in competition) |
| Constant nodes not counted toward `memory` | Standalone `Constant` nodes vanished from `params/memory` | Fixed by enabling `constant_folding=True` (we already do) |

We are on 1.0.1 with constant folding on; none of these affect our current submissions.

### Bugs/quirks that may still affect us

- **`ConstantOfShape` with data-derived shape collapses to 0 under `shape_infer(None)`.** `kq5y` showed: `ConstantOfShape(shape = Cast(input[:,:,0:1,0:1]) * 30)` runtime-produces `[1,10,30,30]` but the scorer's `shape_infer(None)` (zero-filled dummy input) infers `[1,10,0,0]` → downstream Add reported as 0 MACs vs actual 9,000. Yu Luo confirmed this is by-design on the onnx-tool side ("you can't use the default input tensor as all zeros") — the gap is on the scorer side, which passes `None`. Constant folding can't help when the shape depends genuinely on input data. **Status: a real and exploitable accounting gap that survives the 2026-04-28 update.** Yiheng Wang (3rd in competition) explicitly switched to 1.0.1 *to avoid* exploiting bugs. **Owner stance to adopt: do not exploit; ethical play only.**
- **`EinSum` without ellipsis does not broadcast** (Yu Luo). `EinSum(shape=(A,B,C,D), shape=(1,1,1,1), "ijkl,ijkl->ijkl")` produces `(1,1,1,1)` per onnx-tool but runtime broadcasts to `(A,B,C,D)`. Per ONNX spec only ellipsis-form equations broadcast, so this isn't a bug per se. Don't rely on it.
- **`Pow(shape=(A,1), shape=(1,A))` may infer as `(A,1)` instead of `(A,A)`** (`shinh`'s hypothesis; not confirmed in thread). Worth a local PoC if we ever use Pow with mismatched broadcast shapes.

### 2026-05-02 update — additional exploits surfaced from competitor notebooks

Surveying the 14 publicly-shared notebooks at `primary_sources/` (catalog in `primary_sources/00_INDEX.md`) surfaced these items:

- **`Compress-with-fake-zero-batch` PoC confirms the mechanism (2026-05-02).** Built a minimal ONNX model: `input [1,10,30,30] → Compress(input, cond=[False]*30, axis=2) → Identity → output`. Result with both `shape_infer(real)` and `shape_infer(None)`: profiler reports `Compress` output shape as `[1,10,0,30]` → 0 MACs / 30 memory bytes / 30 params. **Mechanism confirmed:** any all-False (or value-derived-to-False under zero-input) condition initializer collapses Compress's downstream tensor's shape, so any subsequent op profiles 0 MACs.

  **Bypass-mechanism investigation (2026-05-02 followup): three hypotheses tested, all REJECTED.** `Compress` is on `_EXCLUDED_OP_TYPES` in `neurogolf_utils.py:51` (verified). The check at `:218` walks `g.nodemap.keys()` after `graph_reorder_nodes()` + `shape_infer(None)` + `profile()`, with `constant_folding=True` enabled in `onnx_tool.loadmodel(...)`. Tested:
  1. **Bare `Compress(input, cond=initializer_all_False)` → Identity → output.** `score_network` returns `(None, None, None)` with `Error: Op type Compress is not permitted.` Constant folding does not remove a Compress whose only constant input is the condition.
  2. **Compress with derived condition (`Less(ones, ones)` → always-False bool tensor; both inputs initializers).** Same rejection. Constant folding does not pre-evaluate the `Less → Compress` chain to elide Compress.
  3. **Dead-code Compress (output bypasses it; only Identity reaches output).** Same rejection. The ban check walks ALL nodes in `g.nodemap`, not just those reachable from outputs. Pruning unreachable code is not part of `graph_reorder_nodes()`.

  Tangential finding: unknown op types (e.g., `op_type="UnknownOp"`) escape the banned-op check entirely (the list is exact-name-match), but `shape_infer` fails immediately on them with `NotImplementedError`, so they're not usable as a Compress wrapper.

  **What remains untested:** (a) running against a pre-2026-04-21 onnx-tool version (the historical scorer that may have had a banned-op-check bug); (b) registering a custom shape-inference handler for an unknown op_type that internally returns a zero-batch tensor; (c) Compress embedded in a `Function` body (but `Function` itself is also banned, so not viable). The exploit may simply be historical (worked against an earlier scorer) — the "Bug-Free Blend [LB 5126]" notebook explicitly avoids it for safety, suggesting it was active at some point but may now be patched. **Owner stance: do not exploit.** PoC scripts: 2026-05-02 followup session log.

- **"Positive-cost wrappers" — fully decoded as 3 layered exploits.** Source: `hack_submission.py` inside `primary_sources/neurogolf-acking-multiple-tasks-part-3.ipynb` (Russian-commented; literally named "хакер ONNX" / "ONNX hacker"). The script takes any submission.zip and applies these three transformations to every task ONNX outside a known-problematic skip list:

  - **Hack 1 — Static→symbolic dim conversion on `output` and `value_info` (NOT on `input`).** Replaces every `dim_value=N` with `dim_param=f"d{N}"`, clearing the value. The validator's `shape_infer(None)` then evaluates symbolic dims as 0, collapsing memory/macs accounting downstream. This is the "dynamic-shape under-counting" exploit named in the bug-free notebook.
  - **Hack 2 — Initializer → Constant node conversion.** For every initializer that isn't a graph output, replace it with a `Constant` node that has the same value. Pre-fold-fix scorers undercounted Constant nodes' contribution; the 2026-04-28 host metric update added constant_folding which should now re-fold these into proper initializers, but the script's targets predate that fix and may still benefit on stale scorers.
  - **Hack 3 — Output wrapped via `Shape → ConstantOfShape(value=0) → Add(internal, cos)`.** For each graph output, redirect the producer to an internal name, then add three ops: `Shape(internal) → shape_t`, `ConstantOfShape(shape_t, value=0) → cos_t`, `Add(internal, cos_t) → output_name`. Functionally identity (adding zeros). Cost-wise: `cos_t`'s shape is data-derived from `Shape(internal)`, so kq5y's exploit fires — `shape_infer(None)` collapses cos_t's shape to 0, the downstream `Add` charges based on the collapsed shape (0 MACs instead of 9k), and the output looks tiny in the profile. **This is the cleanest "positive-cost wrapper" — 3 added ops that strictly increase runtime cost yet decrease the profiled cost.**

  All three are accounting gaps, not runtime bugs. The model still runs correctly — the validator's `verify_subset` passes — but `score_network` reports a much lower number than the true cost. **Owner stance reaffirmed: ethical play only — do not implement these.** Documenting them here so we (a) understand competitor scoring discrepancies, (b) can detect if our own submissions accidentally trigger any (we don't currently use `Constant` nodes, dynamic dim_param, or `ConstantOfShape` — we're clean by construction), and (c) can avoid being misled by competitor cost-tables.
- **I/O tensor-naming repair.** The "Safe Blend" notebook (`primary_sources/neurogolf-logic-driven-ensembe.ipynb`) calls out that some upstream candidate submissions have non-standard `input`/`output` tensor names, and the safe blend repairs them. **Action:** verify our own `build_submission.py` always emits `"input"` and `"output"` as the tensor names. Per `neurogolf_utils.py` validator: required exactly. A quick `python -c "import onnx; m=onnx.load('submissions/task001.onnx'); print(m.graph.input[0].name, m.graph.output[0].name)"` settles it. If we ever start with someone else's ONNX, this is the first thing to fix.

- **`Compress-with-fake-zero-batch` exploit.** Named explicitly in the "Bug-Free Blend [LB 5126]" notebook (`primary_sources/neurogolf-2026-4-27-lb-5126.ipynb`) as a profiler quirk that the bug-free version deliberately avoids. `Compress` is on the disallowed-op list per `neurogolf_utils.py`, so the trick must be a way to use it that the validator misses — most likely making the `Compress` op disappear under shape inference by setting a zero-batch dim or similar. **Not yet PoC'd locally**; ~30 min reverse-engineering would confirm the mechanism and let us catalogue + ignore (per ethical-play stance from §"Bugs/quirks that may still affect us" above).
- **April 21 metric update** — predates the April 28 one we already had. Same notebook references "the host's announced rescoring fixes (April 21st update)" without further detail. Worth searching the discussion forum for the announcement to know what changed (likely the original onnx-tool bug-fix patches that kicked off the broader thread).
- **Silent-fail tasks confirmed in the wild.** The "Blended 401-Task" notebook (`primary_sources/neurogolf-2026-blended-401-tasks-4-27.ipynb`) reports task363 passed all visible-pair benchmarks locally but scored 0 on Kaggle's hidden test until replaced with a custom 265/265 version for +10.14 LB. Not strictly a cost-metric exploit, but relevant to our cost-prediction calibration: a 0.01 local-vs-Kaggle precision proves the **scoring** is calibrated; it does NOT prove the **correctness** is guaranteed against the private set. Captured in `03_PROJECT.md` Open bugs.

### Cost ladder cheat sheet (from `primary_sources/neurogolf-2026-rule-based-onnx-solver.ipynb`, Imaad Mahmood)

The Master Ensemble Solver v7 notebook (est. LB ~6,285) opens with this cleanly tabulated cheat sheet, useful as a target:

| Architecture | Cost | Score |
|---|---|---|
| Static Identity (cost=0) | 0 | **25.0** |
| Channel Gather | ~45–54 | **21.0–21.2** |
| Slice/Pad chains | ~200–500 | **19–21** |
| Spatial Gather | ~4,563 | **16.6** |
| 1×1 Conv | ~90,500 | **13.6** |
| 3×3 CNN | ~814,500 | **11.4** |

We currently get **22.7 pts on color-perm Gather** (matches "Channel Gather"), **19.09–19.88 pts on Slice+Pad** (matches "Slice/Pad chains"), and **13.71 pts on marker-corners** (multi-op, sits between Spatial Gather and 1×1 Conv). The notebook also explicitly notes: **"Lower local cost ≠ better LB score. Always bisect on LB before accepting a swap."** Reaffirms our silent-fail-task lesson.

### Competitive landscape — top public scores

Sidebar links in the same Kaggle session show publicly-shared open solutions at:
- `neurogolf-6645-39-public-score-open-solution.ipynb` — 6,645.39 pts
- `neurogolf-6233-36-public-score-open-solution.ipynb` — 6,233.36 pts
- `neurogolf-2026-blended-401-tasks-4-27.ipynb` — pre-update blend
- `neurogolf-2026-rule-based-onnx-solver.ipynb` — rule-based reference
- `neurogolf-2026-4-27-lb-5126.ipynb` — 5,126 pts (pre-2026-04-28)
- `neurogolf-4808-21-post-apr-28-update.ipynb` — 4,808.21 pts (post-2026-04-28)
- `neurogolf-acking-multiple-tasks-part-3.ipynb` — multi-task technique series
- `neurogolf-all-task-logic-complexity-map.ipynb` — task-complexity map (could inform our mining)

Our local-pre-submission is 229.04 (12/400). The top shared solution is ~29× higher. **Action item for a low-effort session: read 1-2 of the public open-solution notebooks** (especially the 6,645 one and the all-task-logic-complexity-map) to extract pattern ideas we haven't tried.

## Open follow-ups (push to A.4)

- Does the validator accept opset 21 (and therefore int4/uint4)?
- ~~Does `onnx-tool==1.0.1` count sparse_initializer params by NNZ or by dense element count?~~ **RESOLVED 2026-05-30 (task337 sub-floor probe).** The live scorer is NOT onnx-tool — it's `neurogolf_utils.calculate_params`, which counts sparse params as `math.prod(sparse_init.values.dims)` = **NNZ** (indices uncounted). BUT this is moot for our op vocabulary: a sparse initializer is **rejected** as a `Gather` index (and `Conv`/`QLinearConv` weight) at three independent scorer gates — `checker.check_model(full_check=True)` (Gather forbids sparse `Tind`), `shape_inference(strict_mode=True)` (rank mismatch), and `sanitize_model` (renames the node input but not the sparse tensor → dangling ref). `score_network_v2 → None`. **Sparse buys 0 usable params.** Also note: the "pack initializers as uint8 → 4× params" lever in the table above is **stale** — `calculate_params` counts ELEMENTS, dtype-agnostic, so dtype packing saves nothing on params (only `memory`, via cast intermediates). Full proof: `The_400_Tasks/Task_by_Task/task337.md`.
- Forum `/discussion/692827` "scoring discrepancies" loophole — patched in 1.0.1? (Different thread from "Issues in onnx-tool" above; that one doesn't reference 692827.)
- Does ConstantOfShape survive constant_folding in 1.0.1, or get materialized to a Constant initializer first? (Partially addressed above — folding probably materializes when shape is a constant; collapses to 0 when shape is data-derived.)

---
<a id="a3_ban_workarounds"></a>

# A.3 — Ban workarounds

The validator rejects on three classes of constructs: banned ops (`Loop, Scan, NonZero, Unique, Script, Function, Compress`), subgraph-bearing nodes (`If`, `Loop` body attributes, anything with `GRAPH`/`GRAPHS` attr type), and non-stdlib opset domains. Plus the per-file 1,509,949-byte cap (exclusive `>`). For each, the question is: what can we use instead, and at what cost?

## Classification of bans

| Ban | Why it's banned | Why we'd want it | What we use instead |
|---|---|---|---|
| `Loop` | `calculate_memory()` returns `None` on subgraph attrs | iterative algorithms (flood-fill, NCA, message-passing) | **unrolled stacked ops** (B.6) |
| `Scan` | same — body subgraph | sequential reduction with state | unrolled |
| `If` | body subgraph | data-dependent branching | `Where` (mux), `Mul` by mask, branch-free arithmetic |
| `Function` | local custom ops | reuse a subgraph multiple times | inline the subgraph each time, share initializers by name |
| `NonZero` | dynamic-output-shape | find positions of nonzero entries | TopK on a flat reshape, or ArgMax+threshold |
| `Unique` | dynamic-output-shape | dedup values | precomputed lookup; static input cardinality (max 12 colors) |
| `Compress` | dynamic-output-shape | mask-select | `Where(mask, value, fill)` — same shape, no compression |
| `Script` | embedded code attr (rare) | not useful to us | n/a |
| Custom domain | only `""`/`"ai.onnx"` allowed | imported third-party ops | rewrite using stdlib |
| Subgraph attrs | `GRAPH`/`GRAPHS` types | implicit because all subgraph-bearing ops are banned | n/a |

The common thread: all bans target **dynamic shapes** or **subgraph control flow**. Static, shape-determined dataflow is fine.

## Loop / Scan → unrolled graphs

Anything iterative becomes a fixed-depth stack of identical ops. For our 30×30 grids, the maximum interesting iteration count is bounded:

- A flood-fill across a 30×30 grid converges in ≤ 30 + 30 = 60 dilations (worst-case Manhattan diameter).
- A neural cellular automaton converges in similar bounds.
- Power-iteration for connectivity (binary OR-dilation) converges in `ceil(log2(diameter))` ≈ 6 iterations.

**Cost analysis for unrolling:**
- Each unrolled step that touches the activation costs `output_elems = 9000` MACs minimum (one binary or per-element op). 60 steps = 540k MACs → score `25 - log(540000) ≈ 11.8` pts.
- 6 power-iterations of dilation → 54k MACs → score ≈ 14.1 pts.

**Weight sharing in the unroll:** if every step uses the same kernel, the initializer is referenced N times by name and counts once in `params`. Only `macs` and intermediate `memory` scale with depth.

**Memory pressure:** every intermediate activation is held in `value_info` and so is counted by validator's custom `memory` metric. A 60-deep unroll writes 60 intermediate `[1,10,30,30]` tensors = 60 × 36000 = 2,160,000 bytes of memory before any folding. Big. Use `Identity` + reuse the *same* tensor name only if onnx-tool's value_info dedup permits — risky; safer to keep depth low.

**Direct path to short unrolls:** topological log2-style schedules. Power-iteration: dilate, then dilate the dilation, then dilate the doubled-radius dilation. After 6 iterations the dilation radius is 2^6 = 64 ≥ 30 — fully covers the grid. **Six unrolled depthwise-dilate Convs is feasible.** Per-conv cost = `Cout·1·9·900 = 8100` MACs (depthwise 3×3) → 6 × 8100 = 48,600 MACs → score ≈ 14.2 pts/task.

(See B.6 for the full iterative-without-Loop catalog.)

## If → branch-free arithmetic

`If` is implicitly banned because its bodies are subgraph attributes. Replacements:

- **Mask-multiply.** `output = mask * branch_a + (1 - mask) * branch_b`. Two Muls + one Add + one Sub = 4 element-wise ops = 36k MACs on `[1,10,30,30]`. Mask comes from `Greater`/`Equal` + `Cast`-to-float32 — also free of MACs that aren't already paid.
- **`Where`.** `Where(mask, a, b)` is a single node, billed at 1 MAC/output_elem = 9k MACs. **Cheaper than mask-multiply by 4×.** Always prefer Where.
- **`Min`/`Max`.** For "if a > b then a else b" → `Max(a, b)` directly, 1 MAC/elem.
- **Both branches always computed.** No way around this without `If`. Cost = max(branch_a_cost, branch_b_cost) actually equals branch_a_cost + branch_b_cost. Plan: keep both branches as cheap as possible, ideally with shared substructure.

**Conditional branching via input-content discrimination:**
We don't actually need conditionals at *runtime* — the project's `build_submission.py` does pattern detection at *build time* and emits one ONNX file per task. Each ONNX is unconditional. Conditioning on input content within the ONNX is rarely needed.

## Function → inline + named initializer sharing

`Function` is what `onnx-script` and many exporters emit for reusable subgraphs. Banned. Replacement:

- **Inline every use.** Copy the same node sequence with the same initializer names. Initializer bytes count once thanks to name dedup (A.2).
- **No `local`/custom domain.** The `domain` field on every node must be `""` or `"ai.onnx"`. Watch for `com.microsoft.*` (onnxruntime contrib ops like `MatMulNBits`, `QuickGelu`, `BiasGelu`) — these are not in stdlib and will be rejected.

**Exporter risk:** `torch.onnx.export(..., dynamo=True)` (the new dynamo-based exporter) tends to inline functions; the older TorchScript-based exporter sometimes emits Functions for `nn.Module` calls. Strip them in post-export surgery if any appear, or use the older tracing exporter (see PyTorch hygiene below).

## NonZero / Compress / Unique workarounds

These are output-shape-dynamic ops. Our static-shape requirement means we always produce fixed-size outputs anyway, so we don't need to literally dedup or filter — we mask.

| Pattern | Banned op | Static workaround |
|---|---|---|
| "find positions where mask==1" | NonZero | Don't extract positions. Apply the mask in-place: `Where(mask, value, fill)` keeps full grid shape. |
| "select top-k positions" | NonZero + Gather | `TopK(values, k)` returns sorted top-k values + indices in static shape `[k]`. Allowed. |
| "argmax over an axis" | (NonZero of mask) | `ArgMax` directly. Allowed. **Verify with onnx-tool that it bills as expected** — A.4 follow-up. |
| "dedup colors in a grid" | Unique | We know there are at most 12 colors. Compute `OneHot` over 12 channels via Gather of an identity 12×12 LUT, then `ReduceSum` over spatial → presence vector of length 12. Static. |
| "boolean filter / count nonzero" | Compress + ReduceSum | `Cast(mask, float32)` + `ReduceSum`. Static count, single op. |
| "compress along axis" | Compress | Replace the would-be-compressed positions with a sentinel (color 10) using `Where`; downstream consumers know to ignore color 10. |
| "find unique rows" | Unique | Same trick as colors: there are at most 30 distinct rows, each a `[1,10,30]` vector. Often we can hash rows via a fixed projection: `MatMul(rows, random_vec)` gives a scalar per row, then `Equal` against expected hashes — static. |

The general principle: every "find-where" is replaceable with a "mark-where" because we always carry the full grid through.

## Per-file 1,509,949-byte cap

Per-file means: one `.onnx` file per task. **External_data is not viable** (A.2 conclusion). The exclusive `>` check means 1,509,949 bytes is the maximum allowed; the 1,509,950th byte breaks the bound.

What contributes to file bytes (in serialized protobuf, not just `params`):

1. Initializer raw_data — the dominant term for any non-shape-only solution.
2. Node definitions — op_type string, node name, attribute keys/values, input/output name lists.
3. Tensor names — every reference is the full string each time; long names cost.
4. Graph metadata — `doc_string`, `metadata_props`, `value_info` for non-IO tensors.
5. Opset imports — small fixed cost.
6. Model metadata — producer name/version, IR version. Tiny.

**The model proto vs total bytes question:** for a single-file submission with no external_data, model proto bytes == total bytes. So the 1.44 MiB cap is on the proto size of the .onnx file as serialized.

**Initializer overhead:** each `TensorProto` has a small fixed overhead (name, dims, data_type, raw_data length prefix). For a 1-byte raw_data tensor, the overhead might triple the apparent size. For 10kB raw_data, overhead is negligible.

**Recipe to fit the cap:**
- Pre-compress weights via dtype narrowing (uint4/uint8) — A.2.
- Strip doc_strings, metadata_props, value_info before save (`onnx.helper.set_model_props` clears; manual proto editing for value_info).
- Single-character node names: set `node.name = ""` everywhere. The validator doesn't care about names.
- Single-character tensor names where possible (avoid producer-style `onnx::Conv_42`).
- Fold any redundant Reshape/Identity (they cost a few hundred bytes each in proto).

## PyTorch export hygiene (avoiding accidental bans)

Tracing-based export (`torch.onnx.export(model, dummy, opset_version=K, dynamo=False)`):
- Python `if`/`for` are evaluated at trace time and disappear into the linear trace.
- Loops with data-dependent counts trace to **a fixed unroll for the dummy input's count** — exactly what we want.
- Will not emit `Loop`/`If`/`Function` for normal Python control flow.
- May still emit `If` for in-place operations on conditional tensors. Avoid `tensor[mask] = value` style.

Dynamo-based export (`dynamo=True`):
- Uses `torch.export` which understands `torch.cond`. If you use `torch.cond`, you'll get `If`. Don't use it.
- Inlines `nn.Module` calls more aggressively — fewer accidental Functions.
- Strict shape constraints — surfaces dynamic-shape issues at export time.

**Tracing wins for our use case.** The exporter's "limitation" of unrolling loops becomes a feature when Loop is banned anyway.

**Things that emit banned ops by accident:**
- `tensor[mask]` (boolean indexing) → `NonZero` + `Gather`. Avoid; use `Where` with a fill instead.
- `torch.unique(...)` → `Unique`. Avoid; static dedup via OneHot+ReduceMax.
- `torch.nonzero(...)` → `NonZero`. Avoid; use `TopK`/`ArgMax` if you need indices.
- `tensor.view(-1)` followed by index extraction → may emit `Gather` of dynamic indices, OK but ensure static.
- `torch.compile`-style guards, dynamic shape rebinding → may emit `If`. Stay in tracing mode.
- `torch.jit.script(model)` followed by export → emits `Loop`/`If` faithfully. **Never script.**

**Post-export surgery checklist:**
1. `onnx.checker.check_model(m)` — validates structure, not our specific bans.
2. Walk `m.graph.node` and assert `n.op_type not in {"Loop","Scan","If","NonZero","Unique","Function","Script","Compress"}`.
3. Assert `n.domain in {"", "ai.onnx"}`.
4. Assert no attribute has type `GRAPH` or `GRAPHS` (`onnx.AttributeProto.GRAPH = 5`, `.GRAPHS = 9`).
5. Assert all `value_info` and IO `TypeProto.tensor_type.shape.dim` entries have `dim_value` set (no `dim_param`).
6. Run `onnx_tool.Model(m, {"constant_folding": True})`; if `calculate_memory()` returns None, the model would fail the validator.

## Construction route: skip PyTorch entirely

For cases where we hand-design the graph (color permutation, transpose, etc.), build directly with `onnx.helper.make_node` / `make_tensor` / `make_graph`. No exporter touches the model — zero risk of banned ops creeping in. The project already does this in `_PATTERNS`.

For learned weights, use the smallest export path: train in PyTorch, capture the *weight tensors only*, then build the ONNX graph manually with those weights as initializers. This way we control every node and every attribute.

## Summary of replacements at a glance

```
Loop / Scan      → unrolled stacked ops (B.6)
If               → Where(mask, a, b)
Function         → inline + named-initializer sharing
NonZero          → TopK or ArgMax
Compress         → Where with sentinel/fill
Unique           → OneHot + ReduceMax (static cardinality)
custom domain    → rewrite using stdlib only
external_data    → don't use; pack uint4/uint8 inside the .onnx
```

Every workaround has a fixed-shape, banned-op-free equivalent for our task shape. The cost ranges from "free" (`If` → `Where` is even cheaper than mask-multiply) to "log-iteration unroll" (Loop → ~6 stacked ops for diameter-30 problems).

---
<a id="a4_grader_replication"></a>

# A.4 — Grader replication

The goal: build a local pipeline that gives the same `(macs, memory, params, points)` numbers the Kaggle grader does, so we can iterate without burning daily submissions. We also need to nail down the ARC-GEN-100K structure, the private-benchmark assumptions, the submission format, and the open `/discussion/692827` loophole.

## Local replication recipe

The validator is `neurogolf_utils.py` (Apache-2.0, Google) with pinned deps `numpy==2.4.4 onnx==1.21.0 onnxruntime==1.24.4 onnx-tool==1.0.1`. The two functions that matter:

- `verify_network(model_path, task_path)` → bool. Loads the ONNX, runs onnxruntime over each `(input, expected_output)` pair across ARC-AGI v1 train+test and the bundled `arc-gen` subset, applies the `(result > 0.0).astype(float)` thresholding, and `np.array_equal`s against the expected.
- `calculate_memory(model_path)` → custom int. Walks `value_info` byte sizes excluding `"input"`/`"output"`, returns `None` if any node attribute is `GRAPH`/`GRAPHS`-typed.

Plus the standard onnx-tool path: `Model(...).graph.shape_infer(...)`, then `.profile()`, then sum `macs` and unique-initializer `params`. Constant folding is on.

### Step-by-step local setup

1. **Pin the deps exactly.** A fresh venv with `numpy==2.4.4 onnx==1.21.0 onnxruntime==1.24.4 onnx-tool==1.0.1`. Different onnx-tool versions count differently — only `1.0.1` matches.
2. **Pull the validator + starter:**
   ```
   kaggle competitions download -c neurogolf-2026 -f neurogolf_utils/neurogolf_utils.py
   kaggle kernels pull mmoffitt/the-2026-neurogolf-championship
   ```
3. **Apply the headless-import workaround.** The validator imports `IPython.display` and `matplotlib.pyplot` at module top. Stub before import:
   ```python
   import sys, types
   sys.modules.setdefault("IPython", types.ModuleType("IPython"))
   sys.modules["IPython.display"] = types.ModuleType("IPython.display")
   sys.modules["IPython"].display = sys.modules["IPython.display"]
   sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
   sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
   import neurogolf_utils  # now safe
   ```
4. **Point the data root.** Validator has `_NEUROGOLF_DIR = "/kaggle/input/competitions/neurogolf-2026/"` hard-coded. Either:
   - mirror the JSONs locally under that exact path, or
   - monkey-patch `neurogolf_utils._NEUROGOLF_DIR = "/path/to/local/data/"` immediately after import (before any function in the module is called).
5. **Replicate the score path.** Don't re-implement `verify_network` and `calculate_memory` — call the imported ones. Re-implement only the points formula:
   ```python
   import math
   def points(macs, memory, params):
       cost = max(1.0, memory + params)   # MACs removed 2026-05-04
       return max(1.0, 25.0 - math.log(cost))
   ```
   Natural log. Floor for correct = 1.0. Incorrect = literal 0 (not the points formula at all).

### Sanity checks that must pass

- A submitted model that's literally `output = input` ("Identity" architecture) should score 25.000 on tasks 179 and 241 (per the COVERED probe), failing all others (incorrect = 0). Run this end-to-end before trusting your local pipeline.
- A 1×1 identity Conv should report cost 90,100 → 13.591 pts on whatever task it solves. Match the COVERED probe table to the byte.
- Submission #3's measured 95.39 ≈ 2·25 + 2·22.6974 anchors the color-perm pattern's expected pts/task at 22.6974. Reproduce locally.

## ARC-GEN-100K bundled subset

ARC-GEN is Moffitt 2025 (arXiv 2511.00162), open-source at `github.com/google/ARC-GEN`. CLI:
```
python3 arc_gen.py generate <task_num> <num_examples>
```
produces N synthetic `(input, output)` pairs that match the original task's transformation logic via mimetic per-task generators. The repo includes a `validate` command confirming all 400 generators reproduce ARC-AGI-1 exactly.

**For us, what matters is what's bundled in `taskNNN.json`.** Per the project docs: each `taskNNN.json` contains a `train` list, a `test` list, and an `arc-gen` list. The grader evaluates the model on **all three** for the public score; private leaderboard adds a held-out set.

**Implications:**
- A model that's correct on `train + test` but fails on `arc-gen` scores 0.
- ARC-GEN samples vary input/output grid sizes within the task's natural variation envelope (e.g. for a transpose task, ARC-GEN will throw 5×5, 7×11, 30×30 grids at us). **Any pattern that only works on 30×30 squares fails — confirmed by the project's hflip/vflip caveat.**
- The variation distribution is mimetic — it tries to match the original task's distribution rather than expand maximally. So a task whose original examples were all square stays mostly square.

**Local generation for robustness validation:**
```
git clone --recurse-submodules https://github.com/google/ARC-GEN.git
python3 arc_gen.py generate 32 1000  # 1000 samples for task 32
```
This is our anti-overfit stress test. If our network for task K solves all 1000 ARC-GEN samples, we have high confidence the bundled `arc-gen` subset (which is some sample of the same generator) also passes.

## Private benchmark — what to assume

The competition page calls it "private held-out" and says it's the basis of the private leaderboard, which decides final standings. Worst-case assumptions to design against:

1. **Same generator, different random seed.** Most likely scenario — Moffitt is the author of both the competition and ARC-GEN. The held-out is presumably a fresh sample from the same generators that produced the bundled `arc-gen` subset.
2. **Possibly with edge cases sampled more aggressively.** Min/max grid sizes, color edge cases, pad-row-only inputs, single-pixel inputs.
3. **Possibly with novel hand-authored variants** outside ARC-GEN's mimetic distribution. Less likely (would break the whole point of bundling ARC-GEN with the data) but possible.

**Design guidance:** any model whose correctness depends on properties of the *visible* examples (specific grid sizes, specific color counts, specific object counts) is risky. Models whose correctness derives from properties of the *transformation* (e.g. the transformation IS always a transpose) survive ARC-GEN-style variation by construction.

The robust operating procedure: if a task's transformation can be characterized as one of our `_PATTERNS` regardless of input shape, ship that solution. If we have to special-case grid size to make it correct, expect partial private-score failure.

## Submission format

- One ONNX file per task, named `taskNNN.onnx` (matching the task's id in the JSONs).
- Fixed input/output names: `"input"`, `"output"`. Hard-coded in the validator's `run_network` and `verify_network`.
- Fixed input shape: `[1, 10, 30, 30]`, dtype `float32`. Fixed by validator. The 10 channels are the ARC color palette (0–9 + channel 10 "no color" pad + channel 11 "too many colors"; channel 10 is the all-30×30 padding sentinel).
- All dims `dim_value` set, no `dim_param`.
- Per-file ≤ 1,509,949 bytes (`>` is an error). 
- Submission is a bundle (zipped/tar) of all `taskNNN.onnx` files. **Bundled or one-per-task isn't fully documented here** — based on the per-file cap and per-file scoring, expect the format to be a directory or zip with 400 files. Not an ensemble per task.
- Daily limit: 5 submissions; final selections: 2; team size: ≤5.

**Multiple ONNX files per task (ensembling):** assumed not permitted. Not directly confirmed, but the per-file cap, per-task scoring formula, and the grader's "load `taskNNN.onnx`" pattern all imply a 1:1 mapping.

**Static-shape per-task fixed shape:** could declare a smaller static shape for a task whose train/test/arc-gen all agree on grid dimensions (e.g. always 5×5). The 10×30×30 input is hardcoded by the validator; the *output* shape is not necessarily — but we still must produce `[1,10,30,30]` output for the validator's `np.array_equal` to compare against the padded ground truth. So output shape is fixed too.

This means the only free static-shape variable is internal activation shapes — and those are determined by our op choices, not declarations.

## The `/discussion/692827` loophole

The forum thread is titled **"Issues in onnx-tool"**. The validator code references it. We have not read it directly (Kaggle web is anti-bot blocked; authenticated `kaggle` CLI for forum threads works for `discussions list/get` if available — TODO).

What we know secondhand:
- A Codex-CLI issue (`openai/codex#19594`, Apr 2026) flagged a NeuroGolf user investigating "unusually high leaderboard scores" — specifically "7600.42" — that "likely come from an ONNX Runtime or profiler/scoring discrepancy, not from normal algorithmic optimization." The same issue lists task IDs `157, 177, 002, 337, 276, 309` as the ones with anomalous score deltas, plus the keyword `ConstantOfShape` as a candidate explanation.
- That score-delta range is consistent with what someone would see if a particular construct convinced onnx-tool's profiler to under-count MACs or memory, dropping the per-task cost into a near-zero range and pushing the per-task points up toward 25.

**Hypothesis space (NOT verified):**
- `ConstantOfShape` may not be properly accounted in onnx-tool 1.0.1 — a downstream consumer's MACs are computed against the resolved shape but the ConstantOfShape itself may dodge the params count. A way to make a tensor "appear" without paying for its bytes.
- Sparse_initializer might be counted by NNZ in the onnx-tool 1.0.1 release but resolved to dense at shape-inference time, letting big sparse weights ride for cheap.
- Subgraph attribute *types* are banned via `calculate_memory` returning None, but the onnx-tool MACs/params functions may walk into them silently — except that `If`/`Loop`/`Scan` themselves are op-banned. There may be other ops with optional graph attrs we haven't enumerated.
- An ORT execution-provider quirk where certain int4/int8/quantized ops produce different output bits than onnxruntime CPU EP, opening a "score correctly without actually computing the transform" angle — unlikely but mentioned.

**Action items:**
1. Pull the discussion thread via authenticated `kaggle` CLI (`kaggle competitions discussions get` or similar) — this is the highest-priority unread document.
2. Local repro the scenario by trying ConstantOfShape-heavy graphs and observing whether onnx-tool 1.0.1 systematically under-counts.
3. Confirm whether onnx-tool 1.0.1 patched the discrepancy — release notes between the version that exposed it and 1.0.1 would tell us. The repo is "Inactive" per Snyk so 1.0.1 is the last release.

## Scoring-precision details (from COVERED + spec re-read)

- `memory+params` is a sum, not max. So compounding small ops adds linearly, no headroom from a single dominant term.
- The `max(1.0, ...)` on cost prevents log of 0.
- The `max(1.0, 25.0 - log(cost))` floors correct submissions at 1.0 (enabling >1.0 rewards small-cost only).
- Incorrect submissions return literal 0 (not 0.001 like predecessor Code Golf 2025) — confirmed empirically. So a partially-correct heuristic that gets 95% of cases right is **worse** than a no-op that scores 1.0 on the correct tasks: the wrong cases drop us all the way to 0 there. **Any pattern detector must err on the side of skipping ambiguous tasks** (output the safe identity baseline when uncertain).
- The max cost that scores any positive points: `25 - log(cost) > 0` → `log(cost) < 25` → `cost < e^25 ≈ 7.2e10`. Anything below this is in the points window. Above this floors at 1.0. Effectively no upper bound issue for our model sizes.

## Open follow-ups for the next research wave

- **`/discussion/692827`** full text via authenticated kaggle CLI.
- **opset 21 acceptance** by the validator (gates uint4/int4 — see A.2). Test by submitting a single-task model with opset 21 and observe if validator rejects pre-score.
- **subbed onnx version**: `onnx==1.21.0` is more recent than the ones that introduced int4/uint4 (1.17). So the lib supports them; the question is whether the model's declared opset_import version triggers any rejection in the validator's checker.
- **ConstantOfShape MAC behavior** under onnx-tool 1.0.1 specifically — fold + count or skip.
- **how zipped/folder submissions are unpacked** — minor but worth confirming: directory layout, naming convention, missing-file behavior (is missing a task an automatic 0 or rejected upload?).

---
<a id="leaderboard-scoring-discrepancy"></a>

# Source: Leaderboard scoring discrepancy / evaluator update

## Public evidence

1. **`https://github.com/openai/codex/issues/19594`** — bug report filed April 25, 2026 by user `bianfu125` working on NeuroGolf. Verbatim relevant quotes:

   > "The 7600.42 score likely comes from an ONNX Runtime or profiler/scoring discrepancy, not from normal algorithmic optimization."

   > "investigating why a competition score jumped from around 5148 to around 7600"

   > Search terms used: `"7600|5148|score jump|scoring|accounting|runtime|profile|onnx_tool|ConstantOfShape|dynamic|dtype|initializer|leaderboard|task157|task177|task002|task337|task276|task309"`

   The grep terms suggest the suspected loophole involved **`ConstantOfShape` + dynamic dtypes + initializer accounting** — an onnx-tool / onnxruntime accounting edge case. Specific tasks under investigation: **157, 177, 002, 337, 276, 309**.

2. **`https://weeklykagglenews.substack.com/p/weekly-kaggle-news-333`** — Japanese-language Kaggle news roundup, May 1, 2026. Translation of relevant section:

   > "On The 2026 NeuroGolf Championship, around the 29th, the leaderboard was updated as part of changes to fix evaluator deficiencies. It is a 'golf' competition where small neural networks are developed for the theme of image transformation tasks."

   So the leaderboard recalibration occurred **April 29, 2026** following an evaluator fix.

## Inferred consequences for cost modeling

The fact that an exploit existed via `ConstantOfShape + dtype tricks` strongly suggests the **pre-fix** validator was undercounting some category of cost. After the fix, those costs are presumably counted correctly. Practical consequences:

1. **Treat cost numbers from before April 29 as suspect.** The COVERED probe table (dated 2026-05-01, after the fix) is fine. Older numbers in cached sources may not be.
2. **`ConstantOfShape` is now likely fully accounted.** If you were relying on it as a "free" tensor builder (broadcasting a scalar to `[1,10,30,30]`), reverify the cost.
3. **Initializer dtype mismatches may have been the loophole.** E.g., declaring an initializer as `float16` but using it in a `float32` op might have caused onnx-tool to undercount memory. After the fix, verify all dtypes are consistent before measurement.
4. **Cross-check against the live Kaggle validator before submitting**, because onnx-tool 1.0.1 in your local environment may not have all the patches the Kaggle-side scorer received in the April 29 update. The Kaggle CLI command from the brief is:

   ```
   kaggle competitions download -c neurogolf-2026 -f neurogolf_utils/neurogolf_utils.py
   ```

   Diff the downloaded file against the version you used previously to see what changed.

## The forum thread itself (`/discussion/692827`)

- Direct fetch is blocked by Kaggle's anti-bot wall (consistent with COVERED's note that "Kaggle web pages remain anti-bot-blocked"). I was unable to retrieve the thread body via web_fetch.
- Authenticated `kaggle` CLI may be able to pull it; the relevant command is roughly:
  ```
  kaggle competitions discussions list -c neurogolf-2026
  kaggle competitions discussions topic -c neurogolf-2026 -i 692827
  ```
  (verify exact CLI flags against `kaggle --help`.)

## What this means for the J.01–J.18 cluster

- **J.01 BitwiseAnd/BitShift**: the unprofilable status I measured may actually be one of the things the April 29 fix targeted. If the fix added registered handlers for these ops, my J.01 verdict needs revision (post-fix, the float-arithmetic path may no longer dominate). **Verify against the live validator after pulling the latest `neurogolf_utils.py`.**
- **J.14 Morton perm**: Gather is correctly accounted in onnx-tool 1.0.1 (verified). No revision needed.
- **J.16 RNS / J.18 Stochastic**: verdicts are independent of the loophole-fix; both fail by the cost-formula structure regardless of evaluator version.


# ===== merged from Ort_Advanced/ort_profiling.md (Phase-2) =====

# ort_profiling.md — how `memory` is actually measured

The Kaggle scorer's `calculate_memory(model, trace_path)` computes the `memory` half of NeuroGolf cost by combining ONNX static shape inference with ORT runtime profiler trace JSON. Understanding what gets counted (and what doesn't) is critical for predicting your candidate's cost.

## The procedure (`neurogolf_utils.calculate_memory`)

1. **Validation**: rejects models with >1 input, >1 output, name collisions i/o ↔ initializer, subgraphs, sequence types, non-default opset domain, function definitions.
2. **Static pass**: walks `graph.input + graph.value_info + graph.output`. For each tensor, computes `num_elements × dtype.itemsize`. Stores in `tensor_memory[name]`.
   - Tensors named literally `'input'` and `'output'` are **SKIPPED** (line 234 of `neurogolf_utils.py`). The model's I/O tensors do not count.
   - Dynamic dims (`dim_param` or `dim_value <= 0`): model rejected, returns None.
   - Tensors not present in either input, value_info, or output: model rejected.
3. **Profiler pass**: opens the ORT profiler trace JSON. For each `cat=="Node"` event, reads `args.output_type_shape`. For each output tensor, computes `itemsize × sum(prod(dims))` and takes `max(static_value_info_bytes, profiler_observed_bytes)`.
4. **Return**: `sum(tensor_memory.values())`.

## Key implications

### Initializers do NOT count toward `memory`.
They count toward `params` (element-count, not byte-count). So smaller dtype on an initializer saves nothing in `params`. Smaller dtype on an **intermediate** (a value_info tensor) DOES save in `memory`.

### Tensors named `'input'` and `'output'` are skipped.
This is why the "F16-output trick" works: declare the graph output as F16 → the F16 buffer becomes the `output` tensor → skipped in memory accounting. The Concat that previously produced an F16 [1,10,30,30] = 18,000 byte intermediate now produces the model output directly → 0 bytes counted.

Cross-ref: `techniques.md` §3 (F16-output trick). task092, task117 stacked this with crop for cumulative +1.12 LB.

### Scalar value_info tensors (shape `[]`) cost `dtype.itemsize` bytes (NOT always 4).
The Python loop default `num_elements=1` for empty shape, then multiplies by `np.dtype(np_dtype).itemsize`. Onnx-tool's `tensor.py:volume([])` returns 0. So:
- `BOOL []` = 1 byte
- `F16 []` = 2 bytes
- `F32 []` / `INT32 []` = 4 bytes
- `INT64 []` = 8 bytes

**Cost predictors based on onnx-tool underestimate by `dtype.itemsize` bytes per scalar intermediate.** The earlier "scalar costs 4 bytes minimum" rule only applied to 4-byte dtypes. Use `BOOL`/`UINT8` scalars where possible for cheap sentinel/threshold value_info entries.

### Runtime vs static disagreement: `max` wins.
If ORT's profiler observes a different (typically larger) shape at runtime than the static value_info says (e.g., due to broadcasting expanding a [1, 1, 30, 30] to [1, 10, 30, 30]), the larger is counted. This is why `g.ClearField("value_info")` matters after surgery — stale value_info that says `[1, 1, 30, 30]` but actual runtime shape is `[1, 10, 30, 30]` will result in the larger profiler-observed value being counted.

### A single-op `input → Op → output` graph has memory = 0.
Zero intermediate value_info entries means zero counted memory. The model's `params` (initializer element count) is the only contribution to `cost`. Pure shape-only ops (Identity, Transpose with no weights, Cast) can achieve `memory = 0` with `params = 0` → cost = 0 → 25 pts (the theoretical maximum per task).

Empirical confirmation: 2026-05-01 mining found `identity` and `transpose` cost 0; 2 transpose tasks (179, 241) shipped with cost 0.

## How to use this for cost prediction

```python
# Sketch of locally measured cost vs Kaggle expected
import onnx
from neurogolf_utils.neurogolf_utils import calculate_memory, calculate_params

model = onnx.load("candidate.onnx")
# Run scoring once to get a trace
# (this is what tools/spread_score_isolated.py automates)
mem = calculate_memory(model, "trace.json")
params = calculate_params(model)
cost = mem + params
points = max(1.0, 25.0 - math.log(cost))
```

Practical strategy:
1. **Inspect your candidate's value_info BEFORE Kaggle submit**. Use the snippets in `ort_errors.md` to list intermediates and their byte sizes.
2. **The biggest intermediate dominates**. If you have one `[1, 10, 30, 30] F32` intermediate (36,000 bytes), reducing it via dtype change (→ F16 = 18,000) or rank reduction (→ [1, 1, 30, 30] = 3,600) is a high-leverage win.
3. **`max(static, profiler)` means broadcasting matters**. Where(cond, x, y) when cond is [1, 1, 30, 30] and x is [1, 10, 30, 30]: output is [1, 10, 30, 30] (the larger). If value_info statically says `[1, 1, 30, 30]` (because cond's shape was used), but profiler sees `[1, 10, 30, 30]` at runtime, the larger value is counted. Always declare value_info to match the runtime shape.

## Calibration: local vs Kaggle

Local `score_v2.score_network_v2` runs ORT with profiling enabled and computes `memory + params` exactly as the validator does. **Per-task cost matches Kaggle within 1 byte in 99%+ of cases**.

But total Kaggle score differs from local prediction depending on submission regime:

- Single-task and small raw-log submissions usually track close to 1.0× the local point delta.
- Large megastacks (100+ tasks) have empirically tracked closer to ~0.48× the summed local delta.
- Hypothesis: leaderboard aggregation, per-task floors, hidden/public weighting, and cross-task memory cascade interaction reduce the effective gain on large stacks.
- **Anomaly worth flagging**: task110 wildcard 2026-05-20 — predicted +0.18 calibrated, actual +12.73 LB. Likely cross-task memory cascade unlock; the candidate's cost reduction unblocked Kaggle-cascade-locked neighbor tasks from ERROR back to scored.

Do not apply the 0.48 multiplier to every candidate. Use ~1.0× for isolated/small submissions, ~0.48× for 100+ task bundles, and treat cascade events as separate step-function anomalies that require single-task wildcard evidence.

## Why local can miss Kaggle issues

The cross-task memory cascade (see `core.md`) only manifests in Kaggle's actual eval pipeline. Local `score_network_v2` runs each task in isolation, so cascade-induced ERRORs aren't reproducible locally. If a candidate passes local verify + score but Kaggle's task slot ERRORs, suspect the cascade. Known trigger task set: `{45, 129, 130, 171, 172, 210, 211, 220, 239-242, 256-262, 309}`.


# ===== merged from Onnx_basics_worth_reviewing.md (now archived) (the 9-line jewel) =====

# Onnx_basics_worth_reviewing.md — scoring mechanics

- `Constant` node with `value` attribute → counts toward `params` via the element count of `attr.t.dims`. Sparse `sparse_value` attribute counts `attr.sparse_tensor.values.dims` (stored values, NOT logical dense shape).
- `graph.initializer` → counts toward `params` via `init.dims`. `graph.sparse_initializer` counts `sparse_init.values.dims` (stored values).
- Both also count toward the file size limit (1.44 MiB binary).
- Prefer `initializer` over `Constant` nodes — same params cost, but no node-overhead in the graph.
- `calculate_memory` RE-INFERS `value_info` via `onnx.shape_inference.infer_shapes(model, strict_mode=True)` before counting. **Real intermediates are counted even if `graph.value_info` is empty in the saved model.** Clearing stale `value_info` before save is required only to avoid load-time conflicts, NOT to hide memory.
- Each value_info entry contributes `num_elements × dtype.itemsize` to the scorer's `memory` count. Scalars cost `dtype.itemsize` bytes (`BOOL []` = 1 B, `F16 []` = 2 B, `F32 []` = 4 B, `INT64 []` = 8 B) — NOT always 4 bytes.
- Declaring EXTRA value_info (dead entries / aliases) for tensors that aren't real intermediates IS wasted bytes. Real inferable intermediates are counted regardless.

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

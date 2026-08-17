# patterns.md — auto-detector reference (25 patterns)

Run all of them across a task with: `.venv/Scripts/python tools/spread_scan.py --tasks taskNNN --summary-only`.

The driver (`tools/spread_scan.py`) iterates every `tools/spread_patterns/pass_*.py`. For each pattern + each task, the verdict is one of:
- `applied` — the rewrite was emitted, byte-exact verify passed, cost dropped.
- `no-op` — pattern doesn't match this task's graph shape.
- `advisory` — pattern reports an opportunity without rewriting (operator/worker decides).
- `regression` — rewrite emitted but cost increased; rejected.
- `verify-failed` — rewrite emitted but byte-exact verify failed on random inputs; rejected.
- `score-failed` — modified model couldn't be scored (ORT error etc.).
- `skipped` — task on the skip-list (e.g. task175 crashes ORT profiling).

Each pattern's source file documents the math, safety, and out-of-scope cases in detail. The table below is the index.

## Math-equivalent rewrites (21)

| Pattern | Matches | Rewrites to | Source |
|---|---|---|---|
| `add_chain_to_sum` | N consecutive `Add` nodes (left-fold) | Single `Sum(N+1 inputs)` | `tools/spread_patterns/pass_add_chain_to_sum.py` |
| `bool_andfold` | `Mul(Cast(bool, F32), Cast(bool, F32))` OR `Mul(Cast(bool), 0/1-float-init)` | `Cast(And(bool, bool), F32)` | `pass_bool_andfold.py` |
| `clip_mul_to_min` | `Clip[0,1](x) → Mul(x, 0/1-mask)` | `Min(x, mask)` (one fewer intermediate) | `pass_clip_mul_to_min.py` |
| `concat_bool_inputs` | `Concat` with all bool-eligible inputs (bool-cast source OR 0/1 init) | `Concat` in BOOL dtype, output BOOL | `pass_concat_bool_inputs.py` |
| `const_to_init` | `Constant` node with value attribute | Same value moved to `graph.initializer` (drops node) | `pass_const_to_init.py` |
| `dead_relu_post_maxpool` | `Relu(Sub(MaxPool(X, sym-pad, stride=1), X))` | `Sub(MaxPool, X)` (Relu is identity) | `pass_dead_relu_post_maxpool.py` |
| `drop_cast_before_pad_bool` | `Cast(bool, F32) → Pad → [graph_out OR Greater(0)]` | Drop Cast; Pad on BOOL; output BOOL | `pass_drop_cast_before_pad_bool.py` |
| `equal_fold_distance_threshold` | `Sub(a, b) → Mul(self, self) → Less<1` | `Equal(a, b)` (integer-equivalent) | `pass_equal_fold_distance.py` |
| `flip_where` | `Mul(b, Sub(1, Where(c, b, 0)))` with binary b | `Where(c, 0, b)` | `pass_flip_where.py` |
| `mul_by_all_ones_drop` | `Mul(x, all_ones_init)` | `x` (Mul dropped) | `pass_mul_by_all_ones_drop.py` |
| `mul_mask_to_where` | `Mul(Cast(bool_op, F32), V)` | `Where(bool_op, V, 0_const)` | `pass_mul_mask_to_where.py` |
| `onnxsim` | onnxsim's full pass (DCE, constant fold, algebraic) | onnxsim's simplified graph | `pass_onnxsim.py` |
| `opset_bump_pad_uint8` | opset-10 `Cast(bool, F32) → Pad(attr)` | opset-11 `Cast(bool, UINT8) → Pad(input)` + output UINT8 | `pass_opset_bump_pad_uint8.py` |
| `or_cast_collapse` | `Add(Cast(bool_a, F32), Cast(bool_b, F32))` with mutex predicates | `Cast(Or(bool_a, bool_b), F32)` | `pass_or_cast_collapse.py` |
| `safe_opt` | `onnx_opt_safe` (dedupe + CSE + DCE + algebraic) + extended elim passes | Simplified graph | `pass_safe_opt.py` |
| `scalar_broadcast_ones` | Uniform-value initializer (non-scalar shape) | Scalar `[1]` init (broadcasts identically) | `pass_scalar_broadcast_ones.py` |
| `separable_mask_collapse` | 2D init `[H, W]` where every row OR every column is identical | `[1, W]` or `[H, 1]` init (broadcast at use) | `pass_separable_mask_collapse.py` |
| `shared_cast_equal_fold` | N parallel `Sub(v, c_k) → Abs → Less<1` chains over same `v` | One shared `Cast(v, INT32)` + N `Equal` nodes | `pass_shared_cast_equal_fold.py` |
| `slice_axis_collapse` | Slice with axis-0 no-op entries | Drop those entries (so dedupe reuses inits across Slices) | `pass_slice_axis_collapse.py` |
| `slice_chain_to_multiaxis` | `Slice(axis=A) → Slice(axis=B)` with A ≠ B | Single `Slice(axes=[A, B])` | `pass_slice_chain_to_multiaxis.py` |
| `slice_cse` | Duplicate `Slice` nodes with identical params | Single Slice; consumers rewired | `pass_slice_cse.py` |

## Advisory / selective (4)

| Pattern | Reports | Source |
|---|---|---|
| `crop_to_size_advisory` | Tasks whose max active grid < 30×30 but graph has `[..., 30, 30]` intermediates. Counts intermediates; flags potential crop-process-pad opportunity. See techniques.md §2 for the false-positive guard. | `pass_crop_to_size_advisory.py` |
| `final_cast_f16_to_f32_advisory` | Graphs ending in `Cast(F16, F32) → output`. Manual Cast-drop saves the output buffer (see techniques.md §3). | `pass_final_cast_f16_to_f32_advisory.py` |
| `fp16_placement_advisory` | Post-Slice and per-channel Gather positions where placement-aware F16 could net positive (see techniques.md §4). Also flags hash-decode `Gather → MatMul` chains as DO-NOT-cast warnings. | `pass_fp16_placement_advisory.py` |
| `fp16_selective` | Runs onnxconverter_common blanket F16 conversion. Keeps only when cost actually drops AND verify passes. Regresses ~134/401 tasks because of blanket placement; prefer `fp16_placement_advisory` + manual application. | `pass_fp16_selective.py` |

## Driver flags

```
tools/spread_scan.py
  --zip <path>                # input submission zip (default submissions/submission.zip)
  --tasks taskNNN,taskMMM     # subset; default = all 401
  --apply                     # write candidate ONNX files to --apply-dir
  --apply-dir <path>          # default submissions/_spread_candidates
  --results-dir <path>        # JSON+CSV+summary go here
  --verify-inputs N           # random inputs per byte-exact verify (default 3)
  --limit N                   # scan first N tasks only
  --summary-only              # don't write reports; just stdout progress
  --skip-tasks task175,...    # default skips task175 (ORT crash)
```

## Counter notes for new pattern authors

When authoring a new `pass_*.py`:

1. **Node-name uniqueness**: every emitted node MUST have a unique `name`. NeuroGolf graphs frequently have unnamed nodes, so `name=f"{node.name or 'foo'}_{op}"` is NOT enough — multiple unnamed nodes collapse to the same fallback. Use a counter (`_uniq("prefix")` that increments). `pass_add_chain_to_sum.py` was bug-fixed for this 2026-05-20.

2. **`g.ClearField("value_info")`**: required after any rewrite that changes intermediate shapes or dtypes. Otherwise ORT load fails on shape-inference mismatch.

3. **Single-use consumer guards**: when dropping an intermediate node, check that its output is consumed ONLY by the node you're rewriting. Otherwise you orphan a needed value. `consumers.get(out, [])` length check.

4. **Output-name preservation**: if the rewritten node feeds the graph output, the new node's output name must match the original. Otherwise the graph output dangles.

5. **Verify with random inputs**: the driver's default 3-random-input verify will catch most bugs. But it's not a substitute for `verify_against_live.py` on the actual visible+arc-gen pair set — that's the staging gate.

6. **Out-of-scope is fine**: document what your pattern DOESN'T handle. The driver runs ~25 patterns per task; a conservative pattern that fires on a narrow shape is better than an aggressive one that misclassifies.

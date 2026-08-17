# traps.md — what will break your model and how to avoid it

Five-minute trap scan. Every trap here was paid for in worker-hours by someone. The library will save you that pain if you read this BEFORE building.

Grouped by phase:
- **Build-time traps** — model fails ORT load or validator rejection
- **Runtime traps** — model loads and runs, but produces wrong output
- **Scoring traps** — model is correct, but costs more than expected (or hits silent-fail on hidden inputs)

---

## TRAP-ESCAPE PLAYBOOK (consolidated 2026-05-27 from task366 v21→v72 and task255 11k/9k→v12)

When a task fails the self-consistency trap (synth-passing UNLOCK ship → 0 LB), the corrective action is NOT more synth, NOT more compression, but **fundamental rule re-derivation**.

### Confirmed escape pattern (two examples)

| Task | Trap victim | Rule descriptors | Escape candidate | Rule descriptors | Outcome |
|---|---|---|---|---|---|
| **task366** | v21 cost 7,484 | "sparse-stamp", "scorer-aware", "not organic structural" | v72 cost 98,213 | "K=3 selected colors", "color-index library", "placement lattice" | +13.50 LB EXACT |
| **task255** | fullsig_pairpack_constkey 11,323 + fullsig_nobg 9,522 | "fullsig", "pair-pack", "constkey", "row-packed signature" | v12 cost 24,531 (structural compression of v1's 41,480 corridor-fill rule) | "corridor-fill", "Chebyshev erosion", "length-12 runs", "topology cleanup" | +14.90 LB |

### The playbook

1. **Acknowledge the trap.** If a structural UNLOCK candidate ships at 0 LB on a confirmed zero-weight canonical, the rule decode is wrong. Don't compress further — the same wrong rule will produce the same 0 LB.
2. **Run the descriptor test.** Read your build script's comments and the worker's handoff. If you see implementation nouns ("hash", "signature", "table", "pair-pack", "sparse-stamp", "memorize", "scorer-aware"), you're in the trap. Successful unlocks use algorithmic nouns ("count", "reflect", "extend", "fill", "match", "select").
3. **Re-derive the rule from visible pairs.** Discard the prior numpy reference and start fresh. Visualize the train/test pairs grouped by some natural feature (output shape, color count, object count). The rule often jumps out when you stop reading your own code.
4. **Map to ARC-AGI-2 family taxonomy** (per K-Arc2 cross-ref note 02, 4 v2 axes):
   - **Symbolic interpretation**: objects stand for something else (e.g., "color = count of holes")
   - **Compositional / multi-step**: rule R then rule R' on output of R
   - **Contextual rule application**: same rule but direction depends on input property
   - **In-context symbol definition**: symbols whose meaning is established within this task
5. **Verify the new rule produces structurally DIFFERENT outputs** from the trap candidate on at least 3-5 synthetic inputs. If your new rule and the old trap rule give the same outputs on visible AND on the synthetic differentiating inputs, you're STILL in the trap — the synth oracle agrees because both rules are observationally equivalent within visible.
6. **Compress within the new structural family.** Reject any compression that re-introduces lookup descriptors. Compress via dtype narrowing, separable Convs (per techniques.md F1 Win 1), shared kernels (per techniques.md C1 Win 1), etc.
7. **Synth gate ≥98% on the structural variant is sufficient** for asymmetric UNLOCK. Don't need 99.95% — asymmetric profile guarantees downside floor.

### When the playbook fails

If you re-derive the rule, get 266/266 visible on the new numpy reference, build ONNX, and STILL ship 0 LB — the rule decode is still wrong (or the architecture has implementation bugs that diverge from the rule). At that point:
- Run differentiating-input synthesis: construct inputs where multiple candidate rules predict DIFFERENT outputs, and check which prediction matches the trap candidate (the wrong one)
- If you can't construct such inputs from your visible knowledge, all your hypotheses are observationally equivalent within visible — escalate to per-axis classification per K-Arc2 note 02
- Confirmed precedent: task233 architecture failed 4 ways (v8 K10 96.4%, v8 K14, v10_h3 UINT8 97.3%, v10_h3 F16 99.95% + byte-equal). At that point the architecture is dead; pivot to a fundamentally different rule hypothesis.

---

## Build-time traps (ORT load fails or validator rejects)

### 1. `Where(bool, bool, bool)` is unsupported

ORT rejects Where with bool data inputs (x, y) even though the spec implies it should work. Rewrite as `Or(And(c, a), And(Not(c), b))` or exploit one-hot structure for the simpler `And(one_hot, pick) + Or(row, col)` form.

Trace: task110 v4 worker hit this mid-rebuild. See `Ort_Advanced/ort_quirks.md §2`.

### 2. OneHot index/depth dtype mismatch

ORT's OneHot accepts INT32 and INT64 indices at `ORT_DISABLE_ALL`, but the scalar `depth` input must
use the same dtype. The old "INT64 only" rule was a task124 v3 misdiagnosis: INT32 indices were paired
with an INT64 depth.

```python
indices = indices.astype(np.int32)
depth = np.array(10, dtype=np.int32)   # match indices; use INT64 depth for INT64 indices
```

Trace: task124 v3 tried INT32 OneHot to save bytes. Failed load. See `Ort_Advanced/ort_quirks.md §3`.

### 3. `g.ClearField("value_info")` after shape-changing surgery

ORT loads cached `value_info` shapes BEFORE running shape inference. If you crop, dtype-change, or otherwise alter intermediate shapes without clearing value_info, ORT fails:

```
Op (Max) [ShapeInferenceError] Incompatible dimensions
```

One line saves the model:

```python
graph.ClearField("value_info")
```

Trace: every crop-to-active-grid rewrite. See `Ort_Advanced/ort_quirks.md §10`.

### 4. opset-10 Equal accepts INT32/INT64 only, not float

Under opset 10, `Equal(float_a, float_b)` fails to load. Cast to INT32 first. To avoid N redundant Casts when comparing one value against N constants, **share a single Cast**:

```python
v_int = Cast(v, INT32)             # ONE cast, fanned out
Equal(v_int, c1_int)               # downstream
Equal(v_int, c2_int)
...
```

Trace: task368 — 10-channel color compare. See `Ort_Advanced/ort_quirks.md §4`.

### 5. Duplicate node names (softened 2026-05-23: sanitize_model rewrites them)

Multiple emitted nodes with the same `name` field can fail direct ORT load:

```
[ONNXRuntimeError] FAIL: two nodes with same node name
```

**Updated 2026-05-23**: NeuroGolf's `sanitize_model` (called by the official scorer before session creation) rewrites every `node.name = node.output[0]` AND sanitizes all input/initializer/node-output names to `safe_name_<N>`. **Duplicate raw `node.name` values are no longer a scorer hard failure** — sanitization handles them.

*…full trace + incident detail: `traps_appendix.md` → "5. Duplicate node names (softened 2026-05-23: sanitize_model rewrites them)"*

### 6. Banned ops (validator rejects regardless of opset)

```
LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS
```

Source: `_EXCLUDED_OP_TYPES` in `neurogolf_utils.py`. These cannot appear in any submitted graph. Control flow is structurally blocked.

### 7. Pad attribute layout is grouped, not interleaved

The `pads` array of length `2 × rank` is ordered:

```
[x1_begin, x2_begin, ..., xN_begin, x1_end, x2_end, ..., xN_end]
```

NOT `[x1_begin, x1_end, x2_begin, x2_end, ...]`. Many workers get this wrong; ORT then fails shape inference.

Trace: `Ort_Advanced/ort_quirks.md §6`.

### 8. Graph output dtype must match producing node's output

After dtype-changing surgery (BOOL/UINT8/F16 outputs), update:

```python
graph.output[0].type.tensor_type.elem_type = TensorProto.BOOL  # (or UINT8, etc.)
```

ORT enforces dtype consistency at load. Mismatch error: `Type Error: Type X of output arg ... does not match expected type Y`.

Trace: `opset_bump_pad_uint8` rewrite hit this. See `Ort_Advanced/ort_errors.md`.

### 8b. `score_network` returns `None` silently — two non-obvious causes (2026-05-29, task209)

`tools/_score_v2.score_network_v2` returns `None` (not an error) when the official `calculate_memory`/`calculate_params` reject the graph. The model can pass `onnx.checker` + load + run in ORT yet still score `None`. Two causes that bit task209's structural build:

- **Dynamic `Slice` (runtime starts/ends) → `calculate_memory` = None.** Strict `shape_inference` leaves the Slice output shape symbolic (it can't fold the runtime starts/ends), so the intermediate has no concrete dim → rejected. **Fix: crop at a dynamic offset with double-`Gather`** — `Gather(x, off+arange, axis=0)` then `axis=1`. Gather's output shape is the (static) index-tensor shape, so it scores fine. This is the structural-crop idiom whenever the crop OFFSET is data-dependent but its SIZE is fixed.
- **Empty initializer (shape `[0]`) → `calculate_params` = None.** Its guard is `any(d <= 0 for d in init.dims)`. The classic offender is passing an empty `roi` tensor to `Resize`. **Fix: skip the optional input with `""`** (e.g. `Resize` inputs `[X, "", scales]`), never an empty initializer.

Debug recipe: call `neurogolf_utils.calculate_memory(sanitized, trace)` and `calculate_params(sanitized)` separately — whichever returns `None` localizes the cause.

### 8c. Sub-INT32 `Gather`/`GatherElements` indices → `score_network` = None (2026-05-30, task155)

Tempting micro-opt: shrink a `[K]` index intermediate by storing it as INT16 (2 B/elem) or INT8/UINT8 (1 B/elem) instead of INT32 (4 B/elem). **The official strict-mode scorer rejects it — `score_network_v2` returns `None`.** ONNX's `Gather` (and `GatherElements`/`GatherND`/`Scatter*`) constrain the index type `Tind` to `{int32, int64}`; `onnx.shape_inference(strict_mode=True)` inside `calculate_memory` enforces this even though plain `onnx.checker` + a non-strict ORT load may pass. Casting a small-int idx → INT64 *immediately before* the Gather is also rejected (the small-int intermediate still fails strict inference). **INT32 is the hard floor for any gather/scatter index: `K` elements cost `4*K` bytes, period.** Worked example: task155 FLIP-V floor is 158 (idx[30] INT32 = 120 B + 30-param range + 8 B to compute the scalar) — every INT16/INT8/UINT8 index attempt scored `None`.

### 8d. `Div`/`Mod` by an input-derived value → SIGFPE on the scorer's zero-grid → submission REJECTED (2026-05-30, task159)

The official scorer **profiles the graph on `np.zeros((1,10,30,30))`** to populate the ORT trace (see core.md). If any `Div` or `Mod` divisor is *computed from input content* (a color count, a bbox span, `red_total`, etc.), it is **0 on the all-zero dummy** → integer division-by-zero → **SIGFPE C-crash → `score_network` returns `None` → the submission is rejected**, even though the graph is correct on every real input. This is distinct from the fp16-SIGFPE in core.md (that's float overflow; this is integer Div-by-zero).

**Fix: clamp the divisor with `Max(divisor, 1)`** (or `Max(..., k_min)`) before the `Div`/`Mod`. It is **behavior-preserving** whenever real inputs never produce the zero case — verify by confirming the divisor's real range (task159: `red_total ∈ {16,28,40,52}` → `mag ∈ {1,2,3,4}`, never 0). task159's `crop_v1` (78,580 → 7,207) would have been rejected without `Max(mag,1)`; the clamp added 1 op and changed nothing on the 54,000-pair byte-equal gate. **Always clamp any input-derived divisor**, and use the subprocess-isolated scorer (`tools/spread_score_isolated.py`) to surface the crash — the in-process scorer just dies.

### 8e. `sparse_initializer` as an op input → `score_network` = None (unscoreable) (2026-05-30, task309 + task276)

Tempting micro-opt: store a `Gather` index (or any initializer that contains zeros) as a `graph.sparse_initializer` so `calculate_params` counts only the **stored non-zero values** (`sum(math.prod(values.dims))`) instead of the dense element count. For a length-10 channel-permutation idx with a single zero entry that is 9 vs 10; for a "minimal" idx (don't-care slots → 0) it could be 2–3. **VERDICT: it does not work — the official scorer returns `None` (model scores 0 on Kaggle).** Confirmed independently on **task309** (swap 5↔7) and **task276** (swap 2↔6). Three independent rejection points in the official `neurogolf_utils` pipeline, any one of which is fatal:

1. **`calculate_memory` does `init.name for init in graph.sparse_initializer`** — but `SparseTensorProto` has **no `.name` field** (the name lives on `sparse.values.name`) → `AttributeError` → `None`.
2. **`onnx.shape_inference.infer_shapes(strict_mode=True)`** can't see a sparse initializer as a tensor input → it mis-infers the consumer's output rank → raises (observed: `(op_type:Gather): Inferred shape and existing shape differ in rank: (3) vs (4)`).
3. **`sanitize_model`** rewrites node inputs + dense `initializer` names to `safe_name_<N>` but **never touches the `sparse_initializer` list** → the consumer's input is renamed while the sparse tensor reference dangles → ORT `INVALID_ARGUMENT: Node input 'safe_name_0' is not a graph input, initializer, or output`.

**Closure — the sanitize gate (point 3) is defeatable, but it doesn't matter (2026-06-01, task164/311).** Point 3 looks like a mere harness bug you could route around, so it's worth shutting the loophole explicitly: `sanitize_model` renames the lone non-I/O node input *deterministically* — with no dense initializers, the consumer's index input becomes `safe_name_0`. **Pre-name the sparse init's value tensor `safe_name_0` and the reference re-links after sanitization** (`node.input` → `safe_name_0`, sparse values stay `safe_name_0`; verified). This defeats point 3 entirely — yet the model still scores `None`, because points 1–2 are *immutable ONNX/ORT op-schema facts*, not harness bugs: ORT refuses to load a Gather whose `indices` is sparse-typed (`INVALID_ARGUMENT`), and `onnx.shape_inference` rejects it (`indices typestr: Tind, has unsupported type: sparse`). **No ONNX op that we can score accepts a sparse-typed tensor input**, so sparse-NNZ param counting is unreachable for any real consumer; the only way to feed a densified sparse value to an op is a `Constant(sparse_value)` node, whose dense output is then a counted **memory** intermediate (e.g. `[30]` INT64 = 240 B → cost 268 on task164). Repro: `The_400_Tasks/Task_by_Task/task164/prototypes/sparse_bypass_probe.py`. Net: the dense initializer is the unique container counted in **params but not memory** — `prod(dims)` is the hard floor for an op-input index, full stop.

**THE TRAP: raw (un-sanitized) `onnxruntime` loads AND runs the sparse model byte-equal** — it densifies the sparse initializer at load, so it looks completely valid until you score it. Always score through `score_network_v2` (which sanitizes first), never trust a bare ORT run. Together with **§8c** (INT32 is the index-dtype floor) and **§15** (dtype doesn't help `params`), the param cost of a `[K]` op-input initializer is exactly its dense element count `K` — there is no storage or dtype trick below it. This is *why* the single-`Gather` channel-permutation family (task016 / 276 / 309 / 337) is floored at **cost 10** (idx[10] = 10 params, memory 0). See the full floor proof in `The_400_Tasks/Task_by_Task/task309.md`.

### 8f. `f16` is the MEMORY floor for a reduction/arith track — uint8/int8/int16 are unsupported (2026-06-01, task377)

When whole-graph fp16 leaves a graph dominated by many small `[1,1,30,30]` intermediates, the tempting next step is uint8 (1 B/elem, half of f16's 2 B) since the values are small integers (0/1 masks, 0–9 colors). **For any tensor that flows through `ReduceMax` / `ReduceSum` / `Mul` / `Sum` (and most arithmetic), uint8 is rejected in opset 11** — `onnx.shape_inference(strict_mode=True)` raises `data typestr: T, has unsupported type: tensor(uint8)` and ORT load fails `INVALID_GRAPH`. The reduce/arith op type-constraint `T` is `{float, double, float16, int32, int64, uint32, uint64}` — **no sub-int32 ints**. So **f16 (2 B) is the hard memory floor** for a reduction/position-finding track; you cannot halve it with uint8/int8/int16. (Confirmed why task377's 2026-05-20 f16 pass did NOT use uint8.) Only **leaf** tensors consumed *exclusively* by uint8-capable ops can narrow: `ArgMax / Where / Greater / Gather / Equal / Squeeze / Cast / Concat / Reshape` DO accept uint8 — e.g. a one-hot output tail `Cast→Pad` narrows f16→uint8 (Pad-11 accepts uint8 but **rejects bool**), and a colour-index canvas used only by Gather+Equal can go uint8 (but watch a downstream `Mul` that forces a Cast-back). Probe before committing: `The_400_Tasks/Task_by_Task/task377/prototypes/probe_uint8_ops.py` tests each op's uint8 acceptance under checker+ORT+scorer in one shot. Pairs with **§8c** (INT32 index floor) and **§15** (dtype ⊥ params): dtype narrowing helps **memory** only, and only down to each op's narrowest supported type.

**`CumSum` is even narrower — it rejects `f16` AND `uint8` (must be `f32`/`f64`/`int32`/`int64`)** (probed 2026-06-01, task216). `CumSum(float16)` fails `INVALID_GRAPH: Type 'tensor(float16)' of input parameter (x) of operator (CumSum) is invalid`. So a span/run-length track built on `CumSum` (`gap = Less(...) → Cast → CumSum → Less<0.5`) pins its `[k]` vectors to **f32 4 B/elem** — you cannot f16-halve them. (Verified in the same graph that `MatMul(f16)` and `ReduceSum(f16)` DO work, so scoring/position tracks narrow fine; only the CumSum chain is stuck at f32.) When the CumSum vectors are tiny (`[20]`/`[40]`), leave them f32 — the risk (ORT C-crash on some f16 graphs) and effort outweigh the ~40 B/vector saving.

---

## Runtime traps (model loads but produces wrong output)

### 9. The 1-ch0 trap (padding cell encoding)

DO NOT compute "is foreground" as `Sub(1, Gather(input, [0], axis=1))`. Padding cells outside the active grid have ALL channels = 0, so `1 - ch0 = 1` there, incorrectly marking padding as foreground.

Correct formulation:

```
full_sum   = ReduceSum(input, axes=[1], keepdims=1)
ch0_val    = Gather(input, [0], axis=1)
nonbg_bool = Greater(Sub(full_sum, ch0_val), 0)
```

Trace: task384 with 1-ch0 approach got 0/266 visible verify. With ReduceSum-Sub-Greater it got 266/266. See `core.md` "The 1-ch0 trap" for the encoder-level explanation.

### 10. OneHot wraps negative indices to the modulo channel

`OneHot(idx=-1, depth=10)` produces a one-hot vector at channel 9 (modulo wrap), NOT all-zeros.

If you want "no output for this cell" (all-zero per-channel), use an OUT-OF-RANGE positive index:

```python
fi = Where(input_active, real_color_idx, 10)   # 10 is out-of-range; OneHot returns all-zero
output = OneHot(fi, depth=10)
```

Trace: task110 worker initially used -1, got channel-9 outputs at padding cells. Switched to fi=10.

### 11. Joint V+H ArgMax for periodic-fill tasks

When scoring per-direction periods (vertical + horizontal), joint ArgMax over combined V+H scores systematically favors whichever has more pair-count. For asymmetric-period pairs (rH ≠ rW), this picks the wrong period.

Solution: ArgMax separately per direction.

Trace: task110 v4 lesson 2026-05-13. See `Algorithm_Library/algorithm_patterns.md` §4.

### 12. Bool reformulation of float graphs (silent-fail on hidden)

Replacing `Where(c, float_x, float_y)` with `Or(And(c, x_bool), And(Not(c), y_bool))` MAY change outputs on inputs that don't appear in the visible+arc-gen set. Hidden test inputs can hit divergent cases.

task080 v5_bool_slices: 231/231 visible + 450 adversarial × 9 modes = 100% pass → **-13.32 LB on Kaggle**.

Don't apply bool reformulations to float algorithms unless the underlying values are provably 0/1.

Trace: `Logs/lessons.md` 2026-05-12 entry on v5_bool_slices.

### 13. Approximating CC as bounding-rectangle

Connected-component analysis is structurally blocked (Unique is banned). Tempting to approximate as "bounding rectangle = room". This is wrong for L-shaped or non-rectangular rooms.

task145 organic ONNX (rectangular-partition CC) passed 267/267 visible → **-1.18 LB on Kaggle**.

If your numpy reference matches visible 100% but uses an approximation of a non-trivial algorithm — generate adversarial inputs and check against LIVE before staging.

Trace: `Algorithm_Library/rule_discovery_methodology.md` "Trap 3".

### 14. `Equal` requires integer-valued inputs for the math identity

`Sub→Mul(self)→Less<1` collapses to `Equal(a, b)` ONLY when both inputs are exact-integer floats. For non-integer inputs (e.g., 1.3 vs 1.5), the math diverges.

The detector checks integer-ness via numpy round-trip. Manual application: trace inputs back to one-hot or color sources before substituting.

### 14b. Naive non-incremental thickness check causes cross-rectangle false positives

When checking "rectangle extends K cells in direction D from a marker", the naive check
```
has_thick_d_k = And(edge_d, shift_AWAY_k(rect))   # is rect cell at distance k from marker?
```
is **silently broken** when there are MULTIPLE rectangles in the input — the check sees through gaps to a DIFFERENT rectangle's cells, falsely reporting thickness.

Correct: incremental contiguous AND chain:
```
has_thick_d_1 = edge_d                                  # base: thickness >= 1 (always for valid edge)
has_thick_d_k = And(has_thick_d_{k-1}, shift_AWAY_k(rect))   # for k >= 2: all intermediate cells must be rect too
```

The chain ensures cells at distances 1, 2, ..., k are ALL rect (contiguous). A gap (k where `shift_AWAY_k(rect) = 0`) terminates the chain — correct behavior for the next rectangle.

**Empirical**: task280 pair 0 — rect1 marker (4, 3) on LEFT edge has true thickness 1. Rect2 occupies row 4 cols 6-9. Naive check at k=3: `shift_LEFT_3(v296)[4, 3] = v296[4, 6] = 1` (rect2's cell) — **false positive**. Incremental check correctly returns 0 because k=2 is 0 (gap at col 5).

**Especially dangerous** because it can pass 100% visible verify when the visible set happens to have only single-object inputs but fail catastrophically on private/hidden test sets with multi-object inputs. **Always run the adversarial check** (`tools/agent_helpers/adversarial_against_live.py`) on multi-object detectors — but treat the result as advisory per `core.md` 2026-05-22 update. The signal you actually care about for hidden-correctness here is **synthetic-LIVE on multi-object inputs**, not raw shift/rotate adversarials.

### 14c. Compact-profile Or-broadcast pollutes padding rows/cols

When using `§22 compact-mask projection`, a compact `[1, 1, 1, W] BOOL` Or-broadcast against `[1, 1, H, 1]` produces an `[1, 1, H, W]` BOOL where ANY column with True in the compact profile becomes True across ALL rows — INCLUDING padding rows outside the active grid. The compact `col_act_b [1, 1, 1, W]` was correctly gating the COLUMN compact, but the resulting `[H, W]` broadcast extends outside the active region.

```
# WRONG: compact gating of col_profile, but result extends to all H rows
col_2d_b = Or(...)               BOOL [1, 9, 1, 30]    # active cols only
result = And(col_2d_b, row_2d_b) BOOL [1, 9, 30, 30]   # includes padding rows
```

**Fix**: gate the FINAL broadcast result by `cell_active = And(col_act_b [1,1,1,W], row_act_b [1,1,H,1])`:
```
nbg_bool [1, 9, 30, 30] = Or(col_2d_b, row_2d_b)
nbg_active = And(nbg_bool, cell_active_broadcast)   # 900 B for 30×30 BOOL
```

The active-area `cell_active` mask can be derived cheaply from the compact profiles themselves via `ReduceMax(profile, over_channels)` → both compact axes → `And` → broadcasts to `[1, 1, H, W]` for only `H*W` bytes (vs 4× that for a full F32 ReduceMax of the input).

**Empirical**: task013 v2 caught this during build. Symptom: stripe pattern extends into bottom padding rows on visible verify; failed 53/267 before the gating fix.

### 14d. "Any-neighbor" vs "same-color-neighbor" semantic divergence on multicolor inputs

When a task's rule is "keep cell IF neighbor matches PROPERTY", the visible set may not distinguish between two implementations of "matches PROPERTY":
- **Any-neighbor** (any cell present nearby) — cheaper Conv kernel (5×5 ones), fewer ops.
- **Same-color-neighbor** (neighbor has the SAME color value) — requires per-channel filtering, more expensive.

These produce identical outputs when inputs are MONOCHROMATIC (only one color present), which is common in visible+arc-gen sets. They DIVERGE on multicolor inputs where same-color-neighbor produces sparser output than any-neighbor.

**Empirical**: task097 v2_any_conv_neighbor_only (cost 159,386 → 83,715, expected +0.64 LB) passed 267/267 visible + 4806/4806 adversarial perturbations (100%) → **Kaggle scored 5714.95 = -13.02 LB regression** (2026-05-16). Diagnostic: 3,000 random multicolor grids showed **84.23% divergence** vs canonical LIVE.

*…full trace + incident detail: `traps_appendix.md` → "14d. "Any-neighbor" vs "same-color-neighbor" semantic divergence on multicolor inputs"*

### 14f. Conv color-index with weights [0..N-1] mistakes padding for color-0 shapes (2026-05-26)

**Pattern**: techniques.md §23 projection-cache via `Conv(input, weight=[0, 1, ..., 9] FLOAT)` produces `color_idx[r, c] = c` for active cell of color `c`, and `color_idx = 0` for padding cells (all-channels-zero). If the task has shape cells with **color 0**, the Equal(color_idx, 0) test returns True for BOTH shape-0 cells AND padding cells — inflating the shape-0 bbox to extend into padding territory → wrong picked → wrong output.

**Empirical (task319, 2026-05-26)**:
- v33 passed 267/267 visible (no color-0 shapes in visible data) at cost 16,250.
- Synth (independent oracle, 35.3% pairs have color-0 shapes) dropped to 80.6% — visible-blind.
- Fixed in v34 by shifting weights to `[1, 2, ..., 10]`: padding → `color_idx = 0`, color `c` → `c+1`. Then Equal vs `shape_indices + 1`. Padding never matches any shape (since shape_indices ∈ [0, 9] → +1 ∈ [1, 10] ≠ 0).
- Cost: +27 bytes for the Add INT32 +1 op (Add UINT8 unsupported at opset 13). v34: 16,716 polished, synth 99.33% PASS.

**Rule**: when using Conv-based color-index (techniques.md §23), ALWAYS shift weights by +1 (or use a sentinel for padding) UNLESS you've verified the task has no color-0 shape cells in **synth** (not just visible — visible can be color-blind).

**Detection**: this is a visible-blind bug. Generic synth with random colors catches it. Independent-oracle synth WITH color permutation catches it. visible-only verification cannot.

### 14g. Index-space recolor of a one-hot parent must reproduce the parent's `Gather(eye,0)→color-0` default, not blank (2026-05-31)

**Pattern**: Path-A byte-equal compression that collapses a one-hot recolor parent to a single-channel color-index canvas. The parent selects a fill color with `mc_i = Cast(ReduceMax(color_idx * mask), INT64)` then paints it via `Gather(eye_10, mc_i)→[1,10,1,1]` one-hot `Where`. When a class has **no marker**, `ReduceMax = 0` → `mc_i = 0` → `eye_10[0]` = **channel-0 one-hot** → the parent paints **color 0** at those cells. If your index-space rewrite uses a `color+1` canvas and an `Equal(out, [1..10])` tail, a `ReduceMax` of 0 maps to value 0 → matches **no** channel → paints **blank** (all-zero). Divergence from the parent exactly on the no-marker cells.

**Why it fires on VALID inputs, not just degenerate ones**: even when the generator guarantees a marker per class (e.g. task324 legality `len(seen)==2`), a parent that classifies markers via a **heuristic vote** (3-way majority) can mis-bucket both markers into one class, emptying the other → `ReduceMax=0` there. task324 compress_v1: without the fix, **37/2000 ARC-GEN valid + 844/2000 adversarial** diverged (all `parent=0` vs `mine=blank`); fully-random was 0 (markers rarely land on a vote-ambiguous bg).

**Fix**: clamp in `+1` space — `mc_p1 = Max(ReduceMax(canvas * mask), 1)`. No-marker (`ReduceMax=0`) → 1 → color 0, matching the parent's `eye[0]`. Markers (value `color+1 ≥ 1`) are unchanged. Cost: 1 scalar `Max` + 1 scalar init.

**Rule**: when byte-mirroring a one-hot/`eye`-Gather recolor parent in index space, replicate its **default-fill** (usually color 0 from the `argmax/ReduceMax`-of-zeros = 0) — don't assume "no marker → leave unchanged". Catch it with a direct A/B vs the parent on ARC-GEN-valid **and** adversarial vote-ambiguous inputs; visible-only and fully-random both miss it.

### 14h. Validate the full one-hot tensor, not an argmax/active-region shortcut (task165, 2026-06-01)

**Pattern**: a local harness decodes candidate and expected outputs with `argmax`, crops to the active
region, or ignores padding rows/cols, then reports 100% even though the scored tensor differs. This is
not byte-equality. The official output contract is the full `[1,10,30,30]` one-hot tensor, including
padding cells and channel exclusivity.

**Task165 failure mode**: an archfall rewrite leaked fill into rows 20..29. An argmax-on-active-region
self-test saw 200k/200k, but full one-hot comparison and `ng_gate visible` exposed the bug (0/265
before the padding fix). The shipped crop variant only became safe after comparing the complete tensor.

**Missing-selector variant from the same task**: if a marker/color is absent, `ReduceMax`/count `M=0`
can turn into an index/color value of 0 and accidentally match background. Do not let "not found" fall
through to color/background 0; route it to an impossible sentinel or guard on `M==1`.

**Rule**: for any Path-A/Path-B rewrite that re-emits one-hot, run at least one check that compares the
raw full output tensor against LIVE/GT, not just a decoded/cropped visualization. Only after that should
argmax views be used for debugging.

### 14i. Negative-pad crop can be officially byte-equal but lose the robustness margin (task138 v10, 2026-06-02)

**Pattern**: after proving the ARC-GEN generator never uses row 0 / col 0, you crop the source canvas
with negative Conv pads. ORT accepts the graph and the official-generator gate can be perfect, but the
candidate no longer represents edge-touching frames that LIVE can handle.

**Task138 measurement**:
- v9 (clean): source canvas `[1,1,25,24]`, crop planes `[1,1,24,23]`, cost **13,204**. Gates:
  266/266 visible, 200k/200k truth, 100% adversarial. Kaggle: **6451.59 -> 6451.71** (+0.12 over v7).
- v10 (rejected): negative-pad-cropped source canvas `[1,1,24,23]`, cost **12,899**. It was byte-equal
  to v9 on official-encoded inputs (2266/2266) and 200k/200k truth, but adversarial-vs-LIVE fell to
  238/266 base and roughly 85% on perturbation modes, because the row0/col0 source no longer existed.

**Rule**: an official-contract-only crop can be a valid captain-option, but do not call it the clean
floor when it removes a behavior LIVE still supports and the gain is tiny. For follow-on compression
after an already hidden-correct ship, use adversarial as a veto when the saved bytes are small and the
assumption is "the generator never hits this boundary." If the captain explicitly accepts the robustness
trade, document the assumption in the manifest; otherwise ship the robust floor.

### 14j. OOD adversarial shift misses are advisory when the generator fixes placement (task383, 2026-06-02)

**Pattern**: a Path-B rewrite is 100% on visible and generator truth, but an adversarial mode shifts the
object into placements the generator never emits. This is not the same risk as a hidden-valid geometry
miss.

Task383 v1 had **266/266 visible** and **200,000/200,000 ARC-GEN truth**, but the adversarial shift mode
had 53/1064 +/-1 divergences: down/right shifts moved content off the top-left-origin placement that the
real generator fixes. Kaggle held and credited the cost ratio: 55,467 -> 25,767, public 6455.23 -> 6455.99.

**Rule**: if the generator contract fixes placement/canvas occupancy and the candidate is 100% on a large
generator-truth corpus, do not over-veto a small OOD shift-mode divergence. Document it as a robustness
trade and submit. This is different from task138 v10, where the saved bytes were tiny and LIVE still
supported edge frames.

### 14k. Parent-equivalent compression gates against the parent, not against a better oracle (task286, 2026-06-02)

**Pattern**: the current hidden-contributing parent is not visible-clean or truth-clean against the
strongest oracle, but the goal is Path-A compression of that parent. In that case, the safety gate is
byte-equality vs the parent behavior, not visible/truth accuracy versus a different rule.

Task286 crop-label safe was byte-equal **5265/5265** vs the current parent and inherited the parent's
local misses (257/265 visible, 4842/5000 truth), yet Kaggle credited it: 63,211 -> 55,933, +0.12. The
risk ladder m21/m22/m23/m24 showed monotonic public scores close to the parent, while the under-20k m2
probe (19,933, visible 16/265, truth 190/5000) failed badly.

**Rule**: for parent-preserving compression, A/B against the exact shipped parent and require byte-equal
or an explicitly accepted risk ladder. For new-rule replacement, use LIVE/GT/adversarial gates. Do not
reject a parent-equivalent compression just because the parent itself is not the best decoded rule.

### 14e. Count-vector hash lookup is hidden-bad on tasks with public weight

**Pattern**: candidate hashes input statistics (color counts, position counts, signature vectors) to an integer key, then looks up a packed output (r0/c0/h/w, or pre-computed transform). Passes 266/266 visible + arc-gen because the 266 official inputs have unique hash keys. **Hidden inputs collide or fall outside the trained hash range → wrong output → -X LB regression.**

**Confirmed cases (2026-05-22 to 2026-05-23)**:
- task077 `lookup8_colorpos_u8_base9_i32tail`: 142,762 → 10,802. 266/266 verify. Submitted → **-13.13 LB Kaggle**.
- task396 `count_hash_lookup`: 109,351 → 3,294. 266/266 verify + 0.3% adversarial. Submitted in 2-stack → **stack lost 11.29 LB; task396 contribution isolated to ≈ -13.40 LB**.
- task370 `mph_w20_i32tail` (minimal-perfect-hash): 261,275 → 4,014. 266/266 verify but mph = lookup architecture. Submitted in 2-stack → **stack -10.50 LB; task370 contribution ≈ -13.04 LB**.
- task157 `bitpack_key_gathernd_*`: similar architecture; task157 happens to be **zero-public-weight** (7/8 memorized-lookup family) so it scored 0.00 instead of negative, but the architectural risk profile is identical.
- **task066 `gatherhash24_bitpacktail`** (2026-05-23T17:11Z calibration): 21,408 → 4,697, 266/266 verify, 81.6% adv. Submitted as single-task wildcard on 5949.01 → returned **5949.01 = ZERO LB delta**. 24-cell sampled hash + 267-entry template table + packed_coords[267,1] = visible-pair-count lookup signature. Initially recorded as "8th zero-public-weight family member" — **OVERTURNED 2026-05-23T18:33Z** (see entry below).
- **task076 `crop11x7_rowcol_hashspill_f16idx_op13_sim`** (2026-05-23T17:25Z calibration): 26,204 → 3,878, 266/266 verify, 0/50 generic adv. Submitted as single-task wildcard on 5949.01 → returned **5949.01 = ZERO LB delta**. 11×7 sampled signature + 291-entry primary_rows + 3× packed_coords[266,1]. Initially "9th zero-weight family member" — **OVERTURNED 2026-05-23T18:33Z** pending true-rule-rewrite test on task076.
- **task277 `h5hard_convtranspose`** (2026-05-23T17:42Z calibration): 46,539 → 2,948, 266/266 verify (NN distilled output, BOOL tail). Submitted as single-task wildcard on 5949.01 → returned **5949.01 = ZERO LB delta**. Conv(5,10×10)+ReLU+ConvTranspose(5,10×10) autoencoder distilled from 266 visible pairs. Initially "10th zero-weight family member" — **OVERTURNED 2026-05-23T18:33Z** pending true-rule-rewrite test on task277.
- **task233 + task192 (2026-06-09, RECURRENCE — reviewer flagged 2nd time):** task233 under-10k "lookups" read **261/2000 truth (13%)** — memorized the 266 public pairs, garbage on the generator distribution; task192's 910 "free-conv" was the same class. **THE RULE TO INTERNALIZE: run `ng_gate truth --full` BEFORE staging ANY candidate — <99% truth (and not byte-equal to a correct LIVE) = a LEARNING ARTIFACT, not a submission; do not stage it as "best candidate."** `truth --full` (or a large fresh ARC-GEN sample) IS the hidden proxy — the hidden test is the generator distribution. **Visible-pass is necessary but NOT sufficient** (a lookup passes 266/266 by construction). On a DETERMINISTIC rule there is no "risky ship for hidden credit" — visible-pass + truth-fail = certain 0 (this is the complement of the VISIBLE-FAIL axiom). Mark the overfit lookup dirs `do_not_submit`. (task233's correct floor is ~25k: dihedral-signature + histogram genuinely needs it; v80 Path-B 25,658, v86 25,362 — nothing correct goes under.)

### 15. Translation/shift via `Pad` is byte-WRONG under the ch0-background encoding (2026-06-01, task053)

A "shift the grid by (dr,dc), zero-fill" rule looks like a free win: a single `Pad` (opset-10 `pads`
is an **attribute** ⇒ 0 params; the Pad output is the graph `output` ⇒ 0 memory) scores **cost 0 =
25.0000 pts, the ceiling**. It is a **trap** whenever the vacated cells must read as *background* and
the encoder one-hots background as `ch0=1` (the standard `convert_to_numpy`: `benchmark[0][color][r][c]
= 1.0` for **every** color including 0). A scalar `Pad` fills the vacated row/col with all-zero, but

*…full trace + incident detail: `traps_appendix.md` → "15. Translation/shift via `Pad` is byte-WRONG under the ch0-background encoding (2026-06-01, task053)"*

### 🔥 MAJOR DIAGNOSTIC REFINEMENT 2026-05-23T18:33Z (task066 wallpath unlock +15.62 LB)

**The "lookup-variant byte-identical to LIVE → 0.00 LB → zero-weight" calibration was WRONG.** Empirical disproof: `task066_wallpath_rule.onnx` (TRUE structural L-path rewrite, 11,769 cost, NOT lookup-family — pure Where/Less/And/Gather/Cast over wall-plane coordinates) shipped at **5949.01 → 5964.63 = +15.62 LB**.

That's 25× the raw log_e prediction (+0.60 LB). It means **task066 has substantial positive public weight that was MASKED by the LIVE graph's own lookup-style memorization**. When I shipped the 4,697 lookup candidate that morning, it scored 0.00 vs LIVE because BOTH models were producing the same memorized-lookup outputs on hidden inputs (neither was capturing the rule). The structural wallpath_rule produces DIFFERENT, CORRECT outputs on hidden → unlocks the latent +15 LB.

**Corrected diagnostic interpretation**:
- "Lookup candidate scores same as LIVE on Kaggle" = LIVE is ALSO in the lookup family. It does NOT mean the task has zero public weight.
- The ONLY valid zero-weight test is **cherry13b Identity stub** (cost 0, returns all-zeros output): if submitting that doesn't regress LB, the task has zero public weight. If it regresses, LIVE was providing some value.

*…full trace + incident detail: `traps_appendix.md` → "🔥 MAJOR DIAGNOSTIC REFINEMENT 2026-05-23T18:33Z (task066 wallpath unlock +15.62 LB)"*

## Scoring traps (model correct but costs more or hides cost)

### 14z. Multi-output models score `None` — the "declare intermediates as outputs" exemption does NOT exist (probed 2026-06-04)

Graph outputs are exempt from `memory`, which suggests declaring expensive intermediates as
EXTRA graph outputs to zero them out. **Probed and dead: `score_network_v2` returns `None`
for any model with a second graph output** (both aux-last and aux-first orderings; the
single-output control scored normally). The official pipeline requires exactly one output.
Do not spend time on multi-output accounting tricks — and treat any candidate that
accidentally exports debug outputs as unscoreable until stripped. Probe artifact:
task119 v6 exploration (`Logs/lessons.md` 2026-06-04).

### 15. Initializer dtype doesn't help `params`

`params` is element COUNT, not byte count. A `[1, 10, 30, 30] F32` init = 9,000 params. A `[1, 10, 30, 30] BOOL` init = 9,000 params (identical).

Use dtype for `memory` (intermediates) only. For `params` savings, use SHAPE compression (separable masks, scalar broadcast, dynamic compute).

### 16. Scalar value_info costs `dtype.itemsize` bytes, not 0 (and not always 4)

A scalar tensor (shape `[]`) costs `num_elements × dtype.itemsize = 1 × dtype.itemsize` in NeuroGolf's `calculate_memory`. Onnx-tool reports 0 for the same shape. The true cost depends on dtype:
- `BOOL []` = 1 byte
- `F16 []` = 2 bytes
- `F32 []` / `INT32 []` = 4 bytes
- `INT64 []` = 8 bytes

The old "scalar costs 4 bytes" rule is only correct for 4-byte dtypes. Use `BOOL`/`UINT8` scalars where possible for cheap sentinel/threshold value_info entries.

Trace: `data/competitions/neurogolf-2026/neurogolf_utils/neurogolf_utils.py:calculate_memory`. See `scoring_internals.md` (profiling section, merged Phase-2).

### 17. `np.zeros(...)` defaults to F64

```python
arr = np.zeros((10, 30, 30))           # F64 — 8 bytes/elem!
arr = np.zeros((10, 30, 30), dtype=np.float32)   # F32 — 4 bytes
```

Same for `np.arange(N)` (returns INT64). Always specify dtype on initializer construction or `params` inflates silently.

Trace: F64-default trap (Numpy_Advanced/ folder removed 2026-05-23; this trap is documented inline here).

### 18. F16 pre-cast of full input regresses

`Cast(input [1,10,30,30] F32, F16)` adds an 18,000-byte F16 intermediate. If downstream uses only 1-3 channels, this is wasted: a per-channel cast saves 14k+ bytes vs pre-cast.

Rule: place F16 Cast AFTER Slice/Gather (the narrowing op), not before. `pass_fp16_selective` (blanket) regresses ~134/401 tasks because it pre-casts.

Trace: task055 measurement — 5,400 bytes per-channel vs 19,800 bytes pre-cast. See `exemplars/03_crop_and_f16/task055_per_channel.md`.

### 19. ReduceSum/Mean of BOOL is REJECTED by ORT (cast first)

```
ReduceSum(bool_tensor, axes=[k])   # ORT rejects at model load (INVALID_GRAPH)
```

Current ORT (1.24.4 under `ORT_DISABLE_ALL`) rejects `ReduceSum(BOOL)`, `ReduceMean(BOOL)`, **and** `ReduceSum(UINT8)`, `ReduceSum(INT8)`. **You must `Cast(BOOL → INT32/INT64/F16)` BEFORE the reduction.** F16 is the cheapest output dtype for ReduceSum from BOOL.

Note: at opset <=17, `ReduceMax(BOOL)` and `ReduceMin(BOOL)` are also rejected, but
`ReduceMax(UINT8/INT8)` and `ReduceMin(UINT8/INT8)` work at opset 13+. For BOOL→max/min under opset
13/17, cast to UINT8 (1 byte) — cheaper than F16. At opset 20, direct `ReduceMax(BOOL)` is
Kaggle-confirmed (task191 v6); `ReduceMin(BOOL)` remains unconfirmed.

### 19b. `ScatterElements`/`ScatterND` with `reduction='add'` rejects F16; `'max'` accepts it (2026-05-30, task018)

ORT (opset 16+) **rejects** an F16 scatter with `reduction='add'`, forcing an F32 data-in cast + F32 result (two full-size F32 casts). But **`reduction='max'` accepts F16**. When each scatter index is written **at most once with a non-negative value** (one-cell-per-color, one-write-per-slot, etc.), `max(0, v) == add`, so swapping `add`→`max` is behavior-identical on valid inputs and lets the whole scatter chain stay F16 — dropping both F32 casts (~−4.6K on task018). It is NOT byte-exact in general (differs only on the multi-write-per-index case), so gate on the generator's one-write invariant via ARC-GEN, not byte-equality. Reusable technique write-up in `techniques.md` "Path-B-only wins from the ARC-GEN generator §A".

### 20. Cross-task memory cascade (Kaggle scorer)

High-memory ONNX in slot N can ERROR slot N+1. Local validators cannot detect this — only Kaggle's eval pipeline manifests it.

Known trigger set: `{45, 129, 130, 171, 172, 210, 211, 220, 239, 240, 241, 242, 256, 257, 258, 259, 260, 261, 262, 309}`. **task110 may be a discovered new trigger** (cost reduction unblocked neighbors for +12.73 LB on 2026-05-20).

If your task is in this set, prefer aggressive memory reduction even if cost looks acceptable.

### 21. Pre-crop false positive on already-cropped models

**MERGED into §26 (2026-06-05 Phase-0)** — same trap, §26 carries the full mechanism, the
6 confirmed false-positive tasks (186/229/266/322/373/380), and the how-to-apply checklist.

### 22. Multi-mode Pad chains can't merge

`Pad(mode='reflect') → Pad(mode='constant')` cannot fold into a single Pad. The two modes have different boundary semantics. Documented dead-end on task240.

Don't waste cycles trying to merge multi-mode Pad chains.

---


## Doctrine blocks — axiom-grade rules (§VISIBLE-FAIL AXIOM, §DUAL-TRUTH, §GRADER-GAP, §CLASS-CHECK)

> **§CLASS-CHECK protocol (2026-06-04, formalized after 064/025): a 2-minute LIVE-truth run classifies any <100%-truth candidate's task and decides bet-vs-park.** Run `ng_gate truth --task N --cand <LIVE-extract> --n 5000`. LIVE < 100% ⇒ AMBIGUOUS/FILTERED-CLEAN class (LIVE earns full hidden credit despite corpus misses) ⇒ candidates in the same noise-band are viable bets (209 +2.76, 064 +0.99 both WON). LIVE == 100% ⇒ DETERMINISTIC class ⇒ a <100% candidate's misses are REAL bugs ⇒ park it (task145 doctrine; 025-k4 parked at 99.88%). Roster — filtered-clean/ambiguous: 209, 064, 018, 187, 157, **079**, **205**, **048** (k6 visible-floor bet at 99.374% truth HELD 2026-06-07 — LIVE missed 84/200k) (v31 +1.88, less-correct 99.89 vs 99.98 absorbed, BitwiseAnd grader-validated same submit); deterministic: 025, 145.
>
> **§CLASS-CHECK DEPTH RECORD (2026-06-06, task079 v36): the filtered-clean band absorbs at least 98.56% corpus — when visible is CLEAN.** v36 (1,382) read corpus 197,112/200,000 (2,888 misses — far below every prior winner: 209 at 99.2%, 064 at 99.80%, 157 at 99.66%) with visible 266/266, and scored FULL hidden credit (+2.87 EXACT → 6568.57). The visible-clean condition is what separates it from the certain-zero class: same round, the 263/266 siblings were skipped as axiom-certain zeros. Read: broad-corpus % is NOT a hidden predictor for filtered-clean tasks in EITHER direction. Visible 100% + LIVE-ambiguous ⇒ fire the bet regardless of corpus %.

> **§GRADER-GAP ROOT CAUSE (2026-06-04, grader-source read): `calculate_memory` = max(static, runtime) per tensor, profiled on REAL train+test+arc-gen inputs.** Static ⇒ local==Kaggle exact, no submit-probe can help; dynamic ⇒ real inputs inflate runtime shapes and max() charges them; symbolic dims ⇒ None (patched). The whole gap ledger reduces to this one line. (techniques §39 MECHANISM CONFIRMED.)

> **§GRADER-GAP UPGRADE (2026-06-04 night, task094 trilogy): dynamic-graph repricing CONVERGES to the true-trace cost — local numbers can be 78× fiction.** v7 (313 local) and v6 (451 local), both dynamic-shape, credited ≈24,386 and ≈19,182 — the same absolute band, regardless of local cost; the static v1 (3,806) landed EXACT and WON by +1.6. Rule: a dynamic graph's local cost predicts nothing; build the static sibling expecting to ship it; bracket-by-submit only when the dynamic route matters. (techniques §39 GAP-LEDGER UPGRADE has the full rules.)

> **§GRADER-GAP CONFIRMATIONS (2026-06-10, own-stack probes): the fiction multiplier varies, the verdict is binary, the probe decides.** Dynamic "1370" task132 repriced to ≈3486 (2.5× fiction, lost to static aditya 1821); task029 v18 dynamic 6796 repriced above its own v17 static floor 8481 (delta -0.09 → kept v17). BUT task091 v10 dynamic 2620 genuinely traced cheap and **WON** (+0.22 over its static floor 3274). So gate-5 (profile≠static) flags the RISK, not the verdict — fire one free own-stack probe and read the delta; the static sibling is the floor either way. Never agonize over a dynamic candidate, never ship one unprobed.

> **§GRADER-GAP (2026-06-04 night, task398 — local cost can be 37× wrong): the grader's trace, not `score_network_v2`, is the cost oracle for exotic graphs.** A scorer-dummy/boolpad/optional-input stack scored 364–519 locally and **≈13,300 on Kaggle, family-wide** (every ladder rung equally re-priced; byte-equal kept it safe, +1.10 banked vs +4.70 predicted). Combined with task014 (dynslice, 1.4× gap): treat local cost as a LOWER BOUND whenever the graph uses dummy branches, optional/default inputs, bool-Pad collapses, or data-dependent shapes — and ship the boring-static sibling in the same round. The fallback-ladder + fixed-base-probe pattern decodes the credited cost in one pass. (techniques §39-ESCALATION.)
>
> **§FREE-OUTPUT BOUNDARY LEAK (2026-06-07, task125): output-writing convs with POSITIVE off-center taps fire phantoms OUTSIDE the active grid.** Content at the grid edge sits inside the 3×3 window of an out-of-grid output cell → phantom fires → official decode mismatch (truth is all-zero out there). task125 v1 passed 204k corpus+visible and still failed 185/277 synthetic border-stress grids. **Negative off-center taps are safe (u8 saturation); center-only positive taps are safe; REFINEMENT (265, 2026-06-07): positive off-center taps are also safe IFF the tapped channel is structurally zero on its last row/col (e.g. an erode plane whose window reads pad there) — border-stress audit still mandatory.** Gate every free-output conv with a border-stress battery (500 synthetic grids, content touching edges) or corpus-certify the margin (§37 style) before staging.

> **§CLIP-DUP PHANTOM class (2026-06-07, task209 v6): shrinking a read plane changes Clip-overflow semantics.** On a full-grid plane, an overflowing clamped read (`Clip(idx,0,EDGE)`) lands on an EMPTY row/col; after cropping the plane tighter, the SAME Clip DUPLICATES the last real row/col — phantom duplicated guide cells made wrong hypotheses pass an exact moment test (4 visible fails). **Any plane-shrink must re-prove every Clip's overflow target is empty OR mask it** — fixes: mask gathered windows beyond the measured extent, or add the plane-edge bound to the validity predicate. Invisible on corpora whose geometry never reaches the edge (task209 needed h=20/t=2 pairs to expose it).

> **§FAMILY-LABEL MISNOMER (2026-06-07, task359 + task266): the DeepSeek family/transformations classification is a HINT, not the rule — decode the ARC-GEN generator; if they conflict, the generator wins.** THREE confirmed in batch-16 (15% of 20 tasks) where trusting the label would have wasted a whole worker session: task359 labeled "Symmetry Completion" is actually DENOISING (every pixel present-but-noisy; all geometry falsified 0/3000) — the symmetry label sends workers to mirror/rotation dead-ends with nothing to complete. task266 labeled "Filling Regions" is actually a fixed LINEAR DIAGONAL SHIFT (no flood at all) — the flood label invites the u8-flood-kit (#88) when a single diagonal-shift Conv is the answer. task038 labeled "Cropping/Object Detection" is actually count-blue-2x2-boxes -> unary 1x5 (no crop at all). **R1 BIG-FISH RECON CONFIRMS THE RATE (2026-06-08): 11 of 40 decoded big-fish tasks (27%) carry misnomer labels** (e.g. 371/362/046/343) — the generator-decode worklist (`Logs/research/R1_bigfish_worklist.csv`) is now the authoritative rule source for the big-fish tier; trust it over the family column. **RATE: 3 of 20 batch-16 labels were misnomers (15%) — the captain-generated technique hint off the label was wrong all 3 times; only the prompt templates "decode the generator FIRST" instruction saved them.** Captain prompt-tailoring off the label is the same trap (my 359 "symmetry" technique hint was wrong). RULE: every prompt says "the family label is a hint; read the generator source first; if it contradicts the label, the generator is truth." A misnomer is most likely when the cheap label-implied technique fails visible while a different decode passes.

> **§SCORER DUMMY-GRID CRASH (2026-06-07, task209): `score_network_v2`/`calculate_memory` run the model on a SYNTHETIC input — crash-proof every data-dependent index.** Garbage geometry (ArgMax of all-zero vector -> extent 29) drove ScatterND out of bounds -> score None. Clamp SEMANTIC scalars once at their source (`gh<=3`, scatter idx clips — identity on real inputs) instead of guarding every consumer; keeps real-input semantics byte-identical.

> **§FREE-OUTPUT ACTIVE-MASK (2026-06-07, task353 v1): same-size Equal-tail rewrites MUST sentinel the out-of-active-grid canvas cells when grid size VARIES.** Canvas value 0 decodes ch0=True — v1 scored officials 0/9 with every intermediate correct. Fix: `Where(act_b, values, 10u8)` with `act_b = Greater(Conv(input, ones[1,10,1,1], negpads), 0.5)` (~+870 B for a 14x12 canvas) — price it INTO the #81 crossover before converting. Two immune classes: **passthrough tails** (#78 Where-as-output — untouched cells come from input itself) and **LIVE-matches-blindly** (047-class: LIVE writes ch0 across the full fixed canvas, corpus never exercises the difference).
>
> **§GENERATOR-INVARIANT vs OFFICIALS-QUIRK doctrine (2026-06-07, task077): ARC-GEN `generate()` constraint loops only run on the `width is None` branch — official pairs are injected via fixed `colors` and can VIOLATE every generated invariant.** Decode the generator FIRST (it defines hidden), but gate every candidate on officials too; the officials' violations form a small enumerable quirk set that becomes explicit repair sub-rules (each re-proven harmless on the corpus).
>
> **§K2 ACT-NOW PAIR (2026-06-06, source-read v1.24.4): (a) `TopK(sorted=0)` order is UNSTABLE — ORT skips the final sort after `nth_element`; only the selected SET is contractual. Always set `sorted=1` (ties = lower index, source-confirmed). (b) Integer arithmetic follows native C++: `Mod(fmod=0)` sign=divisor, integer `Div` truncates toward zero (NOT floor), integer `Pow` casts `std::pow` output. Full compendium: `ort_reality.md` K2 section.**
>
> **§GRADER-GAP CRASH CLASS (2026-06-06, task061 v11/v12): dynamic graphs can CRASH outright, not just misprice.** Dynamic `Pad` (local 1,131) and dynamic `Mod` (local 141) both died at dynamic-buffer REUSE when ORT ran the REAL visible inputs — after scoring fine on dummy shapes. Before staging anything dynamic, run the actual officials end-to-end; a local score on synthetic shapes proves nothing about runtime viability. *Reproduction #3: task354 v11 dynamic-Pad shape-mismatch crash (2026-06-07).* (techniques §39 CRASH ADDENDUM; don't-do details in `task061.md`/`task354.md`.)

> **§VISIBLE-FAIL AXIOM UPGRADE (2026-06-04 night, 3rd proof — task363 probes):** the no_diamond probe passed **synth 200,000/200,000** and still scored the exact fail value from its ONE visible miss (264/265); siblings at 264/265 and 263/265 likewise. Combined with the task193 8-rung ladder and task145: **officials ⊆ hidden — one visible miss = certain 0, and NO generator/synth coverage overrides it. Treat as axiom; do not spend slots re-testing it** (owner-directed probes excepted). *Proofs #4–#5 (2026-06-06, task086 v17/v20 owner-directed probes, visible 115/266 + 52/266): both landed the exact fail value 6543.95 to the cent.*
>
> **§DUAL-TRUTH arithmetic shortcut (2026-06-04 night, task187 scanline, shipped +0.850 exact):** counts alone can certify zero-real-bugs without per-pair inspection — if over the SAME window `ab_divergences == (LIVE_GT_misses − candidate_GT_misses)` and candidate misses ⊆ LIVE misses (shared misses produce identical outputs, hence don't appear as divergences), then every divergence is LIVE-wrong→candidate-right ⇒ task279/377 class, fire. task187: 21 = 29 − 8.

> **§DUAL-TRUTH divergence attribution (2026-06-04, task279): an A/B divergence is not a verdict — attribute it.** Run `ng_gate truth` on BOTH the candidate AND the LIVE extract over the same window. task279 v3: ab 3,265/3,266 looked like a candidate defect; dual-truth read candidate 5000/5000 vs LIVE 4999/5000 ⇒ the divergence was a **LIVE GT-bug the candidate fixes** (task377 more-correct-than-LIVE class) — shipped at its exact prediction. Also: worker ab windows under-sample (the divergence sat past the worker's 2,266-pair window) — captains re-run ab deeper before firing. Counter-direction datapoint (task145, same day): 99.25–99.4% truth candidates on a DETERMINISTIC-class task failed hidden at exact fail-values — the task209 "99.2% held" exception applies ONLY to ambiguous filtered-clean tasks. Truth% near-misses are class-dependent; byte-equal or dual-truth-attributed candidates are not.

> **§VISIBLE-FAIL = CERTAIN ZERO; a probe ladder converts it into an exact bound (2026-06-04, task193).** A candidate that fails ANY official visible pair on a deterministic-rule task scores 0 on hidden with mathematical certainty — hidden is the same distribution and per-task scoring is all-or-nothing. Eight task193 sub-20 crop rungs (visible 96–219/266) all scored **exactly** the package-minus-task fail value (6441.92 ×8), simultaneously proving hidden contains >17×17 grids and closing the crop family permanently. Fire such rungs ONLY as deliberate distribution probes (keeps-highest makes them free) — never as hopes, and never count them as candidates. **Sibling discipline corollary:** when the primary IS byte-equal-vs-LIVE, a worse-cost sibling can never contribute under keeps-highest — skip it (task287 v1 6,468 skipped behind byte-equal v2 3,016). When the primary is NOT byte-equal, ONE different-architecture sibling is a cheap hedge (task244 v4 einsum fired behind v6 selector; both held).


## Submit-time traps

### 23. Sub-100% visible verify ships LB regression

7 of 8 sub-100%-visible-verify wildcards in the 2026-05-19 experiment regressed Kaggle by -13 to -17 LB each. Visible verify is a hard gate, not a soft filter.

Exception: tasks with <10 visible pairs (like task000 with 3 visible pairs). For those, verify is advisory; ship with documentation.

**§23a. HIDDEN-FRAGILE tasks (a logged LB-regression scar) → ship byte-equal to the PROVEN anchor, do not push sub-floor (task382, 2026-06-08).** Some tasks carry a history of hidden regressions (task382 has a −12.29 LB scar). On those, the correct floor is **byte-equal to a Kaggle-VALIDATED prior version** (task382 v22 is byte-equal to v9, a +2.17 exact-match Kaggle anchor — zero hidden risk), even if cheaper non-byte-equal reformulations look feasible. A sub-floor gamble on a known-dangerous task is bad risk/reward: the downside is the full task's points (~15) on a task that has *already* burned you. When `live_costs` note says hidden-fragile / LB-scar, ship the byte-equal compression of the proven anchor and move the worker on — the cheaper levers (here: input-crop +16000B, INT8-Where ORT-blocked) are documented-dead anyway. Contrast the task234 spread (same day): a NON-fragile task CAN take a non-byte-equal bet *if* paired with a byte-equal floor shipped last (the spread protects the downside). Fragile tasks don't even get the bet.

### 24. "Agree with LIVE" can be misleading when LIVE is wrong

Low agree-with-LIVE on adversarial inputs can mean LIVE is wrong, not that your candidate is wrong. Always compute `cand_correct` and `live_correct` separately to check.

task110 v4 (2026-05-12) — adversarial agree-with-LIVE was 8.6% on tile_corruption, but cand_correct > live_correct → candidate was better.

### 25. Calibrated LB ≠ raw log_e LB — split by submission regime

The old "blanket 0.48× multiplier" is misleading for single-task wildcards. Use this regime-split rule (verified across 2026-05-22 to 2026-05-23 ships):

| Regime | Multiplier | Empirical basis |
|---|---|---|
| Single-task wildcard | **1.0× (raw log_e)** | ~21 single-task ships matched raw prediction within ±0.05 LB |
| Small stack (2-7 tasks) | **~1.0× additive** | 3-stack (+6.16), 5-stack (+9.68), 7-stack all exact |
| Large megastack (100+ tasks) | **~0.48×** | megastack153 on 2026-05-20: predicted +54.20, actual +26.13 |
| Cascade unlock | step-function exception | Only confirmed: task110 v8 +12.73 LB |
| Zero-public-weight task | **0.00 actual regardless of cost** | 10 confirmed family members (`§14e`) |
| Hidden-cost-decoupled | **partial or zero** | task284 full decoupling; task234 ~22% partial |

**Practical rule**: for single-task and small-stack work, predict `Σ log_e(orig/new) × 1.0`. Apply 0.48× only when bundling 100+ tasks. Never apply ANY multiplier > 0 to zero-public-weight family tasks.

See `cost_floor_reference.md` "Calibration: local vs Kaggle".

### 26. `crop_to_size_advisory` false positive when only 30×30 tensor is the model output

`tools/spread_scan.py`'s `crop_to_size_advisory` counts `model.graph.value_info + model.graph.output` to detect `[...,30,30]` intermediates. For models that already Slice to a small active region on their first node (and end with a Pad back to 30×30), the ONLY 30×30 tensor detected is the model OUTPUT — which the ORT scorer EXCLUDES from activation memory.

**Why it's a false positive**: Pre-cropping input on these models adds an extra `input_cropped` activation without eliminating any large intermediate. Cost goes UP (~+372 for crop3×3, +492–+612 for larger crops).

**How to apply**:
1. Before attempting crop on a task with `crop_to_size_advisory` verdict, dump the graph and inspect `g.node`.
2. If the first node is a Slice that immediately extracts the small region, and no subsequent nodes produce 30×30 intermediates — skip. The advisory is a false positive.
3. Crop is genuinely useful only when the model has internal 30×30 computation (e.g., task117's `range30`/`c29i`/`c30i` reflection ops that propagate 30×30 throughout). Those tasks need initializer rewrites, not just a pre-crop Slice.

**Verified on GROUP T (2026-05-19)**: task186, task229, task266, task322, task373, task380 all verified M/M but cost increased.

Cross-ref `techniques.md` #2 (Crop-to-active-grid) for the cases where crop DOES work.

### 27. Lookup-family detection by init names ALONE is overtriggering — use table-size + adversarial-%

Hash-keyed initializer names (`hk_offsets`, `hk_values`, `mph_*`, `gate_*`) do NOT automatically signal lookup-family hidden-bad. Confirmed by 2026-05-23 task370 calibration: `task370_linear_valid_count1_active_hybrid_boolpad_sim` had `dzmp_hk_offsets4x9` + `dzmp_hk_values4x9_u8` + `dzmp_gate_priority16_u8` + `dzmp_valid_*16` initializers and shipped CLEAN at +2.52 LB on Kaggle.

**Refined detection rules** (use these instead of name-matching):

1. **Table size relative to visible pair count (266 for NeuroGolf)**:
   - Small structured tables (N < 50 entries) encoding RULE primitives → **SAFE**. Example: 4×9 = 36 = 4 sign-of-(dr,dc) × 9 indices.
   - Visible-pair-sized tables (N ≈ 200–300) indexed by input-statistic hash → **HIDDEN-BAD**. Examples: task077 lookup8 (266 templates), task396 count_hash (266 buckets), task370 mph_w20 (266 mph entries).

2. **Adversarial agreement**: < 1% on randomized inputs is the strong memorization signal. 0/100 (mph_w20) = -13 LB hidden-bad. 91.4% (task370_linear_valid_count1) = clean ship.

3. **Cost ratio**: Reductions to fractions of expected algorithm-rewrite floor are suspicious but not conclusive — task370_linear_valid_count1 at 12,656 (8% of LIVE 157,750) shipped clean.

**Dominant predictive signal**: adversarial %, NOT initializer names. If 266/266 verify + >30% adversarial, ship even if init names look hash-keyed — provided table sizes are small (< 50 entries).

**Generalized signature (post-2026-05-23T17:42Z)**: zero-weight family is NOT restricted to `[N=266, K]` explicit lookup tables. **Trained NN/MLP/CAE/Conv-distilled architectures fit to the visible set share the same zero-weight outcome** — memorization is via gradient-descent weights instead of explicit indices, but the Kaggle hidden-test outcome is identical (0.00 LB). Cherry13b Identity stub is the universal filter — submit it; if Kaggle returns same score, task has zero public weight.

**Major overturn (2026-05-23T18:33Z)**: the "lookup variant scores same as LIVE → task has zero public weight" diagnostic was WRONG. `task066_wallpath_rule.onnx` (TRUE structural L-path rewrite, 11,769 cost, NO 266-entry tables) shipped at **+15.62 LB** despite earlier calibration showing 0.00 LB on a lookup variant. When BOTH LIVE and a lookup candidate produce memorized outputs on hidden inputs, they score identically — but neither captures the rule. A true structural rewrite that produces DIFFERENT (correct) outputs on hidden unlocks the latent weight.

**Confirmed pattern (now 13 unlocks)**: task066 (+15.62), task277 (+15.27), task076 (+15.00), task101 (+15.19), task118 (+15.51), task158 (+14.56), task367 v2 (+14.24), task096 (+15.83), task363 v2 (+14.31), **task133 (+11.57)**, **task324 (+13.90)**, **task182 (+14.62)**, **task219 (+15.94, 2026-05-26T18:07Z)**. Tasks where LIVE has large `[N≈266, K]` initializers may be hiding +5-15 LB of latent public weight. The cherry13b Identity submission is the safe disambiguator before committing worker hours to a structural rewrite.

**CALIBRATION CORRECTION 2026-05-26T18:07Z**: cherry13b zero-weight tasks contribute **0 LB on hidden** (NOT 1.0 floor). The asymmetric ship profile captures the FULL cost-points of the new candidate, not (new - 1.0). Empirically verified across task133/task324/task182/task219 unlocks landing within ±0.01 of `25 - log_e(new_cost)` predicted. The `max(1.0, ...)` formula floor applies to local prediction; actual scoring of failed tasks on Kaggle returns 0.

### 27g-supplement. Self-consistency trap is real — synth-passing candidates can still score 0 LB on hidden

**Confirmed 2026-05-26T18:23Z** via task255 `fullsig_pairpack_constkey_u8` ship at cost 11,323:
- Worker reported synth gates passing on bg-gated version
- 265/265 visible verify
- Asymmetric profile: canonical zero-weight (per §27g), so downside 0 LB
- Shipped → **0 LB delta** on hidden

*…full trace + incident detail: `traps_appendix.md` → "27g-supplement. Self-consistency trap is real — synth-passing candidates can still score 0 LB on hidden"*

### 27f. task219 canonical artifact mystery (2026-05-25) — earlier MLP-decode work was on the wrong artifact

The §28 hash-MLP work documented for "task219 LIVE" (21→264→52→150 RBF MLP) was based on a probe of an artifact that **is NOT the current canonical task219.onnx**. The hash-MLP scan of canonical 6150.31 finds task219 as: **4,810 nodes, cost 10,844, no MatMul, no RBF, structural** with op signature `Slice→Cast→Reshape→Einsum→Cast→Greater→Cast→ArgMax→Cast→Mul`. This is a genuine structural model, not a hash-MLP.

**Implications**:
- task219 canonical scoring at cost 10,844 contributes ~15.71 pts if it generalizes hidden (cherry13b bulk 2026-05-23 confirmed it tested zero-weight at the time it was tested, but the artifact may have been replaced since)
- The earlier worker's E1 MLP-decode investigation was useful empirically (it produced the §28 hash-MLP fingerprint that just located task157/task255), but the task-specific conclusion ("task219 LIVE is a perfect hash") applies to whatever artifact they probed — possibly a local prototype, possibly an earlier canonical, possibly the original LIVE before any ship.
- task219's 4,810 structural nodes is a major compression target if it has hidden weight. The cost is already low (10,844), but the node count suggests significant optimization headroom via §17/§28-techniques.md patterns.

### 27b. Einsum `bchw,bxhw->bc` is count-ArgMax; `ReduceMax(color_index * mask)` is index-ArgMax — they diverge on ties

**Discovered 2026-05-25 on task324 v5**: Einsum-based mc-detection collapsed `color_3d` + `color_f16` + 2x `mc_mul` intermediates (~5.5k bytes saved) but caused a v4-vs-v5 divergence of 75/100 on column-equal-count adversarial inputs and 179/500 on column-unequal-count. Worker correctly caught this with intra-version A/B and shipped v5b (no Einsum) instead, which then unlocked +13.90 LB.

**Root cause**: the two operations have different tiebreak semantics.
- `ReduceMax(color_index * mask, axis=C)` returns the index of the LAST matching channel (highest index wins on ties).
- `Einsum("bchw,bxhw->bc", mask, color_index_expanded)` returns the COUNT-weighted ArgMax: the color with the most "1" cells across the spatial sum.

They produce the same result when one color clearly dominates the mask, but DIVERGE when two colors land in the same majority-class bucket via downstream voting (typical on column-striped inputs where the 4-neighborhood vote returns 0 / ambiguous for cells flanked by opposite-color columns).

**Rule**: if your downstream uses majority-voting on the resulting class index, the count-ArgMax (Einsum) is SEMANTICALLY MORE CORRECT but DEVIATES from index-ArgMax baselines. If the original LIVE algorithm uses `ReduceMax(index*mask)`, your Einsum replacement is NOT algebraically equivalent — it's a different algorithm that happens to agree on row-striped inputs but disagrees on column-striped or mixed-symmetry inputs.

**How to apply**:
- Before substituting `Einsum(bchw,bxhw->bc)` for `ReduceMax(color_index * one_hot_mask)`, run intra-version A/B including **column-striped + tie-bucket** adversarial inputs.
- If A/B diverges, KEEP the ReduceMax form even if it costs ~5k bytes more. The 5k cost is cheaper than a -10+ LB hidden regression.
- task324 v5b was the corrected ship: Slice-then-Cast + plaid + majority simplifications WITHOUT the Einsum substitution. Cost 66,092 instead of v5's 60,528 — 5.5k more cost, but it shipped clean.

Cross-ref: this was caught by intra-version A/B with 1600 mixed adversarial inputs. 0/267 visible verify did NOT catch it — visible inputs lack the column-tie pattern.

### 27c. `synth_verify.py` may be hardcoded to a specific task or version — verify before trusting

**Discovered 2026-05-25 on task110**: `tools/synth_verify.py` (since deleted) had a hardcoded reference to `task110_v16.onnx`, so later workers running it against v23/v25/v28/v29 saw "1160/1200 synth gate PASS" — the script wasn't testing the candidate AT ALL, regardless of `--candidate` args. v28 then shipped on this false-pass signal and regressed -12.80 LB.

**Rule**: any synth-verify tool in `tools/` or `task<NNN>/prototypes/` must be inspected for hardcoded paths or task IDs BEFORE you trust its output.

**How to apply**:
- For any candidate before ship, inspect the synth-verify script you're using. Confirm `--candidate` arg is actually consumed via `argparse` AND used in `onnx.load(args.candidate)` etc.
- Run an inline reference comparison as a sanity check: load candidate + load known-good baseline + compare outputs on 100 inputs. If your synth tool says "PASS" but inline says "DIVERGE", trust inline.
- Add a `# verified: <date> <author>` comment header to any tool that's been audited so future workers can skip the inspection step.

Same root cause as §14e-2026-05-25T16:42Z (canonical-zip A/B baseline rule): TRUSTING TOOLING WITHOUT INSPECTING WHAT IT ACTUALLY TESTS.

### 27d. A/B methodology decision tree — when canonical-zip A/B vs intra-version A/B

**Synthesis of lessons from task110 (-12.80 LB ×2), task133 (+11.57), task324 v5 (caught bug) + v5b (+13.90)**:

| Situation | Correct A/B baseline | Reason |
|---|---|---|
| Compressing an algorithm currently scoring on hidden (positive cost-points, NOT zero-weight) | **Canonical zip extract** (§14e-2026-05-25T16:42Z) | Replacing a working algorithm with broken version = lose canonical's hidden contribution |
| Replacing a confirmed zero-weight canonical (cherry13b Identity = same score) with structural rewrite | **Prior structural version (v(N-1) vs v(N))** | Canonical is at floor; any working structural is an unlock; canonical-vs-candidate A/B WILL diverge by design (different algorithms) |

*…full trace + incident detail: `traps_appendix.md` → "27d. A/B methodology decision tree — when canonical-zip A/B vs intra-version A/B"*

### 27d-supplement. Random-grid A/B false alarms on TASK-SHAPED algorithms

**Discovered 2026-05-26 on task285 keyfirst20 ship**: canonical-zip A/B with random one-hot grids returned 998/1000 differ between canonical (19,538) and same-algorithm-family candidate (17,654). Both were keyfirst+countscore+twostage at different cost points. This LOOKED like the candidate was a different algorithm — but the candidate shipped at +0.10 LB EXACT calibration match, proving they're functionally equivalent on hidden.

**Root cause**: task285's algorithm operates on STRUCTURED inputs (specific anchor + propagation patterns). Random one-hot grids are out-of-distribution — both canonical and candidate handle them as "garbage" with different default behaviors. The 998/1000 random-grid divergence reflected OOD handling, not algorithm difference.

**Refined methodology**:

| Algorithm type | Canonical-zip A/B with random grids | Best test |
|---|---|---|
| General-purpose (task110 v40 / task182 v47 type) | Reliable — 0/1000 means equivalent | random grids |
| Task-shaped / structured-input (task285 type) | UNRELIABLE — high divergence rate even when algos are equivalent | visible verify (266/266 vs expected) + task-shaped synth |
| Replaces zero-weight canonical (task133/324/182/219 type) | Expected to diverge by design (different algorithm than canonical) | intra-version A/B (v(N-1) vs v(N)) + ≥99% synth on both proto/reference AND numpy/constructed |

**Decision rule update for task-shaped algorithms**: if random-grid A/B shows >50% divergence on a SAME-RULE compression candidate (worker verified 266/266 visible AND proto-vs-LIVE matches), don't auto-reject. The right gate is:
1. 266/266 visible verify (both algos)
2. Worker's task-shaped synth shows convergent behavior (>95% match on task285-shaped inputs)
3. Cost direction is correct (compression, not regression)

The +0.10 LB exact-match landing of task285 keyfirst20 confirms this interpretation.

**Recognition signal for task-shaped algorithms**: graph has structural assumptions about input layout (e.g., specific anchor positions, propagation directions, color-role assignments). When you feed random one-hot grids, the algorithm's invariants are violated. Different versions of the same algorithm will violate the invariants in different ways → high random-grid divergence is EXPECTED, not a bug.

### 28. Hash-MLP architectural fingerprint — if LIVE is an MLP, check whether it's a perfect hash before distilling

**Confirmed empirically 2026-05-25 on task219 LIVE** (21→264→52→150). The MLP is NOT a learned rule — it's a **perfect hash table** masquerading as a neural network.

**Architectural signature**:
- First hidden layer is `Squared → Less → Cast` (RBF-style), NOT `Add → ReLU`. Threshold typically ~0.01.
- Hidden1 width ≈ count of visible inputs (264 ≈ 266 for NeuroGolf).
- Exactly 1 detector fires per visible input (`min == max == mean == 1.0` for "detectors active per input").

*…full trace + incident detail: `traps_appendix.md` → "28. Hash-MLP architectural fingerprint — if LIVE is an MLP, check whether it's a perfect hash before distilling"*

### 29. `sanitize_model` rejects Cast(F16→INT32) — "InsertedPrecisionFreeCast" type conflict

**Confirmed 2026-05-25 (task018 v15 F16-forward-transform attempt).**

When using `ORT_DISABLE_ALL` graph optimization, `neurogolf_utils.sanitize_model` inserts "InsertedPrecisionFreeCast" nodes to keep F16 intermediates alive in the graph. If you have a `Cast(F16_tensor → INT32)` node, sanitize_model will:
1. Insert a `PrecisionFreeCast(F16 → F32)` BEFORE the existing Cast
2. But the existing Cast node still annotates its output as something derived from F16
3. ORT type-checker sees output `InsertedPrecisionFreeCast_safe_name_N` typed as `tensor(float)` while expected `tensor(float16)` → `Type Error`

*…full trace + incident detail: `traps_appendix.md` → "29. `sanitize_model` rejects Cast(F16→INT32) — "InsertedPrecisionFreeCast" type conflict"*

### 30. `adversarial_test.py` decoded with `isclose(1.0)` — non-1.0-valued outputs scored 0/all (FIXED 2026-06-06, task055)

The advisory adversarial tool's `decode()` looked for cells `np.isclose(vec, 1.0)`, while the
REAL decoder (`neurogolf_utils.run_network`) thresholds `(result > 0.0)`. Consequence: any
candidate whose set-cells aren't ~1.0 — U8 bitmask outputs (task055 v9/v11 values 3..146),
F16 staircase outputs with values 2 — read as **0/all adversarial** despite being 200k
ab-byte-equal under the true pipeline. BOOL outputs passed only because
`isclose(True, 1.0)` happens to hold. Fixed in place to `vec > 0.0` (matches run_network;
multi-hot → 11 sentinel logic unchanged). If you see an adversarial 0/all on a candidate
that passed visible/truth/ab, suspect the TOOL before the model — and remember the same
file's `encode()` still pads with ch0=1 (differs from the validator; documented in core.md).

### "Fuse the N Convs into one multi-output Conv" when they are a SEQUENTIAL chain (task364, 2026-05-30)

A Path-A prompt promised a huge win by fusing "22 Convs (a per-color detector bank)" into one multi-output Conv, targeting a ~2k floor. **All wrong** — it was a graph-diag op-histogram misread (`Conv:22` read as 22 parallel detectors):
- 14 of the Convs were a **sequential geodesic flood** (`Conv(cross_kernel)→Min→Conv→Min…`, every Conv reading the *previous Min*, all reusing ONE shared kernel init). Sequential-dependent Convs are **unfusable** (a multi-output Conv needs every output channel to read the same input).
- The other 8 were 1-px / shift-2/3/4 kernels feeding live corner cues. Nothing was a droppable "dead detector" (cue chain fired on 237/266 pairs).
- Same family as the task201 "histogram counts read as tensor shapes" trap (techniques.md §2).

**Fix before trusting any "fuse / collapse the bank / crop" premise**: (1) run the per-tensor `calculate_memory` replica (`task364/prototypes/inspect_task201.py <onnx>`) to see where bytes actually live; (2) dump the node graph with real I/O names to see dataflow (parallel vs sequential); (3) instrument suspected-dead branches across all pairs before dropping. For task364 the real lever was a `[1,10,22,22] F32` crop (19,360 B) feeding only 2 channel-gathers → per-channel slice (−19,360, byte-equal). Then a *validated* plus→square relaxation unlocked UINT8 chains (→46,250, +0.617 LB). Sub-20k was unreachable byte-equal. See techniques.md "Win 1c" + `The_400_Tasks/Task_by_Task/task364.md`.

---

## How to use this file

Before starting any ONNX work, read sections 1-22 once (5 minutes). When you hit an error, search by symptom in `Ort_Advanced/ort_errors.md` first; if not found, scan this file's runtime/scoring sections.

## §CROP-CONTRACT (pointer)

The crop/shape-contract ship rules (when a sub-30 canvas is provably safe vs the task193 8-rung lesson) live in `The_400_Tasks/current_strategy.md` §ship-decision. INDEX advertises this anchor; this pointer closes the grep gap (VERIFY P11).

### 14z-b. Grader None = graceful per-task zero (REAL-SUBMIT confirmed 2026-06-05, task198 v17)

A model the local scorer rejects (None) was real-submitted: the submission COMPLETED, scored
exactly base-minus-that-task's-points (6528.92 = 6543.37 − 14.45). **A None-class task zeros
gracefully; the rest of the zip scores normally.** No whole-zip failure mode. (Probe cost one
slot; keeps-highest floored it.)

### §CLASS-CHECK extension (2026-06-05, task157): the filter absorbs BOTH directions

First less-correct-side bet won full credit: candidate 99.660% vs LIVE 99.735% on raw corpus
(−15/20k net) → hidden full marks. Roster: 209, 064, 018, 187, 157. The ambiguous-class rule is
now symmetric — candidates within the noise band ship regardless of which side of LIVE they
fall, provided visible is 100%. (Real-bug candidates still die: 025-k4's deterministic-class
99.88% remains the counter-case.)

### §INPUT-DTYPE-LOCK (2026-06-07, task359): the gate hard-casts input to float32

`tools/ng_harness/session.py::run_raw` feeds `x.astype(np.float32)` for EVERY gate
(visible/truth/ab). So the model input dtype is locked to FLOAT32 — a model declaring a
FLOAT16 (or other) input is rejected by ORT type-checking at every local gate and is
therefore untestable/unshippable, regardless of what the real scorer might accept.
**Consequence for histogram/reduction families:** `ReduceSum`/`ReduceMax` keep input dtype,
so any histogram reduced straight off the input is F32 and cannot be shrunk by declaring a
smaller-dtype input. Casting the full frame to a smaller dtype first costs the full
frame-size intermediate (e.g. `[1,10,30,30]`→F16 = 18000 B, →u8 = 9000 B) — almost always
more than the histogram bytes saved. task359's two F32 histograms (1200 B each) are a hard
floor for this reason. (The F16-input "halve the histogram" idea is dead; logged in
task359 don't-do.)

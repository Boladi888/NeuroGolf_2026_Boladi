# ort_opset_support.md — opset matrix for NeuroGolf

The project's `neurogolf_utils.py` declares `_IR_VERSION=10, _OPSET_IMPORTS=10` as defaults for `build_template_network`. **The validator does NOT enforce opset 10** — confirmed empirically by building opset-11 models that score correctly.

The actual constraint at scoring time is the banned-op list, regardless of opset version.

## Banned ops (validator rejects regardless of opset)

```
LOOP, SCAN, NONZERO, UNIQUE, SCRIPT, FUNCTION, COMPRESS
```

Source: `_EXCLUDED_OP_TYPES` in `neurogolf_utils.py`. These cannot appear in any submitted graph. CC analysis via Unique-based label compaction is permanently blocked.

## Opset 11+ ops verified working under the scorer

Confirmed via opset-bump tests (2026-05-12). All of these score correctly:

- **CumSum** — opset 11 (F32, F64, INT32, INT64). **F16 CumSum requires opset 14** — bumping from opset 11→14 drops the `Cast(F16→F32) → CumSum(F32) → Cast(F32→F16)` roundtrip, saving ~3,200B per CumSum block on NeuroGolf shapes. **Common falsified claim**: "F16 CumSum is unsafe if output feeds Gather as indices." Verify the Gather connection FIRST — `Gather(data, indices)` only cares about precision on the `indices` input (position 1), not `data` (position 0). F16 data + INT64 indices is exact. Empirical: task280 v1 saved -25,600B via opset 14 bump after debunking the wrong claim.
- **ScatterND** — opset 11. Scatter values into a tensor at index locations. task118 uses bool ScatterND (-38,341 bytes).
- **Round** — opset 11. Round half-to-even.
- **TopK with runtime K** — available from **opset 10** (not 11). `TopK-10` already makes K a required input; opset 11 only adds `largest`/`sorted` attributes. Verified at opset 10 under `ORT_DISABLE_ALL` 2026-05-23.
- **Equal expanded** — opset 11 added the full numeric set (UINT*/INT*/FLOAT16/FLOAT/DOUBLE). **BOOL has been in Equal since opset 1** — NOT added at opset 11. Opset 13 added BFLOAT16. Under opset 11+, `Equal(F32, F32)` works; at opset 10 floats are rejected (use INT32 cast).
- **GatherElements / ScatterElements** — opset 11.
- **BitShift** — opset 11.
- **DynamicQuantizeLinear** — opset 11.
- **Clip expanded** — opset 11 made min/max inputs (previously attributes).
- **Pad with input pads** — opset 11. Switches from attribute form to input form. Also accepts UINT8 input dtype.
- **OneHot** — opset 9 originally, multiple subsequent updates. Works fine; INT32 indices are allowed
  when the `depth` scalar is also INT32, and INT64 indices need INT64 depth (see `ort_quirks.md §3`).
- **ReduceMax(BOOL)** — **opset 20**. This is the first Kaggle-confirmed use of a direct BOOL reducer
  in this family: task191 v6 replaced 8 `Equal` + 6 `Or` membership plumbing with two broadcast
  `Equal` tensors and two `ReduceMax(BOOL)` nodes, scoring 6455.23 on 2026-06-02 (+0.29 over the
  current base). Because ReduceMax axes are an input from opset 18 onward, use the axes-input schema.
  At opset <=17, keep the old `Cast(BOOL, UINT8) -> ReduceMax(UINT8)` path.

## When to bump opset

Bumping `opset_import` from 10 → 11 (or higher) is occasionally a net win for cost. Examples:

- **opset 10 Pad-attr + Cast(bool→F32) → opset 11 Pad(input) + Cast(bool→UINT8)** — UINT8 is 1 byte/elem (vs F32 4 bytes), Pad-11+ supports it. Saved -742 bytes on task319. Cross-ref: `techniques.md` rule 6.
- **opset 10 float Equal → opset 11 numeric Equal**: opset 10 `Equal` accepts BOOL/INT32/INT64 only under ORT. BOOL Equal already returns BOOL at opset 10; bump to opset 11+ only when the inputs are F16/F32/F64 or other numeric dtypes rejected by opset 10.
- **CumSum-based algorithms**: replace a chain of cumulative-add ops with one CumSum. task382 used CumSum+Gather for geometric encoding (-73,443 bytes from-scratch rebuild).
- **BOOL membership reductions**: if the graph is already opset-bumped and the membership plane is
  naturally BOOL, opset 20 can remove the `Cast(BOOL, UINT8/F16)` before `ReduceMax`. This is not just
  locally valid: the Kaggle grader accepted task191 v6.

When you bump opset:
- Update `opset_import` for the default domain (`""` or `"ai.onnx"`).
- The IR version (`model.ir_version`) typically stays the same — IR is a separate concept from opset.
- Re-verify: some ops have semantics that changed between opsets (e.g., Pad's `pads` moving from attr to input, ReduceMax's `axes` becoming optional in opset 18+).

## ORT-CPU op × narrow-dtype matrix (2026-06-01, task286 v1 — empirically probed)

For shrinking a one-hot-input graph (the common NeuroGolf shape: `input` F32 [1,10,30,30]),
you want intermediates in UINT8/INT8 (1 B) not F16 (2 B) / F32 (4 B). What ORT CPU actually
accepts at `ORT_DISABLE_ALL` (the strict scoring mode):

| Op | UINT8 / INT8 | Notes |
|---|---|---|
| `Cast` | ✓ | F32↔UINT8↔INT8↔F16↔BOOL↔INT32 all OK |
| `Gather` | ✓ | data any dtype; **indices must be INT32/INT64** (traps.md §8c) |
| `ReduceMax` | ✓ (UINT8/INT8) | one-hot: `ReduceMax(x,axis=ch)` == "active" mask == `ReduceSum`, but stays 1 B |
| `Max` / `Min` | ✓ | UINT8 and INT8 both load |
| `Greater` / `Less` / `Equal` | ✓ → BOOL | the **AND-NOT trick**: `valid AND NOT wall` (0/1 masks) = `Greater(valid_u8, wall_u8)` in ONE op — replaces a 7-op BOOL `Cast/Not/And` chain |
| `MaxPool` | ✓ **opset ≥ 12** | rejected at opset 10/11; **BOOL always rejected** — 1-byte floor for a flood/BFS is INT8/UINT8 |
| `ConvInteger` | ✓ (UINT8 in → INT32 out) | weighted channel-sum (e.g. color-index `[0..9]`) without an F16 input cast; output is INT32 (4 B), narrow it immediately |
| `Mul` / `Add` | ✗ at opset ≤ 13, ✓ **at opset ≥ 14** | the lever for INT8 arithmetic chains; colors/small-ints stay exact |
| `Sub` | ✗ UINT8 **and** INT8 (all opsets) | use `Greater`/`Max` reformulation instead (see AND-NOT trick) |
| `Where` | ✗ INT8 data in ORT CPU | condition BOOL is fine; **the data branches must be F16/F32** — paint/select tails stay F16 |

### Reusable pattern: "UINT8-cast Phase A" for one-hot inputs
`Cast(input, F16)` [10-ch = 18,000 B on 30×30] is usually the single biggest non-algorithmic
tensor. Halve it: `Cast(input, UINT8)` [9,000] and route consumers through the int-friendly
ops above —
- per-cell **color index** → `ConvInteger(xu8, [0..9]_u8)` → INT32 → narrow,
- **active** mask → `ReduceMax(xu8, axis=ch)` (UINT8),
- **channel** picks (bg/wall) → `Gather(xu8, [k])` (UINT8),
- **mask algebra** (`a AND NOT b`, `a OR b`) → `Greater`/`Max` (UINT8→BOOL), never `Sub`/`Mul`.
Caveat: **initializer params are counted by element count, not dtype** (`calculate_params`
sums `prod(dims)`), so DON'T keep both an F16 and an INT8 copy of a mask param — pick one
dtype and derive a BOOL view with `Cast`. task286 v1: 119,906 → 107,150 byte-equal (+0.1125 LB).

## What stays out of scope

Opset 11+ enables some new ops, but the banned list still applies. So:

- **Loops / conditional branching**: blocked. LOOP, SCAN, COMPRESS all banned. No control flow.
- **Functions / custom subgraphs**: blocked. `model.functions` must be empty; nodes cannot have `AttributeProto.GRAPH` or `GRAPHS`.
- **NonZero**: blocked. Cannot extract non-zero indices. Use alternative approaches (ReduceSum + Greater, etc.).
- **Unique**: blocked. Cannot deduplicate values. Label compaction in CC analysis requires alternative.

## Practical guidance

Most NeuroGolf models historically built against opset 10. The validator accepts opset 11+ silently.
As of 2026-06-02, the highest publicly confirmed Kaggle-scored opset is **20** (task191 v6,
`ReduceMax(BOOL)`). Bumping is safe as long as:

1. You don't introduce a banned op.
2. You update any ops whose representation changed (e.g., Pad attr→input form).
3. You run the full verify gate after the bump.

`tools/spread_patterns/pass_opset_bump_pad_uint8.py` is the only auto-detector that bumps opset. Other patterns leave opset_import alone; if you write a new pattern that needs opset 11+ ops, document the opset bump in its docstring and add a guard so it only fires on already-opset-11+ models OR explicitly bumps and re-validates.

# symmetry_and_affine.md — dihedral transforms, symmetry detection, and affine warps

Cheapest ONNX patterns for the 8 dihedral symmetries (4 reflections + 4 rotations), symmetry-flag detection on data, scaling via `Resize`, and arbitrary affine warps via `GridSample` — all for `[1, C, H, W]` tensors under opset ≤18, ORT_DISABLE_ALL.

Distilled from Card A1 research output (originally `Todo/deep-research-report.md`).

---

## Axis conventions (used throughout)

- **H reflection** (top↔bottom) = reverse axis 2.
- **V reflection** (left↔right) = reverse axis 3.
- **Main-diagonal reflection** = swap axes 2 and 3 (`Transpose([0,1,3,2])`). Shape-preserving only when H=W.
- **Anti-diagonal reflection** = main-diagonal + reverse both spatial. Shape-preserving only when H=W.
- **90° CW** = `Transpose([0,1,3,2])` then reverse axis 3 (post-transpose W). Outputs `[1, C, W, H]`.
- **90° CCW** = `Transpose([0,1,3,2])` then reverse axis 2 (post-transpose H). Outputs `[1, C, W, H]`.
- **180°** = reverse both axes 2 and 3 in one Slice.
- **270° CW** = 90° CCW (and vice versa).

Always specify CW vs CCW — both are valid 90° rotations but produce different outputs.

---

## Primitive cheat sheet

| Op | What it does here | Opset min | Cost properties |
|---|---|---:|---|
| `Slice(steps=[-1])` | Reverse along one or more axes | 10 (attribute form), 11 (input form with `INT_MIN` sentinel for dynamic) | 0 params + small INT64 inits; 0 MACs; output is final-rank |
| `Transpose(perm=...)` | Swap axes; `perm` is an attribute (not counted) | 1 | **0 params, 0 init bytes**; 1 intermediate of size C·H·W·b |
| `Gather(axis=k, indices=reversed)` | Pick slices using a reversed index tensor | 1 (`indices` was attr in early opsets), 11 (clean input form) | indices cost = L × 8 bytes (INT64); only competitive for tiny L |
| `Resize` | Spatial scaling with `nearest`/`linear`/`cubic` | 11 (modern form), 18 (`axes` attribute) | ~4 init bytes for scales; 0 MACs for nearest |
| `GridSample` | Coordinate-based remap | 16 | grid is `2·H_out·W_out` F32 elements (constant or generated) |
| `AffineGrid` | Generate sampling grid from 2×3 matrix | **20** | **NOT AVAILABLE** under opset ≤18 ceiling |

---

## Reflection patterns (Q1)

### Horizontal reflection (reverse axis 2)

**Dynamic-friendly winner: `Slice` with negative step.** Opset 11 form uses `INT_MIN` sentinel for `ends` so it works regardless of axis length:

```python
# H flip: y[..., i, j] = x[..., H-1-i, j]
y = Slice(x, starts=[-1], ends=[INT64_MIN], axes=[2], steps=[-1])
```

Cost: 4 INT64 inits (32 bytes) + 0 intermediates (output is final). Total contribution: **32 bytes**.

**Static tiny-axis niche: `Gather`.** Only when H < 4 — for H=3, `Gather(x, [2,1,0], axis=2)` costs 3 × 8 = 24 INT64 bytes for the index. Slice form costs 32 bytes. **Gather wins for H ≤ 3.** For H=30 (NeuroGolf default), Slice always wins.

**`Transpose + Slice`** is strictly worse: adds a CHWb intermediate.

### Vertical reflection (reverse axis 3)

Same as H — substitute `axes=[3]`. Same cost.

### Main-diagonal reflection

**Unbeatable winner: `Transpose(perm=[0,1,3,2])`.**

```python
y = Transpose(x, perm=[0, 1, 3, 2])
```

Cost: **0 params, 0 init bytes**. `perm` is an attribute (attributes don't count toward `param_elements`). Output is final — no intermediate either.

But: output shape is `[1, C, W, H]`. If you need to subsequently compare against the input (e.g., symmetry detection), you must gate with `H == W` because otherwise the shapes don't match for `Equal`.

### Anti-diagonal reflection

**Dynamic baseline: `Transpose + Slice(steps=[-1,-1], axes=[2,3])`.**

```python
t = Transpose(x, perm=[0, 1, 3, 2])
y = Slice(t, starts=[-1, -1], ends=[INT64_MIN, INT64_MIN], axes=[2, 3], steps=[-1, -1])
```

Cost: 1 intermediate (Transpose output) of `C·H·W·b` bytes + ~64 bytes of Slice inits.

For NeuroGolf `[1,10,30,30]` F32: intermediate = 36,000 bytes. BOOL: 9,000 bytes.

**Static-shape alternative: constant-grid `GridSample`.** Score = `2·H·W` F32 elements as param = 1,800 elements for 30×30. Beats Transpose+Slice by ~20× under F32 and ~5× under BOOL — **only if your BOOL chain can tolerate a Cast to F16/F32 around the GridSample** (GridSample data input is float-only — see GridSample section).

---

## Rotation patterns (Q2)

### 180° rotation

**Winner: single `Slice` reversing both axes.**

```python
y = Slice(x,
          starts=[-1, -1],
          ends=[INT64_MIN, INT64_MIN],
          axes=[2, 3],
          steps=[-1, -1])
```

Cost: 8 INT64 inits = 64 bytes. **No intermediate.** This is essentially always the winner.

### 90° CW / 90° CCW

**Dynamic-friendly baseline: `Transpose + Slice` on one post-transpose axis.**

```python
# 90° CW: y[i, j] = x[H-1-j, i]
t = Transpose(x, perm=[0, 1, 3, 2])                                    # → [1, C, W, H]
y = Slice(t, starts=[-1], ends=[INT64_MIN], axes=[3], steps=[-1])      # reverse post-transpose W

# 90° CCW: y[i, j] = x[j, W-1-i]
t = Transpose(x, perm=[0, 1, 3, 2])
y = Slice(t, starts=[-1], ends=[INT64_MIN], axes=[2], steps=[-1])      # reverse post-transpose H
```

Cost: 1 intermediate (`C·H·W·b` bytes) + ~32 bytes of inits.

Non-square: output shape is `[1, C, W, H]`. This means follow-on ops that need same-shape comparison against the input fail unless `H == W`.

**Static-shape alternative: constant-grid GridSample** — same analysis as anti-diagonal. ~20× cheaper for F32 fixed shape; needs Cast for BOOL chains.

### 270° = 90° in the other direction

270° CW ≡ 90° CCW. Pick one convention and document it; don't pay for both.

---

## Symmetry detection from data (Q3, Q4)

### Boolean-safe scaffold (works for any symmetry test)

```python
# Compare original to transformed; return True iff every element matches.
x_flat = Reshape(x, [-1])
y_flat = Reshape(y_transformed, [-1])
eq_bool = Equal(x_flat, y_flat)             # → BOOL
eq_u8   = Cast(eq_bool, UINT8)              # ReduceMin doesn't accept BOOL
all_eq  = ReduceMin(eq_u8, axes=[0], keepdims=0)  # scalar UINT8
flag    = Cast(all_eq, BOOL)                # scalar BOOL
```

Cost: ~16 bytes of axis/shape inits + intermediates (the flattened tensors and the UINT8 reduction temporary). For a [1,10,30,30] tensor: 2 flatten intermediates of 9000 bytes (BOOL) or 36000 bytes (F32) + tiny scalar.

**Why this scaffold:** `ReduceMin`/`ReduceMax`/`ReduceSum` reject BOOL under ORT (see `ort_reality.md §7`). UINT8 is the cheapest valid cast target for boolean reductions. F16 also works (2 bytes/elem) — pick UINT8 for byte minimization on large flattened tensors.

### Square gate (required for non-shape-preserving symmetries)

Main-diagonal, anti-diagonal, 90°, and 270° all produce `[1, C, W, H]` outputs. Comparing against a `[1, C, H, W]` input requires `H == W`:

```python
sh = Shape(x)                              # [4] INT64
h = Gather(sh, indices=[2], axis=0)        # scalar INT64
w = Gather(sh, indices=[3], axis=0)        # scalar INT64
square_flag = Equal(h, w)                  # scalar BOOL
final_flag = And(square_flag, symmetry_flag)
```

Without this gate, the model load fails on non-square inputs with a shape mismatch.

For NeuroGolf where H=W=30 is fixed in graph metadata, this gate is unnecessary — the shape is known statically. Skip the gate when you can.

### Rotation order classification (1, 2, or 4)

After detecting:
- `r180` = invariance under 180°
- `r90`  = invariance under 90° (square-gated)

The classification is purely logical (no extra graph nodes for the math):

```
order4 = r90
order2 = And(r180, Not(r90))
order1 = Not(r180)
```

Mathematical reason: applying 90° twice equals 180°, so `r90 ⇒ r180` (you don't need to test r180 separately when r90 is true). The implication holds at the boolean-flag level.

### Reflection-symmetry detection — the four cases

| Symmetry | Transform | Square required? | Notes |
|---|---|---|---|
| H-mirror | `Slice(steps=[-1], axes=[2])` | No | Most common ARC-style test |
| V-mirror | `Slice(steps=[-1], axes=[3])` | No | Equally common |
| Main-diagonal mirror | `Transpose([0,1,3,2])` | **Yes** | Square gate mandatory |
| Anti-diagonal mirror | `Transpose + Slice both axes` | **Yes** | Same |

---

## Resize (Q5)

See `techniques.md #44 (spatial catalog)` for the full Resize coverage. Recap:

- For **mask data** (BOOL or 0/1 integer), use `mode="nearest"`. Linear/cubic interpolation produces non-{0,1} values and breaks downstream BOOL chains.
- For **continuous data** (F16/F32), `mode="linear"` is acceptable. `mode="cubic"` is rarely worth the kernel cost.
- Use `coordinate_transformation_mode="asymmetric"` + `nearest_mode="floor"` for predictable integer-aligned replication.
- Opset 18 added `axes=[2,3]` so you can name the spatial dims directly instead of passing full-rank `scales=[1,1,sH,sW]`. Saves 2 init elements.

For pure rectangular up-replication (every pixel → k×k block), `Resize(scales=[1,1,k,k], mode=nearest)` is the right tool. For per-pixel replication along one axis only, `Tile` is simpler.

---

## GridSample (Q6) — the affine warp primitive

**Opset 16.** General coordinate-based remap. For 2D inputs:

```
GridSample(X, grid)
attributes:
  mode: "bilinear" (default) / "nearest" / "bicubic"
  padding_mode: "zeros" (default) / "border" / "reflection"
  align_corners: int (0 or 1)
X: [N, C, H_in, W_in]   — F16/F32/F64
grid: [N, H_out, W_out, 2]   — F32 with normalized coords in [-1, 1]
output: [N, C, H_out, W_out]   — same dtype as X
```

The grid is normalized: `grid[n, h, w] = (x, y)` where x and y are in `[-1, 1]`. The output at `(h, w)` is sampled from input position `(y * (H_in-1)/2 + (H_in-1)/2, x * ...)` (with `align_corners=1`).

### Why it's interesting for NeuroGolf

For a **fixed-shape** graph, a **constant** sampling grid can encode any of the 8 dihedral transforms in a single op:

- Param cost: `2 · H_out · W_out` F32 elements = **1800 elements** for 30×30 = 7200 bytes of init, **1800 params**.
- Intermediate: 0 (output is final).
- Total contribution under score formula: 1800.

vs. `Transpose + Slice` for anti-diagonal / 90°/270°:
- Intermediate: `C · H · W · b` bytes = **36000** (F32) or **9000** (BOOL) for NeuroGolf.
- Total: 36000 or 9000.

**For F32 input, constant-grid GridSample wins by ~20× under the score formula. For BOOL input, the comparison breaks down because GridSample doesn't accept BOOL — see gotchas.**

### Gotchas

1. **Float-only data input.** `X` must be F16/F32/F64. BOOL/INT chains need `Cast(BOOL→F16) → GridSample → Cast(F16→BOOL)` around it. The Casts add 2× the data-bytes worth of intermediates. **This often eats the entire score advantage for BOOL chains** — verify carefully.
2. **`align_corners` matters.** With `align_corners=1`, corner pixels map exactly. With `align_corners=0`, the sampling grid is offset by half a pixel. For exact dihedral transforms on grid-aligned data, `align_corners=1` is the right choice.
3. **`mode="nearest"`** for exact integer-position resamples (no interpolation artifacts). `bilinear` introduces float artifacts at non-integer positions — wrong for color-grid tasks.
4. **`padding_mode="zeros"`** — out-of-bounds samples return 0. For pure dihedral transforms with the right grid, no samples should be out-of-bounds, so this attr doesn't affect correctness.
5. **Generated (not constant) grids cost more.** If the grid must be computed from a runtime shape (`Shape → Range → ...`), the intermediates push the cost back up. **Constant grids are the only score win.**
6. **[NEEDS EMPIRICAL]** ORT 1.24's GridSample kernel support under `ORT_DISABLE_ALL`. NeuroGolf docs verified up to opset 14 (`ort_reality.md`); opset 16 is unverified territory.
7. **`AffineGrid` (opset 20) is unavailable** at the opset ≤18 ceiling. You can't generate a grid from a 2×3 affine matrix using the standard op; you'd have to hand-construct it via Range + Mul + Concat.

### Decision: when to use GridSample over Transpose+Slice

**Use GridSample when:**
- Shape is fully static (no runtime shape dependency).
- Input is already F16/F32 (no Cast round-trip needed).
- Score model dominates over runtime latency.
- You can verify it loads under `ORT_DISABLE_ALL` at opset 16+.

**Use Transpose+Slice (or just Slice for 180°) when:**
- Shape is dynamic.
- Data is BOOL or integer.
- You're not 100% sure opset 16 GridSample is accepted by the scorer.
- The 1800-byte grid init pushes you over the 1.44 MiB file limit somewhere.

---

## Decision table — full dihedral group

| Transform | Dynamic-friendly | Static-shape score-optimal (F32) | Static-shape score-optimal (BOOL) | Min opset |
|---|---|---|---|---:|
| Identity | (no-op; rewire) | (no-op) | (no-op) | — |
| H-flip | `Slice(axes=[2], steps=[-1])` | same | same | 11 |
| V-flip | `Slice(axes=[3], steps=[-1])` | same | same | 11 |
| Main-diagonal | `Transpose([0,1,3,2])` | same (0 cost) | same | 1 |
| Anti-diagonal | `Transpose + Slice` | constant-grid `GridSample` | `Transpose + Slice` (BOOL incompatible) | 11 dyn; 16 GridSample |
| 90° CW | `Transpose + Slice(axes=[3])` | constant-grid `GridSample` | `Transpose + Slice` | 11; 16 |
| 90° CCW | `Transpose + Slice(axes=[2])` | constant-grid `GridSample` | `Transpose + Slice` | 11; 16 |
| 180° | `Slice(axes=[2,3], steps=[-1,-1])` | same | same | 11 |

---

## Recommended code snippets

```python
# H flip (dynamic-safe form)
y = Slice(x,
          starts=[-1], ends=[INT64_MIN],
          axes=[2], steps=[-1])

# Main-diagonal reflection (zero-cost)
y = Transpose(x, perm=[0, 1, 3, 2])

# 180° rotation (single-node winner)
y = Slice(x,
          starts=[-1, -1], ends=[INT64_MIN, INT64_MIN],
          axes=[2, 3], steps=[-1, -1])

# Boolean-safe "is input symmetric?" flag
x_flat   = Reshape(x, [-1])
y_flat   = Reshape(y_transformed, [-1])
eq_bool  = Equal(x_flat, y_flat)
eq_u8    = Cast(eq_bool, UINT8)
all_eq   = ReduceMin(eq_u8, axes=[0], keepdims=0)
flag     = Cast(all_eq, BOOL)

# Constant-grid GridSample for fixed-shape anti-diagonal (F32 input only)
# grid_init: [1, 30, 30, 2] F32, precomputed offline so that
#   grid[h, w] = ((W-1-w) normalized, (H-1-h) normalized)
y = GridSample(x_f32, grid_init,
               mode="nearest",
               padding_mode="zeros",
               align_corners=1)
```

---

## Verification checklist

- [ ] Axis 2 = H, axis 3 = W everywhere (consistent with NeuroGolf NCHW).
- [ ] CW vs CCW explicitly stated for every 90°/270° rotation.
- [ ] Non-square handling: square-gate added for main-diagonal, anti-diagonal, 90°, 270° symmetry detection.
- [ ] Boolean reductions always preceded by `Cast(BOOL → UINT8)`.
- [ ] No `Where(bool, bool, bool)` — rewritten as `Or(And(c,a), And(Not(c),b))` (cross-ref `ort_reality.md §2`).
- [ ] GridSample data input is F16/F32/F64 (not BOOL/INT).
- [ ] `align_corners=1` for exact pixel-aligned dihedral transforms.
- [ ] If using opset 16+ (GridSample), verify ORT under `ORT_DISABLE_ALL` accepts it before relying on it in a submission.

---

## TODO — verify empirically

1. **GridSample at opset 16+ under `ORT_DISABLE_ALL`.** Probably works but unverified per `ort_reality.md`.
2. **Constant-grid GridSample vs `Transpose + Slice`** on a real NeuroGolf task — measure actual score delta.
3. **`Slice` with `INT_MIN` sentinel under opset 11.** Verify ORT accepts INT64 min value as the `ends` for full backward slicing.
4. **Cast round-trip cost for BOOL+GridSample chains.** Does `Cast(BOOL→F16) → GridSample → Cast(F16→BOOL)` actually beat `Transpose+Slice` on NeuroGolf scoring, or do the Casts eat the win?
5. **`AffineGrid` confirmed unavailable** at opset ≤18 — but check if a future opset bump to 20 would unlock anything worth the bump cost.

---

## Cross-references

- `techniques.md #44 (spatial catalog)` — `Resize`, `DepthToSpace`, `SpaceToDepth` (`ReverseSequence` is validator-banned — substring "Sequence"; use `Slice` negstep for flips).
- `zero_param_generators.md` — `Trilu`/`EyeLike`/`Range`/`ConstantOfShape` for the grid generation case.
- `op_cost_patterns.md (multi-axis salvage)` — `GatherND`/`ScatterND` if you need per-coordinate transforms instead of dense grid sampling.
- `ort_reality.md` §2 (`Where(bool,bool,bool)` rejection), §7 (BOOL reductions need Cast).
- `ort_reality.md` — banned ops + max-verified opset.
- `exemplars/09_periodicity_detection/task110_periodicity.md` — periodicity is related to symmetry; some patterns overlap.

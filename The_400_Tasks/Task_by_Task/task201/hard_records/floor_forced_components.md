# task201 — forced-component floor / WALL record (2026-07-06, Claude worker batch50)

**Live bar:** 2004 (manifest, sha `c6a590bc3ed1…`, owner Vamshi).
**Effective best (task-local staged):** 1786 → **now 1773** (this pass, byte-equal, staged
`task201_qlinearconvfinaldecode_cost1773`).
**Prompt target:** ~1400 (`floor_est`). **Verdict: WALL — 1400 unreachable by peephole.**

## Why ~1400 is stale
`floor_est ~1400` and the prompt's "kill the F16/F32 decode / dissolve color_rows F32
[1,1,13,30]=1560" narrative describe the **obsolete 4718/2004-era graph**. The current
graph (Vamshi's bit-coded template pipeline + QLinearConv-final, staged this batch line)
is a fundamentally tighter **einsum-reduction** architecture that already dissolved those
tensors. It never materialises any `[1,10,H,W]` plane — it reduces the raw input straight
to `[1,30]`/`[1,2]` via Einsum. Total memory is already 1524.

## Forced-component floor of the CURRENT graph (memory 1524 + params 249 = 1773)
| Component | Bytes | Why forced |
|---|---:|---|
| `row_codes` Einsum `[1,30]` FLOAT | 120 | Per-row bitmask, values to 8191 need ≥4B (F16 exact only to 2048); 30 rows = einsum output dim tied to the 30-tall input. Narrowing needs a Slice (adds a retained tensor) or an input crop `[1,10,13,13]`=6760 — both price ABOVE. See dead-route 1. |
| `features_u8` Concat `[1,2,7,8]` | 112 | 2 channels `(x, x²)` are REQUIRED for the QLinearConv quadratic one-hot `−2(x−a)²+1` that pads 7×8→30×30 for free (output not counted). 1 channel is linear → can't one-hot. See dead-route 2. |
| `RoiAlign` `[2,10,1,1]` FLOAT | 80 | Minimal dynamic 2-pixel color read (L, source-left). Gather-by-row alt = `[1,10,1,30]`=1200. See dead-route 3. |
| ~6× `[7,8]`/`[1,1,7,8]` render tensors | ~336 | The actual 7×8 output assembly (frame_values, label_map, color_code, color_code_sq, features halves, pad). Forced by output size. |
| `[1,13]` code-decode cluster (post-narrow) | ~65 | Greater/Equal/Cast/Where over the 13 candidate rows. |
| `[1,1,5,6]` template bit-unpack | ~90 | BitShift/Mod/Mul occupancy render. |
| ~55 scalar `[1]` rule tensors | ~220 | bbox / corner / orientation / left-power log2 arithmetic = the rule. |
| params (initializers) | 249 | selectors, coord_moments, col_powers, shifts, quant scalars, slice consts. |

There is no ~140–386 B block of pure waste. Sum of forced ≈ current cost.

## Dead routes (priced, do not re-explore)
1. **Crop the `[1,30]` row cluster to 13 rows fully / crop the input.** The `[1,30]`
   tensors are einsum-DIRECT outputs. `row_codes` itself must stay (Gather@node37 reads it
   with indices up to ~pattern_top+4). Slicing it adds `row_codes13[1,13]`=52 + slice
   params on TOP of the retained `row_codes[1,30]`=120. Measured: the *partial* narrow
   (decode-cluster only) nets **−13** (→1773, staged). Trying to also route Gather@37
   through the 13-wide slice risks OOB (`pattern_top+4` can exceed 12). An input crop
   `Slice(input)→[1,10,13,13]` F32 = **6760 B** — catastrophically worse (the whole point of
   this architecture is to never materialise such a plane). **The −13 partial is the floor
   of this lever.**
2. **Drop the `x²` feature channel (−56 B).** QLinearConv is linear in its inputs; a one-hot
   (1 at x==a else 0) is quadratic → needs `x²`. Alternatives: materialise `[1,10,7,8]`=560
   one-hot then pad (worse), or a `[1,1,30,30]`=900 index canvas + free-output Equal (this is
   the task-local `framecropmirror` candidate at 3015, far worse). **DEAD.**
3. **Replace RoiAlign (−80 B) with Gather.** Dynamic per-(r,c) read on `[1,10,30,30]` via
   Gather forces a `[1,10,1,30]`=1200 intermediate. RoiAlign `[2,10,1,1]`=80 is minimal. **DEAD.**

## One remaining hypothesis (Path-B, out of scope)
A color-INDEX representation that avoids the 30-wide per-row bit-code AND the 2-channel
QLinear feature — e.g. sampling the template colours directly and rendering the 7×8 index
without an intermediate one-hot — could in principle undercut, but it changes collision/
decode semantics (hidden-divergence risk) and must be gated vs ARC-GEN ground truth, not vs
LIVE. Not a PEEPHOLE move. Recommend NOT re-opening unless a from-scratch Path-B builder is
explicitly commissioned.

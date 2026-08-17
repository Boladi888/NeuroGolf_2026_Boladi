<!-- INTEGRATED 2026-06-05 -> techniques.md #55 (Phase-0 hotfix 5) -->
# DONE 2026-06-01 — task013 full rewrite: aspect-branch + Mod/Gather periodic fill + broadcast-And dual-orientation assembly (58,000 → 2,774, +3.04 LB, byte-equal to LIVE)

## Result

- Builder: `The_400_Tasks/Task_by_Task/task013/prototypes/build_task013_v1.py`
- Staged: `.hermes/board/staged_submits/task013_rewrite_modgather_t_006ff1dc_on_6442_86/`
- ONNX SHA256: `006FF1DCF5B7F2DF06461268E1527DD5E57AA81F7EEF3063C6EFEC1AB561EAC9`

Cost: 58,000 → **2,774** (memory 2,708 + params 66). Δ = **−55,226 B**, predicted **+3.04 LB** (raw log_e, single-task wildcard).

Validation: visible **267/267** vs LIVE; truth **200,000/200,000** vs ARC-GEN GT; **byte-equal vs LIVE 4267/4267 (0 divergences)**; 3000/3000 fresh ARC-GEN. A from-scratch Path-B rewrite that is *also* byte-equal to LIVE → Path-A-strength, 0 LB downside.

task013 rule (two non-bg anchors → two-colour stripe pattern; vertical stripes, transposed when `xpose`). Three reusable techniques came out of the rewrite.

## Technique 1 — aspect-ratio branch for transpose-symmetric generators (free `xpose` decode)

Many ARC-GEN tasks build a pattern in one orientation then optionally `transpose`. **Read the generator**: if its `width` and `height` ranges are *disjoint* (task013: `width∈[20,30]`, `height∈[6,12]`), then `xpose` is recoverable from the encoded grid's aspect alone — the array is always wider-than-tall or taller-than-wide, never close.

```
col_act = Greater(ReduceMax(input, axes=[1,2]), 0)   # [1,1,1,30]
row_act = Greater(ReduceMax(input, axes=[1,3]), 0)   # [1,1,30,1]
branch  = Greater(ReduceSum(cmaxf_f32), ReduceSum(rmaxf_f32))   # W > H, scalar BOOL
```

Cost ≈ 2 scalars. **No input-derived `Div`/`Mod`** for the branch → dodges the zero-grid SIGFPE rejection (traps §8d). Beats reverse-engineering a `dc`-vs-`dr` / stripe-count rule (which is *wrong* here — see the task013.md don't-do: the old `dc≤dr` rule missed 124/267).

## Technique 2 — runtime-period periodic fill via clamped `Mod` + `GatherElements`

For "tile a 1-D line at a data-dependent period `p`, alternating two markers" (stripes, lattices, repeats), do NOT enumerate distances (LIVE used 30 `Pad` + 32 `Slice`). Compute the period from the two anchor positions and fold the whole line through one gather:

```
pos   = Where(line>0, arange, 0)            # anchor positions on the [1,L] line
cmax  = ReduceMax(pos); psum = ReduceSum(pos)
cmin  = psum - cmax;    dc   = cmax - cmin           # 2 anchors: min = sum - max
m     = Max(2*dc, 1)                                  # CLAMP: zero-grid -> 1, no Mod-by-0 SIGFPE
cminS = Min(cmin, L-1)                                # CLAMP: zero-grid cmin huge -> in-bounds gather
src   = cminS + Mod(arange - cmin, m)                # phase fold (period 2*dc carries both colours)
vg    = Where(arange-cmin >= 0, GatherElements(line, src, axis=1), 0)   # bg left of first anchor
```

`GatherElements(line[1,L], src[1,L])` returns `line[src]` per position — since `line` is non-zero only at the two anchors, `vg` is the full alternating pattern in **O(1) ops, no enumeration, no dynamic Slice** (dynamic Slice → `score=None`, trap §8b). Indices are INT32 (the hard floor, §8c); keep the data line **U8** (`Gather`/`Where`/`Equal`/`Greater` all accept uint8, §8f → 1 B/elem vs 2 B F16).

## Technique 3 — broadcast-`And` dual-orientation assembly (NO `[1,1,30,30]` canvas)

When the `[1,10,30,30]` one-hot output is **rank-1 within each branch** (a colour line broadcast over the perpendicular axis, gated by the active grid), assemble it so the final op's output IS the free graph output and every intermediate keeps a size-1 spatial axis:

```
colpat = Equal(colvec[1,1,1,30], ids[1,10,1,1])   # [1,10,1,30]  (300 B, not 900)
rowpat = Equal(rowvec[1,1,30,1], ids)             # [1,10,30,1]
X = Or(colpat,           And(notbranch, col_act4))  # cols pat   | rows: col-active
Y = Or(And(branch, row_act4), rowpat)               # cols: row-active | rows pat
output = And(X, Y)        # [1,10,30,30]  <- graph output, FREE
```

Sentinel the off-branch line (`colvec = Where(branch & col_active, vg, 10)`) so its `Equal` is all-False — then the select is a plain `Or` (no suppressing `And`). This is **strictly cheaper than the colour-index canvas** for a runtime transpose branch:

| assembly | bytes |
|---|---|
| broadcast-And (4 BOOL planes `[1,10,1,30]`/`[1,10,30,1]`) | **1,200** |
| single-channel canvas `[1,1,30,30]` + Equal, with conditional transpose (3 planes) | 2,700 |
| conditional transpose of the `[1,10,30,30]` output | 9,000 |
| dynamic-shape Reshape to unify orientations | `score=None` (§8b) |

The win: a `[1,10,1,30]` line-pattern (300 B) is 3× cheaper than a `[1,1,30,30]` canvas (900 B); broadcasting the perpendicular axis at the *free* output never materialises the full canvas. Use whenever the per-cell output, within a branch, depends on only one spatial axis (stripes, row/col fills, axis-broadcast recolours).

## Floor note

2,774 ≈ the structural floor (~2,700). Sub-1k is NOT reachable: the 10-channel one-hot output × runtime `W>H` transpose branch forces the four BOOL one-hot planes (1,200 B), and the gather-index machinery (INT32) + input projections (F32, ORT-forced) are also irreducible. Full proof in `The_400_Tasks/Task_by_Task/task013.md` Cost analysis.

## Sorter notes

- `techniques.md` — add three rules: `aspect_ratio_branch` (disjoint width/height ranges → branch = `W>H`, no input-Div), `modgather_periodic_fill` (clamped Mod + GatherElements for runtime-period tiling; cross-ref §8b/§8d/§8c/§8f), `broadcast_and_dual_orientation` (rank-1-per-branch one-hot output → `And(X[1,10,1,30], Y[1,10,30,1])`, avoids the 900 B canvas; cheaper than canvas/transpose).
- `by_symptom.md` — new entries: "rule transposes between two orientations based on grid aspect" → Technique 1; "periodic/stripe pattern with runtime period (LIVE enumerates Pad+Slice distances)" → Technique 2; "one-hot output is a line broadcast over the perpendicular axis (every row/col identical within a branch)" → Technique 3.
- `cost_floor_reference.md` "full-canvas-plane floor taxonomy" — add task013 row: irreducible = 4× BOOL one-hot planes (1,200 B) because 10-ch output × runtime transpose branch; floor ≈ 2,700.

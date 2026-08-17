# DEAD ROUTE - task174 algebraic score selector without a zeroing predicate

**Recorded:** 2026-07-13 by `bo_ng_worker1`.

## Live Bar
- Live bar: 867 profile cost from `python tools/ng_current_bar.py --task 174 --json`.
- Live SHA-256: `64745efcf4d40c5adbe5ad6801204f6466cc835c535593eaa375f5e68834c6fa`.
- Task-local bar at time of this probe: 847 from `staged_candidates/task174_where_count_base64_nb_cost847/manifest.json`.

## Attempted Route
Avoid materializing `eqz BOOL[1,10]` by replacing `sf = Where(eqz, Ns, 0)` with an algebraic score such as `Ns - K*D` or `Ns/(1+K*D)`.

## Failed Gate
- No ONNX candidate was built; the route failed the pre-build official-corpus locator check.
- Exact failure on visible example 0:
  - `Ns-D` selected the correct color but made the row/col locator negative (`sr=-4629846.5`, `sc=-4637324.5`), yielding NaN crop starts instead of row 6, col 3.
  - `Ns/(1+D)` selected the correct color but leaked loser-color weights into the base-64 row/col sum, yielding row 2, col 2 instead of row 6, col 3.

## Why It Is Dead
The selector tensor feeds both `ArgMax` and the weighted row/column locator. A nonzero loser weight can be harmless for `ArgMax` but still moves the logarithmic locator, so the zeroing predicate is load-bearing.

## Remaining Hypothesis
Only a fused symmetry/presence expression that produces exact zeros for every non-winning color, without a new `[1,10]` predicate tensor, can remove the remaining `eqz` 10B.

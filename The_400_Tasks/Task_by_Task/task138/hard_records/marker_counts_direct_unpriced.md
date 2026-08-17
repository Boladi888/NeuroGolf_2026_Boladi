# task138 - DEAD ROUTE: direct marker_counts-to-F16 qdraw probe

**Recorded:** 2026-07-07 by `bo_ng_cdx_worker3`, task138 only.

## Live / bar at time of probe

- Authoritative live from `python tools/ng_current_bar.py --task 138 --json`: **5403**, owner Vamshi, sha256 `c077e34a5e6c47f2818e70d5337d672fd4d8962b36cd2de8d1be32d0eb67c97a`, manifest updated `2026-07-02T02:35:15Z`.
- Lower task-local staged bar before this pass: **5276**, `staged_candidates/task138_squeezevcol_cost5276/task138.onnx`, sha256 `bdda4d889ce179f6fd9fdf363ba3ad639494ce1fdd72c06a51ba65b9ccd05f80`.
- Current staged win after this pass: **5275**, `staged_candidates/task138_castmarkerbool_cost5275/task138.onnx`, sha256 `1ca239368989d2e343d62b02871987e40518a46048ea150c56a795336a4106ab`.

## Attempted route

Starting from the valid 5275 graph, remove `qdraw_b` entirely:

`Cast(marker_counts -> BOOL) -> qdraw_b -> Cast(qdraw_b -> F16) -> qdraw_h`

became:

`Cast(marker_counts -> F16) -> qdraw_h`

The hope was to drop the `[1,10]` BOOL intermediate and rely on decode positivity because `marker_counts` is nonnegative and only the marker color is positive.

## Exact failed gate

After clearing stale `value_info` and re-inferring shapes:

```text
python -c "from tools._score_v2 import score_network_v2; ... task138_marker_counts_direct.onnx ..."
None None

python tools/agent_helpers/profile_real_cost.py --task 138 --cand ...task138_marker_counts_direct.onnx --arcgen 10
FAIL: score_network returned None on real-input trace
```

Raw ORT can execute the model, but the official scorer path returns `(memory=None, params=348)`, so the graph is unpriced.

## Why this route is dead

Unpriced graphs are not valid candidates under the worker protocol. The apparent memory drop is not claimable because `score_network_v2`/`profile_real_cost` cannot produce a cost. Do not rescue this by leaving stale `value_info`; stale shapes would bill the removed `qdraw_b` anyway and lose the intended saving.

## One remaining hypothesis

Only the conservative `Greater(marker_counts, zero_f)` -> `Cast(marker_counts -> BOOL)` rewrite is safe here; it removes `zero_f` without deleting `qdraw_b` and stages at 5275. Further savings need a different priced representation for `qray_h` or a broader bounded positional-code rewrite.

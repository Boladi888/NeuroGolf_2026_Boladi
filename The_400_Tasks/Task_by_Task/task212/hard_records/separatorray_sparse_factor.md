# task212 hard record - separator-ray sparse/factor subset route

Worker: bo_ng_cdx2_worker2
Date: 2026-07-03

## Live bar used

- Source: `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv` refreshed on 2026-07-03, confirmed against `NG_OnnxlyFans_TaskStatus.xlsx`.
- Team-best owner: Vamshi.
- Cost/profile cost: 616/616.
- SHA-256: `ff7a2c2b9d49a698aedd992e7c72a3116382b90dd9c6628d2a6533c1379f28cd`.
- Local score/profile lines:
  - `score_v2`: `(0, 0, 616) 616`
  - `profile_real_cost`: `task212 task212.onnx: static=616 profile=616 (memory 0 + params 616; 265 real inputs)`
  - `visible`: `[visible] 265/265 (100.00%)  PASS`

## Attempted route

Sparse/factor shrink below the single-output `Einsum` bar. Existing task-local prototype
`prototypes/build_task212_sparse_probe.py` converts the nine dense initializers from the
616 graph into sparse initializers, producing
`prototypes/task212_sparse_probe.onnx`.

Theoretical sparse stored values would be 109 params, but the official checker/scorer
path does not accept this representation for the `Einsum` inputs.

## Exact failed gate

- `score_v2`: `None None`
- `visible`: `[visible] ERROR: InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid model. Node input 'safe_name_0' is not a graph input, initializer, or output of a previous node.`

Earlier task-local sidecar `prototypes/probe_manifest.json` also records:

- `checker`: `ShapeInferenceError: Einsum input 1 rank 0 does not match equation indices rank 2`
- `profile_real_cost`: `InvalidArgument: node input 'safe_name_0' is not a graph input, initializer, or output of a previous node`

## Why this route is dead

The 616 graph is a zero-memory direct-output `Einsum`; all cost is dense initializer
element count. Sparse initializers would beat the bar only if ORT and the official
sanitizer/checker preserved them as valid ranked `Einsum` inputs. They do not.

The same sidecar records bounded dense-factor probes that failed to drop below 616:
dropping spatial modes, source features, or target features was infeasible on official
examples, and the shared tetrahedral color-basis rewrite was visible-zero.

Operational crop/prefix/scanline rewrites are not competitive with this bar unless they
delete the direct-output family entirely: any 10x10 crop or channel mask becomes a charged
intermediate, while the current graph has no charged intermediates.

## Remaining hypothesis

A real sub-616 win likely needs a new sign-coded dense factorization of the same rule,
not sparse initializers, subset rank drops, or a scanline rewrite. The next useful probe
is an independent sign-rank search that changes the factor family rather than dropping
one of Vamshi's existing row/source/target/channel factors.

# task118 hard record — cdx_worker2 probes against Vamshi 8,699

- **Live bar used:** 8,699, manifest authoritative via `python tools/ng_current_bar.py --task 118 --json`.
- **Live SHA-256:** `f3b7cfd77d55353ff522b7d172b7dd38a46ac5f3df585dcc1a837c8e96c37247`.
- **Live graph:** `Team_Best_Onnx/task118.onnx`, profile `static=8699 profile=8699 (memory 8575 + params 124; 267 real inputs)`.

## Dead route: crop shrink below 25x27

- **Attempt:** mutate current-best crop from 25x27 to 25x26 (`prototypes/t118_cdx_w26.onnx`) and adjust final pad.
- **Exact failed gate:** `python tools/ng_gate.py visible --task 118 --cand The_400_Tasks/Task_by_Task/task118/prototypes/t118_cdx_w26.onnx` -> `[visible] 265/267 (99.25%)  FAIL`.
- **Why dead:** visible expected outputs already paint row 24 and col 26 (`max_painted_row=24`, `max_painted_col=26` over 267 visible examples), so any smaller current-family crop drops real visible cells. Do not test narrower/tighter crop variants in this representation family.

## Dead route: delete local-max candidate filter

- **Attempt:** remove `Equal(score,pool)->is_max` and `Where(is_max,score,0)->cand`, feeding `score` directly to the cover QLinearConv (`prototypes/t118_cdx_no_max_filter.onnx`).
- **Exact failed gate:** `python tools/ng_gate.py visible --task 118 --cand The_400_Tasks/Task_by_Task/task118/prototypes/t118_cdx_no_max_filter.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.
- **Why dead:** non-center score responses are load-bearing false positives; the candidate-zeroing map cannot be deleted in this score formulation.

## Dead route: fixed cover stencil

- **Attempt:** remove dynamic `selK` (49 B) and always expand with one fixed cover stencil.
- **Exact failed gates:**
  - `cover_k2`: `[visible] 138/267 (51.69%)  FAIL`
  - `cover_k3`: `[visible] 137/267 (51.31%)  FAIL`
- **Why dead:** radius-specific expansion is load-bearing. Fixed radius 2 under-paints radius 3 cases; fixed radius 3 over-paints radius 2 cases.

## Dead route: global max instead of local maxpool

- **Attempt:** replace the `[21,23]` 11x11 MaxPool output with scalar `Where(useArm2,m2,m3)` to save 482 B (`prototypes/t118_cdx_global_max.onnx`).
- **Exact failed gate:** `python tools/ng_gate.py visible --task 118 --cand The_400_Tasks/Task_by_Task/task118/prototypes/t118_cdx_global_max.onnx` -> `[visible] 140/267 (52.43%)  FAIL`.
- **Why dead:** multi-cross examples need local maxima, not only the global maximum center.

## Dead route: static thresholded cover retune

- **Attempt:** bounded visible-corpus search for a retuned single static radius-3 stencil that acts as both score kernel and cover kernel, using a cover-scale threshold to suppress radius-2 outer-ring cells while keeping radius-3 cells.
- **Exact failed probe:** bounded search over K2 `(center,inner)` in `{(1,2),(2,3),(2,4),(3,5),(5,10),(10,20)}`, K3 `(center,inner,outer)` in `{(1,2,3),(1,3,4),(2,3,4),(1,4,5),(2,4,5),(2,5,6),(3,5,7),(5,10,20),(10,20,20)}`, and cover factor `{0.03..1.0}` found **no 267/267 visible hit**.
- **Why dead:** selected radius-2 and radius-3 score magnitudes overlap on visible (`r2 cand 40..160`, `r3 cand 60..220` in current graph), so a single magnitude threshold cannot reliably suppress only the radius-2 outer ring.

## Cost-neutral / no-op checks

- Native 5x5 K2 score kernel is not a free 24-param shave because the current 7x7 K2 initializer is also the selected cover stencil. Padding a 5x5 kernel back to 7x7 or adding a separate cover kernel costs at least the saved params.
- `tools/polish_candidate.py` on the current best, with temp/output forced under `task118/prototypes`, produced `final_cost=8699`; all passes were no-op or no-savings.

## One remaining hypothesis

The remaining sub-8,699 path probably has to avoid a current forced surface rather than delete one map: either a direct free-output placer that writes color 8 while preserving input passthrough without the 10-channel final `Where`, or a different score carrier that avoids the F32 channel-2 crop. Lookup/Cherry13b-style memorization remains off-limits for a reviewer-ready structural packet.

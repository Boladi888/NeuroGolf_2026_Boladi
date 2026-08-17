# task302 reviewer-locked spatial-code fold

- **Live bar**: 1574, Vamshi, sha256 `366237800121a80b62e22e725646ec0d3c7a6a56a5ec3dd71a57d350e6498528`, source `python tools/ng_current_bar.py --task 302 --json` / team manifest.
- **Effective reviewer bar**: 1573, `staged_candidates/task302_regionfillrewrite_cost1573`, sha256 `f39b26ef41e9d40cb7cfd1ecb511d289e57af66377cbb0e188ef87669b21937e`.
- **Locked family attempted**: `Slice ch5 F32 -> Cast U8 -> QLinearConv hidden_spatial_code -> QLinearConv render_from_hidden_spatial_code -> output`. Final render does not read `ch5_u8`; no `Concat(ch5_u8, fill8)`; no scalar `ch5+k*fill8` packing.
- **Synthetic corpus**: 3750 official+synthetic pairs from `build_task302_spatial_code_locked.py`: task JSON train/test/arc-gen plus bounded legal placements and generator-order overlap-skip attempts.

## Attempt 1: 5x5 hidden detector + 5x5 final render

- Predicted cost: 1161 (<1450).
- Hidden plane: one dense active-grid U8 spatial code from local signed 5x5 gray-mask weights, bias 96.
- Result:
  - `attempt k=5: render unique patches=43090`
  - `attempt k=5 failed: final one-channel render MILP failed k=5 channel=0: The problem is infeasible.`
- Verdict: dead before ONNX artifact/gates. A one-channel 5x5 render from this hidden spatial code cannot separate active background channel 0 from the other labels on the official+synthetic patch set.

## Attempt 2: 7x7 hidden detector + 7x7 final render

- Predicted cost: 1425 (<1450, so reviewer fallback condition was met).
- Hidden plane: same one-plane active-grid spatial-code construction with 7x7 local signed weights, bias 96.
- Result:
  - `attempt k=7: render unique patches=143802`
  - `attempt k=7 failed: final one-channel render MILP failed k=7 channel=0: The problem is infeasible.`
- Verdict: dead before ONNX artifact/gates. The wider below-1450 render still cannot linearly classify channel 0 from the hidden code.

## Remaining hypothesis

The locked two-QLinearConv family is walled at the channel-0 render step for the tested 5x5 and 7x7 spatial-code variants. A viable sub-1450 route would need a different one-plane hidden code that makes active background linearly separable without feeding `ch5_u8` into the final render; otherwise the current two-feature stack remains necessary.

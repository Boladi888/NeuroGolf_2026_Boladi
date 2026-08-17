# task386 separable-marginals route dead

- live bar: cost 78, sha256 `9c54eca017da2e3ce23451448e72ac42cbf846b2b13e96e2bad3781978a6cf9e`, source `python tools/ng_current_bar.py --task 386 --json` / team manifest.
- attempted route: replace the current 4x3 per-cell detector state with row/column/color marginals, per prompt route to separable moments.
- command: `python The_400_Tasks/Task_by_Task/task386/prototypes/probe_sub78_routes.py`
- exact failed gate/probe line: `fresh_pairs=400 unique_green_masks=287`; `mask_rank_hist={np.int64(0): 1, np.int64(1): 43, np.int64(2): 156, np.int64(3): 87}`; `row_col_marginal_collisions=3`.
- why dead: fresh ARC-GEN samples contain masks with identical row and column green counts but different 4x3 outputs, so row/column marginals do not determine the output. All 12 cells varied in 400 fresh rows, so the current `hit[1,1,4,3]` per-cell state is not carrying obvious dead cells.
- one remaining hypothesis: a deep final Einsum could only win if it uses fixed structural factors and no charged spatial selectors; here the right-panel column shift and the active 4x3 crop need either Conv attributes or expensive selector matrices.

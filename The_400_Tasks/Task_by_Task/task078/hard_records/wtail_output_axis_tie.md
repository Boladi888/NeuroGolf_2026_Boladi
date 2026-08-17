# task078 W-tail output-axis tie infeasibility

- Live bar used: 82, sha256 `54db7a9ffcc7777a96dc05b7d3a9d2f0dc81c7eb53dd57eeba37688fd03b287f` from `python tools/ng_current_bar.py --task 78 --json` (manifest, Vamshi, 2026-07-12T15:15:54Z).
- Locked route: keep one-node scorer-invisible final `Einsum` (#89), do not retry sparse/no-H/rank-1-Q; attack only `W[2,10]` by tying output axis `o` to a free input channel axis and shrinking dense `W` to `[1,10]`.
- Variants tried, one numeric fit each:
  - `task078_probe_wtail_count_o_sq_cost72.onnx`, equation `nirc,njsc,nosc,pi,pi,hj,qj,hq,qr->norc`; fit left 201 bad unique cases; gate `[visible] 0/266 (0.00%) FAIL`.
  - `task078_probe_wtail_count_o_signed_cost72.onnx`, equation `nirc,njsc,nosc,pi,hj,qj,hq,qr->norc`; fit left 200 bad unique cases; gate `[visible] 0/266 (0.00%) FAIL`.
  - `task078_probe_wtail_current_o_signed_cost72.onnx`, equation `nirc,njsc,norc,pi,hj,qj,hq,qr->norc`; fit left 107 bad unique cases; gate `[visible] 0/266 (0.00%) FAIL`.
  - `task078_probe_wtail_current_count_o_signed_cost72.onnx`, equation `nirc,njsc,norc,nosc,pi,hj,qj,hq,qr->norc`; fit left 107 bad unique cases; gate `[visible] 0/266 (0.00%) FAIL`.
- Why `W[2,10]` tail is load-bearing: tying `o` to `input[nosc]` gates channels 3-9 for free, but every nonzero output-color count shares the same one-term sign, so columns with multiple colors become multi-hot or all-off. Tying `o` to `input[norc]` can only copy the current cell color; it cannot emit color 2 into original zero cells after the column gravity move. The live `W[po]` supplies the signed channel map that separates `0/1/2` logits inside the same multiplicative final term; the 14 dense zero tail elements are the price of making channels 3-9 nonpositive in a full 10-channel ONNX output.
- One remaining hypothesis: only a non-multiplicative/additive construction could combine `current_o` and `count_o` gates without `W[po]`, but that would require an extra node or charged intermediate and is outside the locked one-node final-Einsum class.

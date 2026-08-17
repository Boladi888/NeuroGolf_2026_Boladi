# task321 no-colmask shared-L route

- live bar: 180, sha256 `5ce4cbdc653f2339cbdf91853e67c2f4f9dec1d83a993c730da26fe9ec148e68` (`python tools/ng_current_bar.py --task 321 --json`, manifest source, Vamshi, 2026-07-12T15:15:54Z)
- attempted route: remove the 30-param `C[30]` output-column mask from the current one-node `Einsum`, keeping shared `L[4,30]` and `Q[3,10]`; prototype `prototypes/task321_probe_einsum150_no_colmask_8C066FA1.onnx`
- exact failed gate: `python tools/ng_gate.py visible --task 321 --cand The_400_Tasks/Task_by_Task/task321/prototypes/task321_probe_einsum150_no_colmask_8C066FA1.onnx` -> `[visible] 0/267 (0.00%)  FAIL`
- why dead: without `C`, the shared `L` basis writes the same 4x4 result into panel columns 5..8 and 10..13, so the full 30x30 one-hot tensor is visibly wrong; one visible miss kills this representation family.
- related prebuild check: rank-2 shared channel embedding was tested as a max-margin inequality problem with `v0=(1,0)`; random search and scipy differential evolution/Nelder-Mead stayed below zero margin (best polished min margin about `-1.0015`), so replacing `Q[3,10]` with `Q[2,10]` is not a credible 170-byte candidate.
- one remaining hypothesis: a fundamentally different single-node contraction would need to remove or factor the 30-wide output-column mask without tying source-panel columns to output-panel columns; simple shared-basis, rank-2-channel, and product-count variants do not price below 180.

# task176 tied color/spatial basis dead route

- Live bar used: cost 140, sha256 `36dcb4901656db6303a277633faae8d4cfb63bbe189781b2f2d6851b4477a5ce` from `python tools/ng_current_bar.py --task 176 --json`.
- Attempted route: share a 3-vector spatial basis while tying the color rank `a` directly to the spatial rank in one `Einsum` (`bchw,ca,oa,aq,qh,ap,pw->bohw`), projected cost 122.
- Exact failed gate: `python tools/ng_gate.py visible --task 176 --cand The_400_Tasks/Task_by_Task/task176/prototypes/v_shared_spatial_basis122.onnx` returned `[visible] 0/25 (0.00%)  FAIL`.
- Why dead: the current 140 graph keeps color rank and spatial rank independent (`a` and `k` are separate sums). Tying them destroys the scalar spatial sign field used for both red pass-through and yellow/background split.
- One remaining hypothesis: independent color/spatial rank can still be compressed by sharing only the spatial row/column basis; that hypothesis produced the staged 88-cost candidate.

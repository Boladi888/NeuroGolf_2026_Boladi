# task257 hard record - one-plane scalar priority

Live bar: 198, sha256 `fc196a8a736cb89f21630c72e7c0f8268f392755268fd272c84a849f1c5dd9d0`
from `python tools/ng_current_bar.py --task 257 --json` (`manifest`, owner Vamshi).

Attempted route: one F32 compact plane from `ConvTranspose(input)` with priority
score `8*TL + 4*TR + 2*BL + BR`, then free-output `Conv` thresholds to channels
0/6/8/4/7. Prototype:
`The_400_Tasks/Task_by_Task/task257/prototypes/task257_oneplane_probe.onnx`.

Static score: `(macs=0, memory=64, params=60)`, cost 124.

Exact failed gate: `python tools/ng_gate.py visible --task 257 --cand The_400_Tasks/Task_by_Task/task257/prototypes/task257_oneplane_probe.onnx`
returned `[visible] 0/269 (0.00%)  FAIL`.

Why dead: a single scalar plus final affine thresholds can only isolate threshold
extremes; task257 has five nonempty mutually-exclusive classes `{0,4,6,7,8}`.
Middle-priority classes become nested/multi-hot or missing. Per visible-fail
axiom, do not run truth/ab/battery/profile on this family.

One remaining hypothesis: only a non-linear final decode from a scalar label could
work, but the needed U8/Equal/Pad or small one-hot crop path reprices above the
198 two-plane live graph.

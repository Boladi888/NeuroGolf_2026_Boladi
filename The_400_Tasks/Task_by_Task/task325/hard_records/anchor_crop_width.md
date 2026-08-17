live bar: manifest/team-best task325 cost 1658, sha256 962a051fea10a5c87401b078ce282ee1752ee088e8597640c6b04062e4c02049

attempted route: shrink Aditya anchor-detector ch8 occupancy Slice from [1,1,16,15] to [1,1,16,14], adjusting QLinearConv right pad to keep a 14-column anchor map.

failed gate: `python tools/ng_gate.py visible --task 325 --cand The_400_Tasks/Task_by_Task/task325/prototypes/task325_anchor_crop_w14.onnx` -> `[visible] 263/266 (98.87%)  FAIL`.

why dead: column 14/15 context is load-bearing for at least 3 visible/ARC-GEN cases; narrower occupancy crops are the same representation family and should not be retried.

remaining hypothesis: bypass the materialized occupancy read entirely with direct component-signature scalar contractions or a small legal QLinear/Conv anchor counter that outputs block scalars directly.

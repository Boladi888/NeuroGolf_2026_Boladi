# relation_le_only hard record

- Live/team-best bar: cost/profile 635, sha256 `bb14c2aac494c18167f7eab7cc1ac124e94634011c4d2e81da68a8acb1321c61`, source `Team_Submission/Backend/manifest.csv` and workbook row `task333`.
- Attempted route: remove the `not_equal` plane from `relation_feature`, shrinking it from `[2,10,10]` to `[1,10,10]`, and shrink `relation_decoder` from `[3,2,2]` to `[3,1,1]`.
- Probe artifact: `The_400_Tasks/Task_by_Task/task333/prototypes/task333_relation_le_only_probe.onnx`.
- Cost gate: `score_network_v2` `(0, 0, 526) 526`; profile `task333_relation_le_only_probe.onnx: static=526 profile=526 (memory 0 + params 526; 265 real inputs)`.
- Failed gate: visible `[visible] 0/265 (0.00%)  FAIL`.
- Why dead: the second relation plane is load-bearing. The current decoder uses `le` and `not_equal` together to synthesize `eq`, `>`, and `<`; one-plane variants cannot distinguish equality from strict direction and fail every visible official/arc-gen pair.
- One remaining hypothesis: a fundamentally different top-left/edge-green selector could avoid the two-plane relation decoder, but do not retry one-plane relation-feature variants.

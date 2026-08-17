live bar: 4074 profile_cost, sha256 234c8a210458b062aada6f4a1a9ea75c15a20501af71852b12c148838d23eda3.

Attempted route: replace `Slice(desc_u8 -> desc_center3_u8)` plus 3x3 MaxPool with one 5x5 MaxPool over `desc_u8`, deleting the center crop and its starts/ends params.

Cost if correct: `score_network_v2` `(0, 3960, 92)`, cost 4052. Named delta from `ng_graph_cost_diff`: removes `desc_center3_u8 [1,2,3,3]` (-18 B memory) plus `center_starts`/`center_ends` (-4 params).

Failed gate: `ng_gate visible --task 89 --cand ...task089_payload_full5_probe.onnx --json` -> `263/267 (98.5019%)`, FAIL.

Why dead: some visible descriptors contain a higher-color outer-ring cell that is not the source payload; only the center 3x3 payload read is valid. A corpus check found no smaller fixed rectangle than the full center 3x3 works across the 356 visible source records.

Remaining hypothesis: no payload-read shave unless the payload color is recovered before descriptor assembly or by a non-rectangular center3 max that costs less than the current 18 B slice.

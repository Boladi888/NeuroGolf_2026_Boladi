live bar: 4074 profile_cost, sha256 234c8a210458b062aada6f4a1a9ea75c15a20501af71852b12c148838d23eda3.

Attempted route: delete `desc_targets_u8` and `desc_u8` by feeding `desc_all_u8` directly into the payload-color filter and descriptor transform.

Cost if correct: `score_network_v2` `(0, 3878, 96)`, cost 3974. Named delta from `ng_graph_cost_diff`: removes `desc_targets_u8 [1,2,5,5]` and `desc_u8 [1,2,5,5]` (-100 B memory).

Failed gate: `ng_gate visible --task 89 --cand ...task089_no_desc_target_sub_probe.onnx --json` -> `69/267 (25.8427%)`, FAIL.

Why dead: target-marker 5x5 neighborhoods contaminate the all-marker descriptor; subtracting target descriptors is semantic, not cleanup.

Remaining hypothesis: a cheaper target subtraction would need to subtract in the 5x5 descriptor space directly; building a source mask in 13x13 space is byte-negative.

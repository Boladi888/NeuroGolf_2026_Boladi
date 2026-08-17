live bar: 3922, sha256 b5bcdca3c64c903e9c6279cf6d332563d92bff83e8df5a36601f789cdfa10235 (ng_current_bar.py manifest)

attempted route: replace TopK(9) plus the public-only tenth-cell repair with TopK(10), then remove the special repair writes.

exact failed gate: score/profile ladder stopped at static price. Prototype The_400_Tasks/Task_by_Task/task131/prototypes/task131_top10_probe.onnx scored memory=3269 params=674 cost=3943.

why dead: removing the repair deletes a few scalar tensors, but every green-coordinate vector grows from 9 to 10 elements while the final ScatterND write count stays 50. Net cost delta from ng_graph_cost_diff was +21 B, so it is not a team-best beat and did not merit visible/truth gates.

one remaining hypothesis: a safe way to reduce the final write count below 50 would matter; changing only the TopK width does not.

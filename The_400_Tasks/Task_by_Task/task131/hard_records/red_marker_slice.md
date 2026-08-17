live bar: 3922, sha256 b5bcdca3c64c903e9c6279cf6d332563d92bff83e8df5a36601f789cdfa10235 (ng_current_bar.py manifest)

attempted route: replace the 24-cell red marker GatherND table with first-row and first-column color-2 slices plus scalar ArgMaxes.

exact failed gate: initial score hit the scorer dummy-grid ScatterND OOB trap because all-zero input selected red_line=0. After adding the semantic clamp to 1, prototype The_400_Tasks/Task_by_Task/task131/prototypes/task131_redslice_probe.onnx scored memory=3323 params=628 cost=3951.

why dead: the route saves 46 params by deleting red_anchor_idx, but the two F32 slices and scalar support add 75 B memory. Net cost delta from ng_graph_cost_diff was +29 B, so the sparse 24-cell GatherND remains cheaper.

one remaining hypothesis: only a red-line locator that avoids both the 24-cell INT64 table and materialized F32 row/column slices could help; none is apparent without control flow or a dynamic index tensor that costs more than it saves.

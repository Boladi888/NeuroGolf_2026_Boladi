live bar: 3481 bytes, sha f727187c16c9d3caad99e7cbb71827701f4d9573e2378adc325f354951148c36 (ng_current_bar manifest, 2026-07-02)

attempted route: replace the full 10x10 UINT8 flood seed `lbl_u8` (100 params) with a broadcast low-rank row/column label (10 params) while preserving the accepted five-round MaxPool flood and QLinearConv count tail.

failed gate: `python tools/ng_gate.py visible --task 277 --cand The_400_Tasks/Task_by_Task/task277/prototypes/task277_label_row_desc.onnx` -> `[visible] 188/266 (70.68%)  FAIL`.

why dead: row-only labels collide across separated components, so the propagated component ids merge distinct shapes and the unique-vs-duplicate area comparison miscolors visible examples. Per worker kill rule, sibling row/column broadcast variants were not gated after the first visible miss.

one remaining hypothesis: a generated label from row+column could preserve uniqueness with fewer params only if it avoids adding a charged 10x10 label intermediate; current ONNX Add/broadcast construction would add a 100B tensor and loses against the 100-param initializer.

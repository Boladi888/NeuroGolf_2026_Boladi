# task182 hard record: single-vector XOR

Live bar: 5448 from `python tools/ng_current_bar.py --task 182 --json`, authoritative source `manifest`, owner Vamshi, SHA-256 `536bb9422efecc3bc27ea505a42bc03a62366affd62a19b9c23f7f5abb8e98de`, updated 2026-07-02T02:35:15Z.

Attempted route: `prototypes/task182_singlevector_xor_probe.onnx` built from the 5448 graph. Kept frame localization, framed color read, blue row packing, and render/unpack tail; replaced the four c0-c3 condition blocks with `Gather(shape_idx)` row-template render from all blue center candidates plus one packed 20-row `BitwiseXor` against the blue row vector.

Priced result: `_score_v2` = `(0, 4476, 284)`, cost 4760; SHA-256 `542a8a48afc4e865b877f0935294695d4ddccac1cc58bb907c9b2e07147d790c`. Static graph diff vs local 5448 oracle reports net structural delta -684 by deleting the c0-c3 slice/shift/xor/and chain and `cond_packed`, while adding the row-template/hash LUTs and the four 20-row XOR/filter tensors.

Failed gate: `python tools/ng_gate.py visible --task 182 --cand The_400_Tasks/Task_by_Task/task182/prototypes/task182_singlevector_xor_probe.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.

Why dead: one full-row XOR loses attribution from mismatch cells back to candidate anchors. Rendering the selected shape from every blue center and XOR/filtering against the blue grid keeps or drops cells globally, but it cannot prove which center generated each mismatch. The four per-condition shifted row tests are not just byte scaffolding; they carry anchor-row/anchor-column attribution.

One remaining hypothesis: a different packed exact-row matcher could preserve attribution by computing row-window equality for all five selected shape rows, but that is no longer the single-vector XOR family and likely expands beyond the current four discriminating conditions.

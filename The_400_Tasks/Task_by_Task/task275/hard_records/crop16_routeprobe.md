# crop16_routeprobe dead route

Live bar: cost 586, SHA b7cbfa82e0af50163edf70e1a353e203dc77efeaa29c86e2bf8d5d6c135c6cfa (`ng_current_bar.py --task 275 --json`, manifest-authoritative).

Attempted route: fold the output-route extent to 16 by using [2,16,4] `block_out`/`mod_out` maps and declaring the final Einsum output as [1,10,16,16], with no Pad intermediate.

Exact failed gate: `python tools/ng_gate.py visible --task 275 --cand The_400_Tasks/Task_by_Task/task275/prototypes/task275_crop16_routeprobe_split.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.

Why dead: the split crop computes the correct top-left 16x16 tensor on the first visible pair, but `verify_subset`/`ng_gate visible` compare against a padded [1,10,30,30] output tensor. Shape mismatch alone makes every visible pair fail. The valid split crop prices at 842, above the 586 bar. The naive 362-param shared-map crop is invalid because the same cropped route maps also contract with 30-wide input axes (`Einsum operands could not be broadcast... operand 3 {2,16,4} vs dim 30`).

One remaining hypothesis: preserve the 30x30 output contract and reduce params by algebraic factorization of dense route maps or duplicate probes, not by emitting a cropped output tensor.

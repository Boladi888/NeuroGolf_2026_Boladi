# task038 hard record — Pad-attribute output placement

- live bar: cost 190, sha256 `5a590f3b88143fbb8eecd7ff55b95689a097fc11e58f0e42961ead25981436d5`, source `python tools/ng_current_bar.py --task 38 --json` / manifest.
- attempted route: keep Sho's 190 graph unchanged except replace input-form final `Pad(packed, pd)` with opset-10 attribute-form `Pad`, deleting the 8-param `pd` initializer.
- failed gate: `python tools/ng_gate.py visible --task 38 --cand The_400_Tasks/Task_by_Task/task038/prototypes/task038_v7_pad_attr_opset10.onnx` -> `InvalidGraph: Type 'tensor(bool)' ... Pad ... is invalid`.
- sibling failed gate: `python tools/ng_gate.py visible --task 38 --cand The_400_Tasks/Task_by_Task/task038/prototypes/task038_v7_pad_attr_opset10_u8_bridge.onnx` -> `InvalidGraph: Type 'tensor(uint8)' ... Pad ... is invalid`.
- why dead: attribute-form Pad needs opset 10, but ORT's Pad-10 rejects the cheap BOOL/UINT8 carriers; an F16/F32 bridge would add at least a charged `[1,2,1,5]` cast tensor (20/40 B) to delete only 8 params, so it cannot beat 190.
- one remaining hypothesis: find a non-Pad final placement primitive that embeds `[1,2,1,5]` into `[1,10,30,30]` with <8 params and no counted large intermediate; no such ONNX primitive was identified in this pass.

live bar: manifest cost 901, sha256 c5dbb60f72207c6d5c260d0db689de2efee0a630c2a3ea7ee8b46d0955828feb (`ng_current_bar.py --task 88 --json`)

attempted route: remove the pre-QLinearConv dynamic Pad-to-10 chain and feed `crop_u8` directly into the final QLinearConv, relying on final QLinearConv end pads plus decoder trimming.

failed gate: `python tools/ng_gate.py visible --task 88 --cand The_400_Tasks/Task_by_Task/task088/prototypes/task088_remove_prepad.onnx` -> `[visible] 6/267 (2.25%)  FAIL`

why dead: dynamic-output/no-prepad representation does not preserve official decoded outputs; per worker protocol a visible miss is a certain hidden zero, so no truth/ab/profile variants in this family.

remaining hypothesis: look only for scalar/initializer shaves inside the existing fixed-30 output family, especially replacing a scalar value without changing the prepad chain.

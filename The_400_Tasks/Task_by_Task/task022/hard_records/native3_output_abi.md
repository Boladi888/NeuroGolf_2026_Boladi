# Native 3x3 Output ABI Probe

- Live bar: cost 836, sha256 `b4f77577321a7f2498d8ae033c440936b1ed39f5f547f84bd0caa939b785e753` (`python tools/ng_current_bar.py --task 22 --json`, team manifest/Sho).
- Attempted route: edit Sho's one-node free-output Einsum to emit `[1,10,3,3]` directly, deleting `v_w4` and the final 3-to-30 placement factors. Prototype: `prototypes/task022_native3_probe.onnx`, static cost 803.
- Failed gate: `python tools/ng_gate.py visible --task 22 --cand The_400_Tasks/Task_by_Task/task022/prototypes/task022_native3_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: the scorer/verifier requires the full `[1,10,30,30]` output ABI for this task; native logical 3x3 output is shape-incompatible even when the 3x3 content is the same formulation.
- Remaining hypothesis: preserve the 30x30 ABI and look only for lower-rank/fewer-element factorizations of `v_w3`/`v_w2`; do not retry native-output variants.

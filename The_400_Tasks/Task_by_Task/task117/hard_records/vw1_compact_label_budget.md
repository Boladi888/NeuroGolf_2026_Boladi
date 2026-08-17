# task117 hard record — compact `v_w1[5,15]` via shared crop is DEAD without two free labels

- **Live bar:** 2380, source `python tools/ng_current_bar.py --task 117 --json` / authoritative manifest, owner Sho, sha `79dcba6fb80f7b66ba3f4ed64b9c7d3e4e5358a02972745772b1c6fdc3a0f879`, updated 2026-07-09T01:09:17Z.
- **Attempted route:** shrink center selector `v_w1[5,30]` to `v_w1c[5,15]` and reuse existing crop `v_w0[15,30]` inside the same free-output `Einsum`: `v_w1[s,A] = sum_Z v_w1c[s,Z] * v_w0[Z,A]`. This would save 75 params if exact.
- **Prototype:** `The_400_Tasks/Task_by_Task/task117/prototypes/build_task117_vw1_compact_tied_probe.py`
- **Probe artifact:** `The_400_Tasks/Task_by_Task/task117/prototypes/task117_vw1_compact_tied_probe.onnx`, sha `7b28470f0b8990f42fc0672e9a49cab453c0b8590a8d9b83a9be65d9d8f1591a`.
- **Exact failed gate:** `python tools/ng_gate.py visible --task 117 --cand The_400_Tasks/Task_by_Task/task117/prototypes/task117_vw1_compact_tied_probe.onnx` -> `[visible] 93/265 (35.09%)  FAIL`.
- **Why dead:** the exact factorization needs two independent fresh Einsum labels, one for row center `s` and one for column center `t`. Sho's equation already uses 51/52 ASCII labels; only `Z` is free. Digits/underscores are ignored by ONNX shape inference, and ORT rejects contracted ellipsis (`Inputs have ellipses in them but the provided output subscript does not contain an ellipsis`). The one-label probe forces `s == t`, while official data has `cy != cx` in 173/266 detected X centers.
- **Remaining hypothesis:** a deeper equation refactor that frees a second ASCII label could revive the 75-param shave, but it must preserve independent row/column centers; no one-label compact selector is viable.

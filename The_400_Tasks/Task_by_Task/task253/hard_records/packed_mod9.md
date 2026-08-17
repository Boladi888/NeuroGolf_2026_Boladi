# task253 packed-mod9 dead route

- Live/team-best bar used: cost 367, SHA256 `d9d16502e41c01a6afabc784c000e383f0a7717061dcc597bf9c697fecc66226`, source `python tools/ng_current_bar.py --task 253 --json` / manifest owner Sho.
- Attempted route: replace current `3*(wr % 3) + (wc % 3)` code path with `(wr + 3*wc) % 9`, keeping Sho's ch0 `ScatterElements` center-code overwrite and changing the 4x4 cellmap accordingly.
- Candidate: `The_400_Tasks/Task_by_Task/task253/prototypes/task253_sho367_packed_mod9.onnx`, built by `prototypes/build_task253_sho367_packed_mod9.py`.
- Failed gate: `python tools/ng_gate.py visible --task 253 --cand The_400_Tasks/Task_by_Task/task253/prototypes/task253_sho367_packed_mod9.onnx` -> `[visible] 3/265 (1.13%)  FAIL`.
- Why dead: visible failure is a certain hidden zero under the worker protocol. Do not run truth/AB/battery or build packed-mod9 variants.
- One remaining hypothesis: only non-packed local shaves remain plausible, especially parameter-only Pad/cellmap/index tricks or a byte-equal compression that preserves the existing two independent mod-3 residues.

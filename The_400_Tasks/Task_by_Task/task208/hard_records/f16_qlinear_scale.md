# Dead route: F16 QLinearConv scale input

- **Live bar:** 3752 (manifest authoritative, owner Boladi, sha256
  `df3c70f1cc72907e6d92570be3d511d58249916e836bdf8b35bdf24a87ceedac`,
  updated 2026-07-10T22:11:58Z). `ng_current_bar.py --task 208 --json`
  also flags the local `Team_Best_Onnx` static 3746 reading as stale because
  the authoritative manifest/profile bar is 3752.
- **Attempted route:** keep the dynamic seed scale for the first `QLinearConv`
  as the existing F16 scalar `seed_y_scale_h`, remove the F32 `Cast` to
  `seed_y_scale`, and feed the F16 scalar into `QLinearConv` input 6. If ORT
  accepted it, this would delete one charged 4-byte F32 scalar.
- **Prototype:** `prototypes/task208_w1_f16_qscale_probe.onnx`, built by
  `prototypes/build_task208_w1_f16_qscale_probe.py`, sha256
  `45775a2ce225ff47c5f4b5dc3c7ce890dd1219be19b236dc941f704624588e1a`.
- **Exact failed gate:** `python tools/ng_gate.py visible --task 208 --cand
  The_400_Tasks/Task_by_Task/task208/prototypes/task208_w1_f16_qscale_probe.onnx`
  -> `[visible] ERROR: InvalidGraph: [ONNXRuntimeError] : 10 : INVALID_GRAPH :
  This is an invalid model. Type Error: Type 'tensor(float16)' of input
  parameter (safe_name_48) of operator (QLinearConv) in node (safe_name_49) is
  invalid.`
- **Why dead:** ONNX Runtime requires `QLinearConv` scale inputs to be F32 on
  this stack. The existing live graph's F16 `seed_y_scale_h -> Cast(F32)` is
  not optional; removing it makes the model unloadable before visible
  correctness can run. Per gate ladder, no score/profile/truth/ab was run.
- **One remaining hypothesis:** only non-`QLinearConv` scalar peepholes remain,
  but the 2026-07-14 polish audit on the live 3752 ONNX found no savings.
  Structural floor remains the documented forced surfaces:
  `ch0_inner17` F32 1156, `mask30` BOOL 900, and `mask19_u8 + mask19` 722.

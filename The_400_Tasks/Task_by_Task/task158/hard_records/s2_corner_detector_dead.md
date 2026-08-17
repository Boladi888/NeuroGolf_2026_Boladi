# task158 v134 sparse scale-2 corner detector — DEAD (2026-07-14)

**Live bar:** manifest/team-best `task158.onnx` cost/profile **18,560**, SHA-256
`b3be1ca261dfb2a6c499594461c0036280f0e05e0c233ba9e7126874bad8f32b`.

## Attempted route

Replace the live graph's scale-2 `valid2` detector:

- from dense `detw2 UINT8 [4,2,6,6]` with `w_zero_point=1`, `bias=-15`, exact 2x2-square/ring penalties;
- to sparse `detw2 UINT8 [4,2,2,2]` with `dilations=[5,5]`, `w_zero_point=0`, `bias=-1`, sampling only the two opposite marker-square corners while preserving `valid2` shape `[1,4,20,19]`.

Expected static saving if valid: `detw2` params `288 -> 32` = **-256** cost.

Prototype:
`The_400_Tasks/Task_by_Task/task158/prototypes/task158_v134_s2_corner_probe.onnx`

## Exact failed gate

```text
python tools/ng_gate.py visible --task 158 --cand The_400_Tasks/Task_by_Task/task158/prototypes/task158_v134_s2_corner_probe.onnx
[visible] 221/266 (83.08%)  FAIL
```

## Why it is dead

The dense scale-2 detector's 6x6 signed kernel is not just detecting the two diagonal marker corners; its zero-point penalties enforce the full 2x2 solid-square geometry and reject extra marker-color pixels in the 6x6 stamp window. Corner sampling admits false or shifted scale-2 pairs on visible officials, so it is a certain hidden zero.

## Remaining hypothesis

A legal scale-2 shrink would need a different exact-square proof that preserves the ring/exclusion information without materializing more state than the 256 saved params. A two-stage "2x2 square mask then pair" route is the next conceptual option, but it must price below the current one-shot `valid2` tensor plus 288 params before any gate.

# task013 hard record - tilingmodgatherbreak_682_floor

## Bar
- Team-best/live bar used for this worker pass: 682 = memory 567 + params 115.
- Source: `Team_Submission/Backend/manifest.csv` task013 row and `NG_OnnxlyFans_TaskStatus.xlsx` Tasks row, both updated 2026-07-02T02:35:15Z.
- Team-best owner: Vamshi.
- SHA-256: `115cc8568729b9730b77e5a3418fad325ae0e3e484b80d0f99f50fd250201c66`.
- Local read-only confirmation: `task013_vamshi682_baseline_copy.onnx` scores `(0, 567, 115)`, total 682; `profile_real_cost.py --arcgen 266` reports `static=682 profile=682`.

## Attempted Route
Audited the named tiling mod/gather floor-break route against the 682 scalar-moment graph. The charged target is the final line/output surface:

- `line_feat` F32 `[2,30]` = 240 bytes, produced by casting `line_feat_u8`.
- `line_feat_u8` U8 `[2,30]` = 60 bytes.
- Final output is a single F32 `Einsum` that consumes the F32 `input` directly as a free active-grid gate, avoiding any charged `[1,10,30,30]` active mask.

Probes under `prototypes/bo_ng_worker1_probe_682.py`:

- `task013_probe_u8_final.onnx`: remove F32 `line_feat` and feed U8 `line_feat_u8` to final `Einsum`.
- `task013_probe_f16_line.onnx`: cast `line_feat_u8` to F16 instead of F32.
- `task013_probe_bool_output_decl.onnx`: declare output BOOL while final `Einsum` still produces F32.
- `task013_probe_f16_coord.onnx`: retype `coord_f` initializer to F16.
- `task013_vamshi682_polished.onnx`: run standard polish passes on the baseline.

## Failed Gate
Foreground rerun on 2026-07-03:

```text
task013_vamshi682_baseline_copy.onnx: score=(0, 567, 115) total=682
task013_probe_u8_final.onnx: score=None total=None
task013_probe_f16_line.onnx: score=None total=None
task013_probe_bool_output_decl.onnx: score=None total=None
task013_probe_f16_coord.onnx: score=None total=None
task013 task013_vamshi682_baseline_copy.onnx: static=682 profile=682 (memory 567 + params 115; 267 real inputs)
[visible] 267/267 (100.00%)  PASS
polish_candidate: original_cost=682 final_cost=682
```

No `truth --full`, `ab --full`, or `battery` was run in this pass because no valid candidate beat the 682 bar/profile cost.

## Why This Route Is Dead
The direct dtype/final-line shaves are scorer-dead or ties:

- ONNX/ORT `Einsum` is homogeneous for this graph. The final output op consumes the F32 `input` to gate the active grid for free, so U8/F16 `line_feat` or F16 coordinate factors make the scorer return `None`.
- Declaring BOOL output without changing the producer type is an output type mismatch and scores `None`.
- The active-grid gate is load-bearing. A final Equal/Bitwise family that emits channel matches from U8 line codes without the F32 input gate would paint background channel 0 into padding, causing visible failure unless it adds a replacement active mask.
- A replacement active mask from row/col reductions costs at least the bytes saved from deleting `line_feat` and risks full-canvas intermediates.
- Mechanical polish found no byte savings; polished output is byte-identical to the team-best hash.

## One Remaining Hypothesis
A genuinely different output writer could still break 682 if it combines channel match and active-grid gating in one graph-output op without materializing any full `[1,10,30,30]` intermediate and without requiring F32 `line_feat`. Treat this as a new representation family, not a continuation of the direct U8/F16 final-Einsum shave.

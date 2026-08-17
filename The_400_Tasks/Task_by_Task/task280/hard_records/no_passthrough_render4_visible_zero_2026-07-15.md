# task280 — no-passthrough 4-term render dead route — 2026-07-15

## Live bar
- live_cost = 2618 (memory 1874 + params 744)
- profile_real_cost = 2618 on 267 real inputs
- authoritative source = `python tools/ng_current_bar.py --task 280 --json`, manifest row
- owner = Aditya, updated_utc = 2026-07-04T05:54:11Z
- parent/live sha256 = `ba4701c0455ac7715daa31b1117682876c1e41238176a3395569e980adde36c3`

## Attempted route
Tested the remaining July-12 hypothesis: remove the constant all-ones passthrough term from the final free-output Einsum.

- invalid first build: `prototypes/build_task280_v50_no_passthrough4.py`
  - candidate: `prototypes/tmp_task280_v50_no_passthrough4.onnx`
  - sha256: `8acce79d0e8ad499a624f92917fb5e064269ceada93bfcf0c4c3d162f7fbc059`
  - failed gate: `ng_gate visible` errored before count because shortening the shared `term_select` also corrupted the five marker-moment Einsums:
    - `[visible] ERROR: InvalidArgument ... GatherND ... invalid index found, index = -110`
- corrected build: `prototypes/build_task280_v51_render4_keep_moments.py`
  - candidate: `prototypes/tmp_task280_v51_render4_keep_moments.onnx`
  - sha256: `0d98ba1e9ca5c30167c4ad607d8776c830388e9d33ef4af2040375b3d9eb606e`
  - only the final output Einsum uses `term_select_render4`; the original 5-row `term_select` still feeds moments.

## Exact failed gate
Command:

```powershell
python tools/ng_gate.py visible --task 280 --cand The_400_Tasks/Task_by_Task/task280/prototypes/tmp_task280_v51_render4_keep_moments.onnx
```

Result:

```text
[visible] 0/267 (0.00%)  FAIL
```

Per the task prompt and `traps.md` §VISIBLE-FAIL, this representation family is rejected immediately. No score/profile/truth/ab/safety gates were run after the visible-zero.

## Why dead
The final render needs term 0 to preserve unchanged input cells when none of the generated stripe terms fire. Terms 1..4 cover only the two fill bands and their centerlines; they do not span the full active passthrough region. Removing the all-ones row/col factor makes no-stripe cells decode to all-zero, producing a visible-zero candidate.

Adding a separate passthrough outside the final contraction would require either a full output-shaped additive/fill plane or an equivalent full-rank active placement mechanism, which is larger than the current 20-wide all-ones term shared inside the free-output Einsum.

## One remaining hypothesis
Only revisit if a new free-output contraction class can express `input passthrough + two separable stripe overlays` without a constant active term and without materializing a `[1,10,30,30]` fill/add plane. Otherwise the July-12 wall remains the operative floor proof: live 2618 is already at the forced render/moment/depth floor.

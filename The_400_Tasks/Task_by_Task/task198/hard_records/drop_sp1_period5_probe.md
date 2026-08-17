# DEAD ROUTE - task198 v52 drop-sp1 / two-period classifier probe

Recorded: 2026-07-14 by `bo_ng_cdx_worker2`.

## Live bar

- `python tools/ng_current_bar.py --task 198 --json` -> authoritative manifest bar
  `live_cost=2898`, owner `Sho`, updated `2026-07-10T09:55:00Z`,
  live sha256 `ce164130c94a15ec0e7e357dbe5cb80f2ad1ed096f8b84ad51654b3368a46619`.
- Current local graph: `Team_Best_Onnx/task198.onnx`, graph
  `task198_v52_opset18_padmaxpool`, opset 18, no `Slice` nodes.
- `profile_real_cost --arcgen 266` on the live graph:
  `static=2899 profile=2899 (memory 1259 + params 1640; 266 real inputs)`.

## Attempted route

Low-confidence remaining hypothesis from `slicecastreadtaxkill.md`: maybe fewer than
three ring scales can classify the period lattice cheaply.

I built `The_400_Tasks/Task_by_Task/task198/prototypes/task198_v52_drop_sp1_probe.onnx`
from Sho v52 by forcing `is_sp1` false and deleting the period-5 detector/read path:
`ring5`, `cell_sum5`, `cell_u5`, `align_max5`, `is_sp1`, and the period-5 `Where` arm.

Builder:
`The_400_Tasks/Task_by_Task/task198/prototypes/build_task198_v52_drop_sp1_probe.py`

Candidate sha256:
`4e51ae7de6c080593b653fccbe398bc6783edc7fb4f9ffde7ebaf57a136e20fc`

## Cost probe

- Direct scorer import: `(0, 1041, 1280)`, cost `2321`.
- `ng_graph_cost_diff` vs `Team_Best_Onnx/task198.onnx`:
  `NET COST DELTA -577 -> candidate is CHEAPER`.
- Named saving is dominated by deleting `ring5` (`-360 params`) plus
  `cell_sum5` (`-144 B`), `cell_u5` (`-36 B`), `cell_tmp56_6_u` (`-36 B`),
  `align_max5` (`-1 B`), and `is_sp1` (`-1 B`), with `sp1_forced_false` adding `+1` param.

## Failed gate

Command:
`python tools/ng_gate.py visible --task 198 --cand The_400_Tasks/Task_by_Task/task198/prototypes/task198_v52_drop_sp1_probe.onnx`

Result:
`[visible] 184/266 (69.17%)  FAIL`

Per the visible-fail axiom, this representation is dead. I did not run profile,
truth, A/B, battery, or safety on the candidate.

## Why it is dead

The 577-byte saving is real, but period-5 classification is load-bearing on visible
officials. A two-period v52 classifier that preserves the existing decode/expansion
assembly cannot be reviewer-ready regardless of local cost.

This also reinforces the earlier v52 floor: the current graph's three ring scales are
not optional tail bytes; dropping one scale immediately loses 82 visible pairs.

## Remaining hypothesis

Only a genuinely different joint reduction remains: a detector that uses fewer/smaller
ring reads while also replacing the 720-param `membership_diff` decode, without
materializing a full-resolution plane. Treat that as a new assembly, visible-gate first,
and stop on the first miss. Do not retry simple branch deletion or the old flood/direct
render families.

# INFEASIBLE ROUTE - task198 v52 generated coordinate basis + score-term decode

Recorded: 2026-07-15 by `bo_ng_worker1`.

## Reviewer-locked hypothesis

Start from Sho v52 (`Team_Best_Onnx/task198.onnx`) and keep all three
period/ring detector cases exactly. Do not retry flood/direct-render/9273 or
simple branch deletion. Replace only the terminal decode side:

- `membership_diff[3,30,8]` -> generated coordinate/period basis.
- `score_bucket = Where(paint_bucket, -0.5, 100.0)` -> affine score term
  folded into the terminal free-output `Einsum`.

Relevant routed docs used: techniques #89 (free-output Einsum temporaries are
not charged), #94 (post-opset18 / Pad-wrap only helps when deleting roll
emulation; v52 has none), #118 (basis matrices as Einsum operands), #119
(deep free-output fold), #123 (Mod coordinate idioms), and traps
`INPUT-DTYPE-LOCK` (F32 input forces F32 final-Einsum operands).

## Bar

`python tools/ng_current_bar.py --task 198 --json`:

- authoritative source: manifest
- live cost: `2898`
- owner: `Sho`
- updated: `2026-07-10T09:55:00Z`
- live sha256: `ce164130c94a15ec0e7e357dbe5cb80f2ad1ed096f8b84ad51654b3368a46619`

## Probe

Builder:
`The_400_Tasks/Task_by_Task/task198/prototypes/build_task198_v52_coordbasis_scoreterm_probe.py`

Candidate:
`The_400_Tasks/Task_by_Task/task198/prototypes/task198_v52_coordbasis_scoreterm_probe.onnx`

Candidate sha256:
`68b7c90ba1b03e51a0464a5be953d2f6b95098a14190e43123e824237f3b67da`

Implementation:

- Keeps `ring4`, `ring5`, `ring6_h`, `ring6_v`, `is_sp0`, `is_sp1`, and the
  coarse selected cell map.
- Builds `membership_gen[3,30,8]` with INT32 coordinate math:
  `Mod(row, period)`, `Div(row, period)`, wall-bucket detection, clamp to bucket
  6, then `OneHot(..., depth=8)`.
- Builds an exclusive period selector `[default, sp0, sp1]` because the generated
  basis is an effective basis bank, not v52's additive-difference basis.
- Replaces `score_bucket` with `score_terms[2,1,8,8]` and
  `transform_terms[2,10,10]`; the final output is one `Einsum` with equation
  `qrij,salm,tden,shi,twj,qkc,bkhw->bchw`.

## Gates and cost

- `python tools/ng_gate.py visible --task 198 --cand ...task198_v52_coordbasis_scoreterm_probe.onnx`
  -> `[visible] 266/266 (100.00%)  PASS`
- scorer import -> `(0, 6285, 1121)`, cost `7406`
- `python tools/agent_helpers/profile_real_cost.py --task 198 --cand ... --arcgen 266`
  -> `static=7406 profile=7406 (memory 6285 + params 1121; 266 real inputs)`
- `python tools/ng_graph_cost_diff.py --task 198 --live Team_Best_Onnx/task198.onnx --candidate ...`
  -> `NET COST DELTA +4505 -> candidate is MORE EXPENSIVE`

Named charged-structure deltas from `ng_graph_cost_diff`:

- `membership_diff` initializer removed: `-720 params`
- `membership_gen[3,30,8]` materialized: `+2880 B`
- coordinate INT32 intermediates `coord_mod`, `coord_div`, `coord_div_clamped`,
  `bucket_idx`: `+360 B` each
- `score_terms[2,1,8,8]` materialized: `+512 B`
- `paint_bucket_f32[1,1,8,8]`: `+256 B`
- `score_bucket[1,1,8,8]` removed: `-256 B`
- `transform_s[10,10]` removed: `-100 params`; `transform_terms[2,10,10]`
  added: `+200 params`

## Verdict

The generated coordinate/period basis is functionally exact on visible, but it
converts v52's stored 720-element basis operand into a charged F32 runtime
tensor. Because the terminal `Einsum` also reads the F32 graph input, the basis
cannot be narrowed to F16/U8 without an 18 KB input cast. The score-term fold
also replaces a 256-byte `score_bucket` with a 512-byte term tensor plus a
256-byte cast of `paint_bucket_b`.

Profile cost `7406` is not strictly below manifest bar `2898`, so per reviewer
instructions I stopped here. No truth/A-B/battery/safety gates were run, and no
candidate was staged.

Remaining hypothesis: none inside this locked decode rewrite. A literal
Pad-wrap@19 route is inapplicable because v52 has no Slice/Concat roll
emulation to delete, and runtime Mod/Gather coordinate generation worsens the
same charged-basis problem rather than removing it.

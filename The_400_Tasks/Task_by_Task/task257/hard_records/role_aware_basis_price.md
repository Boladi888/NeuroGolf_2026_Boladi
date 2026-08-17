# task257 hard record - role-aware basis price

Live bar: 114, sha256 `32e5452b9089ab217e13dc4aac064b7807f9397603a66e5a5a945ffa3b0f5ef6`,
from `python tools/ng_current_bar.py --task 257 --json` (`manifest`, owner Vamshi,
updated 2026-07-12T15:15:54Z).

Reviewer locked hypothesis: replace current shared `feat[2,30] + mask[30]`
with a role-aware separable basis inside the final free-output `Einsum`, so input
positions `5..8` still alias into output coordinates `0..3` while output
positions `5..8` are suppressed.

Attempted route: split the shared spatial basis by role:

- `feat_in[2,30] = feat`, preserving source aliases `0..3` and `5..8`.
- `feat_out[2,30] = feat * mask`, suppressing output positions outside `0..3`.
- Delete the two `mask` operands and the `mask[30]` initializer.

Prototype: `The_400_Tasks/Task_by_Task/task257/prototypes/task257_role_split_probe.onnx`.

Gates/pricing:

- `python tools/ng_gate.py visible --task 257 --cand The_400_Tasks/Task_by_Task/task257/prototypes/task257_role_split_probe.onnx`
  returned `[visible] 269/269 (100.00%)  PASS`.
- `python tools/_score_v2.py` via `score_network_v2`: `(macs=0, memory=0, params=144)`, cost 144.
- `python tools/agent_helpers/profile_real_cost.py --task 257 --cand ... --arcgen 266`
  returned `static=144 profile=144 (memory 0 + params 144; 269 real inputs)`.
- `python tools/ng_graph_cost_diff.py --task 257 --live Team_Best_Onnx/task257.onnx --candidate ...`
  returned net `+30`: removed `feat[2,30]` (-60) and `mask[30]` (-30), added
  `feat_in[2,30]` (+60) and `feat_out[2,30]` (+60).

Why this is the locked-route floor certificate: the current basis must use the
same column vector when a position is read as input (`h,w`) and when it is read
as output (`r,c`). Position classes `5..8` must be nonzero as input, but zero as
output. The direct role discriminator is therefore either the existing
`mask[30]` (30 params) or a separate output-role basis. The exact split proves
the latter costs one extra `2x30` table while deleting only the 30-param mask:
net +30, total 144. A factor-only shared-basis route must still encode this
role distinction; otherwise it is the prior no-mask family and visible-fails
0/269 by leaking decoded positives outside the 4x4 answer window.

One remaining hypothesis: only a fundamentally different spatial code with a
sub-30-param role discriminator could beat 114. The obvious legal carriers are
`mask[30]` or `feat_out[2,30]`; both are >= the current mask price, so do not
rebuild this split-role family unless a concrete <30-param discriminator is
specified first.

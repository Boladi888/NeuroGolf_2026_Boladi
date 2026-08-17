# task158 dynamic row-byte renderer — reviewer rework truth wall (2026-07-10)

```json
{
  "verdict": "INFEASIBLE_TRUTH_MISMATCH",
  "candidate_class": "DYNAMIC_ROWBYTE_PATH_B_PROBE",
  "cost": null,
  "live_cost": 23201,
  "effective_team_best_cost": 22550,
  "status": "REJECT_LOCAL"
}
```

## One locked attempt

Implemented exactly one opset-18 dynamic row-byte probe:

`prototypes/task158_dynamic_rowbyte_probe.onnx`

- Four U8 lanes per 26-row, 25-column marker-role mask.
- `BitShift` with explicit cross-byte carry and width masking.
- `BitwiseAnd`/`BitwiseOr`/`BitwiseNot` exact size-1/2/3 solid-square
  anchors with clipped one-cell ring exclusion.
- Ordered two-role correlation at all four rotated diagonal offsets.
- Nine runtime normalized legend-paint-bit gates over fixed packed stamp
  shifts, followed by one U8-lane unpack.
- Terminal free-output F32 `Where(paint_mask, paint_onehot, input)`.

The model has 718 nodes and contains the requested primitives:
`BitShift` 92, `BitwiseAnd` 71, `BitwiseOr` 198, `BitwiseNot` 3,
`Where` 10. It is intentionally a four-input test-vector probe, not a legal
submission candidate. SHA-256:
`466b8984e7f043aa51ecf1b8751469b6dcc5d3e7ec2d4e105c329840645beb88`.

## Exact failed gate — fresh ARC-GEN truth

Command:

```text
python The_400_Tasks/Task_by_Task/task158/prototypes/build_task158_dynamic_rowbyte_probe.py --fresh 4000
```

The fresh block was allocated from `corpus.cache_info(158)` at seed offset
202026 (requested seeds 202026 through 206025). The immediate-stop line was:

```text
FAIL fresh 1123/4000 seed=203149 diff=4; ONNX==packed_python
```

Thus 1,124 cases were executed before the mandated immediate stop. The ONNX
opset-18 output exactly matched an independent four-U8-lane calculation, so
this is not an ONNX lowering, byte-carry, edge-mask, or unpack defect.

Failing case: 23x23, background 8, paint 1, ordered marker roles `(5,6)`,
legend top-left `(14,5)`. Exact size-2 anchors were:

- role 5: `(5,0)`, `(5,8)`, `(14,16)`
- role 6: `(1,4)`, `(1,12)`, `(10,20)`

The real first two sprites pair as `(5,0)<->(1,4)` and
`(5,8)<->(1,12)`, both with the same authored diagonal. All-pairs geometric
correlation also accepts the emergent cross-pair `(5,8)<->(1,4)` at the
opposite diagonal. That creates the extra packed placement `s2_k2=(1,4)` and
incorrectly paints four cells:

| row | col | truth | probe |
|---:|---:|---:|---:|
| 5 | 6 | 8 | 1 |
| 5 | 7 | 8 | 1 |
| 6 | 6 | 8 | 1 |
| 6 | 7 | 8 | 1 |

Exact ring exclusion cannot distinguish a generator-authored pair from this
accidental cross-pair because all four component anchors are independently
valid exact 2x2 squares. Fixing it requires a pairing discriminator outside
the reviewer-locked all-pairs contraction.

## Explicit charged-tensor ledger

The complete ordered ledger names all 717 charged probe tensors, shapes,
dtypes, byte charges, producer ops, and cumulative cost:

`prototypes/task158_dynamic_rowbyte_probe_ledger.json`

Isolated probe accounting (the four oracle inputs treated as free only for the
test-vector model):

- params: 253
- memory: 88,796
- cost: **89,049**
- first `<11,000` crossing: `lane_+1_slice_0079`, a U8
  `[1,2,26,4]` Slice of 208 B at node 55, raises cumulative cost to
  **11,169**.

The explicit current-v128 producer closure from the real F32 `input` through
the two role masks `XYu` and legend cells `E9f` costs another **13,989**
(13,950 memory + 39 params). Its own first crossing is `score` U8
`[1,1,26,25]`, which raises cumulative producer cost from 10,488 to 11,138,
before role-mask completion or any packed kernel. It also omits the additional
paint-vector producer needed by the terminal `Where`. Joining this producer to
the isolated probe is already at least **103,038**, before that missing decode.

The truth mismatch is independently fatal, so no official-input candidate,
score/profile gate, truth/AB gate, safety run, or staging was performed.

## Bar and classification

- Authoritative deployed LIVE: 23,201, SHA
  `cd08cc20468a7a8d5fe3cd69a2896a26340f72bdfa1ec50b1aee4e3cf498e4ce`.
- Lower task-local team-best used: existing v128 22,550 static/profile, SHA
  `5fcb99b35151eefbb62abae199b7f1a01a20376a2354085009eb799ef4857ec0`.
- Prompt stage threshold: strictly below 11,000.
- New candidate: none; candidate-minus-LIVE delta is not applicable.
- `candidate_class=DYNAMIC_ROWBYTE_PATH_B_PROBE`
- `status=REJECT_LOCAL`
- `safety=null`, `x86_required=null`, `kaggle_probe_required=null` because
  the multi-input probe is not a candidate and stopped at fresh truth.

## Remaining hypothesis

No in-scope variant remains under the one-attempt lock. An authored-pair
assignment discriminator could suppress cross-pairs, but it is a different
representation/search and would need a new reviewer direction; it cannot be
added to this rejected all-pairs probe.

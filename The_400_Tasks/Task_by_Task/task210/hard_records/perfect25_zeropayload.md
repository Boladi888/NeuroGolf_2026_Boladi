# Dead route: perfect-25 / cost<=1 zero-payload family (task210)

- **Live bar:** cost **30**, sha256 `aa923630b6479cf1a1f5e7b7bfe2e4947e1c8ba8edba4f31d250b0b2f0c98dde`
  (authoritative source: `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/manifest.csv`, owner Vamshi, 2026-07-02).
- **Attempted route:** the prompt's "perfect-25 ZeroopsweepCeilingAudit" rumor — a single no-payload,
  direct-output op that reproduces the rule at cost <=1.
- **Rule:** vertical mirror-stack of a 3x3 grid. Output rows = `[in0,in1,in2, in2,in1,in0]` then 24
  background rows. LIVE = one node `Gather(input[1,10,30,30], idx INT64[30], axis=2)`, idx =
  `[0,1,2,2,1,0, 3*24]`; Gather output IS the graph output -> memory 0, params 30, cost 30.

## Exact failed gate (2026-07-07, worker bo_ng_cld_worker3)
For cost<=1 the candidate must carry NO initializer (idx[30] already = 30 params). The only single-node,
payload-free, direct-output ops whose result shape stays `[1,10,30,30]` are Identity and the two
shape-preserving Transpose perms. Built + score_v2 (all cost 0) + visible:

| Candidate | score_v2 cost | visible |
|---|---:|---:|
| `identity` | 0 | **0/266** |
| `transpose_0123` (identity perm) | 0 | **0/266** |
| `transpose_0132` (swap the two 30-axes) | 0 | **0/266** |

Builder: `prototypes/zeroparam_sweep_task210.py`; scores `prototypes/zeroparam_sweep_outputs/`.
`tools/ng_gate.py visible --task 210 --cand <c>.onnx --json` -> `correct:0, total:266, pass:false` for all three.
One visible miss = certain hidden 0 (traps §VISIBLE-FAIL) -> family dead; no truth/ab run.

## Why it is dead (forced-component argument)
The output maps 3 distinct input rows into 6 output rows by **duplication + reflection**
(`[0,1,2,2,1,0]`, row 2 adjacent to itself, order mirrored). This is neither an axis permutation nor an
elementwise map, so:
1. No zero-payload op (Identity/Transpose — the complete payload-free, shape-preserving single-node set)
   can express it. All are visible-zero (proven above).
2. A single Gather that produces the output directly needs an index of exactly 30 elements (output row
   count) = 30 params; dtype shrink below INT32 -> scorer returns `None` (INT16/INT8 rejected).
3. A shorter gather (idx[6]) forces a Pad/Concat to reach 30 rows, materializing a counted
   `[1,10,>=6,30]` intermediate (cheapest measured 7215; Slice/Concat/Pad 14433).

Forced floor = **30** = LIVE. Perfect-25 / cost<=1 is infeasible for this rule.

## One remaining hypothesis
None credible. The only way below 30 would be a scorer change that credits <30 params for a 30-element
index, or a payload-free op that ONNX does not currently define. Do not re-explore the zero-payload
family; re-check only if the scorer's param-counting or accepted index dtypes change.

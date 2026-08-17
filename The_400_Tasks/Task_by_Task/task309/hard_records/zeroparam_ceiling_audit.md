# hard_record — task309 zero-param / perfect-25 ceiling audit (DEAD ROUTE)

- **Date:** 2026-07-07
- **Worker:** bo_ng_cld_worker1 (Claude lane)
- **Live bar at audit:** cost **10**, sha256 `c47b834e65872a11ec0cda3c8246948b8365f81f916610b49ed07267815733d9`
  (authoritative source = manifest, owner Vamshi, 2026-07-02). Confirmed via `ng_current_bar.py --task 309`.
- **Prompt hypothesis tested:** ZeroopsweepCeilingAudit "perfect-25 zero-cost direct-output"
  (Identity / Transpose / attr-only op that is correct AND prices <=1).

## Verdict: WALL — floor is cost 10. Not a perfect-25 task.

task309 (ARC `c8f0f002`) recolors orange(7)->gray(5); palette {1,7,8} and gray never
appears in any input, so on the one-hot `[1,10,30,30]` encoding this is a genuine channel
**swap 5<->7** (out ch5 = in ch7, out ch7 = in ch5). This MOVES channel content — no
zero-param attr-only op can express it. The forced floor is the single `Gather` with a
10-element index => `params=10, memory=0, cost=10`.

## Route attempted + exact failed gate (fresh empirical sweep 2026-07-07)
`prototypes/fast_zeroparam_sweep.py` — built + `score_network_v2` + byte-equal-vs-parent
(200 real {1,7,8} inputs) + ARC ground truth (265 pairs from `data/competitions/neurogolf-2026/task309.json`):

| candidate | cost | eq_parent | arc_ok | verdict |
|---|---:|---|---|---|
| Identity | 0 | False | 0/265 | reject (no recolor) |
| Cast(F32) | 0 | False | 0/265 | reject |
| Transpose(0,1,2,3) | 0 | False | 0/265 | reject |
| Transpose(0,1,3,2) (spatial HW swap) | 0 | False | 0/265 | reject |
| Reshape2x5+Transpose | 72009 | False | 0/265 | reject (perfect-shuffle, +intermediates) |
| Reshape5x2+Transpose | 72009 | False | 0/265 | reject |

No candidate is visible-correct at cost <=1 => per kill rule, stopped; NO truth/ab/battery
run on visible-zeros. Identity scoring 0/265 proves task is not near-identity.

## Rumor source explained (competitor cost-1 artifacts)
`prototypes/inspect_competitor_cost1.py`: the three competitor "cost 1" task309 files are
`Greater(input, threshold=100.0)` => all-zero output (one-hot values are 0/1, never >100).
They price cost 1 but are **byte-wrong, 0/265 correct**. The only *correct* competitor
artifact is the identical `Gather(idx=[0,1,2,3,4,7,6,5,8,9])` at **cost 10, 265/265**.
Independent 5th confirmation of the floor.

## One remaining hypothesis (for completeness — low prior)
The floor is `cost = params(Gather idx) = 10`. The only sub-10 correct construction would
require a single op that (a) permutes within the C axis and (b) needs <10 params. None
exists in the ONNX op set at opset 17 (Transpose can't reorder within an axis; a sparse or
short idx is scorer-rejected -> None -> 0 pts, see task309.md Don't-do log). **Do not
re-explore the zero-param / sub-10 family for task309.**

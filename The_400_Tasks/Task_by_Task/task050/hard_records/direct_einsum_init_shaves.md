# task050 — direct-output Einsum floor is 164 (dead routes below)

Live bar (2026-07-09): manifest/team-best **172** (Boladi, sha256 `e596320140a3…`).
This batch shipped **164** by reusing D on the marker side (see below). The graph is ONE
Einsum, memory=0, so **every** byte of cost is initializer numel.

## Current floor: 164
`P[10,2]=20 · G[30,2]=60 · S''[2,3,4]=24 · A[4,4,3]=48 · D[2,2,3]=12` = **164**.

### The −8 that got us here (172 → 164): marker-side D reuse
The 172 graph used the shared `D[2,2,3]` ([1,x]⊗[1,x]→[1,x,x²]) ONLY on the query side and
still paid a dense `S[2,2,2,4]=32` on the marker side (folding d² inline from two raw [1,d]
reads). Routing the marker reads through the SAME D (`df,dF,fFN`) lets a smaller
`S''[2,3,4]=24` select the 4 features. D is already in the graph → free; S drops 32→24 = −8.
Byte-identical output (ab byte_equal=True vs live; max raw diff 0.0).

## Dead route 1 — shrink A's first dim [4,4,3]→[3,4,3] (would be −12)
A is a bilinear form over the 4 row-features {active,count,sum,sqsum}. Its FIRST index (g)
only uses {0,1,2} (nonzeros at g∈{0,1,2}), so the size-4 first dim looks wasteful. But g is
CONTRACTED with the shared selector S''s feature output (4), and the SECOND slot (l) genuinely
needs feature 3 (sqsum, for the `count·sqsum` term A[1,3,0]). Both slots read the SAME S''.
Truncating A forces a SECOND smaller S'' for the first slot → S cost 24→(18+24)=42, A 48→36:
net compute 72→78 = **+6 (LOSE)**. Arithmetic; not built.

## Dead route 2 — factor A[4,4,3] (CP / 2-way / Tucker)
Verified numerically: A mode-ranks = (3,4,3). **Exact CP-rank = 5** (the e1-block spans a rank-3
(v,N) plane → 3 terms; plus e0-diag and e2-diag terms; no non-axis-aligned merge exists without
creating a spurious (1,0,0) entry). CP factor cost = (4+4+3)·5 = 55 > 48. A 2-way factor
`L[4,4,k]·R[k,3]` needs k≥3 (N-mode rank 3) → 19·3 = 57 > 48. Tucker (3,4,3)-core = 36+12 = 48
(ties, no gain). **A=48 is optimal.**

## Dead route 3 — drop a feature (4 → 3) to shrink S'' AND A
Would give S''[2,3,3]=18 + A[3,3,3]=27 (−27 → cost 137). DEAD: the only droppable feature is
`active`, and marker-only 3-feature is the recorded `direct155` probe → **visible 0/271**
(hard_records/no_active_bias_direct155.md). The negative `active` baseline is required so
background cells preserve under the multiplicative `H·V·Pre` argmax decode. count²-baseline
substitute is the same marker-only family (background rows go to score 0, fragile ties) → dead.

## Dead route 4 — shrink G / query-degree / P
- `G[30,2]=60`: 30 rows pinned by the 30×30 output canvas (query r,c index full 0..29;
  crop-to-15 + Pad materializes a charged [1,10,15,15] intermediate ≫ 30 saved → LOSE). The 2
  columns are [1,x]; x² is reconstructed by D (a parabola's sign is not linear-expressible).
  Locked at 60.
- `P[10,2]=20`: 10 = full one-hot color axis (contracted; Slice-to-{0,3,8} materializes an
  intermediate ≫ 20). 2 = minimal output/marker basis rank. Locked at 20.
- `D[2,2,3]=12`: minimal [1,x]²→[1,x,x²] map, now shared 6 ways (4 marker + 2 query). Locked.

## Verdict
164 is the practical floor for the direct-output single-Einsum representation. Any split into a
second node materializes an intermediate whose memory cost dwarfs the params saved. A genuine
sub-164 would need a fundamentally different multiplicative encoding with total numel < 164 at
one node / memory 0 — none found. Crop-process-pad (≥3005) is not competitive.

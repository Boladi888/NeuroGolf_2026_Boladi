# Orphaned staged candidates — triaged 2026-06-21

These were staged under `Task Prompts/2/Staged` and `/6/Staged` instead of the Submit Captain flow.
Triaged + routed: the real wins were ALREADY in `Staged_Boladi` (workers routed them correctly); these were
leftover duplicates + floor-challenge evidence. Owner directive: delete dups, archive the floor proofs here.

## Archived floor-challenge proofs (why cheaper approaches fail — keep for future work on these tasks)
- **task055 sub-1000 wall**: removing the two active-repair gates leaves nonzero section codes in padded
  rows/cols -> visible 0/263. No correct sub-1000 candidate exists for this formulation (floor ~1113-1115).
- **task154 warp family**: Aditya's 3857 sets output = ScatterND(input,...) so BOTH input and output are FREE;
  the warp family must build a full canvas -> best warp 6179 > 3857. 3857 is the floor; warp cannot win.
- **task348 free-output Einsum**: best EXACT free-Einsum = 3120 (~1.84x the 1693 baseline). Omitted-family lever loses.
- **task365 parallel labelling**: parallel = 4902 > sequential-peel 3354 (matches task365_history prediction). Loses.
- **task366 reports** (PROBE_FINDING + REPORT): DISPROVES "task366 is zero-public-weight" — an isolated Kaggle probe
  scored 14.46 = 25-ln(37577) exactly, so it DOES credit. The 37577 byte-equal fp16 polish is shippable (already staged).

## Deleted (redundant/superseded/bad — backed up to the session job tmp; models preserved elsewhere or unwanted)
- **Pure duplicates** (sha-identical to candidates already in Staged_Boladi/_shipped, nothing lost):
  task055@1115, task055@1113, task222@4256, task366@37577, task034@3499.
- **task034 floor note**: flip-decomposition 4963 REJECTED (+1464 worse); 3499 = geometric min for the
  single-ConvTranspose-bank architecture (already shipped at 3499).
- **task041@2548 — no-gain**: the team already has a cheaper **1777** staged. CAVEAT for the captain: task041 is
  messy — manifest.csv shows a STALE 1809 (Aditya), the actual LIVE extract is 2552, and the staged best is 1777.
  Worth reconciling which is truly live; none of these orphans improve it.
- **task177@1464 — HIDDEN-BAD (proven 2026-06-21)**: an isolated Kaggle probe scored **0.00 vs predicted 17.71**
  despite being byte-equal to LIVE on the full 3500-corpus. Reconfirms the task177 "byte-equal-but-zero" scar.
  *Never ship a task177 candidate without an own-stack isolated probe first.*

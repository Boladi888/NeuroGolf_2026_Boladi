# task023 — the "synth threshold is a hidden-correctness CLIFF" rule, now MEASURED on the 200k truth corpus

## What I discovered

traps.md §14e / the 2026-05-27 lessons.md entry already say "Path A synth
calibration is a THRESHOLD not a guideline" (task023 84% synth restructure
broke hidden correctness, −16.58 LB). I put **exact numbers** on that cliff by
running `ng_gate truth --task 23 --full` (200k ARC-GEN ground-truth corpus, the
same generator that produces the hidden test) against every relevant task023
artifact:

| artifact | cost | truth/200k | Kaggle |
|---|---:|---:|---|
| dynamic_v5 (exact beam) | 49,483 | 99.82% | exact, ~14.19 pts |
| LIVE singlescatter_h4 | 4,532 | **95.65%** | **+16.58 SHIPPED** |
| qtiny_h3 | 4,347 | 89.91% | — |
| 3823 restructure | 3,823 | **92.86%** | **−16.58 (reverted)** |

**The cliff for task023 sits between 92.86% and 95.65% truth-corpus accuracy.**
3823 at 92.86% destroyed the contribution; LIVE at 95.65% keeps it. The whole
margin between a +16.58 ship and a −16.58 disaster is ~2.8 percentage points of
200k-corpus accuracy. The per-seed 1k/5k synth gates the original worker used
(84%, 91%, 92.6%) are noisier proxies for this; the 200k truth corpus is the
sharp instrument.

## Why it matters / what's missing from the library

1. **For learned/heuristic structural unlocks (rankers, beam-with-caps), the
   200k truth-corpus % IS the ship/no-ship oracle**, and it is a step function
   near the cliff, not a smooth cost/quality tradeoff. Workers should run
   `ng_gate truth --full` (not just per-seed 1k synth) before claiming a cheaper
   heuristic is "safe enough" — a 3 pp drop on the full corpus can flip the LB
   sign even when 1k-synth smoke looks fine.

2. **A more-correct candidate can credit FEWER points.** task023's exact solver
   (99.82%) costs 49,483 and credits ~14.19 pts; the 95.65% heuristic at 4,532
   credits 16.58. Kaggle evidently credits the task fully at 95.65%, so paying
   11× cost for the last 4% of correctness is net-negative. The lesson for
   already-credited heuristic ships: do NOT "upgrade to exact" unless the exact
   version is also cost-competitive. Check the manifest points before assuming
   more-correct = better.

3. **Stale "zero-weight unlock" prompts:** task023's prompt still says cherry13b
   zero-weight (true of the OLD canonical, cost 12,260). The CURRENT LIVE is
   already credited at 16.58. Whenever a "zero-weight unlock" prompt names a
   task, the FIRST check must be the manifest points column — if it's already a
   high-points credited ship, the downside is NOT 0, it's the full credited
   value, and the EV math inverts. This generalizes the WORKER_PROTOCOL
   "baseline may be STALE / authoritative team-best is manifest.csv" rule
   specifically to the zero-weight family.

I checked traps.md §14e and lessons.md 2026-05-27 — they have the qualitative
rule but not the measured cliff bounds or the "more-correct credits fewer
points" corollary. Suggest extending §14e (or recent_ships) with the measured
92.86%/95.65% cliff and the manifest-points-first check for zero-weight prompts.

## Where

- task023.md "STATUS 2026-06-24 (Worker 7)" section has the full table.
- LIVE: `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task023.onnx`
  (SHA 8c4f45c3…, manifest cost 4532 / 16.5811 pts).
- Gate cmd: `python tools/ng_gate.py truth --task 23 --cand <onnx> --full`.

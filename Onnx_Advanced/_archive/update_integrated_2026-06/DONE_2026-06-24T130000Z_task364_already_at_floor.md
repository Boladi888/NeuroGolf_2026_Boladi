# task364 — DriveToFloor session: team-best already AT floor, prompt baseline stale

**Worker-4, 2026-06-24.** Verdict: **no-candidate** — team-best 14,162 is at the algorithmic floor.

## What I discovered

1. **The task364 prompt's baseline (46,250) and target (<8,000, "2×2 Euler/hole-count") are STALE
   and WRONG.** Team-best per manifest is **14,162** (Boladi, 2026-06-08). The prompt's Euler /
   hole-count hypothesis cannot work: the three sprites (L/U/H) are all line shapes with **0 enclosed
   holes**, so Euler characteristic does not discriminate them. The real discriminator is junction
   degree (H has a degree-3 branch point) + corner/endpoint cues (U vs L), which the shipped graph
   already uses.

2. **The shipped 14,162 graph is a single combined-flood classifier** (31 nodes, opset 13, BOOL out),
   NOT the dual-geodesic F16 graph that task364.md's prose described until today (that was the
   46,250 / 85,718 era). Boladi's win over the old graph: one `seed = Max(junction_seed, U_cues)`
   feeding ONE 5-step masked MaxPool flood, instead of two separate floods. I reverse-engineered it
   node-by-node and corrected the task364.md Rule / Algorithm / Cost-analysis sections.

3. **Floor proof (exact, STATIC graph so local==Kaggle):** census of `score_v2 = (0,14002,160)`:
   - `v56_f32` F32[1,1,20,22] = 1,760 — dtype-locked F32 input read (input is F32 per
     §INPUT-DTYPE-LOCK; Slice preserves dtype). This is the techniques #88 "input bridge floor."
   - `W4` U8[1,4,20,22] = 1,760 — 4 directional T/elbow conv detectors, ALL 4 channels consumed.
   - `cv_pad` U8[1,1,30,30] = 900 — minimal canvas for the broadcast-Equal output.
   - 5-step masked flood = 10 planes × 440 = 4,400 — **BFS-diameter floor** (techniques #88 round-count
     law). I measured max seed→body geodesic distance = 4 over 4,000 real inputs, but dropping even
     ONE MaxPool+Where step breaks **271/4000** byte-equality (drop-2 → 2,417/4000). The unmasked
     final `mp4` leaks across sprites; intermediate `Where(v56_b,…)` re-masks are what prevent
     corruption, so no step is droppable. Same failure class as task048 k=4.
   - Row-byte/uint32 flood packing (#47/#88) inapplicable: crossover only inverts at width ≤ 8;
     task364 width = 22.
   - `polish_candidate.py` = no-op (−0 B). The Min-masked-vs-Where swap saves no plane (the closing
     `Where(v56_b, w1, wB)` selects two non-zero values and needs the BOOL condition).

## Where

- Shipped graph: `K-NeuroGolf-OnnxlyFans/Team_Submission/Backend/Team_Best_Onnx/task364.onnx`
  (sha per manifest `a6ffcdef…05ea7`).
- Floor proof + corrected rule written to `The_400_Tasks/Task_by_Task/task364.md`.

## Why it matters

Prevents the next worker from burning a session chasing a sub-14k Euler/hole-count rewrite that
cannot work, and documents that this `geodesic-flood component-recolor` idea-class bottoms out at
`F32-input-read (1,760) + feature-bank (1,760) + diameter-floored single flood + minimal recolor
canvas`. The only sub-14k path is a different idea-class avoiding the F32 input read or the
component-label propagation — none known, and the task is DETERMINISTIC (truth 100%) so any
alternative must still prove 100% truth.

## Library status

techniques #88 (round-count = BFS diameter; input-bridge F32 floor; width≤8 packing crossover)
already covers every lever I hit — no library gap. This drop is a task364-specific confirmation /
floor record, not a new technique. The task364.md "Don't-do" + "Cost analysis FLOOR PROOF" blocks
carry the durable per-task lesson.

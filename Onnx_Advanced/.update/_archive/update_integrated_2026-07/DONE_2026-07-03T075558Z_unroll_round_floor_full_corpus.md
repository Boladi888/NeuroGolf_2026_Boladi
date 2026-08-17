# Fixed-unroll round count is pinned by the FULL-corpus deep tail, not a subsample

## What I discovered
For a fixed-unroll iterative graph (flood-fill / BFS / morphological dilation with
K hardcoded rounds), the minimal-safe K equals the **maximum convergence round over
the entire truth corpus**, and that max lives in a very rare deep tail. Subsampling
lies: for task251 (enclosed-region fill, bit-packed U16 flood), a fresh 20k sample
showed max convergence = 8, a 30k sample showed 9 (1 grid), and only the **full
cached 200k** revealed max = **10** (2 grids: idx 81470, 109542, both 12×12).

Concretely, correctness of an R{k} graph ⇔ every grid converges within k rounds.
Truth is all-or-nothing (need 200000/200000), so those 2 deep grids force K=10.
The team-best 3740 graph already runs exactly R10 — it is the unroll floor. R9
(−216 B), R8 (−432 B), R7 (−648 B) are all statically cheaper but each fails truth
(R9: 199998/200000). Verified on the real ONNX, not just the model.

**Method that works (fast + exact):** model the graph's dilation in numpy, then run
it over the **cached** corpus grids directly (`corpus._load(task)["in_idx"]` +
`in_shapes`) — no regeneration, ~1 s for 200k — and read `max_convergence_round`.
This is the cheap oracle for "how many unroll rounds does truth actually need"
BEFORE spending a `truth --full` on a doomed reduced-round candidate.

## Second insight — bit-shift dilation is 8-connected, converges faster
task251's team-best does `vertical = reach|up|down` then `all = vertical |
(vertical<<1) | (vertical>>1)`. Vertical-then-horizontal bit spread is an
**8-connected** dilation (it reaches diagonals), so it converges ~1 round faster
than the 4-connected reference BFS. A 4-connected round-count model over-estimates
the required K. Model the ACTUAL op sequence, not the reference semantics, when
predicting unroll depth. (For task251 the 8- vs 4-connected exterior happens to
agree on the corpus because walls are solid — no single-diagonal pinch leaks.)

## Where
task251 (`enclosed-region fill`, live bar cost 3740 sha `e99be2e84e3d`).
Build/probe scripts: `The_400_Tasks/Task_by_Task/task251/prototypes/`
(`build_reduced_rounds.py`, `conv_cached.py`, `behavior_check.py`,
`wall_certificate.py`). Dead-route certificate:
`The_400_Tasks/Task_by_Task/task251/hard_records/reduced_round_unroll.md`.

## Why it matters
Avoids two traps: (1) shipping a reduced-unroll graph that passes a small truth
sample but fails the full 200k on a handful of deep grids; (2) wasting a serialized
`truth --full` on a candidate a 1-second cached-corpus convergence scan already
proves doomed. Calibrates cost_floor_reference for iterative-unroll tasks: the
per-round byte cost × (max convergence round over full corpus) is a hard floor.

## What's missing from the library
I did not find a documented "measure convergence-round over the full cached corpus
to pin fixed-unroll K" recipe, nor a note that bit-shift V-then-H dilation is
8-connected. by_symptom has flood-fill routing but not this round-count-floor test.

## Sorter note: traps.md — add as new dead-end/floor note "fixed-unroll round count = full-corpus deep-tail max; subsample under-counts; verify on cached corpus not a fresh sample"
## Sorter note: cost_floor_reference.md — extend with "iterative-unroll floor = per-round bytes × max convergence round over full 200k (task251: 216 B × 10 = R10 is floor at 3740)"
## Sorter note: op_cost_patterns.md — add note "bit-packed V-then-H shift dilation (vertical|<<1|>>1) is 8-connected, converges ~1 round faster than 4-connected BFS; model the op sequence not the reference when predicting unroll depth"

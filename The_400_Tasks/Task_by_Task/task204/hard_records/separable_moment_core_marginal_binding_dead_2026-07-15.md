# DEAD ROUTE - task204 SeparableMomentCore marginal binding

**Recorded:** 2026-07-15 by `bo_ng_worker1` for `task204`.

## Live bar

- **Authoritative live/team-best cost:** 7682.
- **Source:** `python tools/ng_current_bar.py --task 204 --json` -> `authoritative_source=manifest`, owner `Boladi`, updated `2026-06-08T23:23:49Z`.
- **Live SHA-256:** `d2f77ac3e6bca0c1e5afdb88378da368ae17805bb70bb62651e5d9d10b150c75`.
- **Parent path checked:** `The_400_Tasks/Task_by_Task/task204/prototypes/task204_v2.onnx`.
- **Parent profile:** `task204 task204_v2.onnx: static=7682 profile=7682 (memory 6652 + params 1030; 268 real inputs)`.
- **Parent visible gate:** `[visible] 268/268 (100.00%)  PASS`.

Note: `ng_current_bar.py` reported local `Team_Best_Onnx/task204.onnx` as stale because its recorded cost is 7678 for the same SHA. The manifest/workbook/library agree on 7682, so 7682 is the bar.

## Attempted route

Prompt route: collapse spatial evidence into row/column/color moments or marginals with free-input `Einsum`/`Reduce` ops, then render with separable/free-output machinery near the stated floor estimate.

Prototype probe:

`python The_400_Tasks/Task_by_Task/task204/prototypes/probe_task204_separable_moment_core.py`

## Exact failed pre-build probe

The route fails before an ONNX candidate because independent row/column summaries do not preserve which row square is paired with which column square.

The probe constructs two valid 12x12 inputs:

- A: 3x3 squares at `(1,1)` and `(7,7)`.
- B: 3x3 squares at `(1,7)` and `(7,1)`.

Both inputs have identical blue row and column marginals:

`row = col = (0,3,2,3,0,0,0,3,2,3,0,0)`

But their correct outputs differ in **36 cells**. Any graph that collapses to independent row/column moments before retaining a joint binding state must produce the same downstream state for both and is therefore wrong.

Corpus/fresh probe output:

- bundled pairs: 268.
- bundled pairs with repeated square size: 221.
- bundled pairs with same-size binding hazard: 202.
- fresh rule-shaped samples: 2000, seed 204.
- fresh samples with repeated square size: 1198.
- fresh samples with same-size binding hazard: 1029.

This is the prompt kill rule: correctness needs nonseparable per-square row-column binding after the marginal step.

## Why this is dead

Free-input marginals are cheap, but task204 is not only asking for side-length parity. It must bind each detected row interval to the correct column interval for every square. Same-size repeated squares make row/column marginals ambiguous; a separable cross-product renderer creates false squares at the swapped pairings.

Repairing the route requires reintroducing a joint row-column state. The known cheapest exact joint state is the current v2 detector:

- `topleft_len_u8 [1,8,18,18]` = 2592 bytes.
- 8 channels are the 8 possible side lengths 3..10 over the 18x18 top-left domain.
- Once this tensor is reintroduced, the graph is back at the current v2 core; previous value-coded, split/bucketed, F16 scan, and all-U8 span attempts are documented as no-gain or regressions.

The best possible "moment core + repair" therefore cannot beat 7682 without deleting or shrinking a different load-bearing component. The only remaining large non-U8 component is the forced F32 input read `ch1_det_f32 [1,1,18,20] = 1440`; current ONNX Slice/Gather preserve input dtype, so there is no direct U8 read.

## One remaining hypothesis

Only revisit if a future accepted op path can read the blue channel from the F32 one-hot input directly into U8 or can perform a typed one-channel crop+threshold in one scorer-visible output smaller than `ch1_det_f32 + ch1_det_u8` (currently 1800 total). Otherwise the marginal/moment family must pay a joint binding tensor and stays at or above the v2 floor.

## Verdict

`REJECT_LOCAL`: no candidate built or staged. The separable moment/marginal formulation is non-injective on valid task204 inputs and any exact repair reintroduces the current joint detector floor.

# DEAD ROUTE — task054 "StripRunScatterRewrite" to ~7000 is INFEASIBLE

**Recorded:** 2026-07-08 by `bo_ng_cld_worker1` (Claude lane).
**Do not re-explore the ~7000 structural target.** Probe-backed floor proof below.

## Bars in play (live, re-checked 2026-07-08)
- Authoritative manifest bar: **22,674** (owner Aditya, sha `01a07eb51dba…c99a1`, `ng_current_bar.py --task 54`).
- Best task-local staged rung: **v28 @ 13,301** (sha `8116ec57…630e`), byte-equal to the shipped v25/v26
  function (99.9135% truth — inherits v25's 173/200k detector bugs; NOT a 100% ship).
- 100%-truth floor (v7-lineage, byte-equal to submission.zip): **26,003** (v8).

## The prompt's premise and why it's wrong
Prompt: "run/strip predicates + ONE output color-index surface … does not need … a
ScatterElements-axis-3 formulation," floor_est ~7000, class STRUCTURAL, confidence HIGH.

The `StripRunScatterRewrite` the prompt asks for is **already the shipped architecture**
(v5 built strip-extracted CumSum runfill + ScatterND row-writes on 2026-06-04; v7 added
ScatterElements V-lines; v25 kept it and swapped detection for Einsum poly-moments). The
current champion (13,301) IS "strip runfill + scatter." There is no un-built formulation here.

The ~7000 estimate assumes "one surface" = one materialized plane. It is not, for two reasons:

1. **ONNX ScatterND / ScatterElements / Reshape are FUNCTIONAL, not in-place.** Each compose
   pass reads a canvas and PRODUCES A NEW [30,30] plane. "One logical canvas updated 3×"
   materializes THREE 900-B intermediates.
2. **The rule needs THREE distinct write patterns that cannot share a scatter:** H-line
   row-strips, V-line col-strips, and stamp/legend points. Row-strips and col-strips use
   different broadcast axes; points use a different index rank. They force 3 separate passes.

## Forced-component floor (probe-backed, 2026-07-08)
Skeleton = `Conv color decode → Cast U8 → N×ScatterND → Equal` (ZERO detection, ZERO strip
logic, trivial `[1,4]` index). `score_v2`:

| skeleton | cost | note |
|---|---:|---|
| color + 1 scatter | 5,425 | color4f 3,600 + color4u 900 + 1 plane 900 |
| color + 2 scatters (H+V lines) | 6,325 | +900 |
| **color + 3 scatters (H+V+stamps)** | **7,225** | +900 — **already > 7,000 with no logic** |

Builder: `prototypes/build_task054_floorprobe_2026-07-08.py`
(→ `task054_floorprobe_{1,2,3}scatter.onnx`).

The 5 forced full [30,30] planes = **7,200 B** are each load-bearing in the real champion
(`largest_intermediate`): `color4f` F32 3,600 (F32 one-hot decode — ArgMax I64 = 7,200 worse,
Einsum/Conv = same 3,600), `color4u` U8 900 (cheap-canvas admission), `scat_h` 900 (H-scatter),
`cV` 900 (V-scatter), `final2d` 900 (stamp scatter = output surface).

## The "single surface" alternative is WORSE (formulation #2, measured honest)
Fold ALL writes into ONE point-scatter. The point index must be a COMPUTED intermediate
(points depend on runtime marker rows/cols), charged as memory bytes:

| single point-scatter | cost |
|---|---:|
| N=41 (stamps only, no lines) | 7,101 |
| N=240 (lines only) | 15,260 |
| N=294 (full rule) | **17,474** |

The [294,4] INT64 index alone = 9,408 B. This is the v6 regression (measured 30,299 in
2026-06-04). Strip-writes are cheaper than point-writes precisely because they broadcast.

## Verdict
- **~7000: WALL.** The 3 functional compose planes + F32 color source = 5 planes = 7,200 > 7,000,
  before any detection/strip/index bytes. No formulation removes a forced plane (color decode is
  F32-floored; both line orientations are static-materialized; stamps can't fold into a line pass).
- **Sub-13,301 structural: effectively WALL.** 6 prior sessions + this one ground it to ±12 B
  (v27→v28). Remaining bytes are the forced planes + INT64 [41,2] stamp index (ScatterND rejects
  INT32) + [30,4] ScatterElements index + the CumSum strip core.
- **Only open lever = CORRECTNESS, not structure.** A 100%-truth graph < 26,003 (fix v25's 173
  Einsum-detector misses — "mostly both-orientation marker layouts", first_fail_idx 989) would
  dominate the correct-floor. That is a detector-debug task, out of this structural-rewrite scope.

## One remaining hypothesis (if anyone insists on pushing structure)
The two line scatters (`scat_h` 900 + `cV` 900) are the only "maybe-mergeable" planes. Merging
needs a single write pattern for both H and V spans — impossible without point-enumeration
(→ N≈240 index blowup, see table). Unless a future opset adds an in-place scatter or a
strip-union op, the 3-plane compose stands. Not worth another session.

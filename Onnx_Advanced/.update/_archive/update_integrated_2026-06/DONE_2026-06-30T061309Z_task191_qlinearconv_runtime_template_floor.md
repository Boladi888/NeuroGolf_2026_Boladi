# task191 — QLinearConv with RUNTIME-extracted kernels halves a variable-template D4 match map; "F16 Conv forced" floors are a trap

Session: bo_ng_cld_worker2 (Claude lane), task191, 2026-06-30. Verdict: no strict win
(WALL); but a large stale-floor correction + a transferable INT8 technique surfaced.

## Insight 1 — INT8 QLinearConv works for a RUNTIME-extracted variable template, not just constant kernels

`by_symptom.md` already covers QLinearConv-as-INT8-match, but ONLY for *documented constant*
kernels (the "geometric ray/diagonal" symptom, task136: weight-4 anchor + weight-1 footprint).
task191's team-best graph (Aditya, cost **11,169**, manifest sha12 `e3d77f6d2cbe`) extends this:
the 8 D4-oriented match kernels are **extracted from the input at runtime** (bbox of the blue
"key" via 2 projection `Conv` + `ArgMax` asc/desc → `Gather`/`Transpose`/`Slice` build the 8
orientations of the ≤3×3 yellow pattern into an `[8,1,5,5]` INT8 kernel), then a single
`QLinearConv(yellow_u8, kernel_i8)` produces the `[1,8,23,23]` **UINT8** match map. A second
`QLinearConv` does the stamp/paint. So **QLinearConv composes with technique #54 (runtime
variable-template extraction)** — you can do INT8 correlation even when the template is unknown
until inference. The whole match+stamp runs in INT8/UINT8; no F16 anywhere in the hot path.

Cost consequence: the 8-orientation match map is `[1,8,23,23]` U8 = **4,232 B**, vs the F16
`Conv` equivalent `[1,8,23,23]`/2×`[1,4,23,23]` F16 = **8,464 B**. ~2× on the dominant tensor,
plus the F16 `ConvTranspose` paint path (another ~4k of F16 seeds) collapses to one INT8
`QLinearConv` stamp.

## Insight 2 (META / trap) — "ORT Conv forces F16, so the match map is irreducible" is a FALSE floor

task191's own `task191.md` carried a rigorous-looking floor proof (v3/v5 lineage, worker-7) that
declared the F16 match map `8,464 B irreducible` and the realistic floor `~24–27k`, with the key
load-bearing claim **"F16 dtype FORCED by ORT (Conv/ConvTranspose reject INT/BOOL/UINT8)."** That
claim is true for plain `Conv` but **ignored `QLinearConv`**, which correlates in INT8. Aditya's
INT8 rewrite reached 11,169 — less than HALF the proven "floor." The whole F16 lineage (v1→v7,
85,921→24,807) was superseded.

Lesson: any task whose floor proof rests on "the match/correlation tensor must be F16 because ORT
Conv rejects integers" should be re-audited — if the kernel weights and the data are integer-valued
(here: 0/1 yellow mask, ±1 template), `QLinearConv` halves the match-map dtype and removes the F16
paint path. Candidate re-audit targets: any PatternRecognition/template-match task with an
`[1,K,H,W]` F16 match map and an F16 `ConvTranspose` paint.

## Insight 3 — the genuine forced floor for an N-orientation template-match-and-stamp

Once you're in INT8, the match map `[1,N,H,W]` U8 is a HARD floor when:
- all N orientations are needed for 100%-truth (here N=8 D4; dropping the 4 transpose orientations
  breaks 0.75% of pairs — verified `prototypes/test_4orient.py` — and LIVE is 100% truth, so <100%
  is a non-shippable learning artifact);
- the N channels must be **separately thresholdable** (conv is linear → can't threshold-then-bit-
  pack into one channel) AND each orientation drives a **distinct downstream stamp shape** (so the
  stamp conv genuinely consumes N input channels). Bit-packing N orientations into one U8 fails on
  both counts.
- U8 is already the min dtype; H×W is the full active anchor grid.

So `N·H·W` bytes is irreducible. For task191 that's 8·23·23 = 4,232, plus a forced F32 input read
(`Slice` preserves the F32 input dtype → `[1,1,23,23]` = 2,116) and the 30×30 output mask (900).
Forced core ≈ 8,835; the baseline 11,169 sits ~1.5k above it in entangled runtime-kernel-build /
bbox overhead. No strict win found this session.

## What's missing from the library

- `by_symptom.md` QLinearConv section is scoped to constant kernels — it does not mention the
  runtime-extracted-template composition (Insight 1).
- No library entry warns that "F16 Conv forced" floor proofs are falsifiable by QLinearConv
  (Insight 2) — this directly caused a ~13k stale floor to stand in task191.md until now.

## Sorter note: by_symptom.md "geometric ray/diagonal drawing from anchor blocks" QLinearConv section — extend with "QLinearConv also works for RUNTIME-extracted variable templates (task191): bbox+Gather build the kernel at inference, then INT8 QLinearConv correlate; match map [1,N,H,W] is UINT8 not F16 (~2× saving)."
## Sorter note: traps.md §GRADER-GAP or a new §INT8-MATCH — add as new trap: "'F16 Conv forced → match map irreducible' is a FALSE floor; QLinearConv correlates in INT8 and halves [1,N,H,W] match maps + removes the F16 ConvTranspose paint. task191: a 24,807/26,917 'floor' was actually 11,169. Re-audit any template-match floor proof that assumes F16."
## Sorter note: techniques.md §54 (runtime variable-template extraction) — extend: "the extracted kernel can feed QLinearConv (INT8), not just F16 Conv, when template+data are integer-valued — task191 11,169."

## Insight 4 — algebra-preserving D4 kernel-build collapse: do per-element arithmetic on the SMALL base patch, BEFORE orientation expansion (task191 v9, 11,169 → 10,969, byte-equal)

When a graph builds N oriented conv kernels at runtime via "orient(template) then ±1 arithmetic on
the `[N,1,k,k]` concatenated stack" (task191: 8 D4 orientations, match kernel `2·markers − rect`,
fill kernel `rect − markers`), the arithmetic can often move ONTO the small `[1,1,k,k]` patches
*before* the Concat — because each orientation `T_k` is a linear permutation. This is
**algebra-preserving**: the resulting `[N,1,k,k]` weight is bit-identical, so a QLinearConv match map
and everything downstream is byte-equal and truth stays 100% **by construction** (verified
267/267 byte-equal + ab 200267/200267 + truth 200000/200000). It is graph surgery, NOT a new
algorithm — the reviewer's key point that I initially (wrongly) dismissed as "correctness risk."

Concretely task191's baseline materialized **6** `[8,1,5,5]`-class INT8 intermediates (markers `v284`,
`v_rect`, fill `v285`, match `v286`, flipped fill, transposed fill = 1,200 B); v9 collapses to **3**
(`v286` match `[8,1,5,5]`, `fill_ci` `[1,8,5,5]`, rot180 stamp `[1,8,5,5]` = 600 B) by computing
`fill_k = rect_k − B_k` and `v286_k = B_k − fill_k` per-orientation on `[1,1,5,5]` patches, then
Concat. Net **−200 B** profile (10,969).

CAVEAT (the floor): the saving is partial, not the naive "−600 from killing 4 big tensors". The
arithmetic must live somewhere — collapsing the `[N,1,k,k]` operands forces ~2N small `[1,1,k,k]`
subs (~400 B here), offsetting ~2/3 of the big-tensor win. And two consumers needing **different
weight layouts** (`[8,1,5,5]` match vs `[1,8,5,5]` rot180 stamp) force ≥2 final kernel tensors.

CAVEAT 2 (which sub-tensors orient): only the tensor the runtime flip-index was DESIGNED for
survives the flips. task191's `rev_idx_table` reverses a pattern-sized window; the pattern (`base`)
flips correctly but a solid box mask (`rect`) does NOT (reversing `[1,1,0,0,0]` by `[2,1,0,0,0]` →
`[0,1,1,1,1]`). So fold arithmetic that mixes `base` only; combine `rect` per-orientation. (v8 tried
to orient `rect` → 194/267 mismatch.)

## Sorter note: techniques.md (new rule near §54 / op_cost_patterns) — add "algebra-preserving runtime-kernel-build collapse: move ±1 kernel arithmetic onto the small [1,1,k,k] template patches BEFORE the orientation Concat (linear T_k commutes); bit-identical weights, byte-equal output. Saving is partial (per-orientation small subs offset ~2/3; ≥2 final tensors forced by differing match/stamp layouts). Only fold tensors the runtime flip-index fits — a solid box mask does not survive a pattern-sized reversal window. task191 v9: 11,169→10,969 byte-equal."

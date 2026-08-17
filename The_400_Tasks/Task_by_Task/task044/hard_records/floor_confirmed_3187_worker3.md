# task044 floor confirmed at 3187 (worker3, 2026-07-11)

**live bar:** 3187 bytes, sha `121a7c894f4f5b877d8dc979bfdb3952f9a3c5d8728e05ab28ebe0cc52c6fe2a`
(`ng_current_bar.py --task 44 --json`, manifest authoritative; owner Boladi; no Staged_Boladi sidecar).
Class: **byte-equal-to-current-live shave ladder** (Aditya 3398 -> worker 3298 -> worker2 3187, all ab byte_equal=True vs submission.zip). Ambiguous-class task (LIVE 99.879% truth); gold gate = byte-equal-vs-LIVE.

**Prompt framing was WRONG:** the card is "DepthToSpace block stamp, floor_est ~1200". Decoded reality (2026-06-07) = ARC `228f6490` box-hole shape-fill. `DepthToSpace` has **no ORT row** (`lib_query.py op DepthToSpace` -> "no ORT row") and is irrelevant. The ~1200 floor_est is a DepthToSpace-model artifact and is **provably impossible** for this rule (see forced components below).

## What worker3 tried (all graph-equivalent shaves of the 3187 graph; static cost model verified exact against base = mem 2791 + par 396 = 3187)

| variant | static cost | verdict |
|---|---|---|
| `seq_keyremove` (drop or_181; two sequential Where) | 3187 | **TIE** — key removal already minimal (4 planes -> 4 planes) |
| `arith_fill` (Mul/Add merged box-fill + one Where) | 3687 | WORSE (+500): nested-Where beats arithmetic fill |
| `exact_holes` (holeX_rect & isblack; drop gray-restore) | 3387 | WORSE (+200): rect-fill + gray-restore beats exact-hole detect |
| `onehot_then_pad` (Equal->onehot10 1000B then Pad) | 3390 | WORSE (+203): 900B idx-pad canvas beats 1000B onehot-pad |

Builders: `prototypes/probe_worker3_floor_shaves.py`, `prototypes/probe_worker3_canvas.py`;
outputs in `prototypes/worker3_floor_shaves/`. No candidate scored < 3187.

## Forced-component floor proof (why < ~3100 is a wall for this rule)

Every graph-equivalent transform ties-or-loses because the cost is dominated by structurally-forced tensors:

- **Output canvas = 900B FORCED.** Output is `[1,10,30,30]` BOOL (grader-fixed; ab byte-equal proves it). Producing it needs a 30x30-spatial operand. Cheapest = `idx30 [1,30,30]` UINT8 = 900B. The only alternative (`Equal->onehot10 [1,10,10,10]` 1000B then Pad) is +100B (empirically 3390). The free 30x30 input shape cannot help: 10x10 content cannot broadcast/expand to 30x30 (that is padding, not a 1->N broadcast).
- **Color-index F32 transient = 400B FORCED.** `conv_1 [1,10,10]` F32 = weighted-sum color index off the F32 input via Einsum; any linear op off an F32 input yields F32. The integer/ArgMax alternative requires casting the input to UINT8 first (`[1,10,30,30]` = 9000B) or a cropped one-hot (`[1,10,10,10]` = 4000B) — both far worse.
- **Color-index UINT8 = 100B FORCED** (`idx`; the fill runs in UINT8 — an F32 fill chain is 4x300B worse; dust preservation needs per-cell colors).
- **Crop selector = 300 params FORCED.** `Ssel [30,10]` maps the 30x30 free input down to 10x10 inside every Einsum, avoiding a materialized `[1,10,10,10]` F32 crop (4000B). It is a rank-10 identity submatrix (10 nonzeros, dense 300 elems); sparse-tensor Einsum operands are ORT-dead (`hard_records/sparse_einsum_initializers.md`); rank-10 cannot factor below 300. Anchored by conv_1/grow which need position-preserving crop, so removing it from the 3 signature Einsums saves nothing.
- **Gray mask = 100B FORCED** (`gray_bool`; preserves gray walls + drives box-column detection).

Forced subtotal = 900 + 400 + 100 + 300 + 100 = **1800**. The prompt's ~1200 est is below the canvas+index+crop sum (1600) alone.

Minimal logic on top (empirically at floor): box row/col detect ~285B, hole rects 200B, rank-2 pairwise signature+match ~110B + ~50 params, fill chain 700B (key-removal 300 + 4 Where 400), misc ~46 params. Sum ~= 1341 + 96 params. **Total floor ~= 3187 — the graph is AT it.**

## Remaining (low-probability) hypotheses for a future batch

1. **more-correct-than-LIVE clean rebuild** (numpy ref is 99.943% > LIVE 99.879%, cand_misses ⊆ LIVE_misses). A cleaner graph might trim LIVE's quirky detection (row0|row1 cols, 5/6 threshold, select_last asymmetry), but still pays the 1800 forced + a full box-detect/match/fill, and must re-prove ⊆ over 200k (expensive). Upside is marginal (< ~100B) and speculative — not obviously < 3187.
2. **rank-1 signature** (drop a pair_a/pair_b column, -20 params -> 3167): the signature `100N^2 + N*Sum(col^2) - (Sum col)^2` is a specific quadratic that factors as rank-2; rank-1 is a weaker signature that will collide and diverge from LIVE. Not pursued (poor ROU; would need ab --full to disprove a likely-divergent 20-byte gamble).

**Verdict: WALL at 3187.** No graph-equivalent shave exists; forced components put the floor at the current cost. Do not re-explore DepthToSpace, canvas-shrink, crop-selector-shrink, or fill-fusion — all closed above.

## Pass-2 (forced second pass, 2026-07-11): genuinely-different assembly + grader/library cross-check

Grader source read (`tools/ng_harness/modes.py` + `encode.py` + `session.py`): the model output is compared `np.array_equal(run_raw(out), encode_grid(expected))` where `encode_grid` = **`[1,10,30,30]` one-hot** and `run_raw` normalizes via `(out>0)`. So (a) the output shape `[1,10,30,30]` is grader-FIXED (canvas floor holds), and (b) only the output SIGN pattern matters (#69) — but that does not shrink the 900B canvas.

**Library laws that independently confirm the floor** (`cost_floor_reference.md`):
- "Single-channel color-index canvas = 900 B is the floor for a `[1,10,30,30]` output... **do NOT try to eliminate the canvas itself**." task388 v6 tried a smaller multi-channel BOOL output and scored WORSE — exactly our `onehot_then_pad` 3390 result.
- "Full-canvas Where-overlay tax... `[1,1,30,30]` BOOL = 900 B floor even when the active region is 10×10" (task341/102/129) → #4b/#48 overlay pays 900B PER cond.
- "Canvas+Equal BEATS ScatterND channel-patch" (task070) → channel-domain scatter pays F32 per patched channel.

**Genuinely-different formulation BUILT + MEASURED — #48 output-overlay** (`prototypes/probe_worker3_overlay.py`): use the free `[1,10,30,30]` input as passthrough, overlay edits with nested `Where(cond30, dyn_color_vec, base)`. Measured static **77,306** (mem 76,893 + par 413) — **24× WORSE**. Breakdown: the two intermediate `Where` outputs `ov1`,`ov2` are `[1,10,30,30]` F32 = **36,000 B each** (only the final `Where` is the free output); plus 3× `[1,1,30,30]` conds = 2,700 B. #4b/#48 is a SINGLE-recolor lever (one `Where` == the free output); this task has THREE dynamic-color edits (holeA→colorA, holeB→colorB, keys→black), so K−1=2 full 10-channel intermediates are unavoidable. Even BOOL intermediates → ~20,700 B (6.5× worse). **DEAD.**

**Other genuinely-different assemblies ruled out:** sign-arithmetic (#69) — Equal is already sign-based, no canvas shrink; negpad-Conv crop (#91/#96) — the shared `Ssel` (counted once across all 5 Einsums, scalar-output moment temps are scorer-invisible) beats a Conv that materializes a 400B plane per signature; channel-domain ScatterND — F32-per-channel tax (task070 law). All lose or are already applied. Floor holds at 3187.

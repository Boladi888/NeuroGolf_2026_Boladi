# task065 — archived history

Superseded floor proofs moved out of `task065.md` on 2026-06-22 (WORKER_PROTOCOL task-md-size rule).
Cost ladder: v7 canvas 1,999 → Aditya 687 formula → Aditya 641 U8 bit-code → **w-065 two-term Einsum 625 (current best)**.
The 625 break analysis and the current floor/component proof live in `task065.md`; the blocks below are the
now-obsolete 687-formula floor proof, retained for provenance only.

## Floor proof (Aditya formula class — 2026-06-12) [SUPERSEDED by 641, then 625]

Worker-065 exhaustively audited all micro-shave options for the Aditya 687 formula. None reduce cost below 687.

**Dominant cost surfaces (exact, from ORT_DISABLE_ALL profile trace on size=7 input):**

| Tensor | Size | Avoidable? |
|---|---:|---|
| row_terms [2,30] F16 | 120 B | No — [2,30] required for separable Einsum; smaller idx can't access full 30-dim input |
| col_terms [2,30] F16 | 120 B | No — same as row_terms |
| row_terms_b + col_terms_b [2,30] BOOL each | 120 B | No — Concat-then-Cast is provably cheaper than Cast-then-Concat |
| active_b + dot_row_b + dot_col_b [30] BOOL each | 90 B | No — boolean masks required for row/col term construction |
| counts [1,10] F32 | 40 B | No — must sum over H,W; input is F32 so output is F32 |
| dot_color [1,10] F32 | 40 B | No — position Einsum requires F32 (input is F32; ORT_DISABLE_ALL no type promotion) |
| color_stack [2,10] F16 | 40 B | No — required for final Einsum's color selection |
| color_stack_b [2,10] BOOL | 20 B | No — Concat-then-Cast cheaper |
| n-recovery scalars × 6 | 24 B | No — sqrt/sub/div chain unavoidable |
| idx [30] F32 (params) | 30 params | No — Range op would cost 120B memory vs 30 params (net +90 cost) |
| coeff_h [2,2] F16 + batch_h [1] + one + two (params) | 7 params | No — removing batch_h adds [10,30,30] F16=18kB intermediate (Einsum output no longer free); coeff_h needed for correction term |

**Structural alternatives killed:**

| Alternative | Kill reason |
|---|---|
| F16 dot_color (20B vs 40B) | ORT_DISABLE_ALL forbids implicit upcasts; mixed-type position Einsum fails |
| [1,30] row/col terms | Cannot express base×active×active + dot×dot_r×dot_c in one term; correction term requires t=2 |
| Combined [3,30] BOOL then extract | Extraction requires Slice+Concat intermediates → 720B vs current 450B |
| Remove batch_h (1 param) | Einsum output loses "output" name → [10,30,30] F16 = 18,000B intermediate (not free) |
| ConstantOfShape for one/two | Output [1,1] F32 = 4B memory added per replacement; saves 1 param → +3B net |
| Range for idx (30 params → 3) | Range output [30] F32 = 120B memory; net cost +90 |
| Gather approach (row then col) | [1,10,7,30] F32 = 8,400B at max; profiles at 8,400+ |
| Canvas approach (any variant) | Best canvas route (v7) profiles at 1,999 |
| Direct F32 Slice | Profiles at 2,323 |
| BitwiseAnd free-output | Cannot separate "NOT at dot position" from active region (joint r,w condition) |
| BOOL Einsum | ORT does not support BOOL inputs to Einsum op |

**Floor verdict (687): memory(650) + params(37) = 687 was the 687-formula-class floor.** (Broken to 641 by the U8 bit-code, then to 625 by the two-term Einsum — see task065.md.)

A sub-687 candidate requires a DIFFERENT formula that either (a) eliminates the [2,30] F16 row/col terms without creating larger intermediates, or (b) computes the quadrant crop without reading position from the input. Information-flow argument: the output quadrant depends on the input colors AND positions jointly; any algorithm must either (1) use a full [1,10,30,30] intermediate (≥9,000B) or (2) use the separable Einsum structure which floors at row_terms+col_terms=240B; sub-240B for these surfaces would require BOOL Einsum (not in ORT).

### Independent reconfirmation (2026-06-12 worker-065-1) — 687-formula floor; hard evidence (still valid sub-claims)

Re-audited every surface from scratch and added source-read / ORT-execution proof for the load-bearing claims:

1. **Output is hard-locked to [1,10,30,30] exact one-hot.** The gate (`tools/ng_harness/modes.py:mode_visible` + `session.run_raw`) compares `(model_out > 0)` against `encode.encode_grid(expected)` which is ALWAYS `[1,10,30,30]` (`tools/ng_harness/encode.py:11`), via `np.array_equal`. Any other output shape → `array_equal` False → 0/266. The decode-to-crop path (output `[1,10,7,7]`, idx7) scored **visible 0/266** — confirmed dead. Row/col terms MUST span 30; the active grid being ≤15×15 (generator `size∈1..7`, `task_2dc579da.py`) does NOT let the terms shrink.
2. **The 18,000 B intermediate is only avoided by the single free-output stacked Einsum.** Scorer reads `output` for free (`neurogolf_utils.calculate_memory:234 — if tensor_name in ['input','output']: continue`), but a 2-Einsum split needs to `Add` two `[1,10,30,30]` F16 planes = a charged 18,000 B intermediate. The single stacked free-output Einsum is structurally forced. (The 625 two-term build obeys this: it is ONE Einsum with a k=2 stacked operand.)
3. **F16 dot_color into the position Einsum: ORT-tested FAIL** — `Type parameter (T) of Einsum bound to different types (float and float16)`. Mixed-dtype Einsum is rejected under ORT_DISABLE_ALL. dot_color (40 B) + counts (40 B) are F32-forced because the input is F32.
4. **U8/I8 row/col terms to halve the 240 B: ORT-tested NOT_IMPLEMENTED** for the final Einsum. INT32 works but is 4 B (worse); **F16 (2 B) is the cheapest Einsum-supported dtype.** So 120 B per [30,2] F16 stack is the true minimum (still load-bearing for the 625 floor).
5. **Params is a local minimum under cost=memory+params.** Every reduction trades ≥1 param for ≥4 B memory (idx→Range: +120 B/−27 params net +93; scalar fold: +4 B/−1 param net +3). Net always positive.
